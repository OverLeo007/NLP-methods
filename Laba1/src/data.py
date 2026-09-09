"""Подготовка и структурная проверка поставляемого корпуса RuReviews."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE: Final = 42
SOURCE_COLUMNS: Final = ("review", "sentiment")
SOURCE_LABELS: Final = ("negative", "neautral", "positive")
PUBLIC_LABELS: Final = ("negative", "neutral", "positive")
EXPECTED_SOURCE_ROWS: Final = 90_000
EXPECTED_CLEAN_ROWS: Final = 86_913
EXPECTED_CLASS_COUNTS: Final = {"negative": 28_887, "neutral": 28_768, "positive": 29_258}
EXPECTED_SPLIT_SIZES: Final = {"train": 60_839, "validation": 13_037, "test": 13_037}
EXPECTED_TIES: Final = 180
EXPECTED_RESOLVED_CONFLICTS: Final = 215


class DataValidationError(ValueError):
    """Ошибка нарушения контракта загруженного файла или табличных данных."""


@dataclass(frozen=True)
class CleaningStats:
    input_rows: int
    empty_reviews_removed: int
    tied_texts_removed: int
    conflicting_texts_resolved: int
    output_rows: int
    class_counts: dict[str, int]


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class PreparedDataset:
    cleaned: pd.DataFrame
    split: DatasetSplit
    cleaning: CleaningStats


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame) -> None:
    actual = tuple(frame.columns)
    if actual != SOURCE_COLUMNS:
        raise DataValidationError(
            f"Неверные столбцы корпуса: получено {actual}, ожидалось {SOURCE_COLUMNS}"
        )


def load_source(path: Path) -> pd.DataFrame:
    """Прочитать TSV и проверить его структурный контракт."""
    try:
        frame = pd.read_csv(path, sep="\t", encoding="utf-8", dtype="string")
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise DataValidationError(f"Не удалось прочитать TSV корпуса: {error}") from error
    _require_columns(frame)
    if frame[list(SOURCE_COLUMNS)].isna().any(axis=None):
        raise DataValidationError("Корпус содержит пропуски в review или sentiment")
    unknown_labels = sorted(set(frame["sentiment"]) - set(SOURCE_LABELS) - {"neutral"})
    if unknown_labels:
        raise DataValidationError(f"Корпус содержит неизвестные метки: {unknown_labels}")
    return frame


def clean_reviews(source: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """Применить правила обработки точных дублей и конфликтующих меток."""
    _require_columns(source)
    frame = source.loc[:, list(SOURCE_COLUMNS)].copy()
    if frame["sentiment"].isna().any():
        raise DataValidationError("Корпус содержит пропуски в sentiment")
    unknown_labels = sorted(set(frame["sentiment"]) - set(SOURCE_LABELS) - {"neutral"})
    if unknown_labels:
        raise DataValidationError(f"Корпус содержит неизвестные метки: {unknown_labels}")

    input_rows = len(frame)
    frame["review"] = frame["review"].fillna("").astype("string").str.strip()
    nonempty = frame["review"].str.len().gt(0)
    empty_reviews_removed = int((~nonempty).sum())
    frame = frame.loc[nonempty].copy()
    frame["sentiment"] = frame["sentiment"].replace({"neautral": "neutral"})

    counts = (
        frame.groupby(["review", "sentiment"], sort=True, observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    rows: list[tuple[str, str]] = []
    tied_texts_removed = 0
    conflicting_texts_resolved = 0
    for review, variants in counts.groupby("review", sort=True, observed=True):
        maximum = int(variants["count"].max())
        winners = variants.loc[variants["count"].eq(maximum), "sentiment"].tolist()
        if len(winners) != 1:
            tied_texts_removed += 1
            continue
        if len(variants) > 1:
            conflicting_texts_resolved += 1
        rows.append((cast(str, review), cast(str, winners[0])))

    cleaned = pd.DataFrame(rows, columns=list(SOURCE_COLUMNS))
    cleaned = cleaned.sort_values(list(SOURCE_COLUMNS), kind="stable").reset_index(drop=True)
    class_counts = {label: int((cleaned["sentiment"] == label).sum()) for label in PUBLIC_LABELS}
    stats = CleaningStats(
        input_rows=input_rows,
        empty_reviews_removed=empty_reviews_removed,
        tied_texts_removed=tied_texts_removed,
        conflicting_texts_resolved=conflicting_texts_resolved,
        output_rows=len(cleaned),
        class_counts=class_counts,
    )
    return cleaned, stats


def split_reviews(cleaned: pd.DataFrame, *, random_state: int = RANDOM_STATE) -> DatasetSplit:
    """Создать детерминированные стратифицированные части 70/15/15 без общих текстов."""
    _require_columns(cleaned)
    if cleaned["review"].duplicated().any():
        raise DataValidationError("Перед разбиением тексты должны быть уникальными")
    if cleaned.empty:
        raise DataValidationError("Нельзя разбить пустой корпус")
    unknown_labels = sorted(set(cleaned["sentiment"]) - set(PUBLIC_LABELS))
    if unknown_labels:
        raise DataValidationError(f"Корпус содержит неизвестные публичные метки: {unknown_labels}")

    try:
        train, temporary = train_test_split(
            cleaned,
            test_size=0.30,
            stratify=cleaned["sentiment"],
            random_state=random_state,
        )
        validation, test = train_test_split(
            temporary,
            test_size=0.50,
            stratify=temporary["sentiment"],
            random_state=random_state,
        )
    except ValueError as error:
        raise DataValidationError(
            f"Не удалось выполнить стратифицированное разбиение: {error}"
        ) from error

    result = DatasetSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
    train_reviews = set(result.train["review"].astype(str))
    validation_reviews = set(result.validation["review"].astype(str))
    test_reviews = set(result.test["review"].astype(str))
    if (
        train_reviews & validation_reviews
        or train_reviews & test_reviews
        or validation_reviews & test_reviews
    ):
        raise DataValidationError("Обнаружено пересечение текстов между частями корпуса")
    return result


def prepare_rureviews(path: Path) -> PreparedDataset:
    """Подготовить поставляемый корпус RuReviews и проверить его структуру."""
    source = load_source(path)
    source_counts = source["sentiment"].value_counts().to_dict()
    expected_source_counts = {label: 30_000 for label in SOURCE_LABELS}
    if len(source) != EXPECTED_SOURCE_ROWS or source_counts != expected_source_counts:
        raise DataValidationError(
            "Исходный корпус не соответствует ожидаемым 90 000 строкам "
            "(по 30 000 строк каждого класса)"
        )

    cleaned, cleaning = clean_reviews(source)
    if (
        cleaning.output_rows != EXPECTED_CLEAN_ROWS
        or cleaning.class_counts != EXPECTED_CLASS_COUNTS
        or cleaning.tied_texts_removed != EXPECTED_TIES
        or cleaning.conflicting_texts_resolved != EXPECTED_RESOLVED_CONFLICTS
    ):
        raise DataValidationError(f"Контрольные числа очистки не совпали: {cleaning}")

    split = split_reviews(cleaned)
    sizes = {
        "train": len(split.train),
        "validation": len(split.validation),
        "test": len(split.test),
    }
    if sizes != EXPECTED_SPLIT_SIZES:
        raise DataValidationError(f"Контрольные размеры разбиения не совпали: {sizes}")
    return PreparedDataset(cleaned=cleaned, split=split, cleaning=cleaning)
