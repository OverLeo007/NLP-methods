"""Обучение фиксированной конфигурации и итоговая оценка модели."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Final, cast

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data import PUBLIC_LABELS
from .features import normalize_reviews
from .model import FittedModel, ModelPackageError, fit_model
from .normalization import NormalizationConfig

SELECTED_CONFIG: Final = NormalizationConfig(
    method="stem", remove_stopwords=True, use_synonyms=False
)


class ExperimentError(ValueError):
    """Ошибка несовместимости данных с обучением или оценкой."""


@dataclass(frozen=True)
class TrainingResult:
    """Результат единственного обучения документированной конфигурации."""

    model: FittedModel
    training_seconds: float


@dataclass(frozen=True)
class TestEvaluation:
    """Итоговая оценка фиксированной модели на test-части."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    classification_report: dict[str, object]
    confusion_matrix: tuple[tuple[int, ...], ...]
    predictions: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "classification_report": self.classification_report,
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "macro_f1": self.macro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
        }


def configuration_id(config: NormalizationConfig) -> str:
    return (
        f"method={config.method};stopwords={'on' if config.remove_stopwords else 'off'};"
        f"synonyms={'on' if config.use_synonyms else 'off'}"
    )


def _partition_columns(frame: pd.DataFrame, name: str) -> tuple[list[str], list[str]]:
    if {"review", "sentiment"} - set(frame.columns):
        raise ExperimentError(f"В {name} отсутствуют столбцы review и sentiment")
    reviews = frame["review"].astype(str).tolist()
    labels = frame["sentiment"].astype(str).tolist()
    if not reviews:
        raise ExperimentError(f"{name} не может быть пустой")
    return reviews, labels


def train_selected_model(train: pd.DataFrame) -> TrainingResult:
    """Один раз обучить заранее выбранную конфигурацию на обучающей части."""
    train_reviews, train_labels = _partition_columns(train, "обучающей выборке")
    started = perf_counter()
    try:
        model = fit_model(train_reviews, train_labels, SELECTED_CONFIG)
    except ModelPackageError as error:
        raise ExperimentError(f"Модель не обучилась: {error}") from error
    return TrainingResult(model=model, training_seconds=perf_counter() - started)


def evaluate_selected_model(model: FittedModel, test: pd.DataFrame) -> TestEvaluation:
    """Оценить фиксированную модель на отложенной test-части."""
    test_reviews, test_labels = _partition_columns(test, "test-выборке")
    predictions = model.classifier.predict(
        model.vectorizer.transform(normalize_reviews(test_reviews, model.config))
    )
    report = cast(
        dict[str, object],
        classification_report(
            test_labels,
            predictions,
            labels=PUBLIC_LABELS,
            output_dict=True,
            zero_division=0,
        ),
    )
    macro = report["macro avg"]
    if not isinstance(macro, dict):
        raise ExperimentError("Не удалось получить macro-метрики test-выборки")
    matrix = confusion_matrix(test_labels, predictions, labels=PUBLIC_LABELS)
    return TestEvaluation(
        accuracy=float(accuracy_score(test_labels, predictions)),
        macro_precision=float(macro["precision"]),
        macro_recall=float(macro["recall"]),
        macro_f1=float(macro["f1-score"]),
        classification_report=report,
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix.tolist()),
        predictions=tuple(str(label) for label in predictions),
    )
