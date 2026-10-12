"""Подготовка пользовательского табличного источника к каноническому корпусу."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from LabsCommon.corpus import VALID_SPLITS, CorpusError, CorpusRow, validate_rows, write_canonical
from LabsCommon.dataset_adapter import PreparationError, load_adapter, run_adapter
from LabsCommon.dataset_config import DatasetConfigError, TableDatasetConfig, load_table_config


@dataclass(frozen=True, slots=True)
class PreparationSummary:
    """Краткий результат без технических деталей реализации."""

    input_path: Path
    config_path: Path
    output_path: Path
    input_rows: int
    output_rows: int
    dropped_rows: int
    class_counts: dict[str, int]
    split_counts: dict[str, int]
    source_name: str = "table"
    metadata: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source_name,
            "input": str(self.input_path),
            "config": str(self.config_path),
            "output": str(self.output_path),
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "dropped_rows": self.dropped_rows,
            "class_counts": self.class_counts,
            "split_counts": self.split_counts,
            "metadata": self.metadata or {},
        }


def resolve_config(input_path: Path, explicit: Path | None) -> Path:
    """Выбрать явный TOML либо одноимённый sidecar- файл."""
    if explicit is not None:
        return explicit
    candidate = input_path.with_suffix(".toml")
    if candidate.is_file():
        return candidate
    raise PreparationError(
        f"Для источника {input_path} не найден sidecar TOML {candidate}; укажите --config"
    )


def prepare_table(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    *,
    adapter_parameters: Mapping[str, object] | None = None,
) -> PreparationSummary:
    """Импортировать CSV/TSV, применив политики в зафиксированном порядке."""
    try:
        config = load_table_config(config_path)
        if config.adapter.path is not None:
            return _prepare_adapter(
                input_path, output_path, config_path, config, adapter_parameters
            )
        raw_rows = _read_source(input_path, config)
        rows, dropped = _map_rows(raw_rows, config)
        rows = _deduplicate(rows, config)
        rows = _assign_splits(rows, config)
        rows = _sample(rows, config)
        rows = validate_rows(rows)
        _check_expectations(rows, config)
        write_canonical(output_path, rows)
    except (CorpusError, DatasetConfigError) as error:
        raise PreparationError(str(error)) from error
    return PreparationSummary(
        input_path,
        config_path,
        output_path,
        len(raw_rows),
        len(rows),
        len(raw_rows) - len(rows),
        dict(sorted(Counter(row.label for row in rows if row.label is not None).items())),
        dict(sorted(Counter(row.split for row in rows if row.split is not None).items())),
    )


def _prepare_adapter(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    config: TableDatasetConfig,
    adapter_parameters: Mapping[str, object] | None,
) -> PreparationSummary:
    """Пропустить внешний результат через общий контракт и единую атомарную запись."""
    assert config.adapter.path is not None
    parameters = dict(config.adapter.parameters)
    parameters.update(adapter_parameters or {})
    result = run_adapter(load_adapter(config.adapter.path), input_path, parameters)
    try:
        if not all(isinstance(row, CorpusRow) for row in result.rows):
            raise PreparationError("Адаптер должен вернуть последовательность CorpusRow")
        rows = validate_rows(result.rows)
        _check_expectations(rows, config)
        write_canonical(output_path, rows)
        delete_after = result.metadata.get("delete_after_write")
        if isinstance(delete_after, str):
            try:
                Path(delete_after).unlink()
            except OSError as error:
                raise PreparationError(
                    f"Не удалось удалить архив после подготовки: {error}"
                ) from error
    except CorpusError as error:
        raise PreparationError(str(error)) from error
    return PreparationSummary(
        input_path,
        config_path,
        output_path,
        int(result.metadata.get("input_rows", len(rows))),
        len(rows),
        int(result.metadata.get("dropped_rows", 0)),
        dict(sorted(Counter(row.label for row in rows if row.label is not None).items())),
        dict(sorted(Counter(row.split for row in rows if row.split is not None).items())),
        source_name="adapter",
        metadata=dict(result.metadata),
    )


def _read_source(input_path: Path, config: TableDatasetConfig) -> list[dict[str, str]]:
    try:
        with input_path.open("r", encoding=config.dataset.encoding, newline="") as stream:
            reader = csv.DictReader(stream, delimiter=config.dataset.delimiter, strict=True)
            if reader.fieldnames is None:
                raise PreparationError("Источник не содержит заголовка")
            required = set(config.columns.text)
            for name in (config.columns.identifier, config.columns.label, config.columns.split):
                if name is not None:
                    required.add(name)
            missing = sorted(required - set(reader.fieldnames))
            if missing:
                raise PreparationError(f"В источнике отсутствуют столбцы: {', '.join(missing)}")
            result: list[dict[str, str]] = []
            for row in reader:
                if None in row:
                    raise PreparationError("Источник содержит строку с лишними полями")
                result.append({name: value or "" for name, value in row.items()})
            return result
    except (OSError, UnicodeError, csv.Error) as error:
        raise PreparationError(f"Не удалось прочитать источник {input_path}: {error}") from error


def _map_rows(
    raw_rows: Iterable[dict[str, str]], config: TableDatasetConfig
) -> tuple[list[CorpusRow], int]:
    result: list[CorpusRow] = []
    dropped = 0
    for source in raw_rows:

        def clean(value: str) -> str:
            return value.strip() if config.cleaning.trim else value

        text_parts = [clean(source[name]) for name in config.columns.text]
        text = config.columns.joiner.join(text_parts)
        identifier = clean(source[config.columns.identifier]) if config.columns.identifier else ""
        label = clean(source[config.columns.label]) if config.columns.label else None
        split = clean(source[config.columns.split]) if config.columns.split else None
        if label is not None:
            label = config.cleaning.labels.get(label, label)
        required = [text if any(text_parts) else ""]
        if config.columns.identifier is not None:
            required.append(identifier)
        if label is not None:
            required.append(label)
        if split is not None:
            required.append(split)
        if any(not value for value in required):
            if config.cleaning.empty == "drop":
                dropped += 1
                continue
            raise PreparationError("После очистки источник содержит пустое обязательное значение")
        if not identifier:
            identifier = f"sha256:{_digest(text)}"
        if split is not None and split not in VALID_SPLITS:
            raise PreparationError(f"Недопустимое значение split: {split}")
        result.append(CorpusRow(identifier, text, label, split))
    return result, dropped


def _deduplicate(rows: list[CorpusRow], config: TableDatasetConfig) -> list[CorpusRow]:
    grouped: dict[str, list[CorpusRow]] = defaultdict(list)
    for row in rows:
        grouped[row.text].append(row)
    result: list[CorpusRow] = []
    for text, group in grouped.items():
        if len(group) == 1:
            result.extend(group)
            continue
        labels = {row.label for row in group}
        if len(labels) > 1:
            if config.cleaning.conflicts == "error":
                raise PreparationError(f"Конфликтующие метки у дубликата текста: {text!r}")
            counts = Counter(row.label for row in group)
            top = max(counts.values())
            winners = sorted(label for label, count in counts.items() if count == top)
            if len(winners) > 1:
                if config.cleaning.tie == "drop":
                    continue
                raise PreparationError(f"Ничья конфликтующих меток у текста: {text!r}")
            selected = min(
                (row for row in group if row.label == winners[0]), key=lambda row: row.identifier
            )
            result.append(selected)
            continue
        if config.cleaning.duplicates == "error":
            raise PreparationError(f"Повторный текст в источнике: {text!r}")
        if config.cleaning.duplicates == "drop":
            continue
        result.append(min(group, key=lambda row: row.identifier))
    return result


def _assign_splits(rows: list[CorpusRow], config: TableDatasetConfig) -> list[CorpusRow]:
    if not config.split.enabled:
        return rows
    if any(row.split is not None for row in rows):
        raise PreparationError("Нельзя одновременно задать столбец split и [split].enabled")
    by_label: dict[str, list[CorpusRow]] = defaultdict(list)
    for row in rows:
        if row.label is None:
            raise PreparationError("Стратифицированное разбиение требует меток")
        by_label[row.label].append(row)
    result: list[CorpusRow] = []
    portions = (
        ("train", config.split.train),
        ("validation", config.split.validation),
        ("test", config.split.test),
    )
    for label, group in sorted(by_label.items()):
        ordered = sorted(group, key=lambda row: (_digest(row.identifier), row.identifier))
        random.Random(f"{config.split.random_state}:{label}").shuffle(ordered)
        assigned = _allocate(len(ordered), portions)
        result.extend(
            CorpusRow(row.identifier, row.text, row.label, split)
            for row, split in zip(ordered, assigned, strict=True)
        )
    return result


def _allocate(size: int, portions: tuple[tuple[str, float], ...]) -> list[str]:
    targets = [(name, size * portion) for name, portion in portions]
    counts = {name: int(value) for name, value in targets}
    left = size - sum(counts.values())
    for name, _ in sorted(
        targets, key=lambda item: (item[1] - int(item[1]), item[0]), reverse=True
    )[:left]:
        counts[name] += 1
    return [name for name, _ in portions for _ in range(counts[name])]


def _sample(rows: list[CorpusRow], config: TableDatasetConfig) -> list[CorpusRow]:
    if config.sampling.per_class is None and config.sampling.per_split is None:
        return rows
    groups: dict[tuple[str, str], list[CorpusRow]] = defaultdict(list)
    for row in rows:
        groups[(row.label or "", row.split or "")].append(row)
    result: list[CorpusRow] = []
    for _key, group in groups.items():
        limit = config.sampling.per_class
        if config.sampling.per_split is not None:
            split_limit = config.sampling.per_split
            if isinstance(split_limit, Mapping):
                split_limit = split_limit.get(group[0].split or "", 0)
            limit = min(limit, split_limit) if limit is not None else split_limit
        result.extend(
            sorted(group, key=lambda row: (_digest(row.identifier), row.identifier))[:limit]
        )
    return result


def _check_expectations(rows: list[CorpusRow], config: TableDatasetConfig) -> None:
    if config.expect.rows is not None and len(rows) != config.expect.rows:
        raise PreparationError(f"Ожидалось строк: {config.expect.rows}, получено: {len(rows)}")
    classes = {row.label for row in rows if row.label is not None}
    if config.expect.classes is not None and len(classes) != config.expect.classes:
        raise PreparationError(
            f"Ожидалось классов: {config.expect.classes}, получено: {len(classes)}"
        )
    counts = Counter(row.split for row in rows if row.split is not None)
    if config.expect.splits:
        portions = tuple(
            (name, config.expect.splits.get(name, 0) / 100)
            for name in ("train", "validation", "test")
        )
        expected_counts = Counter(_allocate(len(rows), portions))
        for split, _ in portions:
            if counts[split] == expected_counts[split]:
                continue
            raise PreparationError(
                f"Ожидалось строк {split}: {expected_counts[split]} "
                f"({config.expect.splits.get(split, 0):g}%), получено: {counts[split]}"
            )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
