"""Строгая неизменяемая TOML-конфигурация табличного корпуса."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Final


class DatasetConfigError(ValueError):
    """Ошибка схемы TOML, обнаруженная до обработки источника."""


def _freeze(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(values))


def _table(
    source: Mapping[str, object], name: str, *, required: bool = False
) -> Mapping[str, object]:
    value = source.get(name)
    if value is None:
        if required:
            raise DatasetConfigError(f"Отсутствует обязательная таблица [{name}]")
        return {}
    if not isinstance(value, dict):
        raise DatasetConfigError(f"[{name}] должен быть таблицей")
    return value


def _unknown(values: Mapping[str, object], allowed: set[str], name: str) -> None:
    extras = sorted(set(values) - allowed)
    if extras:
        raise DatasetConfigError(f"Неизвестные ключи в [{name}]: {', '.join(extras)}")


def _string(
    values: Mapping[str, object], name: str, table: str, *, required: bool = False
) -> str | None:
    value = values.get(name)
    if value is None:
        if required:
            raise DatasetConfigError(f"Отсутствует [{table}].{name}")
        return None
    if not isinstance(value, str) or not value:
        raise DatasetConfigError(f"[{table}].{name} должен быть непустой строкой")
    return value


def _integer(
    values: Mapping[str, object], name: str, table: str, *, minimum: int = 0
) -> int | None:
    value = values.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DatasetConfigError(f"[{table}].{name} должен быть целым числом не меньше {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    format: str
    encoding: str
    delimiter: str


@dataclass(frozen=True, slots=True)
class ColumnSettings:
    text: tuple[str, ...]
    identifier: str | None = None
    label: str | None = None
    split: str | None = None
    joiner: str = " "


@dataclass(frozen=True, slots=True)
class CleaningSettings:
    trim: bool = True
    empty: str = "error"
    labels: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    duplicates: str = "error"
    conflicts: str = "error"
    tie: str = "error"


@dataclass(frozen=True, slots=True)
class SplitSettings:
    enabled: bool = False
    train: float = 0.8
    validation: float = 0.0
    test: float = 0.2
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class SamplingSettings:
    per_class: int | None = None
    per_split: int | Mapping[str, int] | None = None


@dataclass(frozen=True, slots=True)
class ExpectSettings:
    rows: int | None = None
    classes: int | None = None
    splits: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class AdapterSettings:
    path: str | None = None
    parameters: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class TableDatasetConfig:
    dataset: DatasetSettings
    columns: ColumnSettings
    cleaning: CleaningSettings
    split: SplitSettings
    sampling: SamplingSettings
    expect: ExpectSettings
    adapter: AdapterSettings


TOP_LEVEL: Final = {"dataset", "columns", "cleaning", "split", "sampling", "expect", "adapter"}


def load_table_config(path: Path) -> TableDatasetConfig:
    """Загрузить TOML и отклонить неизвестные или несовместимые настройки."""
    try:
        with path.open("rb") as stream:
            source = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DatasetConfigError(f"Не удалось прочитать TOML {path}: {error}") from error
    _unknown(source, TOP_LEVEL, "корень")
    dataset = _parse_dataset(_table(source, "dataset", required=True))
    columns = _parse_columns(_table(source, "columns", required=True))
    cleaning = _parse_cleaning(_table(source, "cleaning"))
    split = _parse_split(_table(source, "split"))
    sampling = _parse_sampling(_table(source, "sampling"))
    expect = _parse_expect(_table(source, "expect"))
    adapter = _parse_adapter(_table(source, "adapter"))
    if split.enabled and columns.label is None:
        raise DatasetConfigError("[split].enabled требует [columns].label")
    if sampling.per_class is not None and columns.label is None:
        raise DatasetConfigError("[sampling].per_class требует [columns].label")
    if sampling.per_split is not None and not split.enabled and columns.split is None:
        raise DatasetConfigError("[sampling].per_split требует split")
    return TableDatasetConfig(dataset, columns, cleaning, split, sampling, expect, adapter)


def _parse_adapter(values: Mapping[str, object]) -> AdapterSettings:
    _unknown(values, {"path", "parameters"}, "adapter")
    path = _string(values, "path", "adapter")
    parameters = values.get("parameters", {})
    if not isinstance(parameters, dict) or not all(isinstance(key, str) for key in parameters):
        raise DatasetConfigError("[adapter].parameters должен быть таблицей")
    return AdapterSettings(path, _freeze(parameters))


def _parse_dataset(values: Mapping[str, object]) -> DatasetSettings:
    _unknown(values, {"format", "encoding", "delimiter"}, "dataset")
    format_name = _string(values, "format", "dataset", required=True)
    if format_name not in {"csv", "tsv"}:
        raise DatasetConfigError("[dataset].format должен быть csv или tsv")
    encoding = _string(values, "encoding", "dataset") or "utf-8"
    delimiter = _string(values, "delimiter", "dataset") or ("," if format_name == "csv" else "\t")
    if len(delimiter) != 1:
        raise DatasetConfigError("[dataset].delimiter должен состоять из одного символа")
    return DatasetSettings(format_name, encoding, delimiter)


def _parse_columns(values: Mapping[str, object]) -> ColumnSettings:
    _unknown(values, {"id", "text", "label", "split", "joiner"}, "columns")
    text = values.get("text")
    if isinstance(text, str):
        fields = (text,)
    elif isinstance(text, list) and text and all(isinstance(item, str) and item for item in text):
        fields = tuple(text)
    else:
        raise DatasetConfigError("[columns].text должен быть непустой строкой или массивом строк")
    if len(set(fields)) != len(fields):
        raise DatasetConfigError("[columns].text не должен содержать повторов")
    identifier = _string(values, "id", "columns")
    label = _string(values, "label", "columns")
    split = _string(values, "split", "columns")
    joiner = _string(values, "joiner", "columns")
    return ColumnSettings(fields, identifier, label, split, " " if joiner is None else joiner)


def _parse_cleaning(values: Mapping[str, object]) -> CleaningSettings:
    _unknown(values, {"trim", "empty", "labels", "duplicates", "conflicts", "tie"}, "cleaning")
    trim = values.get("trim", True)
    if not isinstance(trim, bool):
        raise DatasetConfigError("[cleaning].trim должен быть true или false")
    empty = _string(values, "empty", "cleaning") or "error"
    duplicates = _string(values, "duplicates", "cleaning") or "error"
    conflicts = _string(values, "conflicts", "cleaning") or "error"
    tie = _string(values, "tie", "cleaning") or "error"
    if empty not in {"error", "drop"} or duplicates not in {"error", "drop", "keep"}:
        raise DatasetConfigError("Политика пустых значений или дублей недопустима")
    if conflicts not in {"error", "majority"} or tie not in {"error", "drop"}:
        raise DatasetConfigError("Политика конфликтующих меток недопустима")
    labels = values.get("labels", {})
    if not isinstance(labels, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in labels.items()
    ):
        raise DatasetConfigError("[cleaning].labels должен быть таблицей строковых отображений")
    return CleaningSettings(trim, empty, _freeze(labels), duplicates, conflicts, tie)


def _parse_split(values: Mapping[str, object]) -> SplitSettings:
    _unknown(values, {"enabled", "train", "validation", "test", "random_state"}, "split")
    enabled = values.get("enabled", False)
    if not isinstance(enabled, bool):
        raise DatasetConfigError("[split].enabled должен быть true или false")
    percentages = _percentages(
        {
            name: values.get(name, default)
            for name, default in (("train", 80), ("validation", 0), ("test", 20))
        },
        "split",
    )
    random_state = _integer(values, "random_state", "split", minimum=0)
    return SplitSettings(
        enabled,
        **{name: value / 100 for name, value in percentages.items()},
        random_state=42 if random_state is None else random_state,
    )


def _percentages(values: Mapping[str, object], table: str) -> dict[str, float]:
    """Проверить проценты частей, включая допустимые имена и сумму 100."""
    _unknown(values, {"train", "validation", "test"}, table)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in values.values()
    ):
        raise DatasetConfigError(f"Проценты [{table}] должны быть числами")
    numbers = {name: float(value) for name, value in values.items()}
    if not all(isfinite(value) for value in numbers.values()):
        raise DatasetConfigError(f"Проценты [{table}] должны быть конечными числами")
    if any(value < 0 for value in numbers.values()) or abs(sum(numbers.values()) - 100) > 1e-9:
        raise DatasetConfigError(
            f"Проценты [{table}] должны быть неотрицательными и давать сумму 100"
        )
    return numbers


def _parse_sampling(values: Mapping[str, object]) -> SamplingSettings:
    _unknown(values, {"per_class", "per_split"}, "sampling")
    per_split = values.get("per_split")
    if isinstance(per_split, dict):
        if set(per_split) - {"train", "validation", "test"} or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in per_split.values()
        ):
            raise DatasetConfigError(
                "[sampling].per_split должен содержать допустимые размеры частей"
            )
        per_split = _freeze(per_split)
    elif per_split is not None:
        per_split = _integer(values, "per_split", "sampling", minimum=1)
    return SamplingSettings(
        _integer(values, "per_class", "sampling", minimum=1),
        per_split,
    )


def _parse_expect(values: Mapping[str, object]) -> ExpectSettings:
    _unknown(values, {"rows", "classes", "splits"}, "expect")
    splits = values.get("splits", {})
    if not isinstance(splits, dict):
        raise DatasetConfigError("[expect].splits должен быть таблицей процентов")
    return ExpectSettings(
        _integer(values, "rows", "expect", minimum=0),
        _integer(values, "classes", "expect", minimum=0),
        _freeze(_percentages(splits, "expect.splits") if splits else {}),
    )
