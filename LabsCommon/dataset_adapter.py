"""Явный публичный контракт нестандартной подготовки корпуса."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .corpus import CorpusRow


class PreparationError(ValueError):
    """Ожидаемая ошибка подготовки, пригодная для показа в CLI."""


@dataclass(frozen=True, slots=True)
class AdapterResult:
    """Детерминированный результат адаптера до общей валидации и записи."""

    rows: Sequence[CorpusRow]
    metadata: Mapping[str, object]


DatasetAdapter = Callable[[Path, Mapping[str, object]], AdapterResult]


def load_adapter(import_path: str) -> DatasetAdapter:
    """Загрузить ровно один явно названный вызываемый объект ``module:object``."""
    if import_path.count(":") != 1:
        raise PreparationError("Путь адаптера должен иметь вид module:object")
    module_name, object_name = import_path.split(":")
    if not module_name or not object_name:
        raise PreparationError("Путь адаптера должен иметь вид module:object")
    try:
        module = importlib.import_module(module_name)
        adapter = getattr(module, object_name)
    except (ImportError, AttributeError) as error:
        raise PreparationError(f"Не удалось загрузить адаптер {import_path}: {error}") from error
    if not callable(adapter):
        raise PreparationError(f"Объект адаптера {import_path} должен быть вызываемым")
    return adapter


def run_adapter(
    adapter: DatasetAdapter, source: Path, parameters: Mapping[str, object]
) -> AdapterResult:
    """Вызвать адаптер с неизменяемыми параметрами и проверить тип результата."""
    try:
        result = adapter(source, MappingProxyType(dict(parameters)))
    except PreparationError:
        raise
    except Exception as error:
        raise PreparationError(f"Адаптер завершился с ошибкой: {error}") from error
    if not isinstance(result, AdapterResult):
        raise PreparationError("Адаптер должен вернуть AdapterResult")
    if not isinstance(result.metadata, Mapping):
        raise PreparationError("Метаданные AdapterResult должны быть таблицей")
    return result
