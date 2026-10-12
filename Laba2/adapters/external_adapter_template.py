"""Минимальный шаблон внешнего адаптера для ``module:object`` в TOML."""

from collections.abc import Mapping
from pathlib import Path

from LabsCommon.corpus import CorpusRow
from LabsCommon.dataset_adapter import AdapterResult


def prepare(source: Path, parameters: Mapping[str, object]) -> AdapterResult:
    """Вернуть канонические строки; параметры доступны только для чтения."""
    del source, parameters
    return AdapterResult((CorpusRow("example-1", "Пример текста", "example", "train"),), {})
