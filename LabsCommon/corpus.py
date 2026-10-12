"""Канонический CSV-контракт корпуса, общий для лабораторных работ."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

CANONICAL_REQUIRED_COLUMNS = ("id", "text")
CANONICAL_OPTIONAL_COLUMNS = ("label", "split")
CANONICAL_COLUMNS = CANONICAL_REQUIRED_COLUMNS + CANONICAL_OPTIONAL_COLUMNS
VALID_SPLITS = frozenset(("train", "validation", "test"))


class CorpusError(ValueError):
    """Нарушение контракта канонического корпуса."""


@dataclass(frozen=True, slots=True)
class CorpusRow:
    """Одна строка канонического корпуса без неявных преобразований."""

    identifier: str
    text: str
    label: str | None = None
    split: str | None = None

    def as_dict(self) -> dict[str, str]:
        """Вернуть совместимое с CSV представление строки."""
        return {
            "id": self.identifier,
            "text": self.text,
            "label": self.label or "",
            "split": self.split or "",
        }


def canonical_columns(rows: Sequence[CorpusRow]) -> tuple[str, ...]:
    """Определить минимальный допустимый набор столбцов для строк."""
    has_label = any(row.label is not None for row in rows)
    has_split = any(row.split is not None for row in rows)
    columns = list(CANONICAL_REQUIRED_COLUMNS)
    if has_label:
        columns.append("label")
    if has_split:
        columns.append("split")
    return tuple(columns)


def validate_rows(rows: Iterable[CorpusRow]) -> list[CorpusRow]:
    """Проверить целиком контракт и вернуть строки в исходном порядке."""
    result = list(rows)
    identifiers: set[str] = set()
    texts: set[str] = set()
    has_label = any(row.label is not None for row in result)
    has_split = any(row.split is not None for row in result)
    for row in result:
        if not row.identifier:
            raise CorpusError("Канонический CSV содержит пустой id")
        if not row.text:
            raise CorpusError("Канонический CSV содержит пустой text")
        if row.identifier in identifiers:
            raise CorpusError("Канонический CSV содержит повторный id")
        if row.text in texts:
            raise CorpusError("Канонический CSV содержит повторный text")
        if has_label and not row.label:
            raise CorpusError("Канонический CSV содержит пустую метку label")
        if has_split and row.split not in VALID_SPLITS:
            raise CorpusError("Канонический CSV содержит недопустимое split")
        identifiers.add(row.identifier)
        texts.add(row.text)
    return result


def rows_from_mappings(rows: Iterable[Mapping[str, str | None]]) -> list[CorpusRow]:
    """Преобразовать CSV-строки в типизированный канонический контракт."""
    converted: list[CorpusRow] = []
    for row in rows:
        converted.append(
            CorpusRow(
                identifier=row.get("id") or "",
                text=row.get("text") or "",
                label=row.get("label") if "label" in row else None,
                split=row.get("split") if "split" in row else None,
            )
        )
    return validate_rows(converted)


def read_canonical_rows(path: Path) -> list[CorpusRow]:
    """Прочитать UTF-8 канонический CSV с проверкой заголовка и строк."""
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            columns = tuple(reader.fieldnames or ())
            if columns not in {
                CANONICAL_REQUIRED_COLUMNS,
                CANONICAL_REQUIRED_COLUMNS + ("label",),
                CANONICAL_REQUIRED_COLUMNS + ("split",),
                CANONICAL_COLUMNS,
            }:
                raise CorpusError(f"Неверные столбцы канонического CSV: {columns}")
            mappings: list[dict[str, str | None]] = []
            for row in reader:
                if None in row:
                    raise CorpusError("Канонический CSV содержит строку с лишними полями")
                mappings.append({name: row.get(name) for name in columns})
    except (OSError, UnicodeError, csv.Error) as error:
        raise CorpusError(f"Не удалось прочитать канонический CSV {path}: {error}") from error
    return rows_from_mappings(mappings)


def read_canonical(path: Path) -> Iterator[dict[str, str]]:
    """Вернуть строки в историческом словарном интерфейсе Laba2."""
    yield from (row.as_dict() for row in read_canonical_rows(path))


def write_canonical(path: Path, rows: Iterable[CorpusRow]) -> tuple[str, ...]:
    """Атомарно записать отсортированный UTF-8 CSV без BOM и частичного результата."""
    verified = validate_rows(rows)
    columns = canonical_columns(verified)
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in sorted(verified, key=lambda item: item.identifier):
                source = row.as_dict()
                writer.writerow({name: source[name] for name in columns})
        os.replace(temporary, path)
        temporary = None
    except (OSError, csv.Error) as error:
        raise CorpusError(f"Не удалось записать канонический CSV {path}: {error}") from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
    return columns
