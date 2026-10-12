"""Потоковая подготовка фиксированного поднабора Lenta.ru."""

from __future__ import annotations

import bz2
import csv
import hashlib
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

from sklearn.model_selection import train_test_split

from LabsCommon.corpus import CorpusRow
from LabsCommon.dataset_adapter import AdapterResult, PreparationError

TOPICS = ("Россия", "Мир", "Спорт", "Экономика")
PER_TOPIC = 250
REQUIRED_COLUMNS = {"url", "title", "text", "topic"}

SQL_CREATE_ITEMS: Final = """
CREATE TABLE items (url TEXT PRIMARY KEY, title TEXT, text TEXT, topic TEXT, digest TEXT)
"""
SQL_CREATE_SEEN_URLS: Final = "CREATE TABLE seen_urls (url TEXT PRIMARY KEY)"
SQL_INSERT_SEEN_URL: Final = "INSERT INTO seen_urls VALUES (?)"
SQL_DELETE_ITEM: Final = "DELETE FROM items WHERE url = ?"
SQL_INSERT_ITEM: Final = "INSERT INTO items VALUES (?, ?, ?, ?, ?)"
SQL_SELECT_SUBSET: Final = """
WITH unique_text AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY title || char(10) || text ORDER BY url
    ) AS text_rank
    FROM items
), ranked AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY topic ORDER BY digest, url
    ) AS topic_rank
    FROM unique_text WHERE text_rank = 1
)
SELECT url, title, text, topic
FROM ranked WHERE topic_rank <= ? ORDER BY url
"""
SQL_COUNT_TOPIC: Final = """
SELECT COUNT(*) FROM (
    SELECT ROW_NUMBER() OVER (
        PARTITION BY title || char(10) || text ORDER BY url
    ) AS rank
    FROM items WHERE topic = ?
) WHERE rank = 1
"""
SQL_COUNT_ITEMS: Final = "SELECT COUNT(*) FROM items"


def prepare(source: Path, parameters: Mapping[str, object]) -> AdapterResult:
    """Подготовить Lenta из CSV или BZ2, сохранив полный CSV по указанному пути."""
    retained = _path_parameter(parameters, "retained_source")
    if not source.is_file():
        raise PreparationError(f"Источник Lenta.ru не найден: {source}")
    _retain_source(source, retained)
    rows, counts = _select_rows(retained)
    rows = _split(rows)
    metadata: dict[str, object] = {
        "source": "lenta",
        "retained_source": str(retained),
        "input_rows": counts.pop("input_rows"),
        "dropped_rows": counts.pop("dropped_rows"),
        "topic_counts": counts,
    }
    if source.suffix == ".bz2":
        metadata["delete_after_write"] = str(source)
    return AdapterResult(rows, metadata)


def _path_parameter(parameters: Mapping[str, object], name: str) -> Path:
    value = parameters.get(name)
    if not isinstance(value, str) or not value:
        raise PreparationError(f"Для адаптера Lenta требуется строковый параметр {name}")
    return Path(value)


def _retain_source(source: Path, retained: Path) -> None:
    """Атомарно оставить полный CSV; BZ2 читается последовательным потоком."""
    if source.suffix != ".bz2":
        if source.resolve() != retained.resolve():
            retained.parent.mkdir(parents=True, exist_ok=True)
            with (
                source.open("rb") as input_stream,
                tempfile.NamedTemporaryFile(
                    "wb", dir=retained.parent, delete=False, prefix=f".{retained.name}."
                ) as temporary,
            ):
                shutil.copyfileobj(input_stream, temporary, 1024 * 1024)
                temporary_path = Path(temporary.name)
            temporary_path.replace(retained)
        return
    retained.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with (
            bz2.open(source, "rb") as input_stream,
            tempfile.NamedTemporaryFile(
                "wb", dir=retained.parent, delete=False, prefix=f".{retained.name}."
            ) as temporary,
        ):
            temporary_path = Path(temporary.name)
            shutil.copyfileobj(input_stream, temporary, 1024 * 1024)
        temporary_path.replace(retained)
        temporary_path = None
    except (OSError, EOFError) as error:
        raise PreparationError(f"Не удалось распаковать Lenta.ru: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _source_rows(path: Path) -> Iterator[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or ())):
                raise PreparationError("CSV Lenta.ru не содержит url, title, text, topic")
            for row in reader:
                if None in row:
                    raise PreparationError("CSV Lenta.ru содержит лишние поля")
                yield {name: row.get(name, "") or "" for name in REQUIRED_COLUMNS}
    except (OSError, UnicodeError, csv.Error) as error:
        raise PreparationError(f"Не удалось прочитать CSV Lenta.ru: {error}") from error


def _select_rows(path: Path) -> tuple[list[CorpusRow], dict[str, int]]:
    database_path: Path | None = None
    connection: sqlite3.Connection | None = None
    input_rows = 0
    try:
        with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as stream:
            database_path = Path(stream.name)
        connection = sqlite3.connect(database_path)
        connection.execute(SQL_CREATE_ITEMS)
        connection.execute(SQL_CREATE_SEEN_URLS)
        for row in _source_rows(path):
            input_rows += 1
            url, title, text, topic = (
                row[name].strip() for name in ("url", "title", "text", "topic")
            )
            if not url:
                continue
            try:
                connection.execute(SQL_INSERT_SEEN_URL, (url,))
            except sqlite3.IntegrityError:
                connection.execute(SQL_DELETE_ITEM, (url,))
                continue
            if not title or not text or topic not in TOPICS:
                continue
            connection.execute(
                SQL_INSERT_ITEM,
                (url, title, text, topic, _digest(url)),
            )
        connection.commit()
        selected = connection.execute(
            SQL_SELECT_SUBSET,
            (PER_TOPIC,),
        ).fetchall()
        counts = {
            topic: connection.execute(
                SQL_COUNT_TOPIC,
                (topic,),
            ).fetchone()[0]
            for topic in TOPICS
        }
        missing = {topic: count for topic, count in counts.items() if count < PER_TOPIC}
        if missing:
            raise PreparationError(f"После фильтрации недостаточно статей по категориям: {missing}")
        retained_rows = connection.execute(SQL_COUNT_ITEMS).fetchone()[0]
        return [
            CorpusRow(url, f"{title}\n{text}", topic) for url, title, text, topic in selected
        ], {"input_rows": input_rows, "dropped_rows": input_rows - retained_rows, **counts}
    except sqlite3.Error as error:
        raise PreparationError(
            f"Не удалось обработать временный индекс Lenta.ru: {error}"
        ) from error
    finally:
        if connection is not None:
            connection.close()
        if database_path is not None:
            database_path.unlink(missing_ok=True)


def _split(rows: list[CorpusRow]) -> list[CorpusRow]:
    labels = [row.label for row in rows]
    train_indices, _ = train_test_split(
        range(len(rows)), test_size=0.2, random_state=42, stratify=labels
    )
    train = set(train_indices)
    return [
        CorpusRow(row.identifier, row.text, row.label, "train" if index in train else "test")
        for index, row in enumerate(rows)
    ]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
