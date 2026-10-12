"""Детерминированная подготовка учебных корпусов для LSA."""

from __future__ import annotations

import bz2
import csv
import hashlib
import os
import sqlite3
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from sklearn.model_selection import train_test_split

from LabsCommon.corpus import CorpusError
from LabsCommon.corpus import read_canonical as read_common_canonical

LENTA_COLUMNS: Final = ("url", "title", "text", "topic", "tags", "date")
CANONICAL_COLUMNS: Final = ("id", "text", "label", "split")
LENTA_TOPICS: Final = ("Россия", "Мир", "Спорт", "Экономика")
LENTA_PER_TOPIC: Final = 250
RUREVIEWS_TRAIN_PER_LABEL: Final = 200
RUREVIEWS_TEST_PER_LABEL: Final = 50
RANDOM_STATE: Final = 42
LENTA_ARCHIVE_BYTES: Final = 346_031_300
LENTA_ARCHIVE_SHA256: Final = "c4f7f771c7c78221978943a65dfabc9cc129a72a5d477635acc16a771d1fda47"


class PreparationError(ValueError):
    """Ошибка структуры или подготовки корпуса."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Вычислить SHA-256 файла без зависимости от внутренностей Laba1."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _copy_bz2_to_csv(archive: Path, retained_source: Path) -> None:
    """Распаковать архив атомарно, не оставляя частичного полного CSV."""
    retained_source.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{retained_source.name}.", dir=retained_source.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with bz2.open(archive, "rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    temporary.write(chunk)
        temporary_path.replace(retained_source)
    except (OSError, EOFError) as error:
        raise PreparationError(f"Не удалось распаковать архив {archive}: {error}") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _open_csv(path: Path) -> Iterable[dict[str, str]]:
    """Открыть UTF-8 CSV с точной схемой Lenta.ru."""
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except (OSError, UnicodeError) as error:
        raise PreparationError(f"Не удалось прочитать CSV {path}: {error}") from error
    try:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != LENTA_COLUMNS:
            raise PreparationError(
                f"Неверные столбцы Lenta.ru: получено {tuple(reader.fieldnames or ())}, "
                f"ожидалось {LENTA_COLUMNS}"
            )
        for row in reader:
            if None in row:
                raise PreparationError("CSV Lenta.ru содержит строку с лишними полями")
            yield {name: row[name] or "" for name in LENTA_COLUMNS}
    except (csv.Error, UnicodeError) as error:
        raise PreparationError(f"Некорректный CSV Lenta.ru: {error}") from error
    finally:
        stream.close()


def _write_canonical(path: Path, rows: Iterable[dict[str, str]]) -> None:
    """Записать канонический CSV атомарно с корректным экранированием полей."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=CANONICAL_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _split_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    labels = [row["label"] for row in rows]
    train_indices, test_indices = train_test_split(
        list(range(len(rows))), test_size=0.20, random_state=RANDOM_STATE, stratify=labels
    )
    train_set = set(train_indices)
    result = [
        dict(row, split="train" if index in train_set else "test") for index, row in enumerate(rows)
    ]
    return sorted(result, key=lambda row: row["id"])


def _lenta_rows(retained_source: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Отобрать статьи через временную БД, не удерживая полный корпус в памяти."""
    database_path: Path | None = None
    connection: sqlite3.Connection | None = None
    source_rows = 0
    valid_rows = 0
    try:
        with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as temporary:
            database_path = Path(temporary.name)
        connection = sqlite3.connect(database_path)
        connection.execute(
            "CREATE TABLE candidates (url TEXT PRIMARY KEY, title TEXT NOT NULL, text TEXT NOT NULL, "
            "topic TEXT NOT NULL, url_hash TEXT NOT NULL)"
        )
        connection.execute("CREATE TABLE duplicate_urls (url TEXT PRIMARY KEY)")
        for row in _open_csv(retained_source):
            source_rows += 1
            url, title, text, topic = (
                row[name].strip() for name in ("url", "title", "text", "topic")
            )
            if topic not in LENTA_TOPICS or not url or not title or not text:
                continue
            valid_rows += 1
            if connection.execute("SELECT 1 FROM duplicate_urls WHERE url = ?", (url,)).fetchone():
                continue
            try:
                connection.execute(
                    "INSERT INTO candidates(url, title, text, topic, url_hash) VALUES (?, ?, ?, ?, ?)",
                    (url, title, text, topic, _hash_text(url)),
                )
            except sqlite3.IntegrityError:
                connection.execute("DELETE FROM candidates WHERE url = ?", (url,))
                connection.execute("INSERT INTO duplicate_urls(url) VALUES (?)", (url,))
        connection.commit()
        selected = connection.execute(
            "WITH unique_texts AS ("
            " SELECT url, title, text, topic, url_hash,"
            " ROW_NUMBER() OVER (PARTITION BY title || char(10) || text ORDER BY url) AS text_rank"
            " FROM candidates"
            "), ranked AS ("
            " SELECT url, title, text, topic, url_hash,"
            " ROW_NUMBER() OVER (PARTITION BY topic ORDER BY url_hash, url) AS topic_rank"
            " FROM unique_texts WHERE text_rank = 1"
            ") SELECT url, title, text, topic FROM ranked WHERE topic_rank <= ? ORDER BY url",
            (LENTA_PER_TOPIC,),
        ).fetchall()
        counts = {
            topic: int(
                connection.execute(
                    "SELECT COUNT(*) FROM ("
                    " SELECT topic, ROW_NUMBER() OVER (PARTITION BY title || char(10) || text ORDER BY url) AS rank"
                    " FROM candidates WHERE topic = ?) WHERE rank = 1",
                    (topic,),
                ).fetchone()[0]
            )
            for topic in LENTA_TOPICS
        }
        missing = {topic: count for topic, count in counts.items() if count < LENTA_PER_TOPIC}
        if missing:
            raise PreparationError(f"После фильтрации недостаточно статей по категориям: {missing}")
        rows = [
            {"id": url, "text": f"{title}\n{text}", "label": topic, "split": ""}
            for url, title, text, topic in selected
        ]
        if len(rows) != len(LENTA_TOPICS) * LENTA_PER_TOPIC:
            raise PreparationError("Не удалось получить ровно 1000 статей Lenta.ru")
        return _split_rows(rows), {"source_rows": source_rows, "valid_rows": valid_rows, **counts}
    except sqlite3.Error as error:
        raise PreparationError(
            f"Не удалось обработать временный индекс Lenta.ru: {error}"
        ) from error
    finally:
        if connection is not None:
            connection.close()
        if database_path is not None and database_path.exists():
            database_path.unlink()


def prepare_lenta(
    archive: Path, retained_source: Path, output: Path, *, delete_archive: bool = False
) -> dict[str, object]:
    """Подготовить Lenta.ru и удалить проверенный архив только по явному указанию."""
    if not archive.is_file():
        raise PreparationError(f"Архив Lenta.ru не найден: {archive}")
    archive_hash = sha256_file(archive)
    if archive.stat().st_size != LENTA_ARCHIVE_BYTES or archive_hash != LENTA_ARCHIVE_SHA256:
        raise PreparationError("Архив Lenta.ru не соответствует проверенному выпуску v1.1")
    _copy_bz2_to_csv(archive, retained_source)
    rows, counts = _lenta_rows(retained_source)
    _write_canonical(output, rows)
    if len(list(read_canonical(output))) != 1000:
        raise PreparationError("Повторное чтение поднабора Lenta.ru вернуло неверное число строк")
    if not retained_source.is_file() or not output.is_file():
        raise PreparationError("Не найдены сохранённые результаты подготовки Lenta.ru")
    result = {
        "source": "lenta",
        "version": "v1.1",
        "source_url": "https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/tag/v1.1",
        "archive_sha256": archive_hash,
        "retained_source": file_info(retained_source),
        "output": file_info(output),
        "counts": {**counts, "train": 800, "test": 200},
    }
    if delete_archive:
        try:
            archive.unlink()
        except OSError as error:
            raise PreparationError(
                f"Подготовка завершена, но не удалось удалить архив {archive}: {error}"
            ) from error
    return result


def prepare_rureviews_subset(input_path: Path, output: Path) -> dict[str, object]:
    """Подготовить контрольный набор через общий табличный контракт."""
    from .preparation import prepare_table

    profile = Path(__file__).resolve().parents[1] / "profiles" / "rureviews.toml"
    try:
        summary = prepare_table(input_path, output, profile)
    except ValueError as error:
        raise PreparationError(str(error)) from error
    return {
        "source": "rureviews",
        "input": file_info(input_path),
        "output": file_info(output),
        "counts": summary.split_counts,
        "cleaning": {"input_rows": summary.input_rows, "output_rows": summary.output_rows},
    }


def file_info(path: Path) -> dict[str, object]:
    """Вернуть размер и SHA-256 файла для воспроизводимой сводки."""
    return {"path": str(path), "bytes": os.path.getsize(path), "sha256": sha256_file(path)}


def read_canonical(path: Path) -> Iterable[dict[str, str]]:
    """Сохранить прежний импорт, используя единый контракт LabsCommon."""
    try:
        yield from read_common_canonical(path)
    except CorpusError as error:
        raise PreparationError(str(error)) from error
