from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from Laba2.src.adapters.lenta import _select_rows
from Laba2.src.data import PreparationError, _lenta_rows, read_canonical


def _write_lenta(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("url", "title", "text", "topic", "tags", "date")
        )
        writer.writeheader()
        writer.writerows(rows)


def test_lenta_selection_is_deterministic_and_resolves_duplicate_text(tmp_path: Path) -> None:
    rows = []
    for topic in ("Россия", "Мир", "Спорт", "Экономика"):
        for index in range(250):
            rows.append(
                {
                    "url": f"https://example.test/{topic}/{index:03d}",
                    "title": f"{topic} {index}",
                    "text": f"Текст {topic} {index}",
                    "topic": topic,
                    "tags": "",
                    "date": "2020-01-01",
                }
            )
    rows.append(
        {
            "url": "https://example.test/a",
            "title": "Россия 0",
            "text": "Текст Россия 0",
            "topic": "Россия",
            "tags": "",
            "date": "2020-01-01",
        }
    )
    source = tmp_path / "lenta.csv"
    _write_lenta(source, rows)

    first, first_counts = _lenta_rows(source)
    second, second_counts = _lenta_rows(source)

    assert first == second
    assert first_counts == second_counts
    assert len(first) == 1000
    assert {row["split"] for row in first} == {"train", "test"}
    assert {row["split"] for row in first if row["label"] == "Россия"} == {"train", "test"}
    assert sum(row["id"] == "https://example.test/a" for row in first) == 1


def test_lenta_adapter_excludes_every_row_with_a_repeated_url(tmp_path: Path) -> None:
    rows = []
    russia_urls = []
    for topic in ("Россия", "Мир", "Спорт", "Экономика"):
        count = 251 if topic == "Россия" else 250
        for index in range(count):
            url = f"https://example.test/{topic}/{index:03d}"
            rows.append(
                {
                    "url": url,
                    "title": f"{topic} {index}",
                    "text": f"Текст {topic} {index}",
                    "topic": topic,
                    "tags": "",
                    "date": "2020-01-01",
                }
            )
            if topic == "Россия":
                russia_urls.append(url)
    repeated_url = min(russia_urls, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    rows.append(
        {
            "url": repeated_url,
            "title": "Повторная статья",
            "text": "У повторного URL другой текст",
            "topic": "Россия",
            "tags": "",
            "date": "2020-01-01",
        }
    )
    source = tmp_path / "lenta.csv"
    _write_lenta(source, rows)

    selected, counts = _select_rows(source)

    assert len(selected) == 1000
    assert all(row.identifier != repeated_url for row in selected)
    assert counts["dropped_rows"] == 2
    assert counts["Россия"] == 250


def test_lenta_rejects_insufficient_topic_and_bad_schema(tmp_path: Path) -> None:
    source = tmp_path / "lenta.csv"
    source.write_text("url,title,text,topic,tags,date\nu,t,x,Россия,,d\n", encoding="utf-8")
    with pytest.raises(PreparationError, match="недостаточно"):
        _lenta_rows(source)

    invalid = tmp_path / "invalid.csv"
    invalid.write_text("url,title\nu,t\n", encoding="utf-8")
    with pytest.raises(PreparationError, match="столбцы"):
        _lenta_rows(invalid)


def test_read_canonical_rejects_duplicate_text_and_mixed_split(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    corpus.write_text(
        "id,text,label,split\na,текст,метка,train\nb,текст,метка,test\n", encoding="utf-8"
    )
    with pytest.raises(PreparationError, match="повторный text"):
        list(read_canonical(corpus))

    corpus.write_text("id,text,label,split\na,текст,метка,\n", encoding="utf-8")
    with pytest.raises(PreparationError, match="split"):
        list(read_canonical(corpus))


def test_read_canonical_accepts_unlabeled_csv_with_quoted_multiline_text(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    with corpus.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "text"))
        writer.writeheader()
        writer.writerow({"id": "one", "text": "строка, с запятой\nи переводом строки"})
        writer.writerow({"id": "two", "text": "другой текст"})

    rows = list(read_canonical(corpus))

    assert [row["id"] for row in rows] == ["one", "two"]
    assert rows[0]["label"] == ""
    assert rows[0]["text"] == "строка, с запятой\nи переводом строки"


def test_read_canonical_rejects_malformed_quoting(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    corpus.write_text('id,text\na,"незавершённая строка\n', encoding="utf-8")

    with pytest.raises(PreparationError, match="Не удалось прочитать канонический CSV"):
        list(read_canonical(corpus))


def test_read_canonical_rejects_empty_label_column(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    corpus.write_text("id,text,label\na,первый,\nb,второй,\n", encoding="utf-8")

    with pytest.raises(PreparationError, match="пустую метку label"):
        list(read_canonical(corpus))
