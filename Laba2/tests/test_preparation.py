from __future__ import annotations

import bz2
import csv
import hashlib
import json
from pathlib import Path

import pytest

from Laba2.src.cli import main
from Laba2.src.preparation import PreparationError, prepare_table, resolve_config
from LabsCommon.dataset_config import DatasetConfigError, load_table_config


def _write_config(path: Path, *, delimiter: str = ",") -> None:
    path.write_text(
        f'''[dataset]
format = "{"tsv" if delimiter == "\\t" else "csv"}"
delimiter = "{delimiter}"

[columns]
text = ["title", "body"]
label = "kind"
joiner = "\\n"

[cleaning]
empty = "drop"
duplicates = "keep"
''',
        encoding="utf-8",
    )


def test_prepare_table_combines_fields_generates_ids_and_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset.csv"
    config = tmp_path / "dataset.toml"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_config(config)
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("title", "body", "kind"))
        writer.writeheader()
        writer.writerows(
            [
                {"title": " Заголовок ", "body": "текст,\nс переводом", "kind": " news "},
                {"title": "", "body": "", "kind": "empty"},
            ]
        )

    summary = prepare_table(source, first, config)
    prepare_table(source, second, config)

    assert summary.output_rows == 1
    assert summary.dropped_rows == 1
    assert first.read_bytes() == second.read_bytes()
    rows = list(csv.DictReader(first.open(encoding="utf-8", newline="")))
    text = "Заголовок\nтекст,\nс переводом"
    assert rows == [
        {
            "id": f"sha256:{hashlib.sha256(text.encode()).hexdigest()}",
            "text": text,
            "label": "news",
        }
    ]


def test_prepare_prefers_explicit_config_over_sidecar_and_formats_result(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "dataset.csv"
    sidecar = tmp_path / "dataset.toml"
    explicit = tmp_path / "other.toml"
    output = tmp_path / "result.csv"
    source.write_text("body\nтекст\n", encoding="utf-8")
    sidecar.write_text("[dataset]\nformat='csv'\n[columns]\ntext='missing'\n", encoding="utf-8")
    explicit.write_text("[dataset]\nformat='csv'\n[columns]\ntext='body'\n", encoding="utf-8")

    assert resolve_config(source, None) == sidecar
    assert (
        main(
            [
                "prepare",
                "--input",
                str(source),
                "--output",
                str(output),
                "--config",
                str(explicit),
                "--format",
                "json",
            ]
        )
        == 0
    )
    compact = json.loads(capsys.readouterr().out)
    assert compact["config"] == str(explicit)
    assert (
        main(
            [
                "prepare",
                "--input",
                str(source),
                "--output",
                str(output),
                "--config",
                str(explicit),
                "--format",
                "pretty-json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == compact


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        ("[dataset]\nformat='csv'\nextra=true\n[columns]\ntext='body'\n", "Неизвестные"),
        ("[dataset]\nformat='csv'\n[columns]\ntext='missing'\n", "отсутствуют столбцы"),
    ],
)
def test_prepare_rejects_bad_config_or_source_without_partial_output(
    tmp_path: Path, toml: str, message: str
) -> None:
    source = tmp_path / "source.csv"
    config = tmp_path / "source.toml"
    output = tmp_path / "output.csv"
    source.write_text("body\nтекст\n", encoding="utf-8")
    config.write_text(toml, encoding="utf-8")

    with pytest.raises(PreparationError, match=message):
        prepare_table(source, output, config)

    assert not output.exists()
    assert not list(tmp_path.glob(".output.csv.*"))


def test_prepare_rejects_conflicting_labels_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "source.tsv"
    config = tmp_path / "source.toml"
    output = tmp_path / "output.csv"
    source.write_text("body\tkind\nодин\ta\nодин\tb\n", encoding="utf-8")
    config.write_text(
        "[dataset]\nformat='tsv'\n[columns]\ntext='body'\nlabel='kind'\n",
        encoding="utf-8",
    )

    with pytest.raises(PreparationError, match="Конфликтующие"):
        prepare_table(source, output, config)

    assert not output.exists()


def test_prepare_rejects_empty_configured_id_without_generating_fallback(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    config = tmp_path / "source.toml"
    output = tmp_path / "output.csv"
    source.write_text("id,body\n,текст\n", encoding="utf-8")
    config.write_text(
        "[dataset]\nformat='csv'\n[columns]\nid='id'\ntext='body'\n",
        encoding="utf-8",
    )

    with pytest.raises(PreparationError, match="пустое обязательное значение"):
        prepare_table(source, output, config)

    assert not output.exists()


@pytest.mark.parametrize(
    ("name", "value"), [("train", "nan"), ("validation", "inf"), ("test", "-inf")]
)
def test_table_config_rejects_non_finite_split_fraction(
    tmp_path: Path, name: str, value: str
) -> None:
    config = tmp_path / "source.toml"
    config.write_text(
        f"[dataset]\nformat='csv'\n[columns]\ntext='body'\n[split]\n{name}={value}\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetConfigError, match="конечными числами"):
        load_table_config(config)


def test_prepare_rejects_malformed_csv_quoting_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    config = tmp_path / "source.toml"
    output = tmp_path / "output.csv"
    source.write_text('body\n"незавершённая строка\n', encoding="utf-8")
    config.write_text("[dataset]\nformat='csv'\n[columns]\ntext='body'\n", encoding="utf-8")

    with pytest.raises(PreparationError, match="Не удалось прочитать источник"):
        prepare_table(source, output, config)

    assert not output.exists()


@pytest.mark.parametrize(
    ("size", "train", "validation", "test", "expected"),
    [(20, 65, 10, 25, (13, 2, 5)), (8, 62.5, 12.5, 25, (5, 1, 2))],
)
def test_prepare_splits_percentages_per_class_with_validation(
    tmp_path: Path,
    size: int,
    train: float,
    validation: float,
    test: float,
    expected: tuple[int, int, int],
) -> None:
    source = tmp_path / "source.csv"
    config = tmp_path / "source.toml"
    output = tmp_path / "output.csv"
    source.write_text(
        "body,kind\n"
        + "".join(
            f"текст {label} {index},{label}\n" for label in ("a", "b") for index in range(size)
        ),
        encoding="utf-8",
    )
    config.write_text(
        "[dataset]\nformat='csv'\n[columns]\ntext='body'\nlabel='kind'\n"
        f"[split]\nenabled=true\ntrain={train}\nvalidation={validation}\ntest={test}\n"
        f"[expect]\nsplits={{train={train},validation={validation},test={test}}}\n",
        encoding="utf-8",
    )

    result = prepare_table(source, output, config)

    parts = ("train", "validation", "test")
    assert result.split_counts == {
        split: count * 2 for split, count in zip(parts, expected, strict=True)
    }
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for label in ("a", "b"):
        for split, count in zip(parts, expected, strict=True):
            assert sum(row["label"] == label and row["split"] == split for row in rows) == count


@pytest.mark.parametrize("size", [6, 12])
def test_expect_split_percentages_round_without_fixed_row_count(tmp_path: Path, size: int) -> None:
    source = tmp_path / "source.csv"
    config = tmp_path / "source.toml"
    output = tmp_path / "output.csv"
    source.write_text(
        "body,kind,split\n"
        + "".join(
            f"текст {index},a,{'train' if index < size * 2 // 3 else 'test'}\n"
            for index in range(size)
        ),
        encoding="utf-8",
    )
    base = "[dataset]\nformat='csv'\n[columns]\ntext='body'\nlabel='kind'\nsplit='split'\n"
    config.write_text(base + "[expect]\nsplits={train=67,test=33}\n", encoding="utf-8")

    assert prepare_table(source, output, config).output_rows == size
    original = output.read_bytes()
    config.write_text(base + "[expect]\nsplits={train=50,test=50}\n", encoding="utf-8")

    with pytest.raises(PreparationError, match="Ожидалось строк train"):
        prepare_table(source, output, config)
    assert output.read_bytes() == original


@pytest.mark.parametrize("table", ["split", "expect.splits"])
@pytest.mark.parametrize(
    "values",
    [
        "train=true,test=20",
        "train='80',test=20",
        "train=-10,test=110",
        "train=nan,test=20",
        "train=inf,test=20",
        "train=800,test=200",
        "train=0.8,test=0.2",
        "train=80,test=20,other=0",
    ],
)
def test_table_config_rejects_invalid_percentages(tmp_path: Path, table: str, values: str) -> None:
    config = tmp_path / "source.toml"
    settings = (
        "[split]\n" + values.replace(",", "\n") + "\n"
        if table == "split"
        else "[expect]\nsplits={" + values + "}\n"
    )
    config.write_text(
        "[dataset]\nformat='csv'\n[columns]\ntext='body'\n" + settings,
        encoding="utf-8",
    )

    with pytest.raises(DatasetConfigError):
        load_table_config(config)


def test_expect_split_percentages_reject_unexpected_validation_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    config = tmp_path / "source.toml"
    output = tmp_path / "output.csv"
    source.write_text("body,kind,split\nпервый,a,train\nвторой,a,validation\n", encoding="utf-8")
    config.write_text(
        "[dataset]\nformat='csv'\n[columns]\ntext='body'\nlabel='kind'\nsplit='split'\n"
        "[expect]\nsplits={train=50,test=50}\n",
        encoding="utf-8",
    )

    with pytest.raises(PreparationError, match="Ожидалось строк validation: 0"):
        prepare_table(source, output, config)
    assert not output.exists()


def _lenta_source(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("url", "title", "text", "topic"))
        writer.writeheader()
        for topic in ("Россия", "Мир", "Спорт", "Экономика"):
            for number in range(250):
                writer.writerow(
                    {
                        "url": f"https://example/{topic}/{number:03}",
                        "title": f"{topic} {number}",
                        "text": f"текст {number}",
                        "topic": topic,
                    }
                )


def test_lenta_adapter_selects_exact_subset_and_removes_archive_after_success(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    archive = tmp_path / "lenta-ru-news.csv.bz2"
    retained = tmp_path / "retained.csv"
    output = tmp_path / "output.csv"
    config = tmp_path / "lenta-ru-news.toml"
    _lenta_source(source)
    archive.write_bytes(bz2.compress(source.read_bytes()))
    config.write_text(
        "[dataset]\nformat='csv'\n[columns]\ntext='text'\n[adapter]\npath='Laba2.src.adapters.lenta:prepare'\nparameters={retained_source='"
        + str(retained).replace("\\", "/")
        + "'}\n[expect]\nrows=1000\nclasses=4\nsplits={train=80,test=20}\n",
        encoding="utf-8",
    )

    result = prepare_table(archive, output, config)

    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    assert len(rows) == 1000
    assert result.split_counts == {"test": 200, "train": 800}
    assert result.class_counts == {"Россия": 250, "Мир": 250, "Спорт": 250, "Экономика": 250}
    assert retained.exists()
    assert not archive.exists()


def test_external_adapter_is_loaded_only_by_explicit_path(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    config = tmp_path / "external.toml"
    output = tmp_path / "output.csv"
    source.write_text("ignored", encoding="utf-8")
    config.write_text(
        "[dataset]\nformat='csv'\n[columns]\ntext='text'\n[adapter]\npath='Laba2.adapters.external_adapter_template:prepare'\nparameters={}\n",
        encoding="utf-8",
    )

    assert prepare_table(source, output, config).output_rows == 1
    assert list(csv.DictReader(output.open(encoding="utf-8", newline="")))[0]["id"] == "example-1"


def test_rureviews_profile_has_fixed_sampling_and_cli_json_formats(tmp_path: Path, capsys) -> None:
    source = tmp_path / "rureviews.tsv"
    output = tmp_path / "output.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("review", "sentiment"), delimiter="\t")
        writer.writeheader()
        for label in ("negative", "neautral", "positive"):
            writer.writerows(
                {"review": f"{label} отзыв {number}", "sentiment": label} for number in range(400)
            )

    arguments = [
        "prepare",
        "--profile",
        "rureviews",
        "--input",
        str(source),
        "--output",
        str(output),
    ]
    assert main([*arguments, "--format", "json"]) == 0
    compact = json.loads(capsys.readouterr().out)
    assert main([*arguments, "--format", "pretty-json"]) == 0
    assert json.loads(capsys.readouterr().out) == compact
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    assert compact["split_counts"] == {"test": 150, "train": 600}
    assert {row["label"] for row in rows} == {"negative", "neutral", "positive"}
