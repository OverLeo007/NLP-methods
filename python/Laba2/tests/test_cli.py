from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from Laba2.src.cli import main
from Laba2.src.model import save_model, train_model


def _corpus(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "text", "label"))
        writer.writeheader()
        writer.writerows(
            [
                {"id": "1", "text": "Кошка ловит мышь дома", "label": "home"},
                {"id": "2", "text": "Собака охраняет дом ночью", "label": "home"},
                {"id": "3", "text": "Хоккейная команда забила шайбу", "label": "sport"},
                {"id": "4", "text": "Игрок выиграл хоккейный матч", "label": "sport"},
            ]
        )


def test_train_analyze_file_and_topics_emit_json(tmp_path: Path, capsys) -> None:
    corpus = tmp_path / "corpus.csv"
    model = tmp_path / "model.joblib"
    query = tmp_path / "query.txt"
    _corpus(corpus)
    query.write_text("Хоккейный матч команды", encoding="utf-8")

    assert main(["train", "--corpus", str(corpus), "--model", str(model), "--format", "json"]) == 0
    train_payload = json.loads(capsys.readouterr().out)
    assert train_payload["labeled"] is True
    assert (
        main(
            [
                "analyze",
                "--model",
                str(model),
                "--file",
                str(query),
                "--top-k",
                "2",
                "--format",
                "json",
            ]
        )
        == 0
    )
    analysis = json.loads(capsys.readouterr().out)
    assert analysis["documents"][0]["id"] == "3"
    assert main(["topics", "--model", str(model), "--top-terms", "2", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["lsa_components"] == 3


def test_analyze_reports_missing_model(tmp_path: Path, capsys) -> None:
    assert main(["analyze", "--model", str(tmp_path / "missing.joblib"), "--text", "текст"]) == 1
    assert "Ошибка данных" in capsys.readouterr().err


def test_analyze_rejects_nonfinite_lda_state_without_traceback(tmp_path: Path, capsys) -> None:
    corpus = tmp_path / "corpus.csv"
    package = tmp_path / "model.joblib"
    _corpus(corpus)
    model = train_model(corpus)
    model.lda.scalings_[0, 0] = float("nan")
    save_model(model, package)

    assert main(["analyze", "--model", str(package), "--text", "Хоккейная команда"]) == 1
    error = capsys.readouterr().err
    assert "Ошибка данных" in error
    assert "преобразование LDA" in error
    assert "Traceback" not in error
    assert "NaN" not in error


@pytest.mark.parametrize("format_name", ("text", "json", "pretty-json"))
def test_model_commands_support_all_formats_and_clear_errors(
    tmp_path: Path, capsys, format_name: str
) -> None:
    corpus = tmp_path / "corpus.csv"
    model = tmp_path / "model.joblib"
    query = tmp_path / "query.txt"
    _corpus(corpus)
    query.write_text("Хоккейная команда", encoding="utf-8")

    assert (
        main(["train", "--corpus", str(corpus), "--model", str(model), "--format", format_name])
        == 0
    )
    train_output = capsys.readouterr().out
    assert train_output.strip()
    assert (
        main(
            [
                "analyze",
                "--model",
                str(model),
                "--text",
                "Хоккейная команда",
                "--format",
                format_name,
            ]
        )
        == 0
    )
    analysis_output = capsys.readouterr().out
    assert analysis_output.strip()
    assert (
        main(["analyze", "--model", str(model), "--file", str(query), "--format", format_name]) == 0
    )
    assert capsys.readouterr().out.strip()
    assert main(["topics", "--model", str(model), "--format", format_name]) == 0
    assert capsys.readouterr().out.strip()
    if format_name != "text":
        assert json.loads(analysis_output)["documents"]
    assert main(["analyze", "--model", str(model), "--text", "несуществующее", "--top-k", "0"]) == 1
    assert "положительным" in capsys.readouterr().err


def test_evaluate_writes_protocol_and_keeps_all_formats_semantically_consistent(
    tmp_path: Path, capsys
) -> None:
    corpus = tmp_path / "corpus.csv"
    protocol = tmp_path / "evaluation.json"
    with corpus.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "text", "label", "split"))
        writer.writeheader()
        writer.writerows(
            [
                {"id": "a1", "text": "кошка ловит мышь", "label": "home", "split": "train"},
                {"id": "a2", "text": "собака охраняет дом", "label": "home", "split": "train"},
                {
                    "id": "b1",
                    "text": "хоккейная команда забила шайбу",
                    "label": "sport",
                    "split": "train",
                },
                {
                    "id": "b2",
                    "text": "игрок выиграл хоккейный матч",
                    "label": "sport",
                    "split": "train",
                },
                {"id": "t1", "text": "кошка дома", "label": "home", "split": "test"},
                {"id": "t2", "text": "хоккейный матч", "label": "sport", "split": "test"},
            ]
        )

    rendered: dict[str, object] = {}
    for format_name in ("json", "pretty-json"):
        assert (
            main(
                [
                    "evaluate",
                    "--corpus",
                    str(corpus),
                    "--output",
                    str(protocol),
                    "--format",
                    format_name,
                ]
            )
            == 0
        )
        rendered[format_name] = json.loads(capsys.readouterr().out)

    saved = json.loads(protocol.read_text(encoding="utf-8"))
    assert rendered == {"json": saved, "pretty-json": saved}
    assert main(["evaluate", "--corpus", str(corpus), "--output", str(protocol)]) == 0
    text = capsys.readouterr().out
    assert "train_documents" in text
    assert "skipped_queries" in text

    blocked_output = tmp_path / "directory"
    blocked_output.mkdir()
    assert (
        main(
            [
                "evaluate",
                "--corpus",
                str(corpus),
                "--output",
                str(blocked_output),
                "--format",
                "json",
            ]
        )
        == 1
    )
    assert "Не удалось сохранить JSON-протокол" in capsys.readouterr().err
    assert not list(tmp_path.glob(".directory.*.json"))


def test_all_five_commands_support_three_formats_without_stdout_progress(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.csv"
    config = tmp_path / "source.toml"
    prepared = tmp_path / "prepared.csv"
    model = tmp_path / "model.joblib"
    protocol = tmp_path / "evaluation.json"
    source.write_text(
        "body,kind,split\nкошка ловит мышь,home,train\nсобака охраняет дом,home,train\n"
        "хоккейная команда забила шайбу,sport,train\nигрок выиграл матч,sport,train\n"
        "кошка дома,home,test\nхоккейный матч,sport,test\n",
        encoding="utf-8",
    )
    config.write_text(
        "[dataset]\nformat = 'csv'\n[columns]\ntext = 'body'\nlabel = 'kind'\nsplit = 'split'\n",
        encoding="utf-8",
    )

    commands = {
        "prepare": ["prepare", "--input", str(source), "--output", str(prepared)],
        "train": ["train", "--corpus", str(prepared), "--model", str(model)],
        "analyze": ["analyze", "--model", str(model), "--text", "хоккейная команда"],
        "topics": ["topics", "--model", str(model)],
        "evaluate": ["evaluate", "--corpus", str(prepared), "--output", str(protocol)],
    }
    rendered: dict[str, dict[str, object]] = {}
    for name, command in commands.items():
        rendered[name] = {}
        for format_name in ("text", "json", "pretty-json"):
            assert main([*command, "--format", format_name]) == 0
            output = capsys.readouterr()
            assert output.err == ""
            assert output.out.strip()
            if format_name != "text":
                rendered[name][format_name] = json.loads(output.out)
        assert rendered[name]["json"] == rendered[name]["pretty-json"]
