from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest

from Laba1.src import cli
from Laba1.src.model import build_metadata, fit_model, save_model_package
from Laba1.src.normalization import NormalizationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_model_package(path: Path) -> None:
    reviews = [
        "плохой товар",
        "плохой товар",
        "обычный товар",
        "обычный товар",
        "хороший товар",
        "хороший товар",
    ]
    labels = ["negative", "negative", "neutral", "neutral", "positive", "positive"]
    model = fit_model(reviews, labels, NormalizationConfig(method="stem"))
    metadata = build_metadata(corpus_sha256="a" * 64)
    save_model_package(path, model, metadata)


def _run_cli(*arguments: str, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "Laba1", *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
        text=True,
    )


def test_public_entry_point_exposes_help() -> None:
    result = _run_cli("--help")

    assert result.returncode == 0
    assert "download" not in result.stdout
    assert "train" in result.stdout
    assert "vocabulary" in result.stdout
    assert "analyze" in result.stdout


def test_train_help_exposes_only_packaged_corpus_contract() -> None:
    result = _run_cli("train", "--help")

    assert result.returncode == 0
    assert "--artifacts" in result.stdout
    assert "--verbose" in result.stdout
    assert "--data" not in result.stdout
    assert "rureviews-profile" not in result.stdout


def test_text_and_equivalent_utf8_file_have_stable_schema_and_prediction(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    text_path = tmp_path / "review.txt"
    _write_model_package(model_path)
    text_path.write_text("Платье отличное", encoding="utf-8")

    text_result = _run_cli("analyze", "--model", str(model_path), "--text", "Платье отличное")
    file_result = _run_cli("analyze", "--model", str(model_path), "--file", str(text_path))

    assert text_result.returncode == 0, text_result.stderr
    assert file_result.returncode == 0, file_result.stderr
    text_payload = json.loads(text_result.stdout)
    file_payload = json.loads(file_result.stdout)
    assert text_payload == file_payload
    assert set(text_payload) == {"configuration", "label", "probabilities"}
    assert text_payload["configuration"] == {
        "method": "stem",
        "stopwords": False,
        "synonyms": False,
    }
    probabilities = text_payload["probabilities"]
    assert tuple(probabilities) == ("negative", "neutral", "positive")
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert text_payload["label"] == max(probabilities, key=probabilities.__getitem__)


def test_unknown_words_are_accepted_and_empty_or_invalid_file_input_is_rejected(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.joblib"
    invalid_encoding = tmp_path / "invalid.txt"
    _write_model_package(model_path)
    invalid_encoding.write_bytes(b"\xff\xfe\x00")

    unknown = _run_cli("analyze", "--model", str(model_path), "--text", "крокозябра")
    empty = _run_cli("analyze", "--model", str(model_path), "--text", "___ ...")
    invalid = _run_cli("analyze", "--model", str(model_path), "--file", str(invalid_encoding))

    assert unknown.returncode == 0, unknown.stderr
    assert len(json.loads(unknown.stdout)["probabilities"]) == 3
    assert empty.returncode == 1
    assert "не содержит токенов" in empty.stderr
    assert invalid.returncode == 1
    assert "UTF-8" in invalid.stderr


def test_argument_errors_and_damaged_model_return_nonzero_without_traceback(tmp_path: Path) -> None:
    broken_model = tmp_path / "broken.joblib"
    malformed_model = tmp_path / "malformed.joblib"
    broken_model.write_bytes(b"not a joblib package")
    _write_model_package(malformed_model)
    malformed_payload = joblib.load(malformed_model)
    malformed_payload["class_order"] = None
    joblib.dump(malformed_payload, malformed_model)

    missing_input = _run_cli("analyze", "--model", str(broken_model))
    conflicting_input = _run_cli(
        "analyze", "--model", str(broken_model), "--text", "текст", "--file", "review.txt"
    )
    broken = _run_cli("analyze", "--model", str(broken_model), "--text", "текст")
    malformed = _run_cli("analyze", "--model", str(malformed_model), "--text", "текст")

    assert missing_input.returncode != 0
    assert "--text" in missing_input.stderr
    assert conflicting_input.returncode != 0
    assert "not allowed" in conflicting_input.stderr
    assert broken.returncode == 1
    assert "Ошибка модели" in broken.stderr
    assert "Traceback" not in broken.stderr
    assert malformed.returncode == 1
    assert "Ошибка модели" in malformed.stderr
    assert "Traceback" not in malformed.stderr


def test_train_command_uses_packaged_rureviews_and_saves_package(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviews = [
        "плохой товар",
        "плохой товар",
        "обычный товар",
        "обычный товар",
        "хороший товар",
        "хороший товар",
    ]
    labels = ["negative", "negative", "neutral", "neutral", "positive", "positive"]
    config = NormalizationConfig(method="stem", remove_stopwords=True)
    model = fit_model(reviews, labels, config)
    result = SimpleNamespace(model=model, training_seconds=0.01)
    evaluation = SimpleNamespace(
        accuracy=0.8,
        macro_f1=0.8,
        predictions=("negative", "neutral", "positive"),
        as_dict=lambda: {"accuracy": 0.8, "macro_f1": 0.8},
        confusion_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )
    prepared = SimpleNamespace(
        split=SimpleNamespace(train=[object()], validation=[object()], test=[object()])
    )
    artifacts = tmp_path / "artifacts"
    profile_calls: list[Path] = []
    training_calls: list[object] = []
    evaluation_calls: list[tuple[object, object]] = []

    def prepare_rureviews(path: Path) -> SimpleNamespace:
        profile_calls.append(path)
        return prepared

    def train_model(train: object) -> SimpleNamespace:
        training_calls.append(train)
        return result

    def evaluate(selected_model: object, test: object) -> SimpleNamespace:
        evaluation_calls.append((selected_model, test))
        return evaluation

    monkeypatch.setattr(cli, "prepare_rureviews", prepare_rureviews)
    monkeypatch.setattr(cli, "train_selected_model", train_model)
    monkeypatch.setattr(cli, "evaluate_selected_model", evaluate)
    monkeypatch.setattr(
        cli, "_save_confusion_matrix", lambda value: tmp_path / "confusion_matrix.png"
    )
    exit_code = cli.main(["train", "--artifacts", str(artifacts)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    package_path = artifacts / "model.joblib"
    assert exit_code == 0
    assert profile_calls == [PROJECT_ROOT / "Laba1" / "data" / "rureviews.tsv"]
    assert training_calls == [prepared.split.train]
    assert evaluation_calls == [(model, prepared.split.test)]
    assert captured.err == ""
    assert payload["model"] == str(package_path)
    assert payload["configuration"] == {
        "method": "stem",
        "stopwords": True,
        "synonyms": False,
    }
    assert package_path.exists()
    assert (artifacts / "metrics.json").exists()
    assert (artifacts / "predictions.json").exists()
    loaded = joblib.load(package_path)
    assert "schema_version" not in loaded
    assert loaded["metadata"]["corpus_profile"] == "rureviews"

    verbose_artifacts = tmp_path / "verbose-artifacts"
    verbose_exit_code = cli.main(["train", "--artifacts", str(verbose_artifacts), "--verbose"])
    verbose = capsys.readouterr()
    json.loads(verbose.out)
    assert verbose_exit_code == 0
    assert "Чтение и очистка" in verbose.err
    assert "Test: accuracy=0.800000" in verbose.err
    assert "Артефакты" in verbose.err
