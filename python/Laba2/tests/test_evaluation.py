from __future__ import annotations

import csv
from pathlib import Path

import pytest

from Laba2.src import model
from Laba2.src.model import ModelError, evaluate_corpus


def _write_corpus(path: Path, rows: list[dict[str, str]], *, split: bool) -> None:
    fields = ("id", "text", "label", "split") if split else ("id", "text", "label")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_evaluation_fits_only_train_and_calculates_known_metrics(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    _write_corpus(
        corpus,
        [
            {"id": "a1", "text": "яблоко груша сад", "label": "fruit", "split": "train"},
            {"id": "a2", "text": "яблоко сладкая груша", "label": "fruit", "split": "train"},
            {"id": "b1", "text": "мяч футбол гол", "label": "sport", "split": "train"},
            {"id": "b2", "text": "футбол команда мяч", "label": "sport", "split": "train"},
            {"id": "t1", "text": "яблоко груша", "label": "fruit", "split": "test"},
            {"id": "t2", "text": "футбол мяч", "label": "sport", "split": "test"},
        ],
        split=True,
    )

    protocol = evaluate_corpus(corpus)

    assert protocol["split"] == {
        "kind": "provided",
        "random_state": 42,
        "train_documents": 4,
        "test_documents": 2,
    }
    assert protocol["retrieval"]["skipped_queries"] == 0
    assert protocol["retrieval"]["tfidf"] == {"precision_at_5": 0.4, "hit_at_5": 1.0}
    assert protocol["retrieval"]["lsa"] == {"precision_at_5": 0.4, "hit_at_5": 1.0}
    assert {item["id"] for example in protocol["examples"] for item in example["results"]} <= {
        "a1",
        "a2",
        "b1",
        "b2",
    }


def test_evaluation_makes_deterministic_stratified_split_and_skips_oov(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    rows = [
        {"id": f"a{index}", "text": f"яблоко груша {index}", "label": "fruit"} for index in range(5)
    ] + [{"id": f"b{index}", "text": f"мяч футбол {index}", "label": "sport"} for index in range(5)]
    _write_corpus(corpus, rows, split=False)

    first = evaluate_corpus(corpus)
    second = evaluate_corpus(corpus)

    assert first["split"] == second["split"]
    assert first["examples"] == second["examples"]


def test_evaluation_counts_oov_test_query_as_skipped(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    _write_corpus(
        corpus,
        [
            {"id": "a1", "text": "яблоко груша сад", "label": "fruit", "split": "train"},
            {"id": "a2", "text": "яблоко сладкая груша", "label": "fruit", "split": "train"},
            {"id": "b1", "text": "мяч футбол гол", "label": "sport", "split": "train"},
            {"id": "b2", "text": "футбол команда мяч", "label": "sport", "split": "train"},
            {"id": "t1", "text": "яблоко груша", "label": "fruit", "split": "test"},
            {"id": "t2", "text": "космодром ракета", "label": "sport", "split": "test"},
        ],
        split=True,
    )

    protocol = evaluate_corpus(corpus)

    assert protocol["retrieval"]["evaluated_queries"] == 1
    assert protocol["retrieval"]["skipped_queries"] == 1


def test_evaluation_fits_only_train_and_excludes_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus.csv"
    rows = [
        {"id": "a1", "text": "яблоко груша сад", "label": "fruit", "split": "train"},
        {"id": "a2", "text": "яблоко сладкая груша", "label": "fruit", "split": "train"},
        {"id": "b1", "text": "мяч футбол гол", "label": "sport", "split": "train"},
        {"id": "b2", "text": "футбол команда мяч", "label": "sport", "split": "train"},
        {
            "id": "v1",
            "text": "секретный validation термин",
            "label": "fruit",
            "split": "validation",
        },
        {"id": "t1", "text": "яблоко груша", "label": "fruit", "split": "test"},
        {"id": "t2", "text": "футбол мяч", "label": "sport", "split": "test"},
    ]
    _write_corpus(corpus, rows, split=True)
    original_train = model._train_rows
    fitted_rows: list[dict[str, str]] = []

    def capture_train(rows_for_fit: list[dict[str, str]], model_id: str) -> model.SemanticModel:
        fitted_rows.extend(rows_for_fit)
        return original_train(rows_for_fit, model_id)

    monkeypatch.setattr(model, "_train_rows", capture_train)

    protocol = evaluate_corpus(corpus)

    assert [row["id"] for row in fitted_rows] == ["a1", "a2", "b1", "b2"]
    assert protocol["split"]["train_documents"] == 4
    assert protocol["split"]["test_documents"] == 2


def test_evaluation_reports_undefined_silhouette_reason(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    _write_corpus(
        corpus,
        [
            {"id": "a1", "text": "яблоко груша сад", "label": "fruit", "split": "train"},
            {"id": "a2", "text": "яблоко сладкая груша", "label": "fruit", "split": "train"},
            {"id": "b1", "text": "мяч футбол гол", "label": "sport", "split": "train"},
            {"id": "b2", "text": "футбол команда мяч", "label": "sport", "split": "train"},
            {"id": "t1", "text": "яблоко груша", "label": "fruit", "split": "test"},
            {"id": "t2", "text": "футбол мяч", "label": "sport", "split": "test"},
        ],
        split=True,
    )

    protocol = evaluate_corpus(corpus)

    assert protocol["silhouette"]["lsa"] == {
        "value": None,
        "reason": "для silhouette нужны минимум два класса и больше образцов",
    }
    assert protocol["silhouette"]["lda"] == protocol["silhouette"]["lsa"]


def test_evaluation_rejects_unlabeled_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    corpus.write_text("id,text\na,яблоко груша\nb,мяч футбол\n", encoding="utf-8")

    with pytest.raises(ModelError, match="полностью размеченного"):
        evaluate_corpus(corpus)
