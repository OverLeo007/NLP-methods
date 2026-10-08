from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from Laba2.src.model import ModelError, analyze, load_model, save_model, topics, train_model


def _write_corpus(path: Path, *, labeled: bool) -> None:
    fields = ("id", "text", "label") if labeled else ("id", "text")
    rows = [
        ("a", "Красный сад цветет весной", "nature"),
        ("b", "Зеленый лес растет летом", "nature"),
        ("c", "Футбольная команда забила гол", "sport"),
        ("d", "Спортсмен выиграл футбольный матч", "sport"),
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for identifier, text, label in rows:
            row = {"id": identifier, "text": text}
            if labeled:
                row["label"] = label
            writer.writerow(row)


def test_model_uses_same_normalization_and_survives_save_load(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    _write_corpus(corpus, labeled=True)
    model = train_model(corpus)
    before = analyze(model, "Футбольные матчи спортсменов", 4)
    path = tmp_path / "model.joblib"
    save_model(model, path)
    after = analyze(load_model(path), "Футбольные матчи спортсменов", 4)

    assert before["query_tokens"] == list(model.normalize("Футбольные матчи спортсменов"))
    assert before["documents"] == after["documents"]
    assert before["model"] == after["model"]
    assert before["model"]["id"] == model.model_id
    assert before["model"]["lsa_components"] == model.lsa_matrix.shape[1]
    assert before["model"]["versions"] == model.versions
    assert np.allclose(before["lsa"], after["lsa"])
    assert np.allclose(before["lda"], after["lda"])
    assert len(before["lsa"]) == model.lsa_matrix.shape[1]
    assert model.settings["tfidf"]["max_features"] == 20_000
    assert model.lda_matrix is not None
    assert model.lda_matrix.shape[1] == min(1, model.lsa_matrix.shape[1])
    assert np.isfinite(model.tfidf_matrix.data).all()
    assert np.isfinite(model.lsa_matrix).all()


def test_unlabeled_model_has_no_lda_and_rejects_oov_query(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    _write_corpus(corpus, labeled=False)
    model = train_model(corpus)

    result = analyze(model, "футбольная команда", 2)
    assert result["lda"] is None
    assert [document["id"] for document in result["documents"]] == ["c", "d"]
    with pytest.raises(ModelError, match="отсутствуют в словаре"):
        analyze(model, "несуществующее слово", 2)


def test_load_rejects_damaged_model_package(tmp_path: Path) -> None:
    path = tmp_path / "damaged.joblib"
    path.write_bytes(b"not a joblib package")

    with pytest.raises(ModelError, match="Не удалось загрузить"):
        load_model(path)


def test_load_rejects_nonfinite_tfidf_values(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    package = tmp_path / "model.joblib"
    _write_corpus(corpus, labeled=True)
    model = train_model(corpus)
    model.tfidf_matrix.data[0] = np.nan
    save_model(model, package)

    with pytest.raises(ModelError, match="TF-IDF содержит неконечные значения"):
        load_model(package)


@pytest.mark.parametrize(
    ("attribute", "message"),
    (
        ("vectorizer.idf_", "параметры TF-IDF"),
        ("svd.components_", "компоненты LSA"),
        ("lda.scalings_", "преобразование LDA"),
    ),
)
def test_load_rejects_nonfinite_transform_state(
    tmp_path: Path, attribute: str, message: str
) -> None:
    corpus = tmp_path / "corpus.csv"
    package = tmp_path / "model.joblib"
    _write_corpus(corpus, labeled=True)
    model = train_model(corpus)
    target: object = model
    for name in attribute.split("."):
        target = getattr(target, name)
    target.flat[0] = np.nan
    save_model(model, package)

    with pytest.raises(ModelError, match=message):
        load_model(package)


def test_loading_package_does_not_refit_transforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus.csv"
    package = tmp_path / "model.joblib"
    _write_corpus(corpus, labeled=True)
    save_model(train_model(corpus), package)

    def fail_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("load_model не должен вызывать fit")

    monkeypatch.setattr("sklearn.feature_extraction.text.TfidfVectorizer.fit", fail_fit)
    monkeypatch.setattr("sklearn.decomposition.TruncatedSVD.fit", fail_fit)
    assert load_model(package).documents[0].identifier == "a"


def test_topics_keeps_component_signs_separate(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    _write_corpus(corpus, labeled=False)
    model = train_model(corpus)
    model.svd.components_[0] = np.abs(model.svd.components_[0])

    result = topics(model, 3)

    assert result["topics"][0]["negative"] == []
    assert all(
        item["weight"] > 0 for component in result["topics"] for item in component["positive"]
    )
    assert all(
        item["weight"] < 0 for component in result["topics"] for item in component["negative"]
    )


def test_analysis_has_full_payload_and_stable_top_k(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    _write_corpus(corpus, labeled=True)
    result = analyze(train_model(corpus), "футбольная команда", 2)

    assert set(result) == {
        "model",
        "normalization",
        "settings",
        "query_tokens",
        "tfidf",
        "lsa",
        "lda",
        "documents",
    }
    assert result["lda"] is not None
    assert len(result["documents"]) == 2
    assert all(
        {"id", "label", "fragment", "rank", "tfidf_cosine", "lsa_cosine"} <= set(item)
        for item in result["documents"]
    )
    assert [item["rank"] for item in result["documents"]] == [1, 2]
    with pytest.raises(ModelError, match="положительным"):
        analyze(train_model(corpus), "футбольная команда", 0)
