from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from Laba1.src.model import (
    ModelPackageError,
    build_metadata,
    fit_model,
    load_model_package,
    save_model_package,
)
from Laba1.src.normalization import NormalizationConfig


@pytest.fixture
def training_data() -> tuple[list[str], list[str]]:
    reviews = [
        "плохой брак",
        "ужасный брак",
        "плохой товар",
        "обычный товар",
        "средний товар",
        "обычный выбор",
        "хороший товар",
        "отличный товар",
        "прекрасный выбор",
    ]
    labels = ["negative"] * 3 + ["neutral"] * 3 + ["positive"] * 3
    return reviews, labels


def test_fit_uses_specified_vectorizer_classifier_and_public_class_order(
    training_data: tuple[list[str], list[str]],
) -> None:
    reviews, labels = training_data

    model = fit_model(reviews, labels, NormalizationConfig())

    assert model.vectorizer.ngram_range == (1, 1)
    assert model.vectorizer.min_df == 2
    assert model.vectorizer.max_features == 50_000
    assert model.vectorizer.lowercase is False
    assert model.vectorizer.token_pattern is None
    assert model.classifier.alpha == 1.0
    assert model.classifier.fit_prior is True
    assert model.class_order == ("negative", "neutral", "positive")
    probabilities = model.predict_probabilities(["плохой товар"])[0]
    assert len(probabilities) == 3
    assert sum(probabilities) == pytest.approx(1.0)


def test_package_without_schema_version_round_trips(
    tmp_path: Path,
    training_data: tuple[list[str], list[str]],
) -> None:
    reviews, labels = training_data
    model = fit_model(reviews, labels, NormalizationConfig(method="stem"))
    metadata = build_metadata(corpus_sha256="a" * 64)
    path = tmp_path / "model.joblib"

    save_model_package(path, model, metadata)

    payload = joblib.load(path)
    loaded = load_model_package(path)
    assert "schema_version" not in payload
    assert loaded.model.config == model.config
    assert loaded.metadata["corpus_sha256"] == "a" * 64
    assert loaded.metadata["corpus_profile"] == "rureviews"
    assert loaded.model.predict_probabilities(["плохой товар"]) == (
        model.predict_probabilities(["плохой товар"])
    )


def test_package_rejects_missing_and_incompatible_structure(
    tmp_path: Path,
    training_data: tuple[list[str], list[str]],
) -> None:
    reviews, labels = training_data
    model = fit_model(reviews, labels, NormalizationConfig(method="stem"))
    path = tmp_path / "model.joblib"
    save_model_package(path, model, build_metadata(corpus_sha256="a" * 64))

    missing = tmp_path / "missing.joblib"
    joblib.dump({"config": {}}, missing)
    with pytest.raises(ModelPackageError, match="отсутствуют поля"):
        load_model_package(missing)

    incompatible = tmp_path / "incompatible.joblib"
    payload = joblib.load(path)
    payload["classifier"].alpha = 0.5
    joblib.dump(payload, incompatible)
    with pytest.raises(ModelPackageError, match="Параметры классификатора"):
        load_model_package(incompatible)
