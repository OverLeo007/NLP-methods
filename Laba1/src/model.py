"""Обучение и проверяемое сохранение модели тональности."""

from __future__ import annotations

import os
import pickle
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from string import hexdigits
from typing import Final

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from .data import PUBLIC_LABELS, RANDOM_STATE
from .features import build_vectorizer, normalize_reviews
from .normalization import NormalizationConfig, NormalizationError

DEPENDENCY_NAMES: Final = ("joblib", "pandas", "pymorphy3", "scikit-learn", "snowballstemmer")


class ModelPackageError(ValueError):
    """Ошибка неполного или несовместимого пакета модели."""


@dataclass(frozen=True)
class FittedModel:
    """Обученные vectorizer и классификатор вместе с публичным контрактом."""

    config: NormalizationConfig
    vectorizer: CountVectorizer
    classifier: MultinomialNB
    class_order: tuple[str, ...] = PUBLIC_LABELS

    def predict_probabilities(self, reviews: list[str]) -> list[list[float]]:
        """Вычислить вероятности в фиксированном публичном порядке классов."""
        documents = normalize_reviews(reviews, self.config)
        probabilities = self.classifier.predict_proba(self.vectorizer.transform(documents))
        positions = [list(self.classifier.classes_).index(label) for label in self.class_order]
        return probabilities[:, positions].tolist()


@dataclass(frozen=True)
class LoadedModelPackage:
    model: FittedModel
    metadata: dict[str, object]


def fit_model(reviews: list[str], labels: list[str], config: NormalizationConfig) -> FittedModel:
    """Обучить словарь и MultinomialNB строго на одной обучающей части."""
    if len(reviews) != len(labels) or not reviews:
        raise ModelPackageError("Для обучения требуются непустые тексты и соответствующие метки")
    unknown_labels = sorted(set(labels) - set(PUBLIC_LABELS))
    if unknown_labels:
        raise ModelPackageError(f"Неизвестные метки обучения: {unknown_labels}")

    vectorizer = build_vectorizer()
    try:
        matrix = vectorizer.fit_transform(normalize_reviews(reviews, config))
    except ValueError as error:
        raise ModelPackageError(f"Не удалось построить словарь обучения: {error}") from error
    classifier = MultinomialNB(alpha=1.0, fit_prior=True)
    classifier.fit(matrix, labels)
    if tuple(classifier.classes_) != PUBLIC_LABELS:
        raise ModelPackageError("Обучающая выборка должна содержать все три публичных класса")
    return FittedModel(config=config, vectorizer=vectorizer, classifier=classifier)


def build_metadata(*, corpus_sha256: str, trained_at: datetime | None = None) -> dict[str, object]:
    """Сформировать metadata локального пакета без отдельной версии схемы."""
    timestamp = trained_at or datetime.now(UTC)
    return {
        "corpus_sha256": corpus_sha256.lower(),
        "corpus_profile": "rureviews",
        "dependencies": {name: version(name) for name in DEPENDENCY_NAMES},
        "random_state": RANDOM_STATE,
        "trained_at": timestamp.isoformat(),
    }


def _config_payload(config: NormalizationConfig) -> dict[str, object]:
    return {
        "method": config.method,
        "remove_stopwords": config.remove_stopwords,
        "use_synonyms": config.use_synonyms,
    }


def _require_metadata(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise ModelPackageError("Пакет модели содержит некорректные metadata")
    required = {"corpus_sha256", "corpus_profile", "dependencies", "random_state", "trained_at"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ModelPackageError(f"В metadata модели отсутствуют поля: {', '.join(missing)}")
    if (
        not isinstance(metadata["corpus_sha256"], str)
        or len(metadata["corpus_sha256"]) != 64
        or set(metadata["corpus_sha256"]) - set(hexdigits)
    ):
        raise ModelPackageError("В metadata модели некорректен SHA-256 корпуса")
    if metadata["corpus_profile"] != "rureviews":
        raise ModelPackageError("Пакет модели создан не для поставляемого RuReviews")
    if not isinstance(metadata["dependencies"], dict) or set(DEPENDENCY_NAMES) - set(
        metadata["dependencies"]
    ):
        raise ModelPackageError("В metadata модели отсутствуют версии зависимостей")
    if not all(
        isinstance(metadata["dependencies"][name], str) and metadata["dependencies"][name]
        for name in DEPENDENCY_NAMES
    ):
        raise ModelPackageError("В metadata модели некорректны версии зависимостей")
    if metadata["random_state"] != RANDOM_STATE:
        raise ModelPackageError("Пакет модели создан с несовместимым seed")
    if not isinstance(metadata["trained_at"], str):
        raise ModelPackageError("В metadata модели отсутствует дата обучения")
    try:
        datetime.fromisoformat(metadata["trained_at"])
    except ValueError as error:
        raise ModelPackageError("В metadata модели некорректна дата обучения") from error
    return dict(metadata)


def _validate_payload(payload: object) -> LoadedModelPackage:
    if not isinstance(payload, dict):
        raise ModelPackageError("Пакет модели должен быть объектом с полями")
    required = {"config", "vectorizer", "classifier", "class_order", "metadata"}
    missing = sorted(required - set(payload))
    if missing:
        raise ModelPackageError(f"В пакете модели отсутствуют поля: {', '.join(missing)}")
    config_data = payload["config"]
    if not isinstance(config_data, dict):
        raise ModelPackageError("В пакете модели некорректна конфигурация нормализации")
    try:
        config = NormalizationConfig(**config_data)
    except (NormalizationError, TypeError) as error:
        raise ModelPackageError("В пакете модели некорректна конфигурация нормализации") from error
    class_order_data = payload["class_order"]
    if not isinstance(class_order_data, list) or not all(
        isinstance(label, str) for label in class_order_data
    ):
        raise ModelPackageError("Пакет модели содержит неверный порядок публичных классов")
    class_order = tuple(class_order_data)
    if class_order != PUBLIC_LABELS:
        raise ModelPackageError("Пакет модели содержит неверный порядок публичных классов")
    vectorizer = payload["vectorizer"]
    if not all(
        hasattr(vectorizer, name) for name in ("transform", "analyzer", "ngram_range", "min_df")
    ):
        raise ModelPackageError("Пакет модели не содержит совместимый vectorizer")
    if (
        vectorizer.ngram_range != (1, 1)
        or vectorizer.min_df != 2
        or vectorizer.max_features != 50_000
        or vectorizer.lowercase is not False
        or vectorizer.token_pattern is not None
        or vectorizer.analyzer is not str.split
    ):
        raise ModelPackageError("Параметры vectorizer не соответствуют контракту")
    classifier = payload["classifier"]
    if not isinstance(classifier, MultinomialNB):
        raise ModelPackageError("Пакет модели не содержит MultinomialNB")
    if classifier.alpha != 1.0 or classifier.fit_prior is not True:
        raise ModelPackageError("Параметры классификатора не соответствуют контракту")
    if tuple(classifier.classes_) != class_order:
        raise ModelPackageError("Классы классификатора не соответствуют публичному порядку")
    metadata = _require_metadata(payload["metadata"])
    return LoadedModelPackage(
        model=FittedModel(
            config=config,
            vectorizer=vectorizer,
            classifier=classifier,
            class_order=class_order,
        ),
        metadata=metadata,
    )


def save_model_package(path: Path, model: FittedModel, metadata: Mapping[str, object]) -> None:
    """Атомарно сохранить пакет после проверки структуры, принимаемой загрузчиком."""
    path = Path(path)
    payload = {
        "config": _config_payload(model.config),
        "vectorizer": model.vectorizer,
        "classifier": model.classifier,
        "class_order": list(model.class_order),
        "metadata": dict(metadata),
    }
    _validate_payload(payload)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".joblib", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            joblib.dump(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
    except (OSError, ValueError) as error:
        raise ModelPackageError(f"Не удалось сохранить пакет модели {path}: {error}") from error


def load_model_package(path: Path) -> LoadedModelPackage:
    """Загрузить пакет модели и отклонить неполный или несовместимый контракт."""
    try:
        payload = joblib.load(Path(path))
    except (OSError, ValueError, EOFError, IndexError, KeyError, pickle.UnpicklingError) as error:
        raise ModelPackageError(f"Не удалось прочитать пакет модели {path}: {error}") from error
    return _validate_payload(payload)
