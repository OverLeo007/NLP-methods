"""Обучение, сохранение и применение семантической модели LSA."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import sklearn
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from LabsCommon.normalization import NormalizationConfig, TextNormalizer

from .data import PreparationError, read_canonical, sha256_file

RANDOM_STATE: Final = 42
MAX_FEATURES: Final = 20_000


class ModelError(ValueError):
    """Ошибка обучения, загрузки или применения модели."""


@dataclass(frozen=True)
class CorpusDocument:
    """Документ, сохранённый для поиска после обучения."""

    identifier: str
    text: str
    label: str | None


@dataclass
class SemanticModel:
    """Пакет обученных преобразований и представлений обучающего корпуса."""

    documents: list[CorpusDocument]
    model_id: str
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    lda: LinearDiscriminantAnalysis | None
    tfidf_matrix: sparse.csr_matrix
    lsa_matrix: np.ndarray
    lda_matrix: np.ndarray | None
    normalization: dict[str, object]
    settings: dict[str, object]
    versions: dict[str, str]

    def normalize(self, text: str) -> tuple[str, ...]:
        """Повторить фиксированную нормализацию, применённую на обучении."""
        normalizer = TextNormalizer(
            NormalizationConfig(method="stem", remove_stopwords=True, use_synonyms=False)
        )
        return normalizer.normalize(text).normalized_tokens


def _normalized_documents(rows: list[dict[str, str]]) -> tuple[list[str], list[CorpusDocument]]:
    normalizer = TextNormalizer(
        NormalizationConfig(method="stem", remove_stopwords=True, use_synonyms=False)
    )
    normalized: list[str] = []
    documents: list[CorpusDocument] = []
    for row in rows:
        tokens = normalizer.normalize(row["text"]).normalized_tokens
        if not tokens:
            raise ModelError(f"Документ {row['id']!r} пуст после нормализации")
        normalized.append(" ".join(tokens))
        documents.append(CorpusDocument(row["id"], row["text"], row["label"] or None))
    return normalized, documents


def _validate_labels(documents: list[CorpusDocument]) -> list[str] | None:
    labels = [document.label for document in documents]
    if all(label is None for label in labels):
        return None
    if any(label is None for label in labels):
        raise ModelError("Корпус смешивает размеченные и неразмеченные документы")
    result = [label for label in labels if label is not None]
    counts = {label: result.count(label) for label in sorted(set(result))}
    if len(counts) < 2:
        raise ModelError("Для LDA нужны минимум два класса")
    if min(counts.values()) < 2:
        raise ModelError("Для LDA нужны минимум два документа каждого класса")
    return result


def _train_rows(rows: list[dict[str, str]], model_id: str) -> SemanticModel:
    """Обучить модель только на явно переданных строках корпуса."""
    if len(rows) < 2:
        raise ModelError("Для TF-IDF и LSA нужны минимум два документа")

    normalized, documents = _normalized_documents(rows)
    labels = _validate_labels(documents)
    vectorizer = TfidfVectorizer(
        analyzer=str.split,
        lowercase=False,
        max_features=MAX_FEATURES,
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        smooth_idf=True,
        norm="l2",
    )
    try:
        tfidf = vectorizer.fit_transform(normalized).tocsr()
    except ValueError as error:
        raise ModelError(f"Не удалось обучить TF-IDF: {error}") from error
    feature_count = tfidf.shape[1]
    if feature_count < 2:
        raise ModelError("Для TF-IDF и LSA нужны минимум два различных нормализованных термина")
    components = min(50, len(documents) - 1, feature_count - 1)
    if components < 1:
        raise ModelError("Недостаточно данных для хотя бы одной LSA-компоненты")
    svd = TruncatedSVD(n_components=components, random_state=RANDOM_STATE)
    lsa = svd.fit_transform(tfidf)
    if not np.isfinite(lsa).all():
        raise ModelError("LSA дала нечисловые координаты")

    lda: LinearDiscriminantAnalysis | None = None
    lda_vectors: np.ndarray | None = None
    if labels is not None:
        lda = LinearDiscriminantAnalysis(solver="svd")
        try:
            lda_vectors = lda.fit_transform(lsa, labels)
        except ValueError as error:
            raise ModelError(f"Не удалось обучить LDA: {error}") from error
        if lda_vectors.shape[1] < 1 or not np.isfinite(lda_vectors).all():
            raise ModelError("LDA не сформировала допустимое разделяющее преобразование")

    return SemanticModel(
        documents=documents,
        model_id=model_id,
        vectorizer=vectorizer,
        svd=svd,
        lda=lda,
        tfidf_matrix=tfidf,
        lsa_matrix=lsa,
        lda_matrix=lda_vectors,
        normalization={"method": "stem", "stopwords": True, "synonyms": False},
        settings={
            "tfidf": {
                "max_features": MAX_FEATURES,
                "min_df": 1,
                "max_df": 1.0,
                "sublinear_tf": True,
                "smooth_idf": True,
                "norm": "l2",
            },
            "lsa": {"random_state": RANDOM_STATE, "components": components},
            "lda": {
                "enabled": labels is not None,
                "components": None if lda_vectors is None else int(lda_vectors.shape[1]),
            },
        },
        versions={"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    )


def train_model(corpus: Path) -> SemanticModel:
    """Обучить TF-IDF, LSA и при наличии меток supervised LDA на всём CSV."""
    try:
        rows = list(read_canonical(corpus))
    except PreparationError as error:
        raise ModelError(str(error)) from error
    return _train_rows(rows, sha256_file(corpus))


def _evaluation_split(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    labels = [row["label"] for row in rows]
    if not labels or any(not label for label in labels):
        raise ModelError("Оценка доступна только для полностью размеченного корпуса")
    with_split = [row for row in rows if row["split"]]
    if with_split:
        if len(with_split) != len(rows):
            raise ModelError("CSV для оценки смешивает строки с split и без него")
        train_rows = [row for row in rows if row["split"] == "train"]
        test_rows = [row for row in rows if row["split"] == "test"]
        if not train_rows or not test_rows:
            raise ModelError("CSV с split должен содержать непустые train и test")
        return train_rows, test_rows, "provided"
    try:
        train_rows, test_rows = train_test_split(
            rows, test_size=0.20, stratify=labels, random_state=RANDOM_STATE
        )
    except ValueError as error:
        raise ModelError(
            f"Не удалось создать стратифицированное разбиение 80/20: {error}"
        ) from error
    return train_rows, test_rows, "stratified_80_20"


def _silhouette(vectors: np.ndarray | None, labels: list[str]) -> dict[str, object]:
    if vectors is None:
        return {"value": None, "reason": "LDA не обучалась"}
    class_count = len(set(labels))
    if class_count < 2 or len(labels) <= class_count:
        return {
            "value": None,
            "reason": "для silhouette нужны минимум два класса и больше образцов",
        }
    try:
        return {"value": float(silhouette_score(vectors, labels)), "reason": None}
    except ValueError as error:
        return {"value": None, "reason": str(error)}


def _retrieval_example(
    model: SemanticModel, query: dict[str, str], tfidf: sparse.csr_matrix, lsa: np.ndarray
) -> dict[str, object]:
    tfidf_scores = cosine_similarity(tfidf, model.tfidf_matrix)[0]
    lsa_scores = cosine_similarity(lsa, model.lsa_matrix)[0]
    order = sorted(
        range(len(model.documents)),
        key=lambda index: (-lsa_scores[index], model.documents[index].identifier),
    )[:5]
    return {
        "query_id": query["id"],
        "label": query["label"],
        "results": [
            {
                "id": model.documents[index].identifier,
                "label": model.documents[index].label,
                "tfidf_cosine": float(tfidf_scores[index]),
                "lsa_cosine": float(lsa_scores[index]),
            }
            for index in order
        ],
    }


def evaluate_corpus(corpus: Path) -> dict[str, object]:
    """Оценить TF-IDF и LSA на test без подгонки преобразований по test."""
    try:
        rows = list(read_canonical(corpus))
    except PreparationError as error:
        raise ModelError(str(error)) from error
    train_rows, test_rows, split_kind = _evaluation_split(rows)
    model = _train_rows(train_rows, f"evaluation:{sha256_file(corpus)}")
    test_texts = [row["text"] for row in test_rows]
    test_labels = [row["label"] for row in test_rows]
    normalized = [" ".join(model.normalize(text)) for text in test_texts]
    test_tfidf = model.vectorizer.transform(normalized).tocsr()
    test_lsa = model.svd.transform(test_tfidf)
    test_lda = model.lda.transform(test_lsa) if model.lda is not None else None

    tfidf_precision = 0.0
    lsa_precision = 0.0
    tfidf_hits = 0
    lsa_hits = 0
    skipped = 0
    examples: list[dict[str, object]] = []
    for index, row in enumerate(test_rows):
        query_tfidf = test_tfidf[index]
        if query_tfidf.nnz == 0:
            skipped += 1
            continue
        query_lsa = test_lsa[index : index + 1]
        tfidf_scores = cosine_similarity(query_tfidf, model.tfidf_matrix)[0]
        lsa_scores = cosine_similarity(query_lsa, model.lsa_matrix)[0]
        tfidf_order = sorted(
            range(len(model.documents)),
            key=lambda item: (-tfidf_scores[item], model.documents[item].identifier),
        )[:5]
        lsa_order = sorted(
            range(len(model.documents)),
            key=lambda item: (-lsa_scores[item], model.documents[item].identifier),
        )[:5]
        tfidf_relevant = sum(model.documents[item].label == row["label"] for item in tfidf_order)
        lsa_relevant = sum(model.documents[item].label == row["label"] for item in lsa_order)
        tfidf_precision += tfidf_relevant / 5
        lsa_precision += lsa_relevant / 5
        tfidf_hits += int(tfidf_relevant > 0)
        lsa_hits += int(lsa_relevant > 0)
        if len(examples) < 3:
            examples.append(_retrieval_example(model, row, query_tfidf, query_lsa))
    evaluated = len(test_rows) - skipped
    if evaluated == 0:
        raise ModelError("Ни один test-запрос не имеет терминов в словаре train")
    return {
        "corpus": {"path": str(corpus), "sha256": sha256_file(corpus)},
        "split": {
            "kind": split_kind,
            "random_state": RANDOM_STATE,
            "train_documents": len(train_rows),
            "test_documents": len(test_rows),
        },
        "parameters": {
            "max_features": MAX_FEATURES,
            "random_state": RANDOM_STATE,
            "top_k": 5,
            "normalization": model.normalization,
        },
        "model": {
            "features": int(model.tfidf_matrix.shape[1]),
            "lsa_components": int(model.lsa_matrix.shape[1]),
            "lda_components": None if test_lda is None else int(test_lda.shape[1]),
            "versions": model.versions,
        },
        "retrieval": {
            "evaluated_queries": evaluated,
            "skipped_queries": skipped,
            "tfidf": {
                "precision_at_5": tfidf_precision / evaluated,
                "hit_at_5": tfidf_hits / evaluated,
            },
            "lsa": {"precision_at_5": lsa_precision / evaluated, "hit_at_5": lsa_hits / evaluated},
        },
        "silhouette": {
            "lsa": _silhouette(test_lsa, test_labels),
            "lda": _silhouette(test_lda, test_labels),
        },
        "examples": examples,
    }


def save_model(model: SemanticModel, path: Path) -> None:
    """Сохранить пакет модели в указанном пользователем файле."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, path)
    except (OSError, ValueError, TypeError) as error:
        raise ModelError(f"Не удалось сохранить модель {path}: {error}") from error


def _finite_numeric_state(
    value: object, description: str, expected_shape: tuple[int, ...] | None = None
) -> None:
    """Проверить числовое состояние преобразования, используемое после загрузки."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ModelError(f"Пакет модели повреждён: {description} содержит неконечные значения")
    if expected_shape is not None and array.shape != expected_shape:
        raise ModelError(f"Пакет модели повреждён: неверная размерность {description}")


def load_model(path: Path) -> SemanticModel:
    """Загрузить проверенный пакет без повторного обучения."""
    try:
        model = joblib.load(path)
    except Exception as error:
        # Joblib может передать исключения pickle разных типов для повреждённого файла.
        raise ModelError(f"Не удалось загрузить модель {path}: {error}") from error
    if not isinstance(model, SemanticModel):
        raise ModelError("Файл не содержит пакет семантической модели Laba2")
    if not model.documents or not sparse.isspmatrix_csr(model.tfidf_matrix):
        raise ModelError("Пакет модели повреждён: не совпадает обучающий корпус")
    document_count = len(model.documents)
    feature_count = model.tfidf_matrix.shape[1]
    if model.tfidf_matrix.shape[0] != document_count or feature_count < 2:
        raise ModelError("Пакет модели повреждён: не совпадает обучающий корпус")
    _finite_numeric_state(model.tfidf_matrix.data, "TF-IDF")
    _finite_numeric_state(
        getattr(model.vectorizer, "idf_", None), "параметры TF-IDF", (feature_count,)
    )

    n_components = getattr(model.svd, "n_components", None)
    if not isinstance(n_components, (int, np.integer)) or n_components < 1:
        raise ModelError("Пакет модели повреждён: неверное число LSA-компонент")
    component_count = int(n_components)
    _finite_numeric_state(
        getattr(model.svd, "components_", None), "компоненты LSA", (component_count, feature_count)
    )
    for attribute in ("explained_variance_", "explained_variance_ratio_", "singular_values_"):
        _finite_numeric_state(
            getattr(model.svd, attribute, None), f"LSA.{attribute}", (component_count,)
        )
    if model.lsa_matrix.shape != (document_count, component_count):
        raise ModelError("Пакет модели повреждён: неверные размерности TF-IDF или LSA")
    _finite_numeric_state(model.lsa_matrix, "LSA-координаты")
    if model.lda is None:
        if model.lda_matrix is not None:
            raise ModelError("Пакет модели повреждён: LDA-векторы есть без LDA")
    else:
        if model.lda_matrix is None or model.lda_matrix.shape[0] != document_count:
            raise ModelError("Пакет модели повреждён: неверные LDA-векторы")
        lda_components = model.lda_matrix.shape[1]
        if lda_components < 1:
            raise ModelError("Пакет модели повреждён: неверные LDA-векторы")
        _finite_numeric_state(model.lda_matrix, "LDA-векторы")
        if getattr(model.lda, "n_features_in_", None) != component_count:
            raise ModelError("Пакет модели повреждён: неверное число признаков LDA")
        _finite_numeric_state(getattr(model.lda, "xbar_", None), "центр LDA", (component_count,))
        _finite_numeric_state(
            getattr(model.lda, "scalings_", None),
            "преобразование LDA",
            (component_count, lda_components),
        )
        for attribute in ("means_", "priors_", "coef_", "intercept_"):
            _finite_numeric_state(getattr(model.lda, attribute, None), f"LDA.{attribute}")
    if not isinstance(model.settings, dict) or not isinstance(model.versions, dict):
        raise ModelError("Пакет модели повреждён: отсутствуют настройки или версии")
    return model


def model_summary(model: SemanticModel, path: Path) -> dict[str, object]:
    """Вернуть устойчивую JSON-сводку обученного пакета."""
    return {
        "model": str(path),
        "model_id": model.model_id,
        "documents": len(model.documents),
        "features": int(model.tfidf_matrix.shape[1]),
        "lsa_components": int(model.lsa_matrix.shape[1]),
        "lda_components": None if model.lda_matrix is None else int(model.lda_matrix.shape[1]),
        "labeled": model.lda is not None,
        "normalization": model.normalization,
        "settings": model.settings,
        "versions": model.versions,
    }


def analyze(model: SemanticModel, text: str, top_k: int) -> dict[str, object]:
    """Преобразовать запрос и отсортировать документы по LSA-сходству."""
    if top_k < 1:
        raise ModelError("--top-k должен быть положительным целым числом")
    tokens = model.normalize(text)
    if not tokens:
        raise ModelError("Запрос пуст после нормализации")
    tfidf = model.vectorizer.transform([" ".join(tokens)]).tocsr()
    if tfidf.nnz == 0:
        raise ModelError("Токены запроса отсутствуют в словаре обучающего корпуса")
    lsa = model.svd.transform(tfidf)
    lda = model.lda.transform(lsa) if model.lda is not None else None
    names = model.vectorizer.get_feature_names_out()
    sparse_vector = [
        {"term": str(names[index]), "weight": float(weight)}
        for index, weight in zip(tfidf.indices, tfidf.data, strict=True)
    ]
    tfidf_scores = cosine_similarity(tfidf, model.tfidf_matrix)[0]
    lsa_scores = cosine_similarity(lsa, model.lsa_matrix)[0]
    order = sorted(
        range(len(model.documents)),
        key=lambda index: (-lsa_scores[index], model.documents[index].identifier),
    )
    results: list[dict[str, object]] = []
    for place, index in enumerate(order[:top_k], start=1):
        document = model.documents[index]
        fragment = " ".join(document.text.split())[:200]
        item: dict[str, object] = {
            "id": document.identifier,
            "fragment": fragment,
            "rank": place,
            "tfidf_cosine": float(tfidf_scores[index]),
            "lsa_cosine": float(lsa_scores[index]),
        }
        if document.label is not None:
            item["label"] = document.label
        results.append(item)
    return {
        "model": {
            "id": model.model_id,
            "documents": len(model.documents),
            "features": int(model.tfidf_matrix.shape[1]),
            "lsa_components": int(model.lsa_matrix.shape[1]),
            "lda_components": None if model.lda_matrix is None else int(model.lda_matrix.shape[1]),
            "versions": model.versions,
        },
        "normalization": model.normalization,
        "settings": model.settings,
        "query_tokens": list(tokens),
        "tfidf": sparse_vector,
        "lsa": [float(value) for value in lsa[0]],
        "lda": None if lda is None else [float(value) for value in lda[0]],
        "documents": results,
    }


def topics(model: SemanticModel, top_terms: int) -> dict[str, object]:
    """Показать положительные и отрицательные нагрузки терминов SVD."""
    if top_terms < 1:
        raise ModelError("--top-terms должен быть положительным целым числом")
    names = model.vectorizer.get_feature_names_out()
    output: list[dict[str, object]] = []
    for number, component in enumerate(model.svd.components_, start=1):
        positive = sorted(
            (index for index in range(len(names)) if component[index] > 0),
            key=lambda index: (-component[index], names[index]),
        )[:top_terms]
        negative = sorted(
            (index for index in range(len(names)) if component[index] < 0),
            key=lambda index: (component[index], names[index]),
        )[:top_terms]
        output.append(
            {
                "component": number,
                "positive": [
                    {"term": str(names[index]), "weight": float(component[index])}
                    for index in positive
                ],
                "negative": [
                    {"term": str(names[index]), "weight": float(component[index])}
                    for index in negative
                ],
            }
        )
    return {"lsa_components": len(output), "topics": output}
