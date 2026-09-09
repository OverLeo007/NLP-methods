"""Подготовка признаков для нормализованных моделей тональности «мешок слов»."""

from __future__ import annotations

from collections.abc import Iterable

from sklearn.feature_extraction.text import CountVectorizer

from .normalization import NormalizationConfig, TextNormalizer


def normalize_reviews(reviews: Iterable[str], config: NormalizationConfig) -> list[str]:
    """Вернуть нормализованный документ с пробелами для каждого входного отзыва.

    Один нормализатор намеренно используется внутри одной части данных: кэш лемм ускоряет
    обучение, но изменяемое состояние не разделяется между конфигурациями.
    """
    normalizer = TextNormalizer(config)
    return [" ".join(normalizer.normalize(review).normalized_tokens) for review in reviews]


def build_vectorizer() -> CountVectorizer:
    """Построить заданное спецификацией частотное представление униграмм."""
    return CountVectorizer(
        analyzer=str.split,
        lowercase=False,
        max_features=50_000,
        min_df=2,
        ngram_range=(1, 1),
        token_pattern=None,
    )
