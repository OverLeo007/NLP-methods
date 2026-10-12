from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from Laba1.src.data import (
    DataValidationError,
    clean_reviews,
    load_source,
    split_reviews,
)


def test_load_source_requires_exact_schema_and_known_labels(tmp_path: Path) -> None:
    wrong_schema = tmp_path / "wrong.tsv"
    wrong_schema.write_text("text\tsentiment\nA\tpositive\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="столбцы"):
        load_source(wrong_schema)

    wrong_label = tmp_path / "label.tsv"
    wrong_label.write_text("review\tsentiment\nA\tmixed\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="метки"):
        load_source(wrong_label)


def test_cleaning_corrects_label_resolves_majority_and_removes_tie() -> None:
    source = pd.DataFrame(
        [
            (" A ", "negative"),
            ("A", "positive"),
            ("A", "negative"),
            ("B", "negative"),
            ("B", "positive"),
            ("C", "neautral"),
            ("C", "neautral"),
            ("D", "positive"),
            ("   ", "positive"),
        ],
        columns=["review", "sentiment"],
    )

    cleaned, stats = clean_reviews(source)

    assert list(cleaned.itertuples(index=False, name=None)) == [
        ("A", "negative"),
        ("C", "neutral"),
        ("D", "positive"),
    ]
    assert stats.input_rows == 9
    assert stats.empty_reviews_removed == 1
    assert stats.tied_texts_removed == 1
    assert stats.conflicting_texts_resolved == 1
    assert stats.output_rows == 3
    assert stats.class_counts == {"negative": 1, "neutral": 1, "positive": 1}


def test_cleaning_does_not_depend_on_input_order() -> None:
    source = pd.DataFrame(
        [("B", "positive"), ("A", "negative"), ("A", "negative")],
        columns=["review", "sentiment"],
    )
    first, _ = clean_reviews(source)
    second, _ = clean_reviews(source.sample(frac=1, random_state=7))

    pd.testing.assert_frame_equal(first, second)


@pytest.fixture
def corpus() -> pd.DataFrame:
    rows = [
        (f"{label}-{number:03d}", label)
        for label in ("negative", "neutral", "positive")
        for number in range(100)
    ]
    return pd.DataFrame(rows, columns=["review", "sentiment"])


def test_split_is_stratified_deterministic_and_disjoint(corpus: pd.DataFrame) -> None:
    first = split_reviews(corpus)
    second = split_reviews(corpus)

    assert (len(first.train), len(first.validation), len(first.test)) == (210, 45, 45)
    for name in ("train", "validation", "test"):
        first_part = getattr(first, name)
        second_part = getattr(second, name)
        pd.testing.assert_frame_equal(first_part, second_part)
        assert first_part["sentiment"].value_counts().to_dict() == {
            "negative": len(first_part) // 3,
            "neutral": len(first_part) // 3,
            "positive": len(first_part) // 3,
        }

    train = set(first.train["review"])
    validation = set(first.validation["review"])
    test = set(first.test["review"])
    assert not (train & validation or train & test or validation & test)


def test_split_rejects_duplicate_reviews(corpus: pd.DataFrame) -> None:
    source = corpus.copy()
    source.loc[1, "review"] = source.loc[0, "review"]

    with pytest.raises(DataValidationError, match="уникальными"):
        split_reviews(source)
