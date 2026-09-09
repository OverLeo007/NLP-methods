from __future__ import annotations

import pandas as pd

from Laba1.src.experiment import (
    SELECTED_CONFIG,
    configuration_id,
    evaluate_selected_model,
    train_selected_model,
)


def _partitions() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = []
    test_rows = []
    for label, words in (
        ("negative", ("ужасный брак", "плохой брак", "плохой товар")),
        ("neutral", ("обычный товар", "средний товар", "обычный выбор")),
        ("positive", ("отличный товар", "прекрасный товар", "хороший выбор")),
    ):
        train_rows.extend((word, label) for word in words * 2)
        test_rows.extend((word, label) for word in words)
    return (
        pd.DataFrame(train_rows, columns=["review", "sentiment"]),
        pd.DataFrame(test_rows, columns=["review", "sentiment"]),
    )


def test_training_uses_one_documented_configuration() -> None:
    train, _ = _partitions()

    result = train_selected_model(train)

    assert result.model.config == SELECTED_CONFIG
    assert configuration_id(result.model.config) == "method=stem;stopwords=on;synonyms=off"
    assert result.training_seconds >= 0.0


def test_final_evaluation_reports_fixed_class_order_and_predictions() -> None:
    train, test = _partitions()
    training = train_selected_model(train)

    evaluation = evaluate_selected_model(training.model, test)

    assert len(evaluation.predictions) == len(test)
    assert len(evaluation.confusion_matrix) == 3
    assert all(len(row) == 3 for row in evaluation.confusion_matrix)
    assert set(evaluation.classification_report) == {
        "negative",
        "neutral",
        "positive",
        "accuracy",
        "macro avg",
        "weighted avg",
    }
