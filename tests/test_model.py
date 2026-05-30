import numpy as np

from src.evaluate import find_best_threshold, score_predictions
from src.preprocessing import make_train_val_test_split


def test_train_val_test_split_keeps_expected_sizes():
    X = np.arange(100).reshape(-1, 1)
    y = np.array([0, 1] * 50)

    X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_test_split(X, y)

    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15
    assert y_train.sum() == 35
    assert y_val.sum() + y_test.sum() == 15


def test_threshold_search_uses_probabilities():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    threshold, validation_f1 = find_best_threshold(y_true, y_proba)

    assert 0.2 < threshold <= 0.8
    assert validation_f1 > 0.99


def test_score_predictions_handles_zero_division():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 0])
    y_proba = np.array([0.1, 0.2])

    scores = score_predictions(y_true, y_pred, y_proba)

    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0
    assert scores["f1"] == 0.0
