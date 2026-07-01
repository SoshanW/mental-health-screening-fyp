"""Tests for compute_metrics (needs sklearn/numpy, no torch)."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")
import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from src.modeling.metrics import compute_metrics  # noqa: E402


def test_perfect_predictions() -> None:
    logits = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    labels = np.array([0, 1, 2])
    out = compute_metrics((logits, labels))
    assert out["accuracy"] == 1.0
    assert out["macro_f1"] == 1.0


def test_matches_sklearn_on_mixed_case() -> None:
    logits = np.array(
        [
            [2.0, 1.0, 0.0],  # pred 0, true 0
            [0.0, 3.0, 1.0],  # pred 1, true 1
            [1.0, 0.0, 2.0],  # pred 2, true 0 (wrong)
            [0.5, 2.0, 0.1],  # pred 1, true 1
        ]
    )
    labels = np.array([0, 1, 0, 1])
    preds = logits.argmax(axis=-1)
    out = compute_metrics((logits, labels))
    assert out["accuracy"] == pytest.approx(accuracy_score(labels, preds))
    assert out["macro_f1"] == pytest.approx(
        f1_score(labels, preds, average="macro", zero_division=0)
    )


def test_accepts_eval_prediction_like_object() -> None:
    class _EvalPred:
        predictions = np.array([[5.0, 0.0], [0.0, 5.0]])
        label_ids = np.array([0, 1])

    out = compute_metrics(_EvalPred())
    assert out["accuracy"] == 1.0
