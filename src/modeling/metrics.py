"""Evaluation metrics for the multi-class baseline.

Single responsibility: compute the headline metrics (accuracy + macro-F1) from raw
logits, in the callback shape the HF ``Trainer`` expects. macro-F1 is reported
because the corpus is heavily imbalanced (depression dominates), so plain accuracy
would flatter a majority-class predictor.

Depends only on numpy + scikit-learn, so it is testable without torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _logits_and_labels(eval_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(logits, labels)`` from either a tuple or an EvalPrediction."""
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred
    # Some models return a tuple of outputs; the logits are the first element.
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return np.asarray(logits), np.asarray(labels)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """HF ``Trainer`` ``compute_metrics`` callback: accuracy + macro-F1.

    Accepts an ``EvalPrediction`` (production) or a plain ``(logits, labels)``
    tuple (tests). Predictions are ``argmax`` over the class logits.
    """
    logits, labels = _logits_and_labels(eval_pred)
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }
