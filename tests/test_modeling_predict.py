"""Tests for predict_softmax + build_prediction_frame using a fake model (needs torch)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.modeling.dataset import TextClassificationDataset  # noqa: E402
from src.modeling.labels import CONDITION_NAMES  # noqa: E402
from src.modeling.predict import build_prediction_frame, predict_softmax  # noqa: E402


class _FakeTokenizer:
    def __call__(self, text, truncation=True, max_length=8, padding="max_length"):
        return {"input_ids": [1] * 4, "attention_mask": [1] * 4}


class _FakeModel(torch.nn.Module):
    """Emits fixed per-row logits favouring class 0; ignores inputs."""

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self._num_labels = num_labels

    def forward(self, **inputs):
        batch = next(iter(inputs.values())).shape[0]
        logits = torch.zeros(batch, self._num_labels)
        logits[:, 0] = 5.0
        return logits


def _test_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "condition": ["depression", "bipolar", "depression"],
            "source": ["low_et_al"] * 3,
            "author_id": ["u1", "u2", "u3"],
        }
    )


def test_predict_softmax_rows_sum_to_one() -> None:
    df = _test_frame()
    ds = TextClassificationDataset(df, _FakeTokenizer(), max_length=4)
    model = _FakeModel(num_labels=len(CONDITION_NAMES))
    probs = predict_softmax(model, ds, batch_size=2, device="cpu")
    assert probs.shape == (3, len(CONDITION_NAMES))
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), rtol=1e-5)


def test_build_prediction_frame_columns_and_argmax() -> None:
    df = _test_frame()
    probs = np.zeros((3, len(CONDITION_NAMES)))
    probs[:, 0] = 1.0  # all predict class 0 == CONDITION_NAMES[0]
    out = build_prediction_frame(df, probs)

    for name in CONDITION_NAMES:
        assert f"prob_{name}" in out.columns
    assert list(out["predicted_condition"]) == [CONDITION_NAMES[0]] * 3
    assert list(out["true_condition"]) == ["depression", "bipolar", "depression"]
    np.testing.assert_allclose(out["confidence"].to_numpy(), np.ones(3))
