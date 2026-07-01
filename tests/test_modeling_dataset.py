"""Tests for TextClassificationDataset using a fake tokenizer (needs torch, no network)."""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.modeling.dataset import TextClassificationDataset  # noqa: E402
from src.modeling.labels import CONDITION_TO_ID  # noqa: E402


class _FakeTokenizer:
    """Returns fixed-length fake ids; no vocab, no network."""

    def __init__(self, seq_len: int = 8) -> None:
        self._seq_len = seq_len

    def __call__(self, text, truncation=True, max_length=8, padding="max_length"):
        n = min(max_length, self._seq_len)
        return {
            "input_ids": [1] * n,
            "attention_mask": [1] * n,
        }


def _frame(with_labels: bool = True) -> pd.DataFrame:
    data = {"text": ["i feel low", "manic week", "counting calories"]}
    if with_labels:
        data["condition"] = ["depression", "bipolar", "eating_disorder"]
    return pd.DataFrame(data)


def test_len_matches_rows() -> None:
    ds = TextClassificationDataset(_frame(), _FakeTokenizer(), max_length=8)
    assert len(ds) == 3


def test_item_has_expected_tensors_and_label() -> None:
    ds = TextClassificationDataset(_frame(), _FakeTokenizer(), max_length=8)
    item = ds[1]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].dtype == torch.long
    assert item["input_ids"].shape[0] == 8
    assert int(item["labels"]) == CONDITION_TO_ID["bipolar"]


def test_no_label_column_omits_labels() -> None:
    ds = TextClassificationDataset(_frame(with_labels=False), _FakeTokenizer(), max_length=8)
    item = ds[0]
    assert "labels" not in item


def test_max_length_truncates() -> None:
    ds = TextClassificationDataset(_frame(), _FakeTokenizer(seq_len=32), max_length=4)
    assert ds[0]["input_ids"].shape[0] == 4
