"""Torch ``Dataset`` wrapping a canonical DataFrame for the HF ``Trainer``.

Single responsibility: turn a ``(text, condition)`` frame into tokenized,
Trainer-ready tensors. Tokenization is *injected* (a real HF tokenizer in
production, a fake double in tests) via a minimal :class:`Tokenizer` Protocol, so
this class needs no network and is unit-testable offline -- the same
dependency-injection discipline as :class:`src.data.base.DatasetLoader`.

This module imports torch, so it is intentionally NOT imported by
``src.modeling.__init__`` at package import time; import it only where torch is
available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import pandas as pd
import torch

from .labels import CONDITION_TO_ID


@runtime_checkable
class Tokenizer(Protocol):
    """Structural interface for the subset of a HF tokenizer we use.

    A real ``transformers`` tokenizer satisfies this; tests provide a tiny fake
    returning fixed-length ``input_ids``/``attention_mask`` so no model download is
    required.
    """

    def __call__(
        self,
        text: str,
        truncation: bool = ...,
        max_length: int = ...,
        padding: str | bool = ...,
    ) -> Mapping[str, Sequence[int]]:
        ...


class TextClassificationDataset(torch.utils.data.Dataset):
    """Wrap a canonical frame as tokenized examples for ``transformers.Trainer``.

    Each item is a dict of ``input_ids``, ``attention_mask`` (and ``labels`` when a
    ``condition`` column is present) as ``torch.Tensor``s -- the format the HF
    ``Trainer`` and ``AutoModelForSequenceClassification`` expect.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Tokenizer,
        max_length: int = 256,
        label_map: Mapping[str, int] = CONDITION_TO_ID,
        text_column: str = "text",
        label_column: str = "condition",
    ) -> None:
        """
        Args:
            df: Frame with at least ``text_column``; ``label_column`` is optional
                (absent at pure inference time).
            tokenizer: Injected tokenizer (real or fake).
            max_length: Truncation length passed to the tokenizer.
            label_map: condition-string -> class-id mapping.
            text_column / label_column: column names to read.
        """
        self._texts: list[str] = df[text_column].fillna("").astype(str).tolist()
        self._tokenizer = tokenizer
        self._max_length = max_length

        self._labels: list[int] | None
        if label_column in df.columns:
            self._labels = [label_map[c] for c in df[label_column].tolist()]
        else:
            self._labels = None

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self._tokenizer(
            self._texts[idx],
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
        )
        item = {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in encoding.items()
        }
        if self._labels is not None:
            item["labels"] = torch.tensor(self._labels[idx], dtype=torch.long)
        return item
