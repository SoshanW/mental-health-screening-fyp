"""Factory for the pretrained tokenizer and sequence-classification model.

Single responsibility: own the two ``from_pretrained`` calls in one place so
``train.py`` and ``predict.py`` never duplicate model-loading logic. Imports
``transformers`` lazily inside the functions so merely importing this module does
not require the deep-learning stack (keeps the package importable for the light
tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .config import ModelConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_tokenizer(name_or_path: str | Path) -> "PreTrainedTokenizerBase":
    """Load a tokenizer from a HF Hub id or a local checkpoint directory."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(name_or_path))


def load_model(
    model_config: ModelConfig,
    name_or_path: str | Path | None = None,
) -> "PreTrainedModel":
    """Load an ``AutoModelForSequenceClassification`` head sized to the POC classes.

    Args:
        model_config: Supplies ``num_labels`` and, by default, the pretrained name.
        name_or_path: Optional override (e.g. a saved checkpoint dir at predict
            time). Falls back to ``model_config.pretrained_name``.
    """
    from transformers import AutoModelForSequenceClassification

    source = str(name_or_path) if name_or_path is not None else model_config.pretrained_name
    return AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=model_config.num_labels
    )
