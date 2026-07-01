"""Modeling layer: the naive MentalBERT multi-class baseline classifier.

Public surface (light, torch-free imports only). The torch/transformers-dependent
modules (``dataset``, ``hf_model``, ``metrics``, ``train``, ``predict``) are NOT
imported here so that ``import src.modeling`` works in a plain pandas environment;
import those modules directly where the deep-learning stack is available.

This layer consumes the canonical frames produced by :mod:`src.data` and never the
reverse (one-directional dependency).
"""

from __future__ import annotations

from .config import ArtifactPaths, ModelConfig, SplitConfig, TrainConfig
from .labels import CONDITION_NAMES, CONDITION_TO_ID, ID_TO_CONDITION, decode_predictions, encode_labels
from .splits import (
    SplitFrames,
    author_grouped_split,
    build_poc_splits,
    prepare_classification_frame,
)

__all__ = [
    "ArtifactPaths",
    "ModelConfig",
    "SplitConfig",
    "TrainConfig",
    "CONDITION_TO_ID",
    "ID_TO_CONDITION",
    "CONDITION_NAMES",
    "encode_labels",
    "decode_predictions",
    "SplitFrames",
    "prepare_classification_frame",
    "author_grouped_split",
    "build_poc_splits",
]
