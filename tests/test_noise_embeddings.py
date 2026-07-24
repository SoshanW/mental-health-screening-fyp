"""Tests for the embedding cache (torch-free) and extraction (torch-guarded)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.config import ArtifactPaths
from src.noise.embeddings import (
    METADATA_COLUMNS,
    EmbeddingConfig,
    build_metadata_frame,
    default_embeddings_dir,
    load_embeddings,
    save_embeddings,
)


def test_embedding_config_rejects_bad_dtype() -> None:
    with pytest.raises(ValueError, match="output_dtype"):
        EmbeddingConfig(output_dtype="float64")


def test_build_metadata_frame_columns_and_row_index() -> None:
    df = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "condition": ["depression", "bipolar", "depression"],
            "source": ["low_et_al"] * 3,
            "author_id": ["u1", "u2", "u3"],
        }
    )
    meta = build_metadata_frame(df)
    assert list(meta.columns) == list(METADATA_COLUMNS)
    assert list(meta["row_index"]) == [0, 1, 2]
    assert list(meta["condition"]) == ["depression", "bipolar", "depression"]
    assert list(meta["extractor"]) == ["finetuned"] * 3  # default


def test_build_metadata_frame_stamps_custom_extractor() -> None:
    df = pd.DataFrame({"text": ["a", "b"], "condition": ["bipolar", "depression"]})
    meta = build_metadata_frame(df, extractor="base")
    assert list(meta["extractor"]) == ["base", "base"]


def test_default_embeddings_dir_distinct_per_extractor(tmp_path) -> None:
    artifacts = ArtifactPaths(root=tmp_path)
    finetuned = default_embeddings_dir(artifacts, "train", "finetuned")
    base = default_embeddings_dir(artifacts, "train", "base")
    # Fine-tuned keeps the original path; base is a distinct sibling so it cannot
    # overwrite the D-034 cache.
    assert finetuned == artifacts.root / "embeddings" / "train"
    assert base != finetuned
    assert base == artifacts.root / "embeddings" / "train__base"


def test_save_load_round_trip(tmp_path) -> None:
    features = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    df = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "condition": ["depression", "bipolar", "schizophrenia"],
            "source": ["low_et_al"] * 3,
            "author_id": ["u1", "u2", "u3"],
        }
    )
    meta = build_metadata_frame(df)
    save_embeddings(tmp_path, features, meta)

    loaded_features, loaded_meta = load_embeddings(tmp_path)
    np.testing.assert_array_equal(loaded_features, features)
    assert list(loaded_meta["condition"]) == ["depression", "bipolar", "schizophrenia"]


def test_load_missing_cache_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_embeddings(tmp_path)


def test_save_length_mismatch_raises(tmp_path) -> None:
    features = np.zeros((3, 4), dtype=np.float32)
    meta = build_metadata_frame(pd.DataFrame({"text": ["a", "b"]}))
    with pytest.raises(ValueError, match="length mismatch"):
        save_embeddings(tmp_path, features, meta)


# --- torch-dependent extraction path (skips where torch is absent) ---------------

torch = pytest.importorskip("torch")

from src.noise.embeddings import extract_pooled_embeddings  # noqa: E402


class _FakeTokenizer:
    def __call__(self, text, truncation=True, max_length=8, padding="max_length"):
        return {"input_ids": [1] * 4, "attention_mask": [1] * 4}


class _PooledOutput:
    def __init__(self, tensor: "torch.Tensor") -> None:
        self.pooler_output = tensor


class _FakeEncoder(torch.nn.Module):
    """Stand-in for BertModel: returns a fixed pooler_output sized to `hidden`."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self._hidden = hidden

    def forward(self, **inputs):
        batch = next(iter(inputs.values())).shape[0]
        vals = torch.arange(batch * self._hidden, dtype=torch.float32).reshape(batch, self._hidden)
        return _PooledOutput(vals)


class _FakeModel(torch.nn.Module):
    """Stand-in for AutoModelForSequenceClassification exposing `.base_model`."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self._encoder = _FakeEncoder(hidden)

    @property
    def base_model(self):  # matches transformers' PreTrainedModel.base_model
        return self._encoder


def test_extract_pooled_embeddings_shape_and_dtype(monkeypatch) -> None:
    import src.modeling.hf_model as hf

    hidden = 6
    monkeypatch.setattr(hf, "load_model", lambda cfg, name_or_path=None: _FakeModel(hidden))
    monkeypatch.setattr(hf, "load_tokenizer", lambda name_or_path: _FakeTokenizer())

    df = pd.DataFrame({"text": ["a", "b", "c", "d", "e"], "condition": ["depression"] * 5})
    feats = extract_pooled_embeddings(
        df, "unused-checkpoint", embed_config=EmbeddingConfig(batch_size=2, device="cpu")
    )
    assert feats.shape == (5, hidden)
    assert feats.dtype == np.float32


def test_extract_pooled_embeddings_float16_cache(monkeypatch) -> None:
    import src.modeling.hf_model as hf

    monkeypatch.setattr(hf, "load_model", lambda cfg, name_or_path=None: _FakeModel(4))
    monkeypatch.setattr(hf, "load_tokenizer", lambda name_or_path: _FakeTokenizer())

    df = pd.DataFrame({"text": ["a", "b"], "condition": ["bipolar", "depression"]})
    feats = extract_pooled_embeddings(
        df, "unused", embed_config=EmbeddingConfig(batch_size=2, output_dtype="float16", device="cpu")
    )
    assert feats.dtype == np.float16
