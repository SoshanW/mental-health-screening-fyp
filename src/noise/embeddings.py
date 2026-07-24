"""Extract and cache MentalBERT pooled embeddings for the noise diagnostics (C1).

Single responsibility: turn the Reddit training split into one fixed feature
matrix -- the pooled, pre-classification-head representation of every post -- and
cache it to disk so the downstream clusterability / transition-matrix diagnostics
(:mod:`src.noise.clusterability` and later stages) reload it without recomputing.
These features are what HOC (Zhu, Song and Liu, 2021) consumes: the geometry of
the posts in embedding space, before any label-dependent head.

DECISION (feature = pooler_output): the extracted feature for each post is
``base_model.pooler_output`` -- the [CLS] token passed through BERT's pooler
dense+tanh. Verified 2026-07-24 against transformers 5.14.1 that, in eval mode,
``model.classifier(model.dropout(pooler_output)) == model(...).logits``; i.e.
pooler_output is exactly the tensor the sequence-classification head consumes.
That is the "representation before the classification head" the diagnostic needs.
An alternative (raw ``last_hidden_state[:, 0]``) was rejected because it is not the
tensor this architecture's head actually reads.

DECISION (run location): extraction needs the Milestone 0 checkpoint and the
persisted ``splits/train.csv``, both of which live on Google Drive from the Colab
training run, not on the local machine. This module is therefore meant to run on
Colab against the Drive checkpoint; the cached ``features.npy`` + ``metadata.csv``
are brought back for local analysis by :mod:`src.noise.clusterability`.

Torch/transformers are imported lazily inside the extraction function so this
module (its config, save/load, and path helpers) stays importable in a plain
pandas environment, mirroring the torch-free/torch-dependent split used across
:mod:`src.modeling`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..modeling.config import ArtifactPaths, ModelConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

#: Filenames written under a cache directory. The float matrix and its aligned
#: per-row metadata are kept side by side so a later stage can reload both and
#: assert they line up.
FEATURES_FILENAME: str = "features.npy"
METADATA_FILENAME: str = "metadata.csv"

#: Columns persisted next to the feature matrix. ``condition`` is the NOISY proxy
#: label the clusterability diagnostic groups by; ``extractor`` records which model
#: produced the features (D-035 control) so downstream analysis cannot silently mix
#: fine-tuned and base embeddings; the rest are for traceability.
METADATA_COLUMNS: tuple[str, ...] = (
    "row_index",
    "author_id",
    "source",
    "condition",
    "extractor",
)

#: HF Hub id of base MentalBERT (no fine-tuning on this project's labels). Used by
#: the D-035 control run. It is a GATED repo, so a Colab run with ``--extractor
#: base`` needs the Hugging Face login (notebook step 4b); the fine-tuned run does
#: not, because it loads a local checkpoint.
BASE_MODEL_ID: str = "mental/mental-bert-base-uncased"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Knobs for pooled-embedding extraction (injected, not global).

    Attributes:
        max_length: Token truncation length. Matches the baseline's 256 cap so the
            embeddings describe the same inputs the classifier was trained on.
        batch_size: Forward-pass batch size at inference time.
        output_dtype: On-disk dtype for the cached matrix. ``"float16"`` halves the
            artefact size and transfer cost; cosine-neighbour ordering is robust to
            it. ``"float32"`` keeps full precision. Analysis upcasts to float32
            regardless (see :mod:`src.noise.clusterability`).
        device: Torch device string, or ``None`` to auto-select cuda/cpu.
    """

    max_length: int = 256
    batch_size: int = 32
    output_dtype: str = "float32"
    device: str | None = None

    def __post_init__(self) -> None:
        if self.output_dtype not in ("float16", "float32"):
            raise ValueError(
                f"output_dtype must be 'float16' or 'float32'; got {self.output_dtype!r}."
            )
        if self.max_length <= 0 or self.batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive.")


def default_embeddings_dir(
    artifacts: ArtifactPaths, split: str = "train", extractor: str = "finetuned"
) -> Path:
    """Cache directory for a split's embeddings under ``<Models>/embeddings/``.

    Derived from :class:`ArtifactPaths.root` so it points at the Drive-backed
    ``Models/`` on Colab and the local ``Models/`` off it, without touching
    :class:`ArtifactPaths` itself.

    DECISION (distinct paths per extractor): the fine-tuned run keeps the original
    ``embeddings/<split>`` path so its existing D-034 cache is never overwritten;
    any other extractor (e.g. base MentalBERT for the D-035 control) gets a
    distinct sibling ``embeddings/<split>__<extractor>`` so the two feature sets
    cannot clobber each other.
    """
    base = artifacts.root / "embeddings"
    if extractor == "finetuned":
        return base / split
    return base / f"{split}__{extractor}"


def extract_pooled_embeddings(
    df: pd.DataFrame,
    checkpoint_dir: str | Path,
    model_config: ModelConfig | None = None,
    embed_config: EmbeddingConfig | None = None,
) -> np.ndarray:
    """Return the ``(n_rows, hidden_size)`` pooled-embedding matrix for ``df``.

    Rows are emitted in the exact order of ``df`` so the result aligns positionally
    with any metadata taken from the same frame. The model is loaded through the
    existing :func:`src.modeling.hf_model.load_model` path (an
    ``AutoModelForSequenceClassification``); the feature is its
    ``base_model.pooler_output`` (see the module DECISION note).

    Args:
        df: Canonical frame with a ``text`` column (labels are ignored here).
        checkpoint_dir: Milestone 0 checkpoint directory (HF Hub id also accepted).
        model_config: Supplies ``num_labels`` for the head; defaults to
            :class:`ModelConfig` sized to the POC classes.
        embed_config: Extraction knobs; defaults to :class:`EmbeddingConfig`.

    Raises:
        RuntimeError: if the loaded model's base does not expose a ``pooler_output``
            (e.g. a non-BERT architecture whose head does not read a pooler). The
            feature definition is BERT-specific by design and fails loudly rather
            than silently substituting a different tensor.
    """
    import torch
    from torch.utils.data import DataLoader

    from ..modeling.dataset import TextClassificationDataset
    from ..modeling.hf_model import load_model, load_tokenizer

    model_config = model_config or ModelConfig()
    embed_config = embed_config or EmbeddingConfig()

    device = embed_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer: "PreTrainedTokenizerBase" = load_tokenizer(checkpoint_dir)
    model: "PreTrainedModel" = load_model(model_config, name_or_path=checkpoint_dir)
    # base_model is the encoder submodule (BertModel for MentalBERT); its
    # pooler_output is the pre-head feature. Verified against transformers 5.14.1.
    encoder = model.base_model
    encoder = encoder.to(device)
    encoder.eval()

    dataset = TextClassificationDataset(df, tokenizer, embed_config.max_length)

    # Feed only model inputs to the encoder, never the label column, and move each
    # batch to the device. Mirrors src.modeling.predict.predict_softmax's collate.
    def _collate(batch: list[dict]) -> dict:
        keys = [k for k in batch[0] if k != "labels"]
        return {k: torch.stack([ex[k] for ex in batch]).to(device) for k in keys}

    loader = DataLoader(
        dataset,
        batch_size=embed_config.batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for inputs in loader:
            outputs = encoder(**inputs)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                raise RuntimeError(
                    "Loaded model's base_model produced no pooler_output; the "
                    "pooler feature is BERT-specific. Got output type "
                    f"{type(outputs).__name__}. Revisit the feature definition for "
                    "this architecture before running the diagnostic."
                )
            chunks.append(pooled.detach().cpu().to(torch.float32).numpy())

    features = np.concatenate(chunks, axis=0)
    return features.astype(embed_config.output_dtype, copy=False)


def build_metadata_frame(df: pd.DataFrame, extractor: str = "finetuned") -> pd.DataFrame:
    """Assemble the per-row metadata persisted next to the feature matrix.

    ``row_index`` records the original position so alignment survives a reload even
    if columns are reordered. ``extractor`` stamps which model produced the features
    (D-035 control) so a later comparison cannot silently mix extractors. Missing
    optional columns are filled with ``"unknown"``.
    """
    n = len(df)
    out = pd.DataFrame({"row_index": np.arange(n, dtype="int64")})
    for col in ("author_id", "source", "condition"):
        out[col] = df[col].to_numpy() if col in df.columns else "unknown"
    out["extractor"] = extractor
    return out[list(METADATA_COLUMNS)]


def save_embeddings(cache_dir: str | Path, features: np.ndarray, metadata: pd.DataFrame) -> Path:
    """Write ``features.npy`` + ``metadata.csv`` under ``cache_dir``; return the dir.

    Torch-free (numpy + pandas only), so a machine that received the cache can
    reload it without the deep-learning stack.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if len(features) != len(metadata):
        raise ValueError(
            f"features/metadata length mismatch: {len(features)} vs {len(metadata)}."
        )
    np.save(cache_dir / FEATURES_FILENAME, features)
    metadata.to_csv(cache_dir / METADATA_FILENAME, index=False)
    return cache_dir


def load_embeddings(cache_dir: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Reload a cached ``(features, metadata)`` pair; assert they are aligned.

    Raises:
        FileNotFoundError: if either artefact is absent.
        ValueError: if the row counts disagree.
    """
    cache_dir = Path(cache_dir)
    feat_path = cache_dir / FEATURES_FILENAME
    meta_path = cache_dir / METADATA_FILENAME
    if not feat_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Expected {FEATURES_FILENAME} and {METADATA_FILENAME} under {cache_dir}; "
            "run `python -m src.noise.embeddings` (on Colab) first."
        )
    features = np.load(feat_path)
    metadata = pd.read_csv(meta_path)
    if len(features) != len(metadata):
        raise ValueError(
            f"Cached features/metadata length mismatch under {cache_dir}: "
            f"{len(features)} vs {len(metadata)}."
        )
    return features, metadata


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract + cache MentalBERT pooled embeddings.")
    p.add_argument("--artifacts-root", type=Path, default=None, help="Models dir (default <repo>/Models).")
    p.add_argument("--extractor", type=str, default="finetuned", choices=("finetuned", "base"),
                   help="Feature extractor: fine-tuned Milestone 0 checkpoint (default) or base "
                        "MentalBERT (D-035 control, needs HF login on Colab).")
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Override the model source (default: <artifacts>/checkpoints/latest for "
                        "finetuned, the base MentalBERT hub id for base).")
    p.add_argument("--split", type=str, default="train", help="Which persisted split CSV to embed.")
    p.add_argument("--max-length", type=int, default=EmbeddingConfig().max_length)
    p.add_argument("--batch-size", type=int, default=EmbeddingConfig().batch_size)
    p.add_argument("--output-dtype", type=str, default=EmbeddingConfig().output_dtype,
                   choices=("float16", "float32"))
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifacts = ArtifactPaths(root=args.artifacts_root) if args.artifacts_root else ArtifactPaths.default()

    # Resolve the model source from the extractor choice (overridable).
    if args.checkpoint_dir is not None:
        source: str | Path = args.checkpoint_dir
    elif args.extractor == "base":
        source = BASE_MODEL_ID
    else:
        source = artifacts.checkpoints_dir / "latest"

    split_path = artifacts.splits_dir / f"{args.split}.csv"
    if not split_path.exists():
        raise SystemExit(
            f"Split not found at {split_path}; run `python -m src.modeling.train` first."
        )
    df = pd.read_csv(split_path)

    embed_config = EmbeddingConfig(
        max_length=args.max_length,
        batch_size=args.batch_size,
        output_dtype=args.output_dtype,
        device=args.device,
    )
    features = extract_pooled_embeddings(df, source, ModelConfig(), embed_config)
    metadata = build_metadata_frame(df, extractor=args.extractor)

    cache_dir = default_embeddings_dir(artifacts, args.split, args.extractor)
    save_embeddings(cache_dir, features, metadata)
    print(
        f"Wrote embeddings ({args.extractor}, source={source}) -> {cache_dir}  "
        f"({features.shape[0]} rows x {features.shape[1]} dims, {features.dtype})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
