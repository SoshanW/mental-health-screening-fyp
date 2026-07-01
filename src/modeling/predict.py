"""CLI entrypoint: produce held-out softmax predictions from a trained checkpoint.

Single responsibility: load the persisted test split and a trained checkpoint, run
the model in eval mode, and write per-row softmax probabilities plus the argmax
prediction to a CSV -- the raw material the later calibration/abstention work
(future increments) will consume. Also prints held-out accuracy + macro-F1.

Self-contained: reads ``splits/test.csv`` written by ``train.py`` and never touches
``Datasets/`` again, so predictions are evaluated on the exact held-out rows.

Run::

    python -m src.modeling.predict
    python -m src.modeling.predict --artifacts-root /content/drive/MyDrive/mh/Models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ArtifactPaths, ModelConfig
from .dataset import TextClassificationDataset
from .labels import CONDITION_NAMES, CONDITION_TO_ID, decode_predictions


def predict_softmax(
    model: object,
    dataset: TextClassificationDataset,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Run ``model`` over ``dataset`` in eval mode; return an ``(n, num_labels)`` array.

    ``model`` is any callable returning an object with ``.logits`` (a HF model) or a
    raw logits tensor. Softmax is applied over the class dimension.
    """
    import torch
    from torch.utils.data import DataLoader

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Only feed model inputs, not the label column, to the forward pass.
    def _collate(batch: list[dict]) -> dict:
        keys = [k for k in batch[0] if k != "labels"]
        return {k: torch.stack([ex[k] for ex in batch]).to(device) for k in keys}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    probs: list[np.ndarray] = []
    with torch.no_grad():
        for inputs in loader:
            outputs = model(**inputs)
            logits = getattr(outputs, "logits", outputs)
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs.append(batch_probs)
    return np.concatenate(probs, axis=0)


def build_prediction_frame(test_df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Assemble the output frame: identity + true label + per-class probs + argmax."""
    out = pd.DataFrame(
        {
            "text": test_df["text"].to_numpy(),
            "author_id": test_df["author_id"].to_numpy(),
            "source": test_df["source"].to_numpy(),
            "true_condition": test_df["condition"].to_numpy(),
        }
    )
    for name in CONDITION_NAMES:
        out[f"prob_{name}"] = probs[:, CONDITION_TO_ID[name]]
    pred_ids = probs.argmax(axis=1)
    out["predicted_condition"] = decode_predictions(pred_ids)
    out["confidence"] = probs.max(axis=1)
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict held-out softmax probabilities.")
    p.add_argument("--artifacts-root", type=Path, default=None, help="Models dir (default <repo>/Models).")
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Trained checkpoint (default <artifacts>/checkpoints/latest).")
    p.add_argument("--max-length", type=int, default=ModelConfig().max_length)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifacts = ArtifactPaths(root=args.artifacts_root) if args.artifacts_root else ArtifactPaths.default()
    checkpoint = args.checkpoint_dir or (artifacts.checkpoints_dir / "latest")

    test_path = artifacts.splits_dir / "test.csv"
    if not test_path.exists():
        raise SystemExit(f"Test split not found at {test_path}; run `python -m src.modeling.train` first.")
    test_df = pd.read_csv(test_path)

    from .hf_model import load_model, load_tokenizer

    model_config = ModelConfig(pretrained_name=str(checkpoint), max_length=args.max_length)
    tokenizer = load_tokenizer(checkpoint)
    model = load_model(model_config, name_or_path=checkpoint)

    dataset = TextClassificationDataset(test_df, tokenizer, args.max_length)
    probs = predict_softmax(model, dataset, batch_size=args.batch_size)

    out = build_prediction_frame(test_df, probs)
    artifacts.predictions_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts.predictions_dir / "test_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote predictions -> {out_path}  ({len(out)} rows)")

    # Held-out metrics.
    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(out["true_condition"], out["predicted_condition"])
    macro_f1 = f1_score(
        out["true_condition"], out["predicted_condition"], average="macro", zero_division=0
    )
    print("\n=== held-out test metrics ===")
    print(f"accuracy: {acc:.4f}")
    print(f"macro_f1: {macro_f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
