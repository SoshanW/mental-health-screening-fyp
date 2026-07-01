"""CLI entrypoint: fine-tune the naive MentalBERT multi-class baseline.

Single responsibility: wire the data layer -> author-grouped splits -> tokenized
datasets -> a HF ``Trainer`` fine-tune, then persist the splits and the trained
checkpoint. This is baseline #1 from Docs/technical-contribution-in-depth.md
(naive fine-tune on clean-assumed proxy labels); noise modeling / calibration /
abstention are separate future increments.

Run (local, Windows)::

    ./.venv/Scripts/python.exe -m src.modeling.train

Run (Colab / Linux, data + Models on Google Drive)::

    python -m src.modeling.train \
        --data-root /content/drive/MyDrive/mh/Datasets \
        --artifacts-root /content/drive/MyDrive/mh/Models --fp16

Fast wiring smoke-test (tiny public model, no MentalBERT download, ~seconds)::

    python -m src.modeling.train \
        --model-name hf-internal-testing/tiny-random-bert \
        --max-train-samples 50 --num-epochs 1

The same code runs locally and on Colab -- only the ``--data-root`` /
``--artifacts-root`` arguments change (see notebooks/train_colab.ipynb).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..data.combine import combine_sources
from ..data.config import DataPaths
from ..data.loaders import DAICWoZLoader, LowEtAlLoader, SWMHLoader
from .config import ArtifactPaths, ModelConfig, SplitConfig, TrainConfig
from .dataset import TextClassificationDataset, Tokenizer
from .metrics import compute_metrics
from .splits import SplitFrames, build_poc_splits


def _load_combined(paths: DataPaths, include_interviewer: bool) -> pd.DataFrame:
    """Load every available source (skipping missing ones) and combine them."""
    loaders = [
        SWMHLoader(paths),
        LowEtAlLoader(paths),
        DAICWoZLoader(paths, include_interviewer=include_interviewer),
    ]
    available = []
    for loader in loaders:
        try:
            frame = loader.load()
        except FileNotFoundError as exc:
            print(f"[skip] source '{loader.source}': {exc}")
            continue
        available.append(_Preloaded(loader.source, frame))
    if not available:
        raise SystemExit("No sources could be loaded; check --data-root.")
    return combine_sources(available)


class _Preloaded:
    """Adapter presenting an already-loaded frame as a DatasetLoader."""

    def __init__(self, source: str, frame: pd.DataFrame) -> None:
        self._source = source
        self._frame = frame

    @property
    def source(self) -> str:
        return self._source

    def load(self) -> pd.DataFrame:
        return self._frame


def _persist_splits(splits: SplitFrames, artifacts: ArtifactPaths) -> None:
    """Write train/val/test frames so predict.py evaluates the exact held-out rows."""
    artifacts.splits_dir.mkdir(parents=True, exist_ok=True)
    splits.train.to_csv(artifacts.splits_dir / "train.csv", index=False)
    splits.val.to_csv(artifacts.splits_dir / "val.csv", index=False)
    splits.test.to_csv(artifacts.splits_dir / "test.csv", index=False)


def build_trainer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_config: ModelConfig,
    train_config: TrainConfig,
    output_dir: Path,
    tokenizer: Tokenizer | None = None,
    model: object | None = None,
):
    """Construct a HF ``Trainer`` over the given frames.

    ``tokenizer`` and ``model`` are injectable; when omitted they are loaded from
    ``model_config`` via :mod:`src.modeling.hf_model`. Injecting them lets a smoke
    test pass tiny fakes without a network download.
    """
    from transformers import Trainer, TrainingArguments

    from .hf_model import load_model, load_tokenizer

    if tokenizer is None:
        tokenizer = load_tokenizer(model_config.pretrained_name)
    if model is None:
        model = load_model(model_config)

    train_ds = TextClassificationDataset(train_df, tokenizer, model_config.max_length)
    val_ds = TextClassificationDataset(val_df, tokenizer, model_config.max_length)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config.num_epochs,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        seed=train_config.seed,
        fp16=train_config.fp16,
        logging_steps=train_config.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to=[],
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune the MentalBERT baseline classifier.")
    p.add_argument("--data-root", type=Path, default=None, help="Datasets dir (default <repo>/Datasets).")
    p.add_argument("--artifacts-root", type=Path, default=None, help="Models dir (default <repo>/Models).")
    p.add_argument("--model-name", type=str, default=ModelConfig().pretrained_name,
                   help="HF model id or local path (override for smoke tests).")
    p.add_argument("--max-length", type=int, default=ModelConfig().max_length)
    p.add_argument("--num-epochs", type=int, default=TrainConfig().num_epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig().batch_size)
    p.add_argument("--learning-rate", type=float, default=TrainConfig().learning_rate)
    p.add_argument("--seed", type=int, default=TrainConfig().seed)
    p.add_argument("--train-frac", type=float, default=SplitConfig().train_frac)
    p.add_argument("--val-frac", type=float, default=SplitConfig().val_frac)
    p.add_argument("--test-frac", type=float, default=SplitConfig().test_frac)
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Subsample the training frame (fast smoke runs).")
    p.add_argument("--include-interviewer", action="store_true",
                   help="Keep Ellie's DAIC turns (DAIC is excluded from this classifier anyway).")
    p.add_argument("--fp16", action="store_true", help="Mixed-precision (GPU).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    data_paths = DataPaths(data_root=args.data_root) if args.data_root else DataPaths.default()
    artifacts = (
        ArtifactPaths(root=args.artifacts_root) if args.artifacts_root else ArtifactPaths.default()
    ).mkdirs()

    model_config = ModelConfig(pretrained_name=args.model_name, max_length=args.max_length)
    train_config = TrainConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        fp16=args.fp16,
    )
    split_config = SplitConfig(
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    print(f"Data root:      {data_paths.data_root}")
    print(f"Artifacts root: {artifacts.root}")
    print(f"Model:          {model_config.pretrained_name}")

    combined = _load_combined(data_paths, include_interviewer=args.include_interviewer)
    splits = build_poc_splits(combined, split_config)
    print(
        f"Split sizes -> train={len(splits.train)}  val={len(splits.val)}  test={len(splits.test)}"
    )

    train_df = splits.train
    if args.max_train_samples is not None:
        train_df = train_df.head(args.max_train_samples)
        print(f"[smoke] subsampled train to {len(train_df)} rows")

    _persist_splits(splits, artifacts)

    trainer = build_trainer(
        train_df=train_df,
        val_df=splits.val,
        model_config=model_config,
        train_config=train_config,
        output_dir=artifacts.checkpoints_dir,
    )
    trainer.train()

    final_dir = artifacts.checkpoints_dir / "latest"
    trainer.save_model(str(final_dir))
    trainer.tokenizer.save_pretrained(str(final_dir))
    print(f"Saved checkpoint -> {final_dir}")

    metrics = trainer.evaluate()
    print("\n=== validation metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
