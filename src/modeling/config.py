"""Injectable configuration for the modeling layer (Dependency Inversion).

Single responsibility: hold the knobs the training/prediction pipeline reads from
-- split ratios, model identity, hyperparameters, and output locations -- as plain
frozen dataclasses that are *passed into* the pipeline rather than referenced as
globals. This mirrors :class:`src.data.config.DataPaths` and is what lets the same
code run against a local ``Models/`` dir or a Google Drive path on Colab.

No torch/transformers import here on purpose: this module stays importable in a
plain pandas environment so the split/label logic can be tested without the heavy
deep-learning stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..data.schema import POC_CONDITIONS


@dataclass(frozen=True)
class SplitConfig:
    """Author-grouped train/val/test ratios and RNG seed.

    Attributes:
        train_frac / val_frac / test_frac: Fractions that must sum to ~1.0.
        seed: Deterministic seed for the grouped shuffle split.
    """

    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split fractions must sum to 1.0; got {total} "
                f"(train={self.train_frac}, val={self.val_frac}, test={self.test_frac})."
            )
        for name, frac in (
            ("train_frac", self.train_frac),
            ("val_frac", self.val_frac),
            ("test_frac", self.test_frac),
        ):
            if frac <= 0:
                raise ValueError(f"{name} must be > 0; got {frac}.")


@dataclass(frozen=True)
class ModelConfig:
    """Identity of the pretrained model and the tokenization length cap.

    Attributes:
        pretrained_name: HF Hub id (or local path) of the base model. Defaults to
            MentalBERT; overridable for smoke tests with a tiny public model.
        max_length: Token truncation limit. Docs/technical-contribution-in-depth.md
            §6 caps fine-tuning at <=256 tokens.
        num_labels: Number of output classes; defaults to len(POC_CONDITIONS).
    """

    pretrained_name: str = "mental/mental-bert-base-uncased"
    max_length: int = 256
    num_labels: int = field(default_factory=lambda: len(POC_CONDITIONS))


@dataclass(frozen=True)
class TrainConfig:
    """Trainer hyperparameters for the naive baseline fine-tune."""

    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    seed: int = 42
    fp16: bool = False
    logging_steps: int = 50


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolved output locations for checkpoints, splits, and predictions.

    Mirrors :class:`src.data.config.DataPaths`: a ``root`` plus derived
    subdirectories filled in by ``__post_init__``. On Colab, build one rooted at a
    Google Drive path so checkpoints survive runtime resets.
    """

    root: Path
    checkpoints_dir: Path = field(default=None)  # type: ignore[assignment]
    splits_dir: Path = field(default=None)  # type: ignore[assignment]
    predictions_dir: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if self.checkpoints_dir is None:
            object.__setattr__(self, "checkpoints_dir", self.root / "checkpoints")
        else:
            object.__setattr__(self, "checkpoints_dir", Path(self.checkpoints_dir))
        if self.splits_dir is None:
            object.__setattr__(self, "splits_dir", self.root / "splits")
        else:
            object.__setattr__(self, "splits_dir", Path(self.splits_dir))
        if self.predictions_dir is None:
            object.__setattr__(self, "predictions_dir", self.root / "predictions")
        else:
            object.__setattr__(self, "predictions_dir", Path(self.predictions_dir))

    @classmethod
    def default(cls, project_root: Path | str | None = None) -> "ArtifactPaths":
        """Build the default layout rooted at ``<project_root>/Models``.

        If ``project_root`` is omitted it is inferred as the repo root (two levels
        up from this file: ``src/modeling/config.py`` -> repo root), matching
        :meth:`src.data.config.DataPaths.default`.
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        return cls(root=Path(project_root) / "Models")

    def mkdirs(self) -> "ArtifactPaths":
        """Create all output directories (idempotent); return self for chaining."""
        for d in (self.checkpoints_dir, self.splits_dir, self.predictions_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self
