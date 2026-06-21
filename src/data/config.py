"""Injectable path configuration (Dependency Inversion).

Single responsibility: hold the on-disk locations the loaders read from, as a
plain dataclass that is *passed into* loaders rather than referenced as a global.
This is what makes the layer testable against fixture directories.

The repository's real default layout is::

    <project-root>/Datasets/
        DAIC-WOZ/              # participant folders + PHQ-8 split CSVs
        RedditMentalHealth/    # SWMH files + Low et al. (Zenodo 3941387) files

but nothing here is hardcoded into the loaders; tests build a ``DataPaths``
pointing at a temporary directory instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    """Resolved filesystem locations for every dataset.

    Attributes:
        data_root: Root directory containing the per-dataset subdirectories.
        reddit_dir: Directory holding both SWMH and Low et al. Reddit files.
        daic_dir: Directory holding DAIC-WoZ participant folders and PHQ-8 splits.
    """

    data_root: Path
    reddit_dir: Path = field(default=None)  # type: ignore[assignment]
    daic_dir: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Derive per-dataset dirs from data_root when not explicitly provided.
        object.__setattr__(self, "data_root", Path(self.data_root))
        if self.reddit_dir is None:
            object.__setattr__(self, "reddit_dir", self.data_root / "RedditMentalHealth")
        else:
            object.__setattr__(self, "reddit_dir", Path(self.reddit_dir))
        if self.daic_dir is None:
            object.__setattr__(self, "daic_dir", self.data_root / "DAIC-WOZ")
        else:
            object.__setattr__(self, "daic_dir", Path(self.daic_dir))

    @classmethod
    def default(cls, project_root: Path | str | None = None) -> "DataPaths":
        """Build the default layout rooted at ``<project_root>/Datasets``.

        If ``project_root`` is omitted, it is inferred as the repository root
        (three levels up from this file: ``src/data/config.py`` -> repo root).
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        return cls(data_root=Path(project_root) / "Datasets")
