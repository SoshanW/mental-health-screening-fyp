"""SWMH loader (source="swmh").

Single responsibility: read the SWMH Reddit corpus from the Reddit directory --
whether it ships as one combined CSV (with a label column) or as per-subreddit
CSVs (label encoded in the filename) -- harmonize the subreddit label to a
:class:`~src.data.schema.Condition`, and emit the canonical schema.

DECISION (column names UNVERIFIED): SWMH files are not present on disk yet, so the
raw column names below could not be checked against real data. They follow the
common SWMH distribution (a ``label`` column and a ``text`` column, optional
``author``/``date``). All of them are overridable via the constructor, and the
loader falls back to deriving the label from the filename when no label column is
present. Confirm these against the real SWMH file once added.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import DataPaths
from ..label_map import harmonize_label
from ..schema import CANONICAL_COLUMNS, UNKNOWN_AUTHOR, validate_schema
from ..text_normalization import NOOP_POLICY, NormalizationPolicy, normalize_text

#: Default discovery glob: any CSV whose name contains "swmh" (any casing). This
#: avoids colliding with the Low et al. ``*_features_tfidf_256.csv`` files that
#: share the Reddit directory.
_FILE_GLOB = "*[Ss][Ww][Mm][Hh]*.csv"


class SWMHLoader:
    """Loader for the SWMH (Suicide Watch & Mental Health) Reddit corpus."""

    def __init__(
        self,
        paths: DataPaths,
        file_glob: str = _FILE_GLOB,
        label_column: str = "label",
        text_column: str = "text",
        author_column: str = "author",
        date_column: str = "date",
        normalization: NormalizationPolicy = NOOP_POLICY,
    ) -> None:
        """
        Args:
            paths: Injected path configuration; reads from ``paths.reddit_dir``.
            file_glob: Glob (relative to the Reddit dir) selecting SWMH files.
            label_column: Column holding the raw subreddit/label, if present.
                When absent from a file, the label is derived from the filename
                stem (per-subreddit-file layout).
            text_column: Column holding the post text.
            author_column: Optional author column; ``unknown`` if missing.
            date_column: Optional date column; ``None`` if missing.
            normalization: Text policy applied to each post (default no-op).
        """
        self._paths = paths
        self._file_glob = file_glob
        self._label_column = label_column
        self._text_column = text_column
        self._author_column = author_column
        self._date_column = date_column
        self._normalization = normalization

    @property
    def source(self) -> str:
        return "swmh"

    def load(self) -> pd.DataFrame:
        files = sorted(self._paths.reddit_dir.glob(self._file_glob))
        if not files:
            raise FileNotFoundError(
                f"No SWMH files matching '{self._file_glob}' were found in "
                f"'{self._paths.reddit_dir}'. Expected the SWMH corpus as either "
                f"one combined CSV with a '{self._label_column}' column or "
                f"per-subreddit CSVs named like 'swmh_depression.csv'. Place the "
                f"SWMH files in that directory (or pass a matching file_glob)."
            )

        frames = [self._read_one(f) for f in files]
        df = pd.concat(frames, ignore_index=True)

        df["condition"] = df["_raw_label"].map(
            lambda s: harmonize_label(self.source, s)
        )
        df = df.dropna(subset=["condition"]).copy()
        df["condition"] = df["condition"].map(lambda c: c.value)

        out = df[list(CANONICAL_COLUMNS)].reset_index(drop=True)
        return validate_schema(out)

    def _read_one(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path, dtype=str)
        if self._text_column not in raw.columns:
            raise FileNotFoundError(
                f"SWMH file '{path}' has no '{self._text_column}' column; "
                f"found columns {list(raw.columns)}. Pass text_column=... to "
                f"point at the post-text column."
            )

        out = pd.DataFrame()
        out["text"] = raw[self._text_column].fillna("").map(
            lambda t: normalize_text(t, self._normalization)
        )
        out["condition"] = None  # filled after harmonization in load()
        out["source"] = self.source

        if self._author_column in raw.columns:
            out["author_id"] = raw[self._author_column].fillna(UNKNOWN_AUTHOR)
        else:
            out["author_id"] = UNKNOWN_AUTHOR

        if self._date_column in raw.columns:
            out["date"] = raw[self._date_column].where(raw[self._date_column].notna(), None)
        else:
            out["date"] = None

        if self._label_column in raw.columns:
            out["_raw_label"] = raw[self._label_column]
        else:
            # Per-subreddit-file layout: derive the label from the filename stem,
            # stripping a leading "swmh" prefix and separators.
            stem = path.stem
            for prefix in ("swmh_", "SWMH_", "swmh-", "SWMH-", "swmh", "SWMH"):
                if stem.startswith(prefix):
                    stem = stem[len(prefix) :]
                    break
            out["_raw_label"] = stem
        return out
