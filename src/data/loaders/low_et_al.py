"""Low et al. Reddit Mental Health Dataset loader (source="low_et_al").

Single responsibility: read the Zenodo 3941387 per-subreddit CSVs from the Reddit
directory, keep only the raw fields MentalBERT needs (subreddit, author, date,
post text), harmonize the subreddit to a :class:`~src.data.schema.Condition`, and
emit the canonical schema.

Engineered feature columns (TF-IDF, readability, LIWC, sentiment, etc.) are
ignored on purpose -- the model learns from raw text.

Methodology note (not a filter): each row is one post from one unique author
(single-post proxy). We deliberately do NOT deduplicate by author.

VERIFIED columns (from real files on disk): ``subreddit``, ``author``, ``date``,
``post``. ``date`` is a string like ``"2019/01/01"`` and is passed through as-is.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import DataPaths
from ..label_map import harmonize_label
from ..schema import CANONICAL_COLUMNS, UNKNOWN_AUTHOR, validate_schema
from ..text_normalization import NOOP_POLICY, NormalizationPolicy, normalize_text

#: Files from this release are named ``<subreddit>_..._features_tfidf_256.csv``.
#: We discover them by suffix so any of the pre/post/year variants are picked up.
_FILE_GLOB = "*_features_tfidf_256.csv"

# Raw columns we read; everything else in the 350-column files is ignored.
_RAW_SUBREDDIT = "subreddit"
_RAW_AUTHOR = "author"
_RAW_DATE = "date"
_RAW_POST = "post"
_USECOLS = [_RAW_SUBREDDIT, _RAW_AUTHOR, _RAW_DATE, _RAW_POST]


class LowEtAlLoader:
    """Loader for the Low et al. Reddit Mental Health Dataset (Zenodo 3941387)."""

    def __init__(
        self,
        paths: DataPaths,
        file_glob: str = _FILE_GLOB,
        normalization: NormalizationPolicy = NOOP_POLICY,
    ) -> None:
        """
        Args:
            paths: Injected path configuration; reads from ``paths.reddit_dir``.
            file_glob: Glob (relative to the Reddit dir) selecting this dataset's
                CSVs. Overridable for fixtures.
            normalization: Text policy applied to each post. Default is a no-op so
                Reddit text keeps its native register, but it is routed through
                the shared normalization module so the policy lives in one place.
        """
        self._paths = paths
        self._file_glob = file_glob
        self._normalization = normalization

    @property
    def source(self) -> str:
        return "low_et_al"

    def load(self) -> pd.DataFrame:
        files = sorted(self._paths.reddit_dir.glob(self._file_glob))
        if not files:
            raise FileNotFoundError(
                f"No Low et al. files matching '{self._file_glob}' were found in "
                f"'{self._paths.reddit_dir}'. Expected the Zenodo 3941387 Reddit "
                f"Mental Health Dataset CSVs (e.g. "
                f"'depression_post_features_tfidf_256.csv'). Download from "
                f"https://zenodo.org/record/3941387 and place them there."
            )

        frames = [self._read_one(f) for f in files]
        df = pd.concat(frames, ignore_index=True)

        df["condition"] = df["_raw_subreddit"].map(
            lambda s: harmonize_label(self.source, s)
        )
        df = df.dropna(subset=["condition"]).copy()
        df["condition"] = df["condition"].map(lambda c: c.value)

        out = df[list(CANONICAL_COLUMNS)].reset_index(drop=True)
        return validate_schema(out)

    def _read_one(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path, usecols=lambda c: c in _USECOLS, dtype=str)
        out = pd.DataFrame()
        out["text"] = raw[_RAW_POST].fillna("").map(
            lambda t: normalize_text(t, self._normalization)
        )
        out["condition"] = None  # filled after harmonization in load()
        out["source"] = self.source
        out["author_id"] = raw[_RAW_AUTHOR].fillna(UNKNOWN_AUTHOR)
        out["date"] = raw[_RAW_DATE].where(raw[_RAW_DATE].notna(), None)
        out["_raw_subreddit"] = raw[_RAW_SUBREDDIT]
        return out
