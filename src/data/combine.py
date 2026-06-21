"""Composition of loaders into one corpus (depends only on the abstraction).

Single responsibility: run a collection of :class:`~src.data.base.DatasetLoader`
implementations, validate each against the canonical schema, and concatenate them
into one DataFrame -- deduplicating on ``text`` while preserving distinct-author
rows.

This module depends on the ``DatasetLoader`` interface, never on concrete loader
classes, so adding a dataset requires no change here (Open/Closed + DIP).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .base import DatasetLoader
from .schema import CANONICAL_COLUMNS, validate_schema


def combine_sources(
    loaders: Iterable[DatasetLoader],
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Load and concatenate every loader's output into one canonical frame.

    Args:
        loaders: Any iterable of objects satisfying the ``DatasetLoader``
            interface. They are treated uniformly (Liskov substitution).
        deduplicate: If ``True``, drop rows that are exact duplicates on the
            ``(text, source, author_id)`` triple. This removes genuinely repeated
            rows without collapsing the single-post-proxy design: two different
            authors posting identical text remain distinct rows.

    Returns:
        A single DataFrame with the canonical columns first, followed by the union
        of any additive columns (e.g. DAIC's PHQ-8 fields) which are ``NaN`` for
        sources that do not provide them.
    """
    frames: list[pd.DataFrame] = []
    for loader in loaders:
        frames.append(validate_schema(loader.load()))

    if not frames:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    combined = pd.concat(frames, ignore_index=True, sort=False)

    if deduplicate:
        combined = combined.drop_duplicates(
            subset=["text", "source", "author_id"]
        ).reset_index(drop=True)

    # Keep canonical columns leading, additive columns trailing.
    ordered = [*CANONICAL_COLUMNS, *[c for c in combined.columns if c not in CANONICAL_COLUMNS]]
    combined = combined[ordered]
    return validate_schema(combined)
