"""Author-grouped train/val/test splitting for the multi-class classifier.

Single responsibility: turn the combined canonical corpus into three
non-overlapping partitions such that (a) only in-scope Reddit multi-class rows are
kept and (b) no author's posts are split across partitions (grouped split → no
train/test leakage through author style).

Two small functions rather than one monolith:
  * ``prepare_classification_frame`` — *what rows* are eligible (filtering).
  * ``author_grouped_split`` — *how* the eligible rows are partitioned.
and ``build_poc_splits`` composes them for callers that want the whole thing.

Pandas + scikit-learn only; no torch import, so this stays fast to test.

DECISION (DAIC excluded): DAIC-WoZ is dropped by default. Its ``condition`` is
always ``"depression"`` (it marks the measured axis, not a multi-class proxy) and
it is held out for the later Reddit→clinical shift evaluation, so folding it into
this classifier's train/val/test would both leak the axis and misuse the label.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ..data.schema import POC_CONDITIONS, Condition
from .config import SplitConfig


def prepare_classification_frame(
    df: pd.DataFrame,
    conditions: Sequence[Condition] = POC_CONDITIONS,
    exclude_sources: Sequence[str] = ("daic_woz",),
) -> pd.DataFrame:
    """Filter the combined corpus down to POC multi-class training rows.

    Drops rows from ``exclude_sources`` and any row whose ``condition`` is not in
    ``conditions``. Also drops rows with empty/whitespace-only ``text``.

    Tolerates POC conditions that have zero rows on disk today (e.g. anxiety and
    suicidality currently have no files): it warns listing which are absent, but
    does not raise. Raises ``ValueError`` only if the filtered frame is empty.
    """
    wanted = {c.value for c in conditions}
    out = df[~df["source"].isin(list(exclude_sources))].copy()
    out = out[out["condition"].isin(wanted)]
    out = out[out["text"].notna() & (out["text"].str.strip() != "")]
    out = out.reset_index(drop=True)

    if out.empty:
        raise ValueError(
            "prepare_classification_frame produced 0 rows. Check that the combined "
            f"frame has rows for conditions {sorted(wanted)} from sources other "
            f"than {tuple(exclude_sources)}."
        )

    present = set(out["condition"].unique())
    missing = sorted(wanted - present)
    if missing:
        warnings.warn(
            f"POC conditions with no rows on disk yet: {missing}. "
            f"Training will proceed on the {len(present)} present classes "
            f"({sorted(present)}); add their data to include them.",
            stacklevel=2,
        )
    return out


@dataclass(frozen=True)
class SplitFrames:
    """The three author-disjoint partitions."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def author_grouped_split(
    df: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
    group_col: str = "author_id",
) -> SplitFrames:
    """Partition ``df`` into train/val/test with no ``group_col`` crossing a split.

    Uses two chained ``GroupShuffleSplit`` passes: first peel off ``train_frac``
    vs. the rest, then divide the rest into val/test at
    ``val_frac / (val_frac + test_frac)``. Grouping on ``group_col`` guarantees an
    author's rows land entirely in one partition (no leakage).

    Raises:
        ValueError: if ``df`` has fewer than 3 unique ``group_col`` values (cannot
            form three non-empty author-disjoint partitions).
    """
    n_groups = df[group_col].nunique()
    if n_groups < 3:
        raise ValueError(
            f"author_grouped_split needs >=3 unique '{group_col}' values to form "
            f"train/val/test; got {n_groups}."
        )

    groups = df[group_col].to_numpy()

    # Pass 1: train vs. (val + test).
    gss1 = GroupShuffleSplit(
        n_splits=1, train_size=config.train_frac, random_state=config.seed
    )
    train_idx, rest_idx = next(gss1.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    rest_df = df.iloc[rest_idx].reset_index(drop=True)

    # Pass 2: split the remainder into val vs. test.
    rest_groups = rest_df[group_col].to_numpy()
    val_share = config.val_frac / (config.val_frac + config.test_frac)
    gss2 = GroupShuffleSplit(
        n_splits=1, train_size=val_share, random_state=config.seed
    )
    val_idx, test_idx = next(gss2.split(rest_df, groups=rest_groups))
    val_df = rest_df.iloc[val_idx].reset_index(drop=True)
    test_df = rest_df.iloc[test_idx].reset_index(drop=True)

    return SplitFrames(train=train_df, val=val_df, test=test_df)


def build_poc_splits(
    df: pd.DataFrame,
    split_config: SplitConfig = SplitConfig(),
    conditions: Sequence[Condition] = POC_CONDITIONS,
    exclude_sources: Sequence[str] = ("daic_woz",),
) -> SplitFrames:
    """Composition helper: filter to POC rows, then author-grouped split."""
    prepared = prepare_classification_frame(df, conditions, exclude_sources)
    return author_grouped_split(prepared, split_config)
