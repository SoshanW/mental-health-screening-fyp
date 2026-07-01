"""Tests for filtering + author-grouped splitting (pandas + sklearn, no torch)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.modeling.config import SplitConfig
from src.modeling.splits import (
    author_grouped_split,
    build_poc_splits,
    prepare_classification_frame,
)


def _frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Build a canonical-ish frame from (text, condition, source, author_id) tuples."""
    return pd.DataFrame(
        [
            {"text": t, "condition": c, "source": s, "author_id": a, "date": None}
            for (t, c, s, a) in rows
        ]
    )


def test_prepare_drops_daic_and_out_of_scope_conditions() -> None:
    df = _frame(
        [
            ("post about low mood", "depression", "low_et_al", "a1"),
            ("clinical interview", "depression", "daic_woz", "300"),  # DAIC -> dropped
            ("venting", "offmychest", "swmh", "a2"),  # not a POC condition -> dropped
            ("manic", "bipolar", "low_et_al", "a3"),
        ]
    )
    out = prepare_classification_frame(df)
    assert set(out["source"]) == {"low_et_al"}
    assert set(out["condition"]) == {"depression", "bipolar"}
    assert "daic_woz" not in set(out["source"])


def test_prepare_drops_empty_text() -> None:
    df = _frame(
        [
            ("real text", "depression", "low_et_al", "a1"),
            ("   ", "bipolar", "low_et_al", "a2"),
        ]
    )
    out = prepare_classification_frame(df)
    assert len(out) == 1


def test_prepare_tolerates_missing_conditions_but_warns() -> None:
    # Only depression present; other POC conditions absent -> warn, not raise.
    df = _frame([("a", "depression", "low_et_al", f"u{i}") for i in range(3)])
    with pytest.warns(UserWarning):
        out = prepare_classification_frame(df)
    assert set(out["condition"]) == {"depression"}


def test_prepare_empty_result_raises() -> None:
    df = _frame([("clinical", "depression", "daic_woz", "300")])  # only excluded source
    with pytest.raises(ValueError):
        prepare_classification_frame(df)


def test_author_grouped_split_no_author_crosses_boundary() -> None:
    # 30 authors, 3 posts each.
    rows = []
    for i in range(30):
        for j in range(3):
            rows.append((f"post {i}-{j}", "depression", "low_et_al", f"author{i}"))
    df = _frame(rows)
    splits = author_grouped_split(df, SplitConfig(seed=7))

    train_a = set(splits.train["author_id"])
    val_a = set(splits.val["author_id"])
    test_a = set(splits.test["author_id"])
    assert train_a.isdisjoint(val_a)
    assert train_a.isdisjoint(test_a)
    assert val_a.isdisjoint(test_a)
    # No rows lost.
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(df)


def test_author_grouped_split_approximate_ratios() -> None:
    rows = [(f"p{i}", "depression", "low_et_al", f"author{i}") for i in range(100)]
    df = _frame(rows)
    splits = author_grouped_split(df, SplitConfig(train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=1))
    assert 70 <= len(splits.train) <= 90
    assert 3 <= len(splits.val) <= 20
    assert 3 <= len(splits.test) <= 20


def test_author_grouped_split_too_few_groups_raises() -> None:
    df = _frame([("a", "depression", "low_et_al", "only_author") for _ in range(5)])
    with pytest.raises(ValueError):
        author_grouped_split(df)


def test_split_config_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        SplitConfig(train_frac=0.7, val_frac=0.2, test_frac=0.2)  # sums to 1.1
    with pytest.raises(ValueError):
        SplitConfig(train_frac=0.8, val_frac=0.2, test_frac=0.0)  # zero fraction


def test_build_poc_splits_end_to_end() -> None:
    rows = []
    for i in range(20):
        rows.append((f"dep {i}", "depression", "low_et_al", f"a{i}"))
        rows.append((f"clinical {i}", "depression", "daic_woz", f"d{i}"))  # excluded
    df = _frame(rows)
    with pytest.warns(UserWarning):  # only depression present among POC
        splits = build_poc_splits(df)
    all_sources = set(splits.train["source"]) | set(splits.val["source"]) | set(splits.test["source"])
    assert all_sources == {"low_et_al"}
