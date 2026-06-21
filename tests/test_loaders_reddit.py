"""Tests for the Reddit loaders (Low et al. and SWMH)."""

from __future__ import annotations

import pytest

from src.data.config import DataPaths
from src.data.loaders import LowEtAlLoader, SWMHLoader
from src.data.schema import CANONICAL_COLUMNS, validate_schema


def test_low_et_al_emits_canonical_schema_and_drops_unknown(low_et_al_file: DataPaths) -> None:
    df = LowEtAlLoader(low_et_al_file).load()

    assert list(df.columns) == list(CANONICAL_COLUMNS)
    validate_schema(df)  # must not raise

    assert set(df["source"]) == {"low_et_al"}
    # unknown subreddit row dropped; 3 known rows remain.
    assert len(df) == 3
    assert set(df["condition"]) == {"depression", "eating_disorder", "bipolar"}
    assert "totallyunknownsub" not in set(df["condition"])
    # author/date passed through.
    assert "alice" in set(df["author_id"])
    assert "2019/01/01" in set(df["date"])


def test_low_et_al_does_not_deduplicate_by_author(data_paths: DataPaths) -> None:
    # Two posts from the same author must both survive (single-post proxy).
    csv = (
        "subreddit,author,date,post\n"
        "depression,sameauthor,2019/01/01,post one\n"
        "depression,sameauthor,2019/01/02,post two\n"
    )
    (data_paths.reddit_dir / "depression_post_features_tfidf_256.csv").write_text(
        csv, encoding="utf-8"
    )
    df = LowEtAlLoader(data_paths).load()
    assert len(df) == 2


def test_low_et_al_missing_files_raises(data_paths: DataPaths) -> None:
    with pytest.raises(FileNotFoundError, match="Zenodo 3941387"):
        LowEtAlLoader(data_paths).load()


def test_swmh_emits_canonical_schema_and_harmonizes(swmh_file: DataPaths) -> None:
    df = SWMHLoader(swmh_file).load()

    assert list(df.columns) == list(CANONICAL_COLUMNS)
    validate_schema(df)

    assert set(df["source"]) == {"swmh"}
    assert len(df) == 3  # randomsub dropped
    assert set(df["condition"]) == {"depression", "suicidality", "offmychest"}


def test_swmh_missing_files_raises(data_paths: DataPaths) -> None:
    with pytest.raises(FileNotFoundError, match="SWMH"):
        SWMHLoader(data_paths).load()
