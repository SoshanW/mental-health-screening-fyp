"""Tests for combine_sources composition and dedup behaviour."""

from __future__ import annotations

import pandas as pd

from src.data.combine import combine_sources
from src.data.config import DataPaths
from src.data.loaders import DAICWoZLoader, LowEtAlLoader, SWMHLoader
from src.data.schema import CANONICAL_COLUMNS, validate_schema


class _FakeLoader:
    """Minimal in-memory DatasetLoader for dedup tests."""

    def __init__(self, source: str, frame: pd.DataFrame) -> None:
        self._source = source
        self._frame = frame

    @property
    def source(self) -> str:
        return self._source

    def load(self) -> pd.DataFrame:
        return self._frame


def _frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["text", "author_id", "condition"])
    df["source"] = "fake"
    df["date"] = None
    return df[list(CANONICAL_COLUMNS)]


def test_combine_dedups_exact_duplicates() -> None:
    frame = _frame(
        [
            ("same text", "a1", "depression"),
            ("same text", "a1", "depression"),  # exact dup -> dropped
            ("other", "a1", "depression"),
        ]
    )
    out = combine_sources([_FakeLoader("fake", frame)])
    assert len(out) == 2


def test_combine_keeps_distinct_author_rows_with_same_text() -> None:
    frame = _frame(
        [
            ("identical post", "author_a", "depression"),
            ("identical post", "author_b", "depression"),  # different author -> kept
        ]
    )
    out = combine_sources([_FakeLoader("fake", frame)])
    assert len(out) == 2
    assert set(out["author_id"]) == {"author_a", "author_b"}


def test_combine_concatenates_multiple_real_sources(
    low_et_al_file: DataPaths,
    swmh_file: DataPaths,
    daic_dataset: DataPaths,
) -> None:
    # All three fixtures point at the same tmp data_paths.
    paths = low_et_al_file
    out = combine_sources(
        [SWMHLoader(paths), LowEtAlLoader(paths), DAICWoZLoader(paths)]
    )
    validate_schema(out)
    assert set(out["source"]) == {"swmh", "low_et_al", "daic_woz"}
    # additive DAIC columns present and NaN for reddit rows.
    assert "phq8_binary" in out.columns
    reddit = out[out["source"] != "daic_woz"]
    assert reddit["phq8_binary"].isna().all()
