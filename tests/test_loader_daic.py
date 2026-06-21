"""Tests for the DAIC-WoZ loader."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.config import DataPaths
from src.data.loaders import DAICWoZLoader
from src.data.schema import CANONICAL_COLUMNS, validate_schema


def _row(df: pd.DataFrame, participant: str) -> pd.Series:
    return df[df["author_id"] == participant].iloc[0]


def test_daic_emits_canonical_plus_additive_columns(daic_dataset: DataPaths) -> None:
    df = DAICWoZLoader(daic_dataset).load()

    # canonical five first, then additive three.
    assert list(df.columns[:5]) == list(CANONICAL_COLUMNS)
    for col in ("phq8_binary", "phq8_score", "split"):
        assert col in df.columns
    validate_schema(df)

    assert set(df["source"]) == {"daic_woz"}
    assert set(df["condition"]) == {"depression"}


def test_daic_concatenates_turns_in_time_order_and_drops_ellie(daic_dataset: DataPaths) -> None:
    df = DAICWoZLoader(daic_dataset).load()  # include_interviewer=False default
    text = _row(df, "300")["text"]

    # Participant turns only, ordered by start_time (8.0 before 10.0).
    assert text == "first thing second thing"
    # Ellie's prompt dropped.
    assert "how are you" not in text
    # non-speech tag and xxx removed by default normalization policy.
    assert "[laughter]" not in text
    assert "xxx" not in text


def test_daic_include_interviewer_keeps_ellie(daic_dataset: DataPaths) -> None:
    df = DAICWoZLoader(daic_dataset, include_interviewer=True).load()
    text = _row(df, "300")["text"]
    assert "how are you" in text
    # still time-ordered: 8.0 (first thing) precedes 10.0 (second thing),
    # Ellie at 5.0 comes first overall.
    assert text == "how are you first thing second thing"


def test_daic_joins_phq8_label(daic_dataset: DataPaths) -> None:
    df = DAICWoZLoader(daic_dataset).load()
    row = _row(df, "300")
    assert row["phq8_binary"] == 1
    assert row["phq8_score"] == 15
    assert row["split"] == "train"


def test_daic_test_split_has_no_phq8_label(daic_dataset: DataPaths) -> None:
    df = DAICWoZLoader(daic_dataset).load()
    row = _row(df, "301")
    assert row["split"] == "test"
    assert pd.isna(row["phq8_binary"])
    assert pd.isna(row["phq8_score"])


def test_daic_ignores_split_participant_without_transcript(daic_dataset: DataPaths) -> None:
    df = DAICWoZLoader(daic_dataset).load()
    # participant 999 is in the dev split but has no transcript on disk.
    assert "999" not in set(df["author_id"])


def test_daic_missing_directory_raises(tmp_path) -> None:
    paths = DataPaths(data_root=tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="DAIC-WoZ"):
        DAICWoZLoader(paths).load()
