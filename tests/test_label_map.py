"""Tests for the label harmonizer."""

from __future__ import annotations

import pytest

from src.data.label_map import harmonize_label
from src.data.schema import Condition


@pytest.mark.parametrize(
    ("source", "raw", "expected"),
    [
        ("swmh", "depression", Condition.DEPRESSION),
        ("swmh", "suicidewatch", Condition.SUICIDALITY),
        ("swmh", "offmychest", Condition.OFFMYCHEST),
        ("low_et_al", "bipolarreddit", Condition.BIPOLAR),
        ("low_et_al", "EDAnonymous", Condition.EATING_DISORDER),  # case-insensitive
        ("low_et_al", "schizophrenia", Condition.SCHIZOPHRENIA),
        ("low_et_al", "  Depression  ", Condition.DEPRESSION),  # trimmed
    ],
)
def test_known_pairs_map_correctly(source: str, raw: str, expected: Condition) -> None:
    assert harmonize_label(source, raw) is expected


@pytest.mark.parametrize(
    ("source", "raw"),
    [
        ("low_et_al", "totallyunknownsub"),
        ("swmh", "bipolarreddit"),  # right label, wrong source spelling
        ("daic_woz", "depression"),  # daic does not go through the subreddit table
        ("low_et_al", None),
    ],
)
def test_unknown_pairs_return_none(source: str, raw: str | None) -> None:
    assert harmonize_label(source, raw) is None
