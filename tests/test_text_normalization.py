"""Tests for the text normalization module."""

from __future__ import annotations

from src.data.text_normalization import (
    DEFAULT_POLICY,
    NormalizationPolicy,
    normalize_text,
)


def test_strips_bracket_tags_angle_and_square() -> None:
    out = normalize_text("i was <laughter> happy [cough] today")
    assert "laughter" not in out
    assert "cough" not in out
    assert out == "i was happy today"


def test_removes_xxx_tokens() -> None:
    out = normalize_text("i said xxx and then XXX again")
    assert "xxx" not in out.lower()
    assert out == "i said and then again"


def test_truecase_off_by_default_leaves_text_lowercase() -> None:
    text = "i i i think um it was fine"
    assert DEFAULT_POLICY.truecase is False
    assert normalize_text(text) == text  # unchanged register


def test_truecase_toggle_has_effect_when_on() -> None:
    policy = NormalizationPolicy(truecase=True)
    assert normalize_text("hello world", policy).startswith("H")


def test_none_is_treated_as_empty() -> None:
    assert normalize_text(None) == ""
