"""Tests for the condition <-> id mapping (torch-free)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.schema import POC_CONDITIONS
from src.modeling.labels import (
    CONDITION_NAMES,
    CONDITION_TO_ID,
    ID_TO_CONDITION,
    decode_predictions,
    encode_labels,
)


def test_ids_are_contiguous_and_match_poc_order() -> None:
    assert list(CONDITION_TO_ID.values()) == list(range(len(POC_CONDITIONS)))
    for i, condition in enumerate(POC_CONDITIONS):
        assert CONDITION_TO_ID[condition.value] == i
    assert CONDITION_NAMES == [c.value for c in POC_CONDITIONS]


def test_encode_decode_round_trip() -> None:
    conditions = pd.Series([c.value for c in POC_CONDITIONS])
    ids = encode_labels(conditions)
    assert decode_predictions(ids.tolist()) == conditions.tolist()


def test_id_to_condition_is_inverse() -> None:
    for value, i in CONDITION_TO_ID.items():
        assert ID_TO_CONDITION[i] == value


def test_encode_unknown_condition_raises() -> None:
    with pytest.raises(KeyError):
        encode_labels(pd.Series(["depression", "not_a_condition"]))


def test_decode_out_of_range_raises() -> None:
    with pytest.raises(KeyError):
        decode_predictions([0, len(POC_CONDITIONS)])
