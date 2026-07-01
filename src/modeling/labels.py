"""Label <-> integer id mapping for the multi-class classifier.

Single responsibility: own the one canonical mapping between the harmonized
:class:`~src.data.schema.Condition` vocabulary (as strings) and the contiguous
integer ids a classifier head needs. This mirrors the single-source-of-truth role
of :mod:`src.data.label_map`, but for the *downstream* id encoding rather than the
raw-label harmonization.

The id order is fixed by the enumeration order of ``POC_CONDITIONS`` so ids are
stable across runs (id 0 == first POC condition, etc.). No torch import here: the
mapping is plain Python and stays testable without the deep-learning stack.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import pandas as pd

from ..data.schema import POC_CONDITIONS

#: condition value (e.g. "depression") -> contiguous class id (0..K-1).
CONDITION_TO_ID: Final[dict[str, int]] = {
    condition.value: i for i, condition in enumerate(POC_CONDITIONS)
}

#: class id -> condition value; the inverse of :data:`CONDITION_TO_ID`.
ID_TO_CONDITION: Final[dict[int, str]] = {
    i: value for value, i in CONDITION_TO_ID.items()
}

#: Condition values in id order; handy for per-class report column headers.
CONDITION_NAMES: Final[list[str]] = [ID_TO_CONDITION[i] for i in range(len(ID_TO_CONDITION))]


def encode_labels(conditions: pd.Series) -> pd.Series:
    """Map a Series of condition strings to integer class ids.

    Raises:
        KeyError: if any value is not a POC condition. We fail loudly rather than
            silently drop, because by this stage the frame should already have been
            filtered to POC conditions (see
            :func:`src.modeling.splits.prepare_classification_frame`).
    """
    unknown = sorted(set(conditions.unique()) - set(CONDITION_TO_ID))
    if unknown:
        raise KeyError(
            f"encode_labels received conditions outside POC_CONDITIONS: {unknown}. "
            f"Filter to POC conditions before encoding."
        )
    return conditions.map(CONDITION_TO_ID).astype("int64")


def decode_predictions(ids: Sequence[int]) -> list[str]:
    """Map integer class ids back to condition strings.

    Raises:
        KeyError: if any id is outside ``0..K-1``.
    """
    out: list[str] = []
    for i in ids:
        key = int(i)
        if key not in ID_TO_CONDITION:
            raise KeyError(f"decode_predictions received out-of-range id {key}.")
        out.append(ID_TO_CONDITION[key])
    return out
