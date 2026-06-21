"""Canonical output schema for the data layer.

Single responsibility: define the harmonized ``Condition`` vocabulary, the exact
set of columns every loader must emit, and a ``validate_schema`` helper that each
loader calls at the end of ``load()`` so that all sources are substitutable
(LSP) and can be concatenated by :mod:`src.data.combine`.

This module owns *what a valid output frame looks like*. It does not know how any
individual dataset is parsed.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

import pandas as pd


class Condition(str, Enum):
    """Harmonized, cross-dataset condition vocabulary (the multi-class axis).

    Subclassing ``str`` so values compare/serialize as plain strings, which keeps
    the emitted ``condition`` column a normal object/str column.
    """

    DEPRESSION = "depression"
    ANXIETY = "anxiety"
    BIPOLAR = "bipolar"
    SUICIDALITY = "suicidality"
    SCHIZOPHRENIA = "schizophrenia"
    EATING_DISORDER = "eating_disorder"
    BPD = "bpd"
    OFFMYCHEST = "offmychest"
    HEALTHY_CONTROL = "healthy_control"


#: The conditions actually used for the proof-of-concept multi-class training.
#: This is the narrower set present in the Reddit data; the full ``Condition``
#: enum is broader on purpose (e.g. OFFMYCHEST / HEALTHY_CONTROL are auxiliary).
POC_CONDITIONS: Final[tuple[Condition, ...]] = (
    Condition.DEPRESSION,
    Condition.ANXIETY,
    Condition.BIPOLAR,
    Condition.SUICIDALITY,
    Condition.SCHIZOPHRENIA,
    Condition.EATING_DISORDER,
)


#: The five columns every loader must emit, in canonical order. DAIC adds more
#: columns on top of these (additive), but these five are always present so the
#: three sources can be concatenated.
CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "text",
    "condition",
    "source",
    "author_id",
    "date",
)

#: Placeholder used when a dataset has no author/participant identifier.
UNKNOWN_AUTHOR: Final[str] = "unknown"


class SchemaError(ValueError):
    """Raised when a loader produces a frame that violates the canonical schema."""


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that ``df`` satisfies the canonical contract; return it unchanged.

    Checks that:

    * all :data:`CANONICAL_COLUMNS` are present,
    * the first five columns appear in canonical order (extra additive columns,
      e.g. DAIC's PHQ-8 fields, may follow),
    * ``condition`` only contains valid :class:`Condition` values (or ``None``).

    Raises:
        SchemaError: if any of the above is violated.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Output frame is missing required columns {missing}; "
            f"got columns {list(df.columns)}"
        )

    leading = list(df.columns[: len(CANONICAL_COLUMNS)])
    if leading != list(CANONICAL_COLUMNS):
        raise SchemaError(
            "The first five columns must be the canonical columns in order "
            f"{list(CANONICAL_COLUMNS)}; got {leading}"
        )

    valid_values = {c.value for c in Condition}
    bad = (
        df.loc[df["condition"].notna() & ~df["condition"].isin(valid_values), "condition"]
        .unique()
        .tolist()
    )
    if bad:
        raise SchemaError(
            f"`condition` column contains values outside the Condition enum: {bad}"
        )

    return df
