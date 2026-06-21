"""Label harmonization: the single source of truth for raw-label -> Condition.

Single responsibility: map each dataset's raw subreddit/label string onto the
shared :class:`~src.data.schema.Condition` vocabulary. SWMH and Low et al. name
the *same* condition differently (e.g. ``bipolar`` vs ``bipolarreddit``), which is
exactly why this table exists and lives in one place.

No code outside this module may branch on raw subreddit strings. Every loader
calls :func:`harmonize_label`; unknown labels return ``None`` and those rows are
dropped by the caller.
"""

from __future__ import annotations

from typing import Final

from .schema import Condition

#: The harmonization table, keyed on ``(source, raw_label)``. Keys are stored
#: lowercase; :func:`harmonize_label` lowercases its inputs before lookup so that
#: real-world casing (e.g. Low et al.'s ``"EDAnonymous"``) resolves correctly.
#:
#: VERIFIED against the real files on disk:
#:   * Low et al. subreddit column values: "depression", "bipolarreddit",
#:     "EDAnonymous", "schizophrenia" (anxiety/suicidewatch/bpd from the same
#:     Zenodo release are mapped here too even though those CSVs are not present
#:     locally, so the table is complete for the full dataset).
#: SWMH raw labels are taken from the documented SWMH subreddit set; the SWMH
#: files are not present on disk yet, so these keys are unverified against real
#: data -- see DECISION note below.
_LABEL_TABLE: Final[dict[tuple[str, str], Condition]] = {
    # --- SWMH (source="swmh") ---
    # DECISION: SWMH raw-label spellings below are unverified (no SWMH files on
    # disk). They follow the documented SWMH subreddit set. Confirm against the
    # real SWMH file's label column when it is added.
    ("swmh", "depression"): Condition.DEPRESSION,
    ("swmh", "anxiety"): Condition.ANXIETY,
    ("swmh", "bipolar"): Condition.BIPOLAR,
    ("swmh", "suicidewatch"): Condition.SUICIDALITY,
    ("swmh", "offmychest"): Condition.OFFMYCHEST,
    # --- Low et al. (source="low_et_al") ---
    ("low_et_al", "depression"): Condition.DEPRESSION,
    ("low_et_al", "anxiety"): Condition.ANXIETY,
    ("low_et_al", "bipolarreddit"): Condition.BIPOLAR,
    ("low_et_al", "suicidewatch"): Condition.SUICIDALITY,
    ("low_et_al", "schizophrenia"): Condition.SCHIZOPHRENIA,
    ("low_et_al", "edanonymous"): Condition.EATING_DISORDER,
    ("low_et_al", "bpd"): Condition.BPD,
}


def harmonize_label(source: str, raw_label: str | None) -> Condition | None:
    """Map a dataset-specific ``raw_label`` to a :class:`Condition`.

    Lookup is case-insensitive and whitespace-trimmed on both ``source`` and
    ``raw_label``. Returns ``None`` for unknown ``(source, raw_label)`` pairs;
    callers drop those rows.
    """
    if raw_label is None:
        return None
    key = (source.strip().lower(), str(raw_label).strip().lower())
    return _LABEL_TABLE.get(key)
