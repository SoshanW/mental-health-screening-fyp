"""DAIC-WoZ loader (source="daic_woz") -- the depression-only clinical-shift target.

Single responsibility: turn the DAIC-WoZ clinical interview corpus into one
canonical row *per participant* (user-level framing), join the PHQ-8 labels from
the top-level split files, and emit the canonical five columns plus the additive
DAIC fields ``phq8_binary``, ``phq8_score`` and ``split``.

This is NOT multi-class training data: every DAIC row has
``condition = "depression"`` to mark the axis being measured, while the actual
evaluation label is ``phq8_binary``. See DECISION notes below.

Text normalization is delegated to :mod:`src.data.text_normalization`; this
loader never cleans text inline.

VERIFIED against real files on disk:
  * ``XXX_TRANSCRIPT.csv`` is TAB-delimited with columns
    ``start_time, stop_time, speaker, value``; speakers are ``Ellie`` and
    ``Participant``; non-speech events use SQUARE brackets (e.g. ``[laughter]``).
  * ``train``/``dev`` split CSVs: ``Participant_ID, PHQ8_Binary, PHQ8_Score,
    Gender, PHQ8_NoInterest, ... PHQ8_Moving``.
  * The ``test`` split CSV is DIFFERENT: columns ``participant_ID, Gender`` only,
    with NO PHQ-8 labels (AVEC2017 withheld them). Test rows therefore get
    ``phq8_binary = None`` and ``phq8_score = None`` -- see DECISION.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import DataPaths
from ..schema import CANONICAL_COLUMNS, validate_schema
from ..text_normalization import (
    DEFAULT_POLICY,
    NormalizationPolicy,
    normalize_text,
)

# --- Transcript file format (VERIFIED) ---
_TRANSCRIPT_GLOB = "*_TRANSCRIPT.csv"
_TRANSCRIPT_DELIMITER = "\t"
_COL_START = "start_time"
_COL_SPEAKER = "speaker"
_COL_VALUE = "value"
_SPEAKER_PARTICIPANT = "Participant"

# --- Split files (VERIFIED) ---
# Map split name -> (filename, participant-id column). The test file uses a
# lower-case 'participant_ID' and carries no PHQ-8 columns.
_SPLIT_FILES: dict[str, tuple[str, str]] = {
    "train": ("train_split_Depression_AVEC2017.csv", "Participant_ID"),
    "dev": ("dev_split_Depression_AVEC2017.csv", "Participant_ID"),
    "test": ("test_split_Depression_AVEC2017.csv", "participant_ID"),
}
_COL_PHQ8_BINARY = "PHQ8_Binary"
_COL_PHQ8_SCORE = "PHQ8_Score"

# DECISION: every DAIC row is tagged with the condition axis being measured
# (depression), regardless of PHQ-8 outcome. The depressed/not-depressed result
# is carried by `phq8_binary`, NOT by the multi-class `condition` field. Confirm.
_DAIC_CONDITION = "depression"

#: Output columns: canonical five + DAIC-additive three.
_DAIC_EXTRA_COLUMNS = ("phq8_binary", "phq8_score", "split")


class DAICWoZLoader:
    """Loader for the DAIC-WoZ depression interview corpus (AVEC2017 splits)."""

    def __init__(
        self,
        paths: DataPaths,
        include_interviewer: bool = False,
        normalization: NormalizationPolicy = DEFAULT_POLICY,
    ) -> None:
        """
        Args:
            paths: Injected path configuration; reads from ``paths.daic_dir``.
            include_interviewer: If ``False`` (default), drop Ellie's prompts and
                keep only ``Participant`` turns. DECISION: participant-only is the
                default; the flag exists so the interviewer-context ablation can
                be run later.
            normalization: Text policy for the concatenated transcript. Defaults
                to :data:`~src.data.text_normalization.DEFAULT_POLICY` (strip
                ``[...]``/``<...>`` non-speech tags and ``xxx``; keep lowercase,
                unpunctuated register).
        """
        self._paths = paths
        self._include_interviewer = include_interviewer
        self._normalization = normalization

    @property
    def source(self) -> str:
        return "daic_woz"

    def load(self) -> pd.DataFrame:
        daic_dir = self._paths.daic_dir
        if not daic_dir.exists():
            raise FileNotFoundError(
                f"DAIC-WoZ directory '{daic_dir}' does not exist. Expected "
                f"participant folders (e.g. '300_P/300_TRANSCRIPT.csv') and the "
                f"PHQ-8 split CSVs. Obtain DAIC-WoZ from "
                f"https://dcapswoz.ict.usc.edu/ (access-restricted)."
            )

        labels = self._load_labels(daic_dir)  # participant_id -> label record
        transcripts = self._load_transcripts(daic_dir)  # participant_id -> text

        if not transcripts:
            raise FileNotFoundError(
                f"No transcript files matching '{_TRANSCRIPT_GLOB}' were found "
                f"under '{daic_dir}'. Expected per-participant folders like "
                f"'300_P/300_TRANSCRIPT.csv'."
            )

        rows: list[dict[str, object]] = []
        for participant_id, text in sorted(transcripts.items()):
            label = labels.get(participant_id)
            if label is None:
                # Transcript present but participant not listed in any split file.
                # DECISION: skip unlabeled-by-split participants rather than guess.
                continue
            rows.append(
                {
                    "text": text,
                    "condition": _DAIC_CONDITION,
                    "source": self.source,
                    "author_id": participant_id,
                    "date": None,
                    "phq8_binary": label["phq8_binary"],
                    "phq8_score": label["phq8_score"],
                    "split": label["split"],
                }
            )

        out = pd.DataFrame(rows, columns=[*CANONICAL_COLUMNS, *_DAIC_EXTRA_COLUMNS])
        # phq8_binary may contain None (test split) -> use pandas nullable Int.
        out["phq8_binary"] = out["phq8_binary"].astype("Int64")
        out["phq8_score"] = out["phq8_score"].astype("Int64")
        return validate_schema(out)

    # -- transcripts -------------------------------------------------------

    def _load_transcripts(self, daic_dir: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        for transcript_path in sorted(daic_dir.rglob(_TRANSCRIPT_GLOB)):
            participant_id = self._participant_id_from_path(transcript_path)
            result[participant_id] = self._concat_turns(transcript_path)
        return result

    def _concat_turns(self, transcript_path: Path) -> str:
        df = pd.read_csv(transcript_path, sep=_TRANSCRIPT_DELIMITER, dtype=str)
        df = df.dropna(subset=[_COL_VALUE])
        if not self._include_interviewer:
            df = df[df[_COL_SPEAKER] == _SPEAKER_PARTICIPANT]
        # Concatenate in time order.
        df = df.sort_values(_COL_START, key=lambda s: s.astype(float))
        joined = " ".join(t for t in df[_COL_VALUE].tolist() if t)
        return normalize_text(joined, self._normalization)

    @staticmethod
    def _participant_id_from_path(transcript_path: Path) -> str:
        # File is named '<id>_TRANSCRIPT.csv'; the id is the leading numeric part.
        return transcript_path.name.split("_", 1)[0]

    # -- labels ------------------------------------------------------------

    def _load_labels(self, daic_dir: Path) -> dict[str, dict[str, object]]:
        labels: dict[str, dict[str, object]] = {}
        for split, (filename, id_col) in _SPLIT_FILES.items():
            path = daic_dir / filename
            if not path.exists():
                # DECISION: a missing split file is tolerated (e.g. test withheld);
                # we still load whichever splits are present.
                continue
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            id_key = cols.get(id_col.lower())
            if id_key is None:
                raise FileNotFoundError(
                    f"DAIC split file '{path}' is missing the participant-id "
                    f"column '{id_col}'; found {list(df.columns)}."
                )
            bin_key = cols.get(_COL_PHQ8_BINARY.lower())
            score_key = cols.get(_COL_PHQ8_SCORE.lower())
            for _, row in df.iterrows():
                participant_id = str(int(row[id_key])) if pd.notna(row[id_key]) else None
                if participant_id is None:
                    continue
                labels[participant_id] = {
                    "split": split,
                    "phq8_binary": (
                        int(row[bin_key])
                        if bin_key is not None and pd.notna(row[bin_key])
                        else None
                    ),
                    "phq8_score": (
                        int(row[score_key])
                        if score_key is not None and pd.notna(row[score_key])
                        else None
                    ),
                }
        return labels
