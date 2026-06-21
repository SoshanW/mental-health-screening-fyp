"""Shared pytest fixtures: tiny synthetic on-disk datasets (no real data)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.config import DataPaths


@pytest.fixture
def data_paths(tmp_path: Path) -> DataPaths:
    """A DataPaths rooted at a fresh temp dir, with empty dataset subdirs."""
    paths = DataPaths(data_root=tmp_path)
    paths.reddit_dir.mkdir(parents=True, exist_ok=True)
    paths.daic_dir.mkdir(parents=True, exist_ok=True)
    return paths


@pytest.fixture
def low_et_al_file(data_paths: DataPaths) -> DataPaths:
    """Write a tiny Low et al. style CSV (with extra ignored feature columns)."""
    csv = (
        "subreddit,author,date,post,n_words,liwc_anxiety\n"
        "depression,alice,2019/01/01,i feel low today,4,0.1\n"
        "EDAnonymous,bob,2019/02/02,counting calories again,3,0.0\n"
        "bipolarreddit,carol,2019/03/03,manic episode last week,4,0.2\n"
        "totallyunknownsub,dave,2019/04/04,should be dropped,3,0.0\n"
    )
    (data_paths.reddit_dir / "depression_post_features_tfidf_256.csv").write_text(
        csv, encoding="utf-8"
    )
    return data_paths


@pytest.fixture
def swmh_file(data_paths: DataPaths) -> DataPaths:
    """Write a tiny combined SWMH style CSV with a label column."""
    csv = (
        "label,author,date,text\n"
        "depression,u1,2018/01/01,cant get out of bed\n"
        "SuicideWatch,u2,2018/02/02,i dont want to be here\n"
        "offmychest,u3,2018/03/03,just venting\n"
        "randomsub,u4,2018/04/04,drop me\n"
    )
    (data_paths.reddit_dir / "swmh_combined.csv").write_text(csv, encoding="utf-8")
    return data_paths


@pytest.fixture
def daic_dataset(data_paths: DataPaths) -> DataPaths:
    """Write a 3-line fake transcript + tiny train/dev/test split files."""
    # Participant 300: transcript with Ellie + Participant turns, out of order,
    # containing a [laughter] non-speech tag and an xxx unintelligible token.
    transcript = (
        "start_time\tstop_time\tspeaker\tvalue\n"
        "10.0\t12.0\tParticipant\tsecond thing [laughter]\n"
        "5.0\t7.0\tEllie\thow are you\n"
        "8.0\t9.0\tParticipant\tfirst thing xxx\n"
    )
    pdir = data_paths.daic_dir / "300_P"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "300_TRANSCRIPT.csv").write_text(transcript, encoding="utf-8")

    # Participant 301: transcript present, lives in the (label-less) test split.
    pdir2 = data_paths.daic_dir / "301_P"
    pdir2.mkdir(parents=True, exist_ok=True)
    (pdir2 / "301_TRANSCRIPT.csv").write_text(
        "start_time\tstop_time\tspeaker\tvalue\n"
        "1.0\t2.0\tParticipant\thello there\n",
        encoding="utf-8",
    )

    # train split: participant 300 labelled depressed.
    (data_paths.daic_dir / "train_split_Depression_AVEC2017.csv").write_text(
        "Participant_ID,PHQ8_Binary,PHQ8_Score,Gender,PHQ8_NoInterest\n"
        "300,1,15,0,2\n",
        encoding="utf-8",
    )
    # dev split: a participant with no transcript on disk (should be ignored).
    (data_paths.daic_dir / "dev_split_Depression_AVEC2017.csv").write_text(
        "Participant_ID,PHQ8_Binary,PHQ8_Score,Gender,PHQ8_NoInterest\n"
        "999,0,3,1,0\n",
        encoding="utf-8",
    )
    # test split: lowercase id col, NO PHQ-8 columns.
    (data_paths.daic_dir / "test_split_Depression_AVEC2017.csv").write_text(
        "participant_ID,Gender\n301,1\n",
        encoding="utf-8",
    )
    return data_paths
