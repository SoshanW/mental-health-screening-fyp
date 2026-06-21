"""CLI entrypoint: load all three sources and print sanity-check counts.

Single responsibility: wire the default :class:`~src.data.config.DataPaths` to the
three loaders, combine them, and print per-source and per-condition value counts
so the load can be eyeballed. Loaders whose data is absent are reported and
skipped rather than aborting the whole run.

Run with::

    python -m src.data
    python -m src.data --data-root /path/to/Datasets
    python -m src.data --include-interviewer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .base import DatasetLoader
from .combine import combine_sources
from .config import DataPaths
from .loaders import DAICWoZLoader, LowEtAlLoader, SWMHLoader


class _PreloadedLoader:
    """Adapter wrapping an already-loaded frame as a ``DatasetLoader``.

    Lets the CLI load each source once (to catch missing-data errors), then feed
    the surviving frames back through ``combine_sources`` without re-reading disk.
    """

    def __init__(self, source: str, frame: pd.DataFrame) -> None:
        self._source = source
        self._frame = frame

    @property
    def source(self) -> str:
        return self._source

    def load(self) -> pd.DataFrame:
        return self._frame


def _build_loaders(paths: DataPaths, include_interviewer: bool) -> list[DatasetLoader]:
    return [
        SWMHLoader(paths),
        LowEtAlLoader(paths),
        DAICWoZLoader(paths, include_interviewer=include_interviewer),
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and summarize the screening corpus.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root Datasets directory (defaults to <repo>/Datasets).",
    )
    parser.add_argument(
        "--include-interviewer",
        action="store_true",
        help="Keep Ellie's prompts in DAIC transcripts (default: participant-only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = DataPaths(data_root=args.data_root) if args.data_root else DataPaths.default()

    print(f"Data root: {paths.data_root}")
    loaders = _build_loaders(paths, include_interviewer=args.include_interviewer)

    available: list[DatasetLoader] = []
    for loader in loaders:
        try:
            frame = loader.load()
            available.append(_PreloadedLoader(loader.source, frame))
        except FileNotFoundError as exc:
            print(f"[skip] source '{loader.source}': {exc}")

    if not available:
        print("No sources could be loaded. Check --data-root and dataset placement.")
        return 1

    combined = combine_sources(available)

    print("\n=== rows per source ===")
    print(combined["source"].value_counts().to_string())

    print("\n=== rows per condition ===")
    print(combined["condition"].value_counts(dropna=False).to_string())

    print("\n=== source x condition ===")
    with pd.option_context("display.max_rows", None):
        print(
            combined.groupby(["source", "condition"], dropna=False)
            .size()
            .rename("count")
            .to_string()
        )

    if "phq8_binary" in combined.columns:
        daic = combined[combined["source"] == "daic_woz"]
        if not daic.empty:
            print("\n=== DAIC phq8_binary x split ===")
            print(
                daic.groupby(["split", "phq8_binary"], dropna=False)
                .size()
                .rename("count")
                .to_string()
            )

    print(f"\nTotal rows: {len(combined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
