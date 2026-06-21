# Data layer — purpose & how to run

This directory (`src/data/`) is the **data-ingestion layer** for the research
project on *condition-dependent proxy-label noise and calibrated abstention for
multi-condition mental-health screening*. The downstream model is MentalBERT.

This layer does **one job**: load three datasets, harmonize their different label
vocabularies into a single schema, and hand back clean pandas DataFrames. It does
**not** train, calibrate, or evaluate anything — that lives downstream and
consumes the frames produced here.

---

## Why this exists (the problem it solves)

The three datasets describe the same conditions using **different words** and come
in **different shapes**:

- **SWMH** (Reddit) calls bipolar `bipolar`; **Low et al.** (Reddit) calls it
  `bipolarreddit`. Same condition, different raw label.
- **Reddit** posts are mixed-case and punctuated; **DAIC-WoZ** clinical
  transcripts are lowercased, unpunctuated, full of disfluencies (`um`, `i i
  think`) and non-speech tags (`[laughter]`). That register gap is part of the
  Reddit→clinical shift the project studies.
- Reddit data is **per-post**; DAIC is a **per-participant** clinical interview
  with PHQ-8 depression labels stored in separate split files.

If every downstream script handled these quirks itself, the label mapping and
text-cleaning rules would be copy-pasted everywhere and drift apart. Instead, this
layer funnels everything through **one label table** and **one text-normalization
policy**, and emits **one canonical schema** so the sources can be concatenated
and compared on equal footing.

### The single canonical schema

Every loader returns a DataFrame with these five columns (always, in this order):

| column      | type           | meaning |
|-------------|----------------|---------|
| `text`      | str            | the post / concatenated participant turns |
| `condition` | str            | harmonized condition label (the multi-class axis) |
| `source`    | str            | `swmh`, `low_et_al`, or `daic_woz` |
| `author_id` | str            | author / participant id (`unknown` if absent) |
| `date`      | str or None    | post date if available |

DAIC adds three more columns on top (additive): `phq8_binary`, `phq8_score`,
`split`. Reddit rows leave those empty.

---

## How the pieces fit (SOLID, one concern per file)

```
src/data/
  schema.py              # Condition enum, canonical column list, validate_schema()
  label_map.py           # the ONE (source, raw_label) -> Condition table + harmonize_label()
  text_normalization.py  # the ONE text-cleaning policy (strip [..]/<..> tags, xxx; optional truecase)
  config.py              # DataPaths: injectable paths (no hardcoded globals)
  base.py                # DatasetLoader interface (just `source` + `load() -> DataFrame`)
  loaders/
    swmh.py              # SWMHLoader
    low_et_al.py         # LowEtAlLoader
    daic_woz.py          # DAICWoZLoader  (per-participant, joins PHQ-8 labels)
  combine.py             # combine_sources(loaders) -> one concatenated, deduped frame
  __main__.py            # CLI: load everything and print sanity-check counts
```

- **Adding a future dataset** = add one new file under `loaders/` implementing the
  `DatasetLoader` interface and register it. No edits to existing loaders or to
  `combine.py`.
- **No code outside `label_map.py`** is allowed to branch on raw subreddit
  strings — everything goes through `harmonize_label()`. Unknown labels return
  `None` and those rows are dropped.
- **Text cleaning lives only in `text_normalization.py`** so the policy is defined
  once and is reusable by every loader.

---

## Where the data goes

Real data lives in `Datasets/` at the repo root (this folder is **git-ignored** —
it is large and license-restricted, so it is never committed):

```
Datasets/
  RedditMentalHealth/    # SWMH files + Low et al. (Zenodo 3941387) CSVs
  DAIC-WOZ/              # participant folders (e.g. 300_P/300_TRANSCRIPT.csv) + PHQ-8 split CSVs
```

Paths are **injected** via `DataPaths`, not hardcoded, so tests run against tiny
fixture directories instead of the real corpus.

> **Current on-disk state (partial):** Low et al. is present for 4 subreddits;
> SWMH files are not on disk yet (its loader fails loudly until they are added);
> DAIC has one participant folder + the three split CSVs.

---

## Setup

Requires **Python 3.11+** and `pandas`. A virtual environment is already created
at `.venv/` (also git-ignored). If you need to recreate it:

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
```

---

## How to run

All commands assume you are in the **repo root**.

### 1. Sanity-check the load (the CLI)

Loads all three sources, harmonizes, combines, and prints per-source and
per-condition counts. Sources whose data is missing are reported and skipped
rather than crashing the run.

```bash
./.venv/Scripts/python.exe -m src.data
```

Useful flags:

```bash
# point at a different data root
./.venv/Scripts/python.exe -m src.data --data-root /path/to/Datasets

# keep Ellie's (interviewer) turns in DAIC transcripts instead of participant-only
./.venv/Scripts/python.exe -m src.data --include-interviewer
```

Example output:

```
=== rows per source ===
low_et_al    139898
daic_woz          1

=== rows per condition ===
depression         113577
eating_disorder     13447
schizophrenia        7680
bipolar              5195
```

### 2. Use it from your own code

```python
from src.data import DataPaths, combine_sources
from src.data.loaders import SWMHLoader, LowEtAlLoader, DAICWoZLoader

paths = DataPaths.default()            # <repo>/Datasets, or DataPaths(data_root=...)
loaders = [LowEtAlLoader(paths), DAICWoZLoader(paths)]
df = combine_sources(loaders)          # one canonical DataFrame
```

### 3. Run the tests

Unit tests use tiny synthetic fixtures (a fake 3-line transcript, a 2-row split
file, etc.) — **not** the real data — so they run anywhere in well under a second.

```bash
./.venv/Scripts/python.exe -m pytest -q
```

They cover: label harmonization (known pairs + unknown→`None`), each loader's
schema, DAIC time-ordering / Ellie-dropping / PHQ-8 join / `include_interviewer`
toggle, text normalization, and `combine_sources` dedup behaviour.

---

## Things to confirm / know

- **DAIC `condition` is always `"depression"`** — it marks the axis being
  measured. The actual depressed/not-depressed label is `phq8_binary`, not the
  multi-class `condition` field.
- **The DAIC test split carries no PHQ-8 labels** (withheld in AVEC2017), so those
  rows have `phq8_binary = <NA>`.
- **Truecasing is OFF by default** — DAIC text keeps its lowercase, unpunctuated
  register on purpose (it's part of what's being studied). A toggle exists for
  later ablation.
- **SMHD and RSDD are intentionally absent** — permanently unavailable under the
  Reddit API ToS change; no loaders exist for them by design.
```
