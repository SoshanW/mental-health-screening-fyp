# Data Harmonization — how it works

This document explains, in depth, **how the three datasets are harmonized into one
clean table**. It is the focused companion to the broader data-layer overview in
[`src/data/README.md`](../src/data/README.md).

> **Scope note.** Harmonization is the *only* thing built so far. It lives entirely
> in `src/data/`. Nothing downstream (model training, calibration, abstention) has
> been implemented yet — those are future increments.

---

## 1. The problem harmonization solves

Three datasets describe the **same mental-health conditions**, but they disagree on
almost everything else:

| | SWMH (Reddit) | Low et al. (Reddit) | DAIC-WoZ (clinical) |
|---|---|---|---|
| **What a "label" is** | subreddit name | subreddit name | PHQ-8 questionnaire score |
| **Name for bipolar** | `bipolar` | `bipolarreddit` | — (depression-only corpus) |
| **Row granularity** | one post | one post | one whole interview per person |
| **Text register** | mixed-case, punctuated | mixed-case, punctuated | lowercase, no punctuation, disfluencies, `[laughter]` tags |
| **Where labels live** | a column | a column | separate split CSV files |

If every downstream script had to know that `bipolar` and `bipolarreddit` mean the
same thing, that rule would be copy-pasted everywhere and drift apart. Harmonization
funnels all of this through **one vocabulary**, **one label table**, and **one text
policy**, and emits **one canonical schema** so the sources can be concatenated and
compared on equal footing.

Harmonization has **two axes**:

1. **Label harmonization** — different words → one shared condition vocabulary.
2. **Register (text) harmonization** — cleaning non-speech noise while *deliberately
   preserving* the Reddit↔clinical style gap that the research studies.

---

## 2. Axis 1 — Label harmonization

### Step 1: one shared vocabulary — `src/data/schema.py`

There is a single canonical list of conditions, the `Condition` enum:

```
depression, anxiety, bipolar, suicidality, schizophrenia,
eating_disorder, bpd, offmychest, healthy_control
```

Every dataset must express its labels in **these terms and no others**.
`validate_schema()` enforces this at the end of every loader — a loader physically
cannot emit a condition outside this set, or it raises `SchemaError`.

`POC_CONDITIONS` is the narrower 6-condition subset used for the proof-of-concept
multi-class task: `depression, anxiety, bipolar, suicidality, schizophrenia,
eating_disorder`.

### Step 2: one translation table — `src/data/label_map.py`

This is the heart of harmonization: a single dictionary keyed on `(source, raw_label)`:

```python
("swmh",      "bipolar")       -> Condition.BIPOLAR
("low_et_al", "bipolarreddit") -> Condition.BIPOLAR      # same condition, different word
("low_et_al", "edanonymous")   -> Condition.EATING_DISORDER
("swmh",      "suicidewatch")  -> Condition.SUICIDALITY
("low_et_al", "depression")    -> Condition.DEPRESSION
...
```

`harmonize_label(source, raw_label)` does a **case-insensitive, whitespace-trimmed**
lookup. The rules that make this trustworthy:

- **Single source of truth.** No code anywhere else is allowed to branch on raw
  subreddit strings — everything goes through this one function. To remap a
  condition, you change one line here and every dataset updates.
- **Unknown labels are dropped, not guessed.** A `(source, raw_label)` pair not in
  the table returns `None`, and the calling loader drops that row. Junk subreddits
  silently disappear rather than corrupt the label space.

This is exactly why SWMH's `bipolar` and Low et al.'s `bipolarreddit` both land on
the identical `Condition.BIPOLAR`.

### Step 3: each loader translates its own raw data

- **`loaders/low_et_al.py`** reads the `subreddit` column, calls
  `harmonize_label("low_et_al", subreddit)`, drops unmapped rows.
- **`loaders/swmh.py`** does the same via `harmonize_label("swmh", label)` (or the
  filename stem when a file has no label column).
- **`loaders/daic_woz.py`** is the special case (see §4).

---

## 3. Axis 2 — Register (text) harmonization

Owned entirely by `src/data/text_normalization.py`, so the cleaning policy lives in
exactly one place and is applied uniformly.

DAIC-WoZ transcripts follow clinical transcription conventions: all lowercase, no
punctuation, spelled-out numbers, **retained disfluencies** (`um`, `i i think`),
non-speech events in **square** brackets (`[laughter]`, `[cough]`), and `xxx` for
unintelligible speech. Reddit text is mixed-case and punctuated.

The policy (`NormalizationPolicy` / `DEFAULT_POLICY`):

| Rule | Default | Why |
|---|---|---|
| Strip bracket tags `[...]` / `<...>` | **ON** | non-speech events aren't language |
| Remove `xxx` unintelligible tokens | **ON** | not real words |
| Collapse whitespace | **ON** | tidy up after removals |
| Truecase / re-punctuate | **OFF** | **keep the register gap on purpose** |

The crucial design decision: **the register gap is preserved, not erased.** Truecasing
is OFF, so normalized DAIC text stays lowercase and unpunctuated. That Reddit↔clinical
style mismatch *is the distribution shift the project studies* — scrubbing it would
delete the very phenomenon under investigation. A toggle exists so the gap can be
ablated later.

> **Documentation correction worth noting:** some project drafts describe DAIC
> non-speech events as being in *angle* brackets (`<laughter>`). The real AVEC2017
> files on disk use **square** brackets (`[laughter]`). The code strips **both**
> styles defensively (see the `DECISION` note in `text_normalization.py`), but prose
> that says "angle brackets" should be corrected to "square brackets."

---

## 4. The DAIC-WoZ special case (labels ≠ multi-class condition)

DAIC-WoZ is a **clinical depression** corpus, not a multi-class one, so it is
harmonized differently — and on purpose:

- Every DAIC row is hard-tagged `condition = "depression"`. This marks *which axis is
  being measured*, **not** a prediction target.
- The real clinical label (the PHQ-8 questionnaire result) rides in **separate
  additive columns**: `phq8_binary`, `phq8_score`, and `split` (train/dev/test).
- One row **per participant** (the whole interview concatenated), joined to the
  PHQ-8 labels from the AVEC2017 split CSVs.
- The **test** split has no PHQ-8 labels (AVEC2017 withheld them), so those rows get
  `phq8_binary = <NA>`.
- An `include_interviewer` flag (default `False`, exposed as `--include-interviewer`)
  keeps or drops the interviewer's ("Ellie's") turns, so the interviewer prompt-leakage
  confound can be controlled by ablation.

So DAIC's *proxy label* (Reddit-style multi-class) and its *clinical label* (PHQ-8)
are **two distinct, aligned signals on the depression axis** — not one label mapping
swallowing both.

---

## 5. One canonical output — `src/data/combine.py`

Every loader emits the same five columns, in this order:

| column | meaning |
|---|---|
| `text` | the post / concatenated participant turns |
| `condition` | harmonized condition (the multi-class axis) |
| `source` | `swmh`, `low_et_al`, or `daic_woz` |
| `author_id` | author / participant id (`unknown` if absent) |
| `date` | post date if available |

DAIC adds `phq8_binary`, `phq8_score`, `split` on top (additive; empty for Reddit
rows). `combine_sources()` concatenates all available loaders and de-duplicates on
`(text, source, author_id)`. Sources whose files are missing are reported and skipped,
not fatal.

---

## 6. End-to-end example

```
RAW (Low et al. CSV row):
  subreddit = "bipolarreddit"
  post      = "Manic episode last week, barely slept."

RAW (SWMH CSV row):
  label = "bipolar"
  text  = "cant tell if im manic again"

RAW (DAIC transcript, participant 320):
  [laughter] i i don't um  xxx  really sleep much

HARMONIZED (canonical rows):
  text="Manic episode last week, barely slept."  condition="bipolar"      source="low_et_al"  ...
  text="cant tell if im manic again"             condition="bipolar"      source="swmh"       ...
  text="i i don't um really sleep much"          condition="depression"   source="daic_woz"   phq8_binary=1 split="train"
```

Both Reddit rows now share `condition="bipolar"` despite their different raw labels;
the DAIC row keeps its clinical register (disfluencies retained, `[laughter]`/`xxx`
stripped) and carries PHQ-8 separately.

---

## 7. How to run / verify

From the repo root:

```bash
# sanity-check the whole harmonized load (per-source + per-condition counts)
./.venv/Scripts/python.exe -m src.data          # Windows
python -m src.data                               # Colab/Linux

# run the tests (synthetic fixtures, sub-second, no real data/network)
./.venv/Scripts/python.exe -m pytest -q
```

### Current verified state (2026-07-01)

```
=== rows per source ===
low_et_al    139898
daic_woz        188

=== rows per condition ===
depression         113764
eating_disorder     13447
schizophrenia        7680
bipolar              5195

Total rows: 140086
```

- **Low et al.**: present for 4 of the 6 POC conditions (depression, bipolar,
  schizophrenia, eating_disorder). `anxiety` and `suicidality` files are not on disk
  yet — their table entries exist, they just have no rows to harmonize today.
- **DAIC-WoZ**: 188 participants load (dev 34, train 107, test 47-unlabeled).
- **SWMH**: no files on disk yet; its loader raises and is skipped by the CLI. Its
  raw column-name assumptions are still unverified until real files arrive.
- Tests: **31/31 pass**.

---

## 8. Where each piece lives

```
src/data/
  schema.py              # Condition enum, canonical columns, validate_schema()
  label_map.py           # the ONE (source, raw_label) -> Condition table
  text_normalization.py  # the ONE text-cleaning policy (register harmonization)
  config.py              # DataPaths: injectable paths (no hardcoded globals)
  base.py                # DatasetLoader interface
  loaders/
    swmh.py              # SWMHLoader
    low_et_al.py         # LowEtAlLoader
    daic_woz.py          # DAICWoZLoader (per-participant; joins PHQ-8 labels)
  combine.py             # combine_sources(loaders) -> one concatenated frame
  __main__.py            # CLI: load everything and print sanity-check counts
```
