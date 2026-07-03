# Project Progress & Roadmap — resume-from-here reference

_Last updated: 2026-07-01. This is the single "where are we / what's next" document.
Read the TL;DR, then jump to §5 "Next action when you resume."_

---

## 0. TL;DR (30-second version)

- **Done & verified:** the data-ingestion + harmonization layer (`src/data/`). Loads
  all three datasets, harmonizes labels, emits one canonical table. 31/31 tests pass;
  `python -m src.data` loads 140,086 rows.
- **Done & verified (2026-07-03):** the MentalBERT baseline classifier
  (`src/modeling/`) code is verified locally — light deps (`scikit-learn`, `numpy`)
  installed in `.venv`, `pytest -q` → **48 passed, 2 skipped** (the 2 skips are the
  torch-only dataset/predict tests). All files are committed and pushed to
  `SoshanW/mental-health-screening-fyp` (`main` @ `c8e79de`).
- **Immediately next:** run the Colab notebook (§5 Step C) to actually fine-tune
  MentalBERT and produce the checkpoint + held-out predictions + per-condition
  confusion matrix. **No training run has happened yet.**
- **Big picture:** this baseline is rung 1 of a longer research pipeline (noise
  modeling → calibrated abstention → shift evaluation). See §7.

---

## 1. What this project is

Multi-condition mental-health screening from text, studying **proxy-label noise**
(Reddit "subreddit = diagnosis" is a noisy label) and **calibrated abstention** (the
model should refuse to answer when unsure), evaluated under a **Reddit → clinical
distribution shift**. Downstream model: **MentalBERT**. Full technical framing lives
in `Docs/technical-contribution-in-depth.md`.

Three datasets:
- **Low et al.** (Reddit, Zenodo 3941387) — per-post, subreddit = proxy label.
- **SWMH** (Reddit) — per-post, subreddit = proxy label. *(not on disk yet)*
- **DAIC-WoZ** (clinical interviews) — per-participant, PHQ-8 depression labels; held
  out for the shift evaluation.

---

## 2. What was already built before this session (the data layer)

`src/data/` — the ingestion + harmonization layer. One responsibility per file:

| File | Role |
|---|---|
| `schema.py` | `Condition` enum, `POC_CONDITIONS` (6-class subset), canonical column list, `validate_schema()` |
| `label_map.py` | the ONE `(source, raw_label) -> Condition` table + `harmonize_label()` |
| `text_normalization.py` | the ONE text-cleaning policy (strips `[laughter]`/`xxx`, preserves the register gap) |
| `config.py` | `DataPaths` — injectable paths, no hardcoded globals |
| `base.py` | `DatasetLoader` Protocol (`source` + `load()`) |
| `loaders/{swmh,low_et_al,daic_woz}.py` | one loader per dataset, each emits the canonical schema |
| `combine.py` | `combine_sources()` — concatenate + dedup |
| `__main__.py` | CLI: `python -m src.data` prints sanity-check counts |

Harmonization is documented in depth in **`Docs/data-harmonization.md`**.

---

## 3. What we did THIS session — step by step (including the tedious bits)

1. **Confirmed the datasets you added are seen by the loaders.** Ran
   `python -m src.data`. Result: DAIC-WoZ jumped from 1 participant to **188**;
   combined corpus = **140,086 rows** (low_et_al 139,898 + daic_woz 188). Per
   condition: depression 113,764 / eating_disorder 13,447 / schizophrenia 7,680 /
   bipolar 5,195.
2. **Ran the existing test suite** — `python -m pytest -q` → **31 passed**.
3. **Updated project memory** with the new on-disk state and the fact that DAIC now
   loads fully; recorded that SWMH is still absent.
4. **Answered the harmonization question** and confirmed label harmonization was
   *already* implemented (it's `label_map.py`), plus flagged two documentation
   corrections: (a) PHQ-8 is a *separate aligned* signal, not folded into the Reddit
   label mapping; (b) DAIC non-speech tags are in **square** brackets `[laughter]`,
   not angle brackets — the code already handles both.
5. **Wrote `Docs/data-harmonization.md`** — a focused deep-dive on how harmonization
   works (label axis + register axis + the DAIC special case).
6. **Decided scope for the next increment:** the naive MentalBERT baseline classifier
   only (rung 1 of the eval grid). Noise modeling / calibration / abstention are
   explicitly deferred.
7. **Decided training happens on Google Colab** (not the local RTX 4060), with data
   and model outputs on **Google Drive**, driven by a ready-to-run notebook. Saved
   this to memory so future sessions keep the code Colab-portable.
8. **Built the modeling layer** `src/modeling/` (all new files below).
9. **Wrote 5 new test files** for the modeling layer.
10. **Added dependencies** to `requirements.txt` and `pyproject.toml`; added `Models/`
    to `.gitignore`.
11. **Wrote the Colab notebook** `notebooks/train_colab.ipynb`.
12. **Did NOT install deps or run any modeling code yet** — that's the resume point.

### Files created this session

```
Docs/data-harmonization.md            # harmonization deep-dive
Docs/progress-and-roadmap.md          # this file
src/modeling/__init__.py              # light public surface (torch-free import)
src/modeling/config.py                # SplitConfig, ModelConfig, TrainConfig, ArtifactPaths
src/modeling/labels.py                # CONDITION_TO_ID / ID_TO_CONDITION, encode/decode
src/modeling/splits.py                # prepare_classification_frame, author_grouped_split, build_poc_splits
src/modeling/dataset.py               # Tokenizer Protocol + TextClassificationDataset (torch)
src/modeling/hf_model.py              # load_tokenizer / load_model factory (transformers)
src/modeling/metrics.py               # compute_metrics (accuracy + macro-F1)
src/modeling/train.py                 # build_trainer + CLI: python -m src.modeling.train
src/modeling/predict.py               # predict_softmax + CLI: python -m src.modeling.predict
tests/test_modeling_labels.py         # id mapping (no torch)
tests/test_modeling_splits.py         # filtering + author-grouped split (no torch)
tests/test_modeling_dataset.py        # dataset wrapping (importorskip torch, fake tokenizer)
tests/test_modeling_metrics.py        # metrics (importorskip sklearn)
tests/test_modeling_predict.py        # predict_softmax + output frame (importorskip torch, fake model)
notebooks/train_colab.ipynb           # Colab runner (mount Drive → clone → install → train → predict)
```

### Files edited this session

```
requirements.txt   # + torch, transformers, accelerate, scikit-learn, numpy (with CUDA-wheel note)
pyproject.toml     # + [project.optional-dependencies].modeling
.gitignore         # + Models/
```

---

## 4. How the new baseline layer is designed (so it makes sense later)

**Design rule:** everything is path-injected and CLI-driven, so the *same* code runs
locally (Windows `.venv`) and on Colab (Linux) with only different `--data-root` /
`--artifacts-root` args. Light modules (`config`, `labels`, `splits`) import without
torch; heavy modules (`dataset`, `hf_model`, `metrics`, `train`, `predict`) pull in
torch/transformers only when used.

**Pipeline flow:**
```
src.data.combine_sources(...)                      # the 140k canonical table
        │
        ▼
prepare_classification_frame()                     # keep POC Reddit rows; DROP daic_woz + non-POC
        │
        ▼
author_grouped_split()                             # train/val/test, NO author across splits (no leakage)
        │                                          # persisted to Models/splits/{train,val,test}.csv
        ▼
TextClassificationDataset + HF Trainer (train.py)  # fine-tune MentalBERT, save Models/checkpoints/latest
        │
        ▼
predict.py                                         # softmax on held-out test → Models/predictions/test_predictions.csv
```

**Why DAIC is excluded from this classifier:** its `condition` is always
`"depression"` (marks the measured axis, not a multi-class label), and it's reserved
for the later shift evaluation. Folding it in would leak/mislabel.

**Outputs land in `Models/`** (gitignored; on Drive when using Colab):
`checkpoints/latest/`, `splits/{train,val,test}.csv`, `predictions/test_predictions.csv`.

---

## 5. ▶ NEXT ACTION when you resume (start here)

**Step A — verify the code (local, fast). ✅ DONE 2026-07-03.** Installed the light
test deps and ran the suite:
```bash
./.venv/Scripts/python.exe -m pip install "scikit-learn>=1.4" "numpy>=1.24"
./.venv/Scripts/python.exe -m pytest -q
```
Result: **48 passed, 2 skipped** (the dataset/predict tests skip — no torch locally),
2 expected warnings (only bipolar/depression present on disk). Green; ready for Colab.

**Step B — (optional) full local check.** Only if you want to run every test locally:
`pip install -r requirements.txt` (⚠️ ~2 GB torch download on Windows — see the
CUDA-wheel note in `requirements.txt`), then `pytest -q`.

**Step C — train on Colab.**
1. Upload your local `Datasets/` to Google Drive at
   `MyDrive/mental-health-fyp/Datasets`.
2. Open `notebooks/train_colab.ipynb` in Colab, set runtime to **GPU**.
3. Run cells top-to-bottom. Cell 6 is a seconds-long smoke test (tiny model); cell 7
   is the real MentalBERT fine-tune; cell 8 produces predictions + metrics.
4. Cell 9 prints the per-condition confusion matrix + precision/recall/F1 report and
   writes `Models/predictions/confusion_matrix.csv`.
5. Results persist on Drive under `Models/`.

> **Note:** already handled — all code + the notebook are committed and pushed to
> `SoshanW/mental-health-screening-fyp` (`main` @ `c8e79de`), so the Colab `git clone`
> picks them up. Nothing further to push before training.

---

## 6. Known open items / caveats

- **Code is verified; training is not.** Local test suite is green (§5 Step A), but no
  MentalBERT fine-tune has run yet — the Colab run (Step C) is the outstanding item.
- **SWMH data is still absent.** Only Low et al. Reddit data feeds the classifier
  today, and only 4 of 6 POC conditions have rows (**anxiety** and **suicidality**
  are missing — the split code warns but does not fail). Adding SWMH / the missing
  subreddits later needs no code change.
- **`mental/mental-bert-base-uncased` availability** on the HF Hub should be
  confirmed on first run; `--model-name` is the escape hatch if the id has moved.
- **Class imbalance** (~20:1 depression-heavy) is real; the baseline reports macro-F1
  precisely because of this. Handling the imbalance is a later step, not the baseline.

---

## 7. Future roadmap — what's left and what each step gives

This baseline is **rung 1**. The full contribution (per
`Docs/technical-contribution-in-depth.md`) is three coupled pieces:

| Phase | What it is | What it gives |
|---|---|---|
| **Now: Baseline (C0)** | Naive MentalBERT multi-class fine-tune on proxy labels; raw softmax confidence | The reference point every later method must beat; the softmax outputs that feed noise estimation |
| **C1 — Condition-dependent noise model** | Estimate a per-condition label-noise transition matrix `T` via Confident Learning (`cleanlab`), with clinically-seeded priors; state identifiability conditions | A principled estimate of *how unreliable each condition's proxy label is* (e.g. bipolar noisier than depression) |
| **C2 — Noise-coupled calibrated abstention** | Temperature-scale for calibration; make the per-condition abstention threshold a function of C1's noise estimate | A system that abstains most on the conditions where labels were least trustworthy — lower selective risk at matched coverage |
| **C3 — Identifiability-under-shift eval** | Synthetic recovery test + per-condition degradation curves + Reddit→clinical (DAIC) shift evaluation | Evidence for *where and why* the method breaks down, matched against C1's theory |
| **Optional — symptom-alignment signal** | Per-post DSM-5/ICD-11 symptom-match score as an extra noise-model feature | Finer, per-post label-reliability estimate (ablation dimension, droppable) |

**End state:** a trustworthy multi-condition screener that (a) knows its labels are
noisy and models that noise per condition, (b) translates label-uncertainty into
honest prediction-time abstention, and (c) has a characterized, theory-backed failure
boundary under the Reddit→clinical shift. Evaluation is per-condition AND pooled:
accuracy/macro-F1, ECE (calibration), risk–coverage/AURC, and the *degradation* of
each under shift.

**Concrete next coding increments after the baseline is verified & trained:**
1. Save/inspect the baseline's held-out softmax + a confusion matrix per condition.
2. Add a `src/noise/` layer: out-of-fold prediction generation + `cleanlab` transition
   matrix estimation (C1).
3. Add calibration + selective-prediction utilities (C2), initially with a uniform
   threshold, then noise-coupled.
4. Wire the DAIC held-out set into a shift-evaluation script (C3).
```
