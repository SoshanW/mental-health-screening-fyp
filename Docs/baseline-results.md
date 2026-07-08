# Baseline results — record of run

_Factual record of the naive MentalBERT multi-class baseline (rung 1 / C0). For a
plain-English explanation of every metric below, see
[`baseline-results-explained.md`](baseline-results-explained.md)._

- **Date:** 2026-07-07
- **Environment:** Google Colab GPU (T4), data + outputs on Google Drive
- **Status:** Baseline of record. Trained & evaluated; artifacts persisted.

---

## 1. Configuration

| Item | Value |
|---|---|
| Base model | `mental/mental-bert-base-uncased` (gated; HF token + accepted terms) |
| Task | Multi-class single-label classification (proxy labels = subreddit) |
| Classes present | `bipolar`, `depression`, `eating_disorder`, `schizophrenia` (4 of 6 POC; `anxiety`, `suicidality` absent from disk) |
| Sources used | Low et al. Reddit only (SWMH absent; DAIC-WoZ excluded by design) |
| Epochs | 3 |
| Batch size | 16 (per device) |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Max sequence length | 256 tokens |
| Mixed precision | fp16 (on) |
| Seed | 42 |
| Checkpointing | `--save-steps 500`, `save_total_limit=2`, `--resume` enabled |
| Split | Author-grouped 0.80 / 0.10 / 0.10 (train/val/test), seed 42 |
| Total optimizer steps | 20,982 (3 × 6,994) |

**Split sizes:** train = 111,892 · val = 13,967 · test = 14,039.

> Note: because step-based saving was used, best-model-at-end is disabled — the saved
> `checkpoints/latest` is the **final-step** model, not the best-validation epoch.

---

## 2. Headline metrics (held-out test set, 14,039 rows)

| Metric | Value |
|---|---|
| Accuracy | **0.9605** |
| **Macro-F1** (headline) | **0.8782** |
| Macro precision | 0.9043 |
| Macro recall | 0.8552 |
| Weighted-F1 | 0.9596 |

Validation macro-F1 at the epoch-1 checkpoint was 0.8664, consistent with the final
result (see §5).

---

## 3. Per-condition metrics (test set)

| Condition | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| depression | 0.9715 | 0.9860 | 0.9787 | 11,351 |
| eating_disorder | 0.9515 | 0.9388 | 0.9451 | 1,421 |
| schizophrenia | 0.8968 | 0.7850 | 0.8372 | 786 |
| bipolar | 0.7972 | 0.7110 | 0.7516 | 481 |
| **macro avg** | 0.9043 | 0.8552 | **0.8782** | 14,039 |
| **weighted avg** | 0.9593 | 0.9605 | 0.9596 | 14,039 |

---

## 4. Confusion matrix (test set)

Rows = true condition, columns = predicted condition. Diagonal = correct.

| true ↓ / pred → | bipolar | depression | eating_disorder | schizophrenia | row total |
|---|---|---|---|---|---|
| **bipolar** | 342 | 116 | 4 | 19 | 481 |
| **depression** | 57 | 11,192 | 55 | 47 | 11,351 |
| **eating_disorder** | 3 | 79 | 1,334 | 5 | 1,421 |
| **schizophrenia** | 27 | 133 | 9 | 617 | 786 |

**Dominant error mode:** minority classes misclassified as `depression`
(bipolar→depression 116, schizophrenia→depression 133, eating_disorder→depression 79).
Depression is both the majority class and the primary confusion sink.

---

## 5. Comparison: 1-epoch fast run vs. 3-epoch full run

| Run | Artifacts | Accuracy | Macro-F1 | bipolar F1 | depression F1 | eating_dis F1 | schizo F1 |
|---|---|---|---|---|---|---|---|
| Fast (1 epoch, batch 32) | `Models_fast/` | 0.9611 | 0.8796 | 0.7492 | 0.9789 | 0.9440 | 0.8463 |
| **Full (3 epochs, batch 16)** | `Models/` | 0.9605 | **0.8782** | 0.7516 | 0.9787 | 0.9451 | 0.8372 |

**Finding:** additional epochs produced **no improvement** (macro-F1 0.878 vs 0.880 is
within run-to-run noise). Interpretation: performance is bounded by proxy-label noise,
not model capacity or training duration — the motivation for the C1 noise-modeling phase.

---

## 6. Artifacts (Google Drive)

Canonical run — `MyDrive/mental-health-fyp/Models/`:
- `checkpoints/latest/` — final fine-tuned MentalBERT (weights + tokenizer)
- `splits/{train,val,test}.csv` — exact author-grouped split used
- `predictions/test_predictions.csv` — per-row softmax probabilities + argmax + true label
- `predictions/confusion_matrix.csv` — the §4 matrix

Fast run — `MyDrive/mental-health-fyp/Models_fast/` (1-epoch, same layout).

---

## 7. Caveats / limitations

- **Proxy labels, not diagnoses.** "Correct" = matches the source subreddit, not a
  clinical diagnosis. Distinctive per-subreddit vocabulary inflates scores relative to
  true diagnostic difficulty.
- **In-distribution only.** Both train and test are Reddit (Low et al.). The
  Reddit→clinical (DAIC-WoZ) shift evaluation is a later phase and is expected to be
  substantially harder.
- **4 of 6 classes.** `anxiety` and `suicidality` have no data on disk; the split code
  warns but does not fail. No code change needed to add them later.
- **Class imbalance ~24:1** (depression vs bipolar) — the reason macro-F1 is the
  headline metric.
- **Final-step, not best-epoch, checkpoint** (a consequence of step-based saving).

---

## 8. Next step

C1 — condition-dependent noise model (`src/noise/`): out-of-fold prediction generation
over the full labeled set + `cleanlab` transition-matrix estimation, seeded with clinical
priors. Note: C1 requires **out-of-fold** predictions (K-fold cross-fit), not the
held-out test predictions recorded here. See `Docs/progress-and-roadmap.md` §7.
