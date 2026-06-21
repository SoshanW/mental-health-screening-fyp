# Technical Contribution — In Depth

_A technical deep-dive into the methodological contributions of the project. Audience: readers comfortable with machine learning (supervisor, examiner, technical reviewer). For the plain-language version, see the companion "Everything In Depth, In Simple Terms" document._

> **Status:** Direction reviewed and confirmed by supervisor; proceeding.

---

## 0. The one-sentence technical claim

We model proxy-label noise in multi-condition mental-health screening as a **condition-dependent class-conditional noise-transition process**, prove the **identifiability conditions** under which that process is recoverable, and **couple the estimated per-condition noise directly into a calibrated selective-prediction (abstention) rule** — so that label-uncertainty propagates into prediction-time caution — then evaluate the whole pipeline's **calibration and risk–coverage degradation under a Reddit → clinical distribution shift**.

The three contributions below are a single mechanism, not three modules. Their through-line: _uncertainty about supervision should flow into honest uncertainty about predictions._

---

## 1. Contribution 1 — Condition-dependent noise-transition model, with identifiability theory

### 1.1 Setup and notation

Let the latent true condition be `y ∈ {1, …, K}` and the observed proxy label (subreddit membership) be `ỹ ∈ {1, …, K}`. The standard class-conditional noise model assumes a transition matrix `T` where

```
T[i, j] = P(ỹ = j | y = i)
```

Confident Learning (Northcutt et al., 2021) estimates `T` (and the joint `Q[i,j] = P(y=i, ỹ=j)`) from the model's out-of-sample predicted probabilities, under the assumption that noise is class-conditional (depends on the true class, not the features directly).

### 1.2 What's new here

Existing mental-health work that addresses label noise at all (e.g. SDCNL — Haque, Reddi & Giallanza, 2021, ICANN, DOI 10.1007/978-3-030-86383-8_35, arXiv:2102.09427 — a binary suicide-vs-depression task with deliberately noise-distribution-agnostic unsupervised label correction) treats noise as **undifferentiated** — one global cleaning process. The contribution is to treat `T` as **condition-dependent and clinically seeded**: the prior on each row of `T` encodes domain knowledge that self-identification reliability differs by disorder. Concretely:

- Rows for depression / anxiety get a prior concentrated near the identity (low off-diagonal mass — reliable self-report).
- Rows for bipolar / schizophrenia get a flatter prior (higher off-diagonal mass — noisier self-report, more confusion with adjacent conditions).

This prior is not hand-waved: where clinical data exists (DAIC-WoZ, depression only), the corresponding row of `T` is _estimated/validated_ against it rather than assumed. For the conditions without clinical anchoring, the prior is treated as a hyperparameter swept over a defensible range, with results reported as a sensitivity analysis (this is the honest scoping — see §1.4).

### 1.3 The identifiability result (the theory-first upgrade)

The key intellectual upgrade over a purely empirical project: rather than only _showing_ that `T` is sometimes recoverable, we _state the conditions under which it is_, and derive why the rare conditions violate them.

Class-conditional `T` is identifiable only under additional structure. The literature gives three standard routes:

- **Anchor points** (Liu & Tao; Patrini et al.): for each class `i`, there exists at least one instance with `P(y=i | x) ≈ 1`. The cleaner and more numerous the anchors, the better `T` is estimated.
- **Irreducibility / separability** conditions (Scott et al.) on the class-conditional distributions.
- **Clusterability** (Zhu et al., "clusterability" / HOC) — nearby instances share a true label, enabling `T` estimation from local consensus without anchors.

The contribution is to connect this to **prevalence**: low-prevalence conditions (bipolar ≈ 5.8k posts vs depression ≈ 117k in the verified data — a ~20:1 ratio) have _fewer and lower-confidence anchor points_, so their rows of `T` are estimated with higher variance and, past a threshold, become effectively unidentifiable. We state this as a predicted breaking point as a function of per-condition anchor availability, and §3 tests whether the empirical breakdown matches the prediction.

This is the move that lifts the work from "applied engineering" to "applied research with a theoretical spine": a derivation that _predicts_ where the method fails, then an experiment that confirms it.

### 1.4 Honest scope

`T`'s depression row is the only one with genuine clinical validation (DAIC-WoZ is PHQ-8, depression-only, and its construct validity is itself contested — Patapati et al., ICMI 2025). Every other row is clinically _unvalidated_ and handled by sensitivity analysis. This is stated as a limitation, not hidden — and it is precisely what motivates C3.

### 1.5 Optional extension: a symptom-alignment auxiliary signal

A practitioner suggestion received during planning ("run the Reddit threads so they align with clinical/textbook symptoms; add RLHF to purely validate and filter the dataset") was evaluated and partly corrected before incorporation. RLHF (Reinforcement Learning from Human Feedback) is, by definition, a _generative-model fine-tuning_ technique — it trains a reward model from human pairwise preference comparisons between a model's own outputs, then optimises the generative policy against that reward, typically via PPO (Christiano et al., 2017; Ouyang et al., 2022). It has no defined role in static dataset validation or filtering, and "add RLHF" as literally stated is not implementable here.

The underlying instinct, however, is sound: incorporate clinical symptom criteria as an additional check on label trustworthiness. Implemented correctly:

```
align(x_i, ỹ_i) ∈ [0, 1]
```

a per-post score measuring how strongly post `x_i`'s content matches DSM-5/ICD-11 symptom criteria for its proxy label `ỹ_i`, computed via a curated clinical symptom lexicon (cheapest, most transparent; preferred default) or, if needed, a validated symptom-extraction tool. This is **not** used as a row filter (`align` below threshold ⇒ drop) — doing so would inject a new, unvalidated selection bias (over-selecting clinically literate writers) and would directly contradict the project's premise that no single proxy signal should be trusted as ground truth without modelling its own reliability.

Instead, `align(x_i, ỹ_i)` is added as one additional input feature to the existing per-post noise posterior in C1 — refining the noise estimate from per-condition (`T[i,:]`) to a finer per-post granularity. It is evaluated identically to every other component: as one further ablation dimension (§4.2), with its own precision/recall reported against a small hand-labelled validation sample before being trusted as a feature, exactly as any other proxy in this project must be validated rather than assumed reliable.

This is scoped as a small, optional enrichment of C1, time-boxed, and dropped without affecting the core deliverable (C1–C3) if it threatens the project timeline.

---

## 2. Contribution 2 — Noise-coupled calibrated abstention (the novelty multiplier)

### 2.1 The two standard pieces (adapted, not novel)

- **Calibration:** temperature scaling (Guo et al., 2017) on a held-out split, evaluated by Expected Calibration Error (ECE) and reliability diagrams.
- **Selective prediction / conformal abstention:** a selective classifier (Geifman & El-Yaniv, 2017) or split-conformal predictor (Angelopoulos & Bates, 2023) that abstains when the non-conformity score exceeds a threshold calibrated to a target coverage `1 − α`, with a finite-sample coverage guarantee.

On their own, each of these is off-the-shelf.

### 2.2 The coupling (this is the contribution)

The novelty is that the **per-condition abstention threshold is a function of that condition's estimated noise level from C1**. Write the estimated noise level for condition `i` as a scalar summary of its `T` row, e.g.

```
η_i = 1 − T̂[i, i]      (estimated off-diagonal mass = how unreliable condition i's proxy is)
```

The selective/conformal threshold for condition `i`, `τ_i`, is then made monotone increasing in `η_i`: noisier conditions get a _stricter_ bar to clear before the system commits, so

```
τ_i = g(α, η_i),   with ∂τ_i/∂η_i > 0
```

Two concrete realisations to evaluate:

1. **Class-conditional conformal:** run split-conformal _per condition_ but allocate the error budget `α_i` as a function of `η_i` (tighter coverage demanded where labels are noisier), keeping overall coverage controlled.
2. **Noise-scaled selective threshold:** a single selective classifier whose per-class score threshold is shifted by `η_i`.

The effect is a single mechanism in which **label-uncertainty (C1) propagates into prediction-time caution (C2)**: the system abstains most exactly on the conditions where its supervision was least trustworthy. This is the precise combination a literature search did not find — existing work does calibrated abstention with a _uniform_ or purely confidence-driven threshold, decoupled from any model of label reliability.

### 2.3 Why this is more than aesthetics

It yields a testable prediction: noise-coupled abstention should achieve **lower selective risk at matched coverage** than uniform abstention, and the gap should be **largest on the noisy conditions**. That is a clean, falsifiable experimental claim (§4), not a design flourish.

---

## 3. Contribution 3 — Empirical identifiability-under-shift analysis

C3 is the experimental complement to C1's theory. Procedure:

1. **Synthetic recovery test (white box):** inject a _known_ condition-dependent `T` into clean-ish labels, then check the estimator recovers it as anchor availability is varied. This is also a unit test (it appears in the test suite), so it does double duty as software verification and research result.
2. **Per-condition degradation curve:** measure estimation error of `T̂[i, :]` as a function of (a) number of posts for condition `i` and (b) anchor-point quality, sweeping from data-rich (depression) to data-poor (bipolar/schizophrenia).
3. **Theory-vs-empirics check:** test whether the empirical breakdown point matches the threshold predicted in §1.3.
4. **Under shift:** repeat the calibration/coverage evaluation across the Reddit → clinical (DAIC-WoZ, depression) gap, reporting degradation rather than a single in-distribution number.

A characterised breaking point that _confirms the theoretical prediction_ is the strongest outcome; a mismatch is also reportable (it would mean the identifiability story is more complex than the standard conditions suggest — itself interesting).

---

## 4. Evaluation: baselines, ablations, metrics

### 4.1 Baselines (strongest-available, not strawmen)

1. Naïve fine-tuned MentalBERT on clean-assumed proxy labels, raw softmax confidence (field default).
2. Generic noise correction: SDCNL-style unsupervised label correction re-implemented and run on the _same_ data (fairer than citing cross-dataset numbers).
3. Post-hoc calibration only (temperature scaling, no noise modelling).
4. Uniform-threshold abstention (selective/conformal with one global threshold) — the direct control isolating the value of noise-coupling.
5. Class-weighting only (handles the 20:1 imbalance, ignores label noise) — separates "imbalance fix" from "noise fix".

### 4.2 Ablation grid

`condition-dependent noise {on, off}` × `calibration {on, off}` × `abstention {none, uniform, noise-coupled}`.

**If the optional symptom-alignment signal (§1.5) is implemented**, it is added as a fourth, independent ablation dimension {with, without} applied only to the condition-dependent-noise arm — i.e. comparing condition-dependent noise modelling alone vs condition-dependent noise modelling + symptom-alignment feature — kept cleanly isolated from the C2/C3 results rather than entangled with the headline comparison.

Reporting the full grid is what licenses precise claims ("noise-coupling reduced selective risk by X at matched coverage, holding calibration fixed") instead of "our system is better".

### 4.3 Metrics (always per-condition AND pooled)

Accuracy / macro-F1; ECE and per-condition ECE; risk–coverage curves and AURC; selective risk at fixed coverage; empirical coverage vs target `1 − α`; and the **degradation** of each under Reddit → clinical shift. All with confidence intervals over repeated seeds/splits.

### 4.4 What counts as a convincing result

- Noise-coupled abstention < uniform abstention in selective risk at matched coverage, most clearly on noisy conditions.
- Empirical identifiability breakpoint (C3) matches C1's theoretical prediction.
- Whole-system calibration degrades more gracefully under clinical shift than every baseline.

Any one is a strong result; a clean negative on the first two is still markable because the experiments are designed to be decisive either way.

---

## 5. What is novel vs adapted (for honest attribution)

| Component                                                                                        | Status                                                                                                    |
| ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| Class-conditional noise estimation (Confident Learning, `cleanlab`)                              | Adapted                                                                                                   |
| Temperature scaling, ECE (Guo et al.)                                                            | Adapted                                                                                                   |
| Split-conformal / selective prediction (Angelopoulos–Bates; Geifman–El-Yaniv; MAPIE/TorchCP)     | Adapted                                                                                                   |
| MentalBERT, SHAP                                                                                 | Adapted                                                                                                   |
| **Condition-dependent, clinically-seeded noise prior**                                           | **Novel**                                                                                                 |
| **Identifiability result tying recoverability to per-condition prevalence/anchors**              | **Novel**                                                                                                 |
| **Coupling of estimated per-condition noise into the abstention threshold**                      | **Novel (central)**                                                                                       |
| **Mechanism-level shift evaluation (noise-coupled vs uniform abstention under Reddit→clinical)** | **Novel**                                                                                                 |
| Symptom-alignment auxiliary signal (§1.5)                                                        | Optional, novel-as-applied-here; lexicon construction is adapted from DSM-5/ICD-11 criteria, not invented |

---

## 6. Compute & feasibility

Everything is fine-tuning-scale (MentalBERT, ≤256 tokens) on modest data (≈150k posts total across conditions), plus CPU-side wrapper methods (CL, conformal, calibration). No pretraining, no multi-GPU. Free Colab/Kaggle suffices. The contribution is methodological, not scale-driven.

---

## 7. The single biggest technical risk

The identifiability story (C1 §1.3) is the most ambitious piece and the one most likely to stall if the formal conditions prove hard to state cleanly for the _condition-dependent_ case. Mitigation: the empirical degradation curve (C3) stands on its own even if the theorem is only stated at the level of the existing anchor-point/clusterability conditions rather than fully re-derived. The project is structured so that a weaker theoretical result still leaves a complete, defensible empirical contribution.
