# C1 -- Condition-dependent noise model: design (`src/noise/`)

_Design doc for the C1 noise layer, written before any implementation. Scope of this
document: `prior.py` (the clinically-seeded prior) and `transition.py` (the estimator that
combines that prior with Confident Learning's confident joint). `oof.py` -- the K-fold
out-of-fold prediction generator that feeds Confident Learning -- is GPU-expensive and is
**deliberately deferred**; it is specified here only at the interface boundary so the two
CPU modules can be built and tested against synthetic data without it._

_Reviewers: this maps to RO6 / RO8 / RO12 and research questions RQ1 and RQ2. Section 3
(identifiability) is the intellectual core and corresponds to the project's identifiability
hypothesis (referred to in planning as "H4"): that a condition-dependent transition matrix
is recoverable only for conditions with sufficient support / anchor availability, and is
expected to fail for the rare, high-noise conditions (bipolar, schizophrenia)._

Companion docs: `Docs/technical-contribution-in-depth.md` (§1 sets up `T`, §2.2 sets up the
C2 threshold function `g`), `Docs/Chapter_01_Introduction_v2.md` (RQ1-RQ4), and
`Docs/baseline-results.md` (the C0 baseline whose out-of-fold softmax C1 will consume).

---

## 1. Purpose and non-goals

**Purpose.** Estimate a condition-dependent class-conditional noise-transition matrix `T`,
where `T[i, j] = P(ỹ = j | y = i)` (`y` = latent true condition, `ỹ` = observed proxy label
= subreddit). The estimate must (a) use Confident Learning's data-driven confident joint
where the data can support it, and (b) fall back to a clinically-seeded prior where it
cannot -- transparently, and per condition. The scalar noise summary this produces,
`η_i = 1 - T̂[i, i]`, is the exact quantity C2's abstention threshold `g(α, η_i)` consumes
(technical-contribution §2.2), so C1's output contract is fixed by C2's input contract.

**Non-goals (this step).**
- No `oof.py`, no GPU code, no `torch` anywhere in `prior.py` / `transition.py`.
- No cleanlab call on real predictions (that needs out-of-fold probabilities, which need
  `oof.py`). `transition.py`'s core estimator takes an already-computed confident-joint
  count matrix, so it is fully testable on synthetic count matrices alone.
- No calibration / abstention logic (that is C2).

---

## 2. Notation and the estimation target

- `K` = number of conditions = `len(POC_CONDITIONS)` = 6. The axis order is fixed by
  `POC_CONDITIONS` (see `src/data/schema.py`), i.e. the same id order the classifier head
  already uses (`src/modeling/labels.py`):

  | id | condition | has data on disk? |
  |----|-----------|-------------------|
  | 0 | depression | yes (majority) |
  | 1 | anxiety | no |
  | 2 | bipolar | yes (rare, ~20:1) |
  | 3 | suicidality | no |
  | 4 | schizophrenia | yes (rare) |
  | 5 | eating_disorder | yes |

  `T` is kept `6 x 6` even though only four conditions currently have data. Rows for
  absent conditions are pure prior; keeping full `K` means adding anxiety/suicidality data
  later needs no reshape. Rows = true condition, columns = observed proxy.

- Confident Learning (Northcutt, Jiang and Chuang, 2021) produces the **confident joint**
  `C`, a `K x K` matrix of counts: `C[i, j]` is the number of examples confidently
  belonging to true class `i` but carrying proxy label `j`, calibrated so row sums track the
  observed noisy-class counts. Write the per-condition **confident support**
  `n_i = Σ_j C[i, j]` -- how many examples the data could confidently attribute to true
  class `i`. `n_i` is small exactly for the rare conditions, and that is the whole problem.

- The data-only estimate cleanlab would report is the row-normalized confident joint,
  `T̂_cl[i, :] = C[i, :] / n_i`. Our estimator replaces this with a prior-aware version.

---

## 3. Identifiability (the core section, RQ2 / "H4")

### 3.1 The problem: plain `T` is not identifiable from noisy labels alone

Liu, Cheng and Zhang (2023) show that a class-conditional transition matrix is **generally
not identifiable from noisy labels alone** without additional structure. The intuition for
this project: from noisy data we can only observe the noisy posterior `P(ỹ | x)`, which
decomposes as

```
P(ỹ = j | x) = Σ_i T[i, j] · P(y = i | x).
```

The clean posterior `P(y | x)` is unobserved, so infinitely many `(T, P(y | x))` pairs
reproduce the same observed `P(ỹ | x)`. The likelihood is **flat** along those directions:
data cannot distinguish them. The standard structural assumptions that restore
identifiability are:

- **Anchor points** (Liu and Tao; Patrini et al., 2017; questioned by Xia et al., 2019):
  for each class `i`, at least one instance with `P(y = i | x) ≈ 1`, which pins down a row
  (or column) of `T`. More and cleaner anchors -> lower-variance `T̂`.
- **Irreducibility / mutual-contamination separability** (Scott et al.).
- **Clusterability / high-order consensus** (Zhu et al., HOC): nearby instances share a
  true label, letting `T` be recovered from local agreement without explicit anchors.

All three degrade with **prevalence**: a condition with few examples has few, lower-
confidence anchors and weaker local consensus. This is the link the project makes explicit
(technical-contribution §1.3): identifiability is not a fixed property of the method, it is
a per-condition property that tracks `n_i`.

### 3.2 What extra structure the clinical prior supplies -- stated precisely

The clinically-seeded prior is the "additional structure" Liu, Cheng and Zhang require, but
it is important to be exact about *what kind*:

- It supplies an **informative prior on each row of `T`**, `m_i = (m_{i1}, ..., m_{iK})`, a
  probability vector encoding expected self-identification reliability and the expected
  direction of confusion for condition `i` (§5). This makes the MAP estimate of `T`
  **unique and stable even where the likelihood is flat**: the non-identified directions of
  the likelihood are resolved by the prior instead of left arbitrary.

- # DECISION: we do NOT claim the prior makes a non-identifiable parameter identifiable in
  the frequentist sense. A prior cannot add information the data lacks; it can only
  determine which of the likelihood-equivalent solutions we report. The honest, precise
  claim is therefore two-regime:
  1. **Data-identified regime** (large `n_i`, e.g. depression): the likelihood is
     informative, dominates the prior, and `T̂[i, :]` is a genuine estimate of `T[i, :]`.
     The prior only regularizes finite-sample variance and washes out asymptotically.
  2. **Prior-dominated regime** (small `n_i`, e.g. bipolar, schizophrenia): the likelihood
     is near-flat, the prior determines `T̂[i, :]`, and the output is effectively the
     clinical assumption, not an empirical finding. This row must be reported as a
     **sensitivity analysis over the prior**, not as identification (technical-contribution
     §1.4).

The estimator (§4) is designed so the transition between these regimes is smooth,
per-condition, and controlled by a single interpretable strength parameter -- which is
precisely what makes the boundary empirically mappable in §6.

### 3.3 Where it is still expected to fail (the H4 prediction)

Even with the prior, recovery of the *true* `T[i, :]` is expected to fail when all of:

- **support is low** (`n_i` small -> likelihood uninformative), and
- **noise is high and asymmetric** (large true off-diagonal mass -> few high-confidence
  anchors even relative to `n_i`; the confident joint is sparse and biased), and
- **the prior is misspecified** (`m_i` far from the true `T[i, :]`).

Under those three, `T̂[i, :] ≈ m_i` = a wrong assumption, and no amount of data corrects it.
By the on-disk statistics this predicts the failure region is **bipolar (id 2) and
schizophrenia (id 4)** -- exactly the conditions of greatest clinical interest and the ~20:1
minority side of the imbalance. Depression (id 0) sits firmly in the data-identified regime;
eating_disorder (id 5), with moderate support and distinctive vocabulary (baseline F1 0.945,
see `baseline-results.md`), is expected to be borderline-identified. This condition-indexed
prediction is the falsifiable content of RQ2 / H4, and §6 is its synthetic test.

---

## 4. Combining the prior with the confident joint (the mechanism)

"Regularize toward the prior" is under-specified. This section names candidate mechanisms,
the properties any valid mechanism must satisfy, and the recommended choice -- treated as a
first-class design decision, exactly as `g` is for C2.

### 4.1 Properties any valid mechanism must satisfy

Let `T̂(C, prior)` be the combined estimator for a confident joint `C` with per-condition
support `n_i`.

- **P1 -- Data limit (support-consistency).** As `n_i → ∞`, `T̂[i, :] → C[i, :] / n_i`
  (cleanlab's data-only estimate). The prior must wash out with data.
- **P2 -- Prior limit.** As `n_i → 0`, `T̂[i, :] → m_i` (the clinical prior row).
- **P3 -- Stochasticity.** For every finite `C`, each row of `T̂` is nonnegative and sums to
  1 (a valid transition matrix).
- **P4 -- Per-condition locality.** Row `i`'s prior influence depends on row `i`'s support
  only. Depression's row must be data-driven while bipolar's is prior-driven, in the same
  matrix.
- **P5 -- Monotone prior influence.** The weight on the prior is non-increasing in `n_i`.
  This gives a single clean knob for the §6 boundary sweep.
- **P6 (desirable) -- Interpretable strength.** The prior strength should be expressible in
  data-equivalent units, so it can double as the sensitivity-analysis axis.

### 4.2 Candidate mechanisms

**Candidate A -- Dirichlet-multinomial pseudocounts on the confident joint (row-wise).**
Treat row `i` of the confident joint as multinomial with a `Dirichlet(α_i · m_i)` prior
(`α_i` = per-condition concentration, `m_i` = prior mean row). The posterior-mean estimate
is the conjugate closed form:

```
T̂[i, :] = (C[i, :] + α_i · m_i) / (n_i + α_i),   with n_i = Σ_j C[i, j].
```

`α_i` reads as "the clinical prior is worth `α_i` confident observations." Satisfies P1
(`n_i → ∞` -> `C[i,:]/n_i`), P2 (`n_i → 0` -> `m_i`), P3 (row is a convex combination of two
probability vectors), P4 and P5 (weight on prior `= α_i / (n_i + α_i)`, decreasing in `n_i`),
and P6 (`α_i` in count units). Closed form, no optimizer, no `torch`, pure `numpy`.

**Candidate B -- Support-weighted convex blend at the matrix level.**

```
T̂[i, :] = w_i · T̂_cl[i, :] + (1 - w_i) · m_i,   w_i = n_i / (n_i + κ_i).
```

Empirical-Bayes / James-Stein shrinkage of cleanlab's *final* row estimate toward the
prior. Satisfies P1-P6 as well. Note it is **algebraically the reduced form of A** when
`T̂_cl[i,:] = C[i,:]/n_i` and `κ_i = α_i`; the two differ in practice only because cleanlab's
confident joint is calibrated (its normalization is not exactly raw `C/n`). B treats
cleanlab as a black box; A intervenes at the sufficient statistic before normalization.

**Candidate C -- Penalized constrained MAP.**

```
T̂ = argmin_T [ NLL(noisy labels | T, model_probs) + Σ_i λ_i · KL(m_i ‖ T[i, :]) ]
    subject to each row of T on the simplex.
```

Most general: does not depend on cleanlab's specific estimator and lets the identifiability
constraints enter directly as the penalty. Satisfies P1-P3 (data term dominates as data
grows; KL term forces `T[i,:] → m_i` as data vanishes; simplex constraint gives P3). But it
needs an optimizer and an explicit likelihood model, departs from the "adapted machinery"
(Confident Learning) the project commits to, and carries more implementation and convergence
risk.

### 4.3 Recommendation

# DECISION: use Candidate A (Dirichlet-multinomial pseudocounts on the confident joint).
Reasons: (1) it intervenes at the natural sufficient statistic cleanlab already computes,
making it the least-invasive extension of adapted machinery; (2) it satisfies all six
properties in closed form -- no optimizer, no `torch`, so it fits the CPU-only,
`numpy`-only constraint and is trivially unit-testable; (3) `α_i` is interpretable in
data-equivalent units and *is* the H4 sweep axis (§6); (4) it degrades honestly -- for
prior-dominated rows the output transparently equals `m_i`, which is exactly the
"scoped-down to sensitivity analysis" behavior RQ2 demands rather than a false estimate.
Candidate B is retained as a validation cross-check (it should agree with A up to cleanlab's
calibration step). Candidate C is deferred as a research extension, to be revisited only if
the confident-joint route proves inadequate for the rare conditions.

The pseudocounts are added to cleanlab's **calibrated** confident joint (so row sums already
reflect observed noisy-class frequencies) before the row-normalization above.

---

## 5. The clinical prior: shape and sourcing

### 5.1 Shape

`m` is a `K x K` row-stochastic matrix in `POC_CONDITIONS` id order; `m[i, :]` is the prior
mean of `T[i, :]` = expected proxy-label distribution given true condition `i`. `α` is a
length-`K` vector of per-condition strengths. Diagonal-heavy rows encode reliable
self-report; flatter rows with directed off-diagonal mass encode known confusion.

### 5.2 Sourcing the numbers (rationale, not invention)

The prior encodes documented clinical structure, and its point values are treated as
**swept hyperparameters over a defensible range**, not asserted constants -- consistent with
the honest-scope position in technical-contribution §1.4. The seeding rationale per row:

- **Depression (0), anxiety (1):** high diagonal (low off-diagonal mass). Most prevalent,
  least stigmatized, highest self-recognition and help-seeking (prevalence/stigma
  asymmetry, Mitchell et al., 2015). Depression's row is additionally the **only** row with
  a genuine clinical anchor: it can be validated against DAIC-WoZ (PHQ-8, depression-only;
  Gratch et al., 2014), controlling for the interviewer-prompt confound (Burdisso et al.,
  2024). All other rows are clinically unvalidated by construction.

- **Bipolar (2):** substantial prior off-diagonal mass directed at **depression**. Bipolar
  disorder is frequently first diagnosed and self-identified as unipolar depression, with
  large reported misdiagnosis rates and multi-year diagnostic delay (Hirschfeld et al.,
  2003 -- *verify pre-submission*). This is also visible in the C0 baseline confusion matrix
  (116 of 481 true-bipolar posts predicted depression; `baseline-results.md` §4), which
  motivates but does not by itself set the prior.

- **Schizophrenia (4):** flatter row with off-diagonal mass toward **depression** and
  **bipolar**. Psychotic-spectrum overlap plus reduced insight / anosognosia lowering
  self-identification reliability (Amador et al., 1994 -- *verify pre-submission*). Baseline:
  133 of 786 true-schizophrenia posts predicted depression.

- **Eating_disorder (5):** moderately high diagonal (distinctive vocabulary; baseline F1
  0.945), with modest off-diagonal mass to depression reflecting high ED-depression
  comorbidity (*verify pre-submission*).

- **Suicidality (3):** strong prior mass toward **depression**, reflecting the r/depression
  vs r/SuicideWatch overlap that is the entire subject of SDCNL (Haque, Reddi and
  Giallanza, 2021).

# DECISION: `prior.py` ships one `default_clinical_prior()` encoding the *structure* above
(which off-diagonals are non-negligible and their direction), with the exact magnitudes and
the `α_i` exposed as constructor arguments so C3 can sweep them. No magnitude is hard-coded
as if it were measured; the default is the center of the swept range. Clinical citations
marked *verify* follow the project-wide pre-submission citation-verification discipline.

---

## 6. Synthetic recovery test as a boundary sweep

The test is the empirical evidence for RQ2 / H4, and runs on synthetic data alone (CPU, no
GPU, no real corpus). It is a **2-D boundary sweep**, not a single pass/fail.

### 6.1 Data-generating process

1. Fix a ground-truth condition-dependent `T*` (`K x K`, row-stochastic) with realistic
   asymmetry (near-identity depression row; heavier, depression-directed bipolar and
   schizophrenia rows).
2. Sample true labels `y` from a class prior that reproduces the on-disk imbalance
   (~20:1 depression:bipolar), scaled by a **support multiplier** (axis 1).
3. Flip `y → ỹ` through `T*` to get proxy labels.
4. Simulate out-of-fold predicted probabilities of controllable quality via a
   separability parameter (interpolating between one-hot-on-true and uniform), standing in
   for the real model's `P(y | x)` without training anything.
5. Compute the confident joint from `(ỹ, probs)`, then run the §4 estimator with a prior
   whose **misspecification** `δ` from `T*` is the second swept axis.

### 6.2 The two sweep axes

- **Axis 1 -- per-condition support** `n_i`: from data-rich (depression-scale) to data-poor
  (hundreds, bipolar-scale).
- **Axis 2 -- prior misspecification** `δ = TV(m_i, T*[i, :])` (total-variation distance):
  from `δ = 0` (perfect prior) to a large `δ` (badly wrong prior).

### 6.3 Metrics and reported surface

Per cell `(support, δ)` and per condition `i`, report row recovery error
`TV(T̂[i, :], T*[i, :])` and the induced noise-summary error `|η̂_i - η*_i|` (the quantity
that actually reaches C2). Emit a 2-D grid marking, per condition, where recovery holds vs
degrades against a stated tolerance.

### 6.4 Predicted pattern (the falsifiable claim)

- **High support, any prior:** recovers `T*` (data dominates; prior washes out -- P1).
- **Low support, accurate prior (`δ ≈ 0`):** recovers (prior carries the row -- P2).
- **Low support, misspecified prior (`δ` large):** **fails** -- `T̂[i,:] ≈ m_i ≠ T*[i,:]`.
  This is the H4 breakdown region, expected to coincide with bipolar/schizophrenia-scale
  support.

If the observed breakdown boundary tracks support and `δ` as predicted, RQ2's identifiability
story is empirically supported; if it does not, that is itself reportable (the identifiability
structure is richer than the standard conditions imply). A subset of §6 doubles as a
deterministic unit test by feeding hand-built confident-joint matrices directly to the
estimator (bypassing steps 1-4), checking P1-P3 exactly.

---

## 7. Module interfaces (design-level signatures)

House style: injectable frozen dataclasses mirroring `src/modeling/config.py`;
`from __future__ import annotations`; full type hints; `numpy` only, no `torch`; US
spelling. Signatures below are the contract, not the implementation.

### 7.1 `src/noise/prior.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from ..data.schema import Condition, POC_CONDITIONS


@dataclass(frozen=True)
class ClinicalNoisePrior:
    """Injectable clinically-seeded prior over the noise-transition matrix ``T``.

    ``conditions`` fixes the axis order (must match POC id order). ``prior_mean`` is a
    row-stochastic ``K x K`` matrix; ``prior_mean[i]`` is the prior mean of ``T[i, :]``.
    ``concentration[i]`` is the Dirichlet strength ``alpha_i`` (prior mass in confident-
    count units) for true condition ``i``.
    """

    conditions: tuple[Condition, ...]
    prior_mean: tuple[tuple[float, ...], ...]   # K x K, each row sums to 1
    concentration: tuple[float, ...]            # length K, alpha_i >= 0

    def __post_init__(self) -> None: ...        # validate square, row-stochastic, non-negative

    @property
    def num_conditions(self) -> int: ...

    def prior_row(self, true_id: int) -> tuple[float, ...]: ...

    def as_matrix(self) -> np.ndarray: ...       # K x K float array (fresh copy)

    def alpha(self) -> np.ndarray: ...           # length-K float array


def default_clinical_prior(
    conditions: Sequence[Condition] = POC_CONDITIONS,
    *,
    # DECISION: structural knobs exposed so C3 can sweep them; defaults are the
    # center of the defensible range, never presented as measured constants.
    bipolar_to_depression: float = ...,
    schizophrenia_to_depression: float = ...,
    schizophrenia_to_bipolar: float = ...,
    eating_disorder_to_depression: float = ...,
    suicidality_to_depression: float = ...,
    reliable_diagonal: float = ...,             # depression/anxiety self-report reliability
    concentration: float | Sequence[float] = ...,  # scalar broadcast or per-condition alpha
) -> ClinicalNoisePrior: ...
```

### 7.2 `src/noise/transition.py`

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .prior import ClinicalNoisePrior


@dataclass(frozen=True)
class TransitionEstimate:
    """Result of combining the confident joint with the clinical prior."""

    matrix: np.ndarray            # K x K estimated T-hat (row-stochastic)
    support: np.ndarray           # length-K confident support n_i
    prior_weight: np.ndarray      # length-K alpha_i / (n_i + alpha_i) -- prior influence
    conditions: tuple[str, ...]   # axis labels, POC id order

    def noise_levels(self) -> np.ndarray: ...   # eta_i = 1 - T-hat[i, i]; the C2 hook


def estimate_transition_matrix(
    confident_joint: np.ndarray,        # K x K calibrated counts (from cleanlab, upstream)
    prior: ClinicalNoisePrior,
) -> TransitionEstimate:
    """Candidate A: row-wise Dirichlet-multinomial posterior mean.

    T-hat[i, :] = (C[i, :] + alpha_i * m_i) / (n_i + alpha_i),  n_i = sum_j C[i, j].
    Pure numpy; deterministic; fully testable on synthetic ``confident_joint`` matrices.
    """
    ...
```

### 7.3 Boundary to the deferred `oof.py`

`oof.py` (later, GPU) produces out-of-fold predicted probabilities over the labeled set and
hands `transition.py` a **confident-joint count matrix** -- nothing more. Concretely, the
deferred wrapper `confident_joint_from_predictions(labels, probs)` will call cleanlab
(CPU, `torch`-free) and return the `K x K` array that `estimate_transition_matrix` already
consumes. Because the core estimator's input is that count matrix, everything in `prior.py`
and `transition.py` is built and tested now, with `oof.py` slotting in behind an unchanged
interface once the K-fold cross-fit run is justified.

Downstream, C2 consumes `TransitionEstimate.noise_levels()` -> `η_i`, feeding
`τ_i = g(α, η_i)` (technical-contribution §2.2). C1's output contract is therefore the
length-`K` `η` vector plus the full `T̂` for the per-condition reporting in RO10.

---

## 8. Open questions for review (before coding)

1. **Prior magnitudes and `α_i`.** §5 fixes the *structure*; the default magnitudes and the
   per-condition `α_i` (especially how much stronger the prior should be for the
   no-data rows, anxiety/suicidality) need sign-off before they enter `default_clinical_prior`.
2. **Depression-row validation.** Confirm the plan to validate `T̂[0, :]` against DAIC-WoZ
   (with the Burdisso prompt-leakage control) belongs in C3, not C1.
3. **Candidate B cross-check.** Agree whether the B-vs-A agreement check is a permanent test
   or a one-off validation.
4. **Tolerance for §6.** The pass/fail tolerance on `TV(T̂[i,:], T*[i,:])` and `|η̂_i - η*_i|`
   sets where "recovery holds"; it should be chosen to reflect the `η` resolution C2 actually
   needs, not an arbitrary threshold.

---

## 9. References

Verified in existing project docs:
- Liu, Y., Cheng, H. and Zhang, K. (2023) 'Identifiability of Label Noise Transition
  Matrix', ICML 2023, PMLR 202, pp. 21475-21496.
- Northcutt, C., Jiang, L. and Chuang, I. (2021) 'Confident Learning: Estimating Uncertainty
  in Dataset Labels', JAIR 70, pp. 1373-1411.
- Xia, X. et al. (2019) 'Are anchor points really indispensable in label-noise learning?',
  NeurIPS 32.
- Patrini, G. et al. (2017) 'Making Deep Neural Networks Robust to Label Noise: a Loss
  Correction Approach', CVPR 2017.
- Mitchell, A. J. et al. (2015) -- prevalence/stigma asymmetry across conditions.
- Gratch, J. et al. (2014) -- DAIC-WoZ.
- Burdisso, S. et al. (2024) -- interviewer-prompt leakage in DAIC.
- Haque, Reddi and Giallanza (2021) -- SDCNL.

To verify before submission (clinical seeding rationale, §5):
- Hirschfeld, R. M. A. et al. (2003) -- bipolar under-recognition / misdiagnosis as
  depression.
- Amador, X. F. et al. (1994) -- insight / anosognosia in schizophrenia.
- Zhu, Z. et al. -- clusterability / HOC; Scott, C. et al. -- mutual-contamination /
  irreducibility (identifiability routes, §3.1).
- Eating-disorder / depression comorbidity source.
