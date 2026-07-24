# Decision Log

**Project:** Condition-Dependent Proxy-Label Noise and Calibrated Abstention for Trustworthy Multi-Condition Mental-Health Screening
**Module:** 6COSC023W, Informatics Institute of Technology (University of Westminster)
**Author:** Soshan
**Supervisor:** Sachindu Jayasinghe
**Repository:** github.com/SoshanW/mental-health-screening-fyp
**Log started:** 2026-07-17 (retrospective; earlier entries reconstructed)

---

## Purpose

This file records **what was decided, why, on what evidence, and what happened when a decision turned out to be wrong.** It follows the Architecture Decision Record (ADR) convention.

It exists for three reasons:

1. **Viva defence.** When an examiner asks "why did you not just estimate the transition matrix from the data?", the answer should be a dated entry with evidence, not a recollection.
2. **Demonstrating rigour.** The reversal entries (D-020 through D-026) are the most valuable part of this file. A project that never changed its mind is a project that never checked.
3. **Not relitigating.** Several directions were considered and closed. They stay closed unless new evidence appears.

### Status legend

| Status | Meaning |
|---|---|
| **ACTIVE** | Current, load-bearing. |
| **SUPERSEDED** | Replaced by a later decision. Kept for the record. |
| **REVERSED** | Found to be wrong. The reversal reasoning is the point. |
| **OPEN** | Not yet decided. Needs action. |
| **CLOSED** | Considered and rejected. Do not reopen without new evidence. |

### Template for new entries

```markdown
## D-0NN · [Title]
**Date:** YYYY-MM-DD · **Status:** ACTIVE · **Category:** [Scope | Data | C1 | C2 | C3 | Engineering | Writing]

**Decision.** One sentence, in the past tense.

**Context.** What prompted this. What was true at the time.

**Reasoning.** Why this rather than the alternatives.

**Evidence.** Citations, measurements, or "none, this is a judgement call."

**Consequences.** What this commits the project to. What it forecloses.

**Links.** Supersedes / superseded by / depends on.
```

### Date honesty

Entries D-001 to D-019 are **reconstructed retrospectively on 2026-07-17** and are marked *undated*. Backfill real dates from the repository git log and supervisor meeting notes before submission. An examiner will not mind reconstructed dates; they will mind invented ones.

---

## Index

| ID | Decision | Status | Category |
|---|---|---|---|
| D-001 | Frame as screening and decision-support, not diagnosis | ACTIVE | Scope |
| D-002 | Three coupled contributions rather than one system | ACTIVE | Scope |
| D-003 | Treat class imbalance as evidence, not an engineering nuisance | ACTIVE | Scope |
| D-004 | Low et al. as the sole Reddit source | ACTIVE | Data |
| D-005 | Exclude SWMH | ACTIVE | Data |
| D-006 | SMHD and RSDD unavailable, do not pursue | CLOSED | Data |
| D-007 | DAIC-WoZ as the depression-only clinical anchor | ACTIVE | Data |
| D-008 | Reject E-DAIC | CLOSED | Data |
| D-009 | Aich et al. as a stretch goal, not a dependency | OPEN | Data |
| D-010 | Author-grouped split, not stratified | ACTIVE | Engineering |
| D-011 | MentalBERT with class-weighted loss as the baseline | ACTIVE | Engineering |
| D-012 | `src/` package in VS Code; notebooks as thin GPU runners | ACTIVE | Engineering |
| D-013 | Follow the "Jazzify" thesis structure | ACTIVE | Writing |
| D-014 | Adopt supervisor's prose rules | ACTIVE | Writing |
| D-015 | Abandon XAI, distillation, GNN and LLM-judge directions | CLOSED | Scope |
| D-016 | Drop "CALM" as a project name | CLOSED | Writing |
| D-017 | Verify novelty by search, never from assumption | ACTIVE | Scope |
| D-018 | Two-regime identifiability framing | ACTIVE | C1 |
| D-019 | Name scope and limitations before examiners raise them | ACTIVE | Scope |
| D-020 | **REVERSED:** "nobody couples noise estimation to abstention" | REVERSED | C2 |
| D-021 | **REVERSED:** the invented `g()` coupling function | REVERSED | C2 |
| D-022 | **REVERSED:** posterior width as the coupling signal | REVERSED | C2 |
| D-023 | **REVERSED:** Dirichlet prior stabilises the conformal quantile | REVERSED | C2/C3 |
| D-024 | **REVERSED:** "every route to the matrix needs clean labels" | REVERSED | C2 |
| D-025 | **REVERSED:** the α_V = 0 proposition with pruning | REVERSED | C2 |
| D-026 | **REVERSED:** PHQ-8 described as clinician-administered | REVERSED | Data |
| D-027 | Build on Sesia et al. rather than compete with it | ACTIVE | C2 |
| D-028 | Elicit in Penso's direction, convert to Sesia's | ACTIVE | C2 |
| D-029 | Keep Mondrian (label-conditional), not marginal coverage | ACTIVE | C2 |
| D-030 | C1 becomes a diagnostic, not an estimator | ACTIVE | C1 |
| D-031 | Do not prune the elicited set (preserve α_V = 0) | OPEN | C2 |
| D-032 | Three-arm experiment as the core evaluation | ACTIVE | C2 |
| D-033 | Read primary sources before any claim becomes load-bearing | ACTIVE | Scope |
| D-034 | Clusterability diagnostic results: prediction falsified | ACTIVE | C1 |
| D-035 | Base-embedding control run to quantify the fine-tuning artifact | ACTIVE | C1 |

---

# Part 1 · Scope and framing

## D-001 · Frame as screening and decision-support, not diagnosis
**Date:** undated · **Status:** ACTIVE · **Category:** Scope

**Decision.** The system is described throughout as screening and decision-support research. It is never described as a diagnostic tool.

**Reasoning.** The labels are proxy labels (subreddit membership), not clinical diagnoses. A system trained on proxy labels cannot make diagnostic claims without a category error. Institutional ethics approval is required before data handling, and the framing must be consistent with what was approved.

**Evidence.** Chancellor and De Choudhury (2020, *npj Digital Medicine* 3:43) document construct-validity failures across the field: of 75 reviewed studies, only 32 (42%) reported enough to be reproducible.

**Consequences.** Every deliverable must use this language consistently. It also constrains the abstention framing: abstention is deferral to a clinician, not a refusal to diagnose.

---

## D-002 · Three coupled contributions rather than one system
**Date:** undated · **Status:** ACTIVE · **Category:** Scope

**Decision.** The project makes three contributions (C1 noise model, C2 calibrated abstention, C3 limits analysis) that depend on each other, rather than building one large system.

**Reasoning.** The marking rubric weights Design at 0.5 and Introduction at 0.4, rewarding methodological contribution over system scale. A coupled trio is defensible as a single intellectual move; three separate gadgets would not be.

**Consequences.** The coupling is the thing being defended. If the coupling fails empirically, the project falls back on C3 (see D-032).

---

## D-003 · Treat class imbalance as evidence, not an engineering nuisance
**Date:** undated · **Status:** ACTIVE · **Category:** Scope

**Decision.** The roughly 20:1 depression-to-bipolar imbalance is treated as supporting evidence for the thesis that condition-dependent noise priors are necessary, not merely as a problem to engineer around.

**Reasoning.** The conditions with the least data are the conditions whose noise is hardest to estimate. That is not a coincidence, it is the argument.

**Consequences.** Strengthened considerably by later findings (see D-030 and the prevalence arithmetic in Part 4). The imbalance turned out to be the reason the competing estimators have no guarantee on three of four conditions.

---

## D-015 · Abandon XAI, distillation, GNN and LLM-judge directions
**Date:** undated · **Status:** CLOSED · **Category:** Scope

**Decision.** The following were considered and closed: pure XAI/conformal-prediction faithfulness evaluation; MetroXAI / measurement-error-aware XAI; calibrated knowledge distillation; conformal GNNs; hallucination probes; LLM-judge debiasing.

**Consequences.** Do not resurrect without an explicit reason. Each was closed for its own reasons and reopening one costs time that the current direction needs.

---

## D-016 · Drop "CALM" as a project name
**Date:** undated · **Status:** CLOSED · **Category:** Writing

**Decision.** "CALM" was rejected as a project name; it is crowded in the broader ML literature. "NoCAB" was flagged as a more distinctive alternative but has not been committed to.

**Status note.** Naming remains uncommitted. Low priority.

---

## D-017 · Verify novelty by search, never from assumption
**Date:** undated · **Status:** ACTIVE · **Category:** Scope

**Decision.** No claim that a gap is open may be made without an actual literature search confirming it. Re-verify before final submission.

**Reasoning.** Established after Claude repeatedly overclaimed open gaps from training knowledge rather than evidence.

**Consequences.** This decision is the reason D-020 and D-024 were caught rather than surviving to the viva. **It has been the single most valuable process decision in the project.** See Part 4 for what it caught.

**Amendment (2026-07-17).** Search is necessary but not sufficient. See D-033: search results and abstracts also proved unreliable. The rule is now *read the primary source*.

---

## D-019 · Name scope and limitations before examiners raise them
**Date:** undated · **Status:** ACTIVE · **Category:** Scope

**Decision.** Every known weakness is stated in the text before an examiner can find it.

**Reasoning.** Stated preference for honest framing over overclaiming. Also strategic: a named limitation is a demonstration of understanding; a discovered one is a failure of it.

**Current instances.** The elicitation-validity assumption (D-031); the Ding pooling tension; the Theorem 1 versus Theorem 2 distinction in the prevalence table; the DAIC-WoZ construct-validity problems.

---

## D-033 · Read primary sources before any claim becomes load-bearing
**Date:** 2026-07-17 · **Status:** ACTIVE · **Category:** Scope

**Decision.** No claim about a paper's content enters the thesis based on its abstract, its citation in another paper, or a search-result snippet. The PDF gets read.

**Context.** Over the course of one working session, four separate claims about the literature were made from abstracts and snippets. **All four were materially wrong** (D-020, D-023, D-024, plus the Penso mischaracterisation). Every claim made after reading the full paper has held.

**Evidence.** The reversal record in Part 4 of this file is the evidence.

**Consequences.** Slower. Also the reason the project's central claim is currently defensible rather than fatally wrong.

---

# Part 2 · Data

## D-004 · Low et al. as the sole Reddit source
**Date:** undated · **Status:** ACTIVE · **Category:** Data

**Decision.** The Low et al. Reddit Mental Health Dataset (Zenodo 3941387, PDDL public domain) is the only Reddit source.

**Composition (independently verified).** depression 117,331; eating disorder (EDAnonymous) 14,577; schizophrenia 8,712; bipolar (bipolarreddit) 5,780.

**Reasoning.** It covers all four target conditions at larger scale than the alternatives, under a permissive licence.

---

## D-005 · Exclude SWMH
**Date:** undated · **Status:** ACTIVE · **Category:** Data

**Decision.** SWMH (Ji et al.) is excluded definitionally, not for convenience.

**Reasoning.** Low et al. covers the same conditions at larger scale. Including both would add no coverage and would complicate the harmonisation table.

**Consequences.** Ji et al. (2022, *Neural Computing and Applications* 34(13), 10309–10319) stays in the literature survey but is **repositioned as prior-art context** rather than as a dataset source: a multi-condition social-media detection model that performs detection without calibration or abstention.

---

## D-006 · SMHD and RSDD unavailable
**Date:** undated (confirmed June 2026) · **Status:** CLOSED · **Category:** Data

**Decision.** Neither dataset is obtainable. The Reddit API terms-of-service change closed the Georgetown access form.

**Consequences.** Do not suggest, do not pursue, do not cite as a "future work" option without noting unavailability.

---

## D-007 · DAIC-WoZ as the depression-only clinical anchor
**Date:** undated (access confirmed and integrated) · **Status:** ACTIVE · **Category:** Data

**Decision.** DAIC-WoZ (Gratch et al., 2014) is the clinical anchor. 188 participants, PHQ-8 labels, depression only.

**Reasoning.** It is the only obtainable clinical corpus for any of the four conditions.

**Known problems, stated up front.** Construct validity is contested from two independent directions: Burdisso et al. (2024, *Findings of ACL*) show many models exploit the interviewer's prompts rather than participant language; Patapati et al. (2025, ICMI Companion, doi:10.1145/3747327.3763034) argue the single self-reported scale means models may detect general distress rather than depression specifically.

**Consequences.** DAIC-WoZ is treated as **a shift target with its own contamination**, not as a gold standard. This framing is a contribution, not a hedge.

---

## D-008 · Reject E-DAIC
**Date:** undated · **Status:** CLOSED · **Category:** Data

**Decision.** E-DAIC was considered and rejected: marginal benefit, not worth a second gated application.

---

## D-009 · Aich et al. as a stretch goal, not a dependency
**Date:** undated · **Status:** OPEN · **Category:** Data

**Decision.** Aich et al. (2025, CLPsych; 644 participants with bipolar/schizophrenia/healthy-control expert diagnoses) is pursued as an aspirational second clinical anchor. In active negotiation with Natalie Parde (parde@uic.edu, UIC) and the UCSD team.

**Reasoning.** If granted, C1's noise estimates for bipolar and schizophrenia could be validated against expert diagnoses rather than remaining assumptions. That would be transformative. But it cannot be a dependency, because access is outside the project's control.

**Status.** Pending. **The project must be complete and defensible without it.**

**Open item.** The specific LLM model names and metric definitions in Aich et al. remain unverified. Check the ACL Anthology PDF (2025.clpsych-1.15) before citing specifics.

---

## D-026 · REVERSED: PHQ-8 described as clinician-administered
**Date:** 2026-07-17 · **Status:** REVERSED · **Category:** Data

**What was claimed.** Project documentation described DAIC-WoZ's PHQ-8 as "a clinician-administered instrument," contrasted against Reddit's self-selected forum membership.

**What is true.** PHQ-8 in DAIC-WoZ is a **self-report questionnaire completed by the participant**, administered within a structured interview protocol.

**Why it mattered.** The contrast being drawn is real but weaker than claimed: self-selected forum membership versus *a validated self-report instrument under a structured protocol*, not versus a clinician's diagnosis. Overstating the anchor's authority runs directly against D-007's framing and against Patapati et al.'s critique, which the project already cites.

**Fixed in.** Technical contribution explainer v2 and v3.

---

# Part 3 · Engineering

## D-010 · Author-grouped split, not stratified
**Date:** undated · **Status:** ACTIVE · **Category:** Engineering

**Decision.** 80/10/10 split grouped by author using scikit-learn's `GroupShuffleSplit` keyed on `author_id`. Not stratified by condition.

**Reasoning.** If the same author's posts appeared in both train and test, the model could learn to recognise that person's writing style rather than the condition, inflating the score.

**What was verified, and what was not.** Author-grouping prevents leakage but does **not** by construction guarantee equal per-condition proportions across piles. This was checked empirically on the actual splits: per-condition proportions held within **0.6 percentage points** across train, validation and test, with roughly **490 bipolar examples in each of validation and test**.

**Honesty note.** The property holds empirically for this run, not by construction. Re-check whenever the data or split seed changes. **Do not claim stratification the code does not perform.**

**Consequences.** The figure of ~490 has become load-bearing. It is the per-condition conformal calibration sample size, and it determines whether C2 produces useful prediction sets or vacuous ones.

---

## D-011 · MentalBERT with class-weighted loss as the baseline
**Date:** undated (Milestone 0 complete) · **Status:** ACTIVE · **Category:** Engineering

**Decision.** Class-weighted MentalBERT fine-tune via a `WeightedLossTrainer` (HuggingFace `Trainer` subclass), inverse-frequency weighting. 3 epochs, batch size 16, 256-token cap.

**Result.** 96% accuracy, 0.8782 macro-F1 on the held-out test set.

**Reasoning.** Plain cross-entropy under 20:1 imbalance would let the model ignore bipolar entirely and still look accurate.

**Consequences.** The current loss handles *imbalance* but knows nothing about *label noise*.

---

## D-012 · `src/` package in VS Code; notebooks as thin GPU runners
**Date:** undated · **Status:** ACTIVE · **Category:** Engineering

**Decision.** Structured, testable, version-controlled code lives in the `src/` package. Colab and Kaggle clone the repo and call package functions. Notebooks are not the primary code home.

**Reasoning.** The rubric weights software engineering at 50%. Notebooks do not demonstrate it.

**Conventions.** Injectable frozen dataclass configs; pure-logic modules avoiding torch imports where possible; US spelling in code (distinct from UK English in thesis prose); no em dashes; `# DECISION:` comments for key assumptions; full type hints; pytest structure mirroring each module; a single `(source, raw_label) → Condition` harmonisation table with no branching on raw strings elsewhere; `validate_schema()` at the end of every `load()`; `include_interviewer: bool = False` on the DAIC loader for ablation.

---

# Part 4 · The novelty claim and its reversals

> **This is the most important part of this file.** The project's central claim was wrong twice and was corrected twice, both times by reading a primary source. An examiner who reads only this section will conclude the work was done carefully. Do not sanitise it.

## D-020 · REVERSED: "nobody couples noise estimation to abstention"
**Date claimed:** undated · **Date reversed:** 2026-07-17 · **Status:** REVERSED · **Category:** C2

**What was claimed.** That C2's coupling of a per-condition noise estimate to a per-condition abstention threshold was unprecedented. Stated in the project proposal and in the technical explainer as: *"No existing system connects a label-noise estimate to an abstention threshold this way."*

**What killed it.** Sesia, M., Wang, Y.X.R. and Tong, X. (2025) 'Adaptive conformal classification with noisy labels', *Journal of the Royal Statistical Society Series B* 87(3), pp. 796–815, doi:10.1093/jrsssb/qkae114.

Their equations (13) to (15) set a per-class conformal threshold as an explicit function of a per-class noise estimate. Their target is label-conditional (Mondrian) coverage. Their contamination model is a general K x K matrix, not uniform noise.

**How it was caught.** Literature search under D-017, then confirmed by reading the full PDF.

**Why it mattered.** The claim was a **gap claim** (nobody has done this). Gap claims are cheap to defend but fatal when false. An examiner familiar with the conformal literature would have ended the discussion there.

**What replaced it.** A **differentiation claim**, then (after D-024) a structural claim. See D-027.

**Lesson recorded.** A named competitor is worth more than an absence. "Sesia et al. correct coverage back to nominal despite noise; I let condition-dependent noise change the policy" is a stronger sentence than "nobody has done this," because it proves the literature was read.

---

## D-021 · REVERSED: the invented `g()` coupling function
**Date reversed:** 2026-07-17 · **Status:** REVERSED · **Category:** C2

**What was claimed.** C2 defined an abstention threshold as `τ_i = g(α, η_i)` with `∂τ_i/∂η_i > 0`: noisier condition, stricter threshold, more abstention.

**Three problems.**

1. **It was not novel** (see D-020).
2. **It pointed the wrong way.** Sesia's Corollary 1 shows that noisy calibration data already makes standard conformal prediction **over-cover**. Noise makes you conservative for free. Sesia's entire contribution is *removing* that excess conservativeness because it produces uselessly large prediction sets. A `g()` that adds more conservativeness pays twice for something already obtained.
3. **It was invented.** "I made up a function" was always the softest point in the defence.

**What replaced it.** The coupling is now emergent rather than designed: a wide elicited region produces wide prediction sets through the borrowed machinery, with no invented function. See D-027.

---

## D-022 · REVERSED: posterior width as the coupling signal
**Date proposed and reversed:** 2026-07-17 (same session) · **Status:** REVERSED · **Category:** C2

**What was proposed.** After D-021, couple the abstention threshold to the **posterior width** of the Dirichlet noise estimate rather than its mean, on the argument that Sesia's Algorithm 2 tightens sets when the confidence region is wider.

**Why it is wrong.** The Dirichlet-multinomial posterior after counts `n` with prior `α` is `Dir(α + n)`. **Posterior width shrinks as `α` grows, with no data at all.** In the prior-dominated regime (bipolar), the posterior concentrates around the prior and the width goes *narrow*. The coupling would read that as confidence and abstain **less** on bipolar. Exactly backwards.

**The deeper problem.** Posterior width cannot distinguish "narrow because the data pinned it down" from "narrow because I asserted it firmly."

**What replaced it.** A **credal envelope**: the set of estimates induced by sweeping over a range of clinically defensible priors. That set is narrow when data dominates and wide when the prior does, which is the behaviour wanted. This led to D-027.

**Why this entry matters.** It was caught within one session, before any code was written. It is a good example of an idea being killed by its own arithmetic.

---

## D-023 · REVERSED: Dirichlet prior stabilises the conformal quantile
**Date reversed:** 2026-07-17 · **Status:** REVERSED · **Category:** C2/C3

**What was claimed.** That C1's Dirichlet prior does double duty: clinical seeding *and* shrinkage that stabilises the Mondrian conformal quantile at low `n`, making it viable at bipolar's ~490 calibration examples. This was proposed as a strong C2 differentiator, citing Ding et al. (2023, NeurIPS) on erratic per-class quantiles.

**Why it is wrong: two different sample sizes.**

| Quantity | Where it lives | Bipolar n |
|---|---|---|
| Transition matrix estimate | **training** split, via confident joint | ~4,800 |
| Conformal quantile | **calibration** split | ~490 |

The prior shrinks the first. It adds no calibration data, so it **cannot touch the second**. Ding's instability hits all experimental arms identically.

**Consequences.** Ding et al. is **not a C2 differentiator**. It was moved to C3, where it belongs, as one of several independent lines predicting breakdown on the same conditions.

**Compounding note.** This claim was made from a search snippet, not the paper. See D-033.

---

## D-024 · REVERSED: "every route to the matrix needs clean labels"
**Date claimed:** 2026-07-17 · **Date reversed:** 2026-07-17 · **Status:** REVERSED · **Category:** C2

**What was claimed.** After D-020, the replacement claim was: Sesia's algorithm needs a noise matrix, and every route to obtaining one requires clean labels, which do not exist for bipolar or schizophrenia. This became the central claim of technical explainer v2.

**What killed it.** Zhu, Z., Song, Y. and Liu, Y. (2021) 'Clusterability as an alternative to anchor points when learning with noisy labels', *Proceedings of the 38th ICML*, PMLR 139, pp. 12912–12923.

Their HOC estimator obtains a **unique** transition matrix from noisy labels alone: no clean data, no anchor points. It uses third-order consensus among 2-nearest-neighbour noisy labels. Code is public at github.com/UCSC-REAL/HOC. Their Theorem 1 proves uniqueness under 2-NN label clusterability, non-singularity and diagonal dominance.

**How it was caught.** The paper was identified as a threat and the PDF was obtained and read.

**Why the reversal improved the project.** The follow-up paper by the same authors supplies the rescue, and it is stronger than the original claim. See D-030.

---

## D-025 · REVERSED: the α_V = 0 proposition with pruning
**Date proposed and reversed:** 2026-07-17 · **Status:** REVERSED (partially) · **Category:** C2

**What was claimed.** A proposition stating that a clinically elicited region, **pruned** against the observed subreddit proportions, satisfies Sesia's Theorem 4 with `α_V = 0`, giving a deterministic containment guarantee.

**The flaw.** The proof asserted that the true model "is by definition consistent with the observed proportions, so it survives the pruning." **This is false.** The observed proportions are estimated from a *finite sample*. The true model is consistent with the *population* proportions, not necessarily with the sample. Sampling noise alone could prune the true model out, after which the region no longer contains the truth and the guarantee collapses.

**Two repairs, both legitimate, and the choice is a real design decision.**

| Option | Containment | α_V | Cost |
|---|---|---|---|
| **(a) Do not prune** | Deterministic | 0 | Wider region, wider prediction sets |
| **(b) Prune against a confidence region for the proportions** | Probabilistic | α_ρ > 0 | Tighter region, but the "tighter than their bootstrap" corollary dies |

**Current position.** Option (a) is stated in the proposition. See D-031, which remains OPEN.

**Note.** With ~140k posts, `α_ρ` can be made tiny. But tiny is not zero, and the entire point of a finite-sample guarantee is not waving that away.

---

# Part 5 · Current position (post-reversal)

## D-027 · Build on Sesia et al. rather than compete with it
**Date:** 2026-07-17 · **Status:** ACTIVE · **Category:** C2

**Decision.** Sesia, Wang and Tong (2025) is the foundation, not the competitor. The contribution fills an input slot their own Theorem 4 leaves open.

**The argument.** Theorem 4 requires a region `[V_low, V_upp]` containing the true noise parameters with probability at least `1 - α_V`, independent of the calibration data. **Nothing in the theorem or its proof cares how that region was constructed.** They fill it with a bootstrap from clean data (their Section 3.3) because that is the natural statistical move. It is not the only legal one.

**Verified by reading.** Their supplementary Section A3 was checked. A3.1 is the single-parameter randomised-response model; A3.2 is a two-parameter label-hierarchy extension. Neither is an open clinical prior. Every route in the paper to a valid region requires either M assumed exactly known (Algorithm 1, their example is differential privacy where the parameter is a design choice) or clean data (Algorithms 2/3, including the simplified scalar case in A3.1.4). **There is no third option in the paper.**

**The claim.** Three routes to the matrix exist in the literature, and all three close on mental-health Reddit text:

| Route | Needs | Why it closes |
|---|---|---|
| Clean data (Sesia §3.3) | Verified true labels | Do not exist for bipolar/schizophrenia; DAIC-WoZ is depression-only |
| Anchor points (Xia et al. 2019; Patrini et al. 2017) | Instances where true class is near-certain from features | Bipolar and depression share language by clinical necessity |
| Clusterability (Zhu et al. 2021, HOC) | 2-NN share the *true* class | Fails on lower-quality features; see D-030 |

**And even if one were open:** all three produce a **point estimate**. Theorem 4 needs a **region**. HOC's finite-sample rate (their Theorem 2) is derived under an assumption that off-diagonal entries are *uniform*, which is precisely what this project rejects.

**Consequences.** This is now the central claim. It is structural, cites the threatening papers as support, and is defensible.

---

## D-028 · Elicit in Penso's direction, convert to Sesia's
**Date:** 2026-07-17 · **Status:** ACTIVE · **Category:** C2

**Decision.** Elicit the contamination model as `P(Ỹ = j | Y = i)` (given the true condition, where does the post go?), then convert to Sesia's parameterisation.

**Reasoning.** Sesia's matrix is `P(Y = l | Ỹ = k)`: "given the label says bipolar, what is the chance the truth is depression?" **Nobody can answer that directly**, because it depends on the unknown true prevalence. Penso's direction is the one a clinician can answer. Penso et al. note this explicitly in their Section 4: Sesia and Clarkson "need to know the marginal class frequencies for both the clean and noisy labels, whereas we do not."

**The conversion.** The *noisy* marginals are simply observable (count subreddit memberships). Elicit `P`, observe `ρ̃`, derive `ρ`, then convert by Bayes. No clean data needed anywhere.

**UNVERIFIED.** The specific derivation `ρ = (Pᵀ)⁻¹ρ̃`, and the consequent observation that inconsistent models could be pruned (see D-025, D-031), **is not taken from any published paper.** It is a derivation produced during this session. **Check the algebra and have the supervisor check it before it enters the thesis as a contribution.**

---

## D-029 · Keep Mondrian (label-conditional), not marginal coverage
**Date:** 2026-07-17 · **Status:** ACTIVE · **Category:** C2

**Decision.** C2 targets label-conditional (Mondrian) coverage, giving the guarantee separately per condition.

**Reasoning.** Marginal coverage would let depression's 117k examples drown out bipolar's 5.8k. A system could hit 90% overall while covering bipolar 40% of the time and the headline number would look fine. Per-condition difference **is** the research question.

**The objection to be ready for.** Bortolotti, T., Wang, Y.X.R., Tong, X., Menafoglio, A., Vantini, S. and Sesia, M. (2025) 'Noise-adaptive conformal classification with marginal coverage', arXiv:2501.18060. Same group, extending to marginal coverage. Their results indicate that while all adaptive methods reach valid coverage, **only the marginal ones produce more informative prediction sets**; the label-conditional versions gain validity without gaining informativeness.

**Response.** They optimise for one global guarantee; this project needs per-condition behaviour. **Run a marginal arm alongside to show the contrast rather than avoiding the point.**

**OPEN.** This paper has not been read in full (rate-limited). Their estimation route may be softer on clean data than the 2024 paper, which would pressure D-027. **Read before finalising Chapter 2.**

---

## D-030 · C1 becomes a diagnostic, not an estimator
**Date:** 2026-07-17 · **Status:** ACTIVE · **Category:** C1

**Decision.** C1 stops being "estimate the matrix with Confident Learning plus a clinical prior" and becomes "**demonstrate, on the actual data, that the available estimators fail**." That demonstration is what licenses C2's alternative.

**Why the change was forced.** D-024. HOC estimates the matrix without clean data, so "you need a prior because the data cannot do it" was not established.

**Why the replacement is stronger.** Zhu, Z., Wang, J. and Liu, Y. (2022) 'Beyond images: label noise transition matrix estimation for tasks with lower-quality features', *Proceedings of the 39th ICML*, PMLR 162.

This paper **exists because HOC fails outside computer vision**. Their abstract: tasks with lower-quality features fail to meet the anchor-point or clusterability condition. They test on **BERT-embedded text**, which is this project's exact setting.

Estimation error (x100, lower is better), their Table 2:

| Dataset | Noise | HOC | Confident Learning | T-Revision | Their fix |
|---|---|---|---|---|---|
| AG's News (BERT), 4-class, 30k each | e ≈ 0.178 | 13.32 | 11.41 | 10.38 | 8.35 |
| AG's News | e ≈ 0.302 | 10.62 | 10.63 | 10.71 | 6.52 |
| Jigsaw (BERT), binary, 9.4:1 imbalance | e ≈ 0.111 | **14.25** | **20.17** | **20.92** | 9.97 |
| Jigsaw | e ≈ 0.2 | 11.28 | 16.44 | 17.10 | 7.66 |

**Two things to say out loud.**

1. **AG's News is the closest published analogue** on class count (4) and size (120k vs ~140k). HOC's error is 0.133 while the quantity being estimated has size 0.178. **The error is roughly 75% of the signal.** On *balanced* data with 30,000 per class. This project has 5,780 bipolar.
2. **Jigsaw is the imbalanced one.** At 11% noise, the trivial estimate "assume no noise" scores 11.1. **HOC scores 14.25: worse than assuming the labels are perfect.** Confident Learning scores 20.17, which for binary at that noise level is exactly a random guess.

**And Confident Learning is C1's own engine.** That is not a footnote; it is why C1 changed role.

**The clinical explanation, which is the elegant part.** HOC needs each post's nearest neighbours in embedding space to share its **true** class. A person who truly has bipolar disorder, writing during a depressive episode, writes like a person with depression. That is *phase predominance*, a documented clinical feature. So their nearest neighbours in MentalBERT space *are* depression posts. **The reason clusterability fails is the same clinical fact that creates the label noise.** The estimator's assumption and the phenomenon under study are in direct conflict.

**Also:** HOC's Assumption 2 requires a dominant diagonal. If diagnostic delay means most truly-bipolar people post in r/depression, bipolar's diagonal is not dominant and Theorem 1 does not apply at all.

**What C1 now does.** Run HOC and `cleanlab` on the actual data. Do they disagree? Are they unstable across seeds? Do they produce clinically implausible matrices? **Every outcome is useful:** inside the elicited envelope is independent corroboration; outside is the measured failure the 2022 paper predicts.

---

## D-031 · Do not prune the elicited set (preserve α_V = 0)
**Date:** 2026-07-17 · **Status:** OPEN · **Category:** C2

**Provisional decision.** Option (a) from D-025: do not prune, keep deterministic containment, accept a wider region.

**Why still OPEN.** The trade is real and unresolved. Option (b) gives tighter sets at the cost of `α_V = α_ρ > 0` and loses the corollary that this route is *tighter* than Sesia's bootstrap. With ~140k posts `α_ρ` would be very small. This needs a decision before Chapter 4 is written, ideally with the supervisor.

**Depends on.** D-028's derivation being correct.

---

## D-018 · Two-regime identifiability framing
**Date:** undated · **Status:** ACTIVE, and substantially strengthened 2026-07-17 · **Category:** C1/C3

**Original decision.** Because a plain class-conditional transition matrix is not identifiable from noisy labels alone (Liu, Cheng and Zhang, 2023, ICML), conditions are split into **data-identified** (large support, e.g. depression) and **prior-dominated** (small support, e.g. bipolar and schizophrenia, where the output leans on the clinical prior and is reported as a sensitivity analysis rather than a firm estimate).

**Strengthened 2026-07-17: the split is now arithmetic, not a framing choice.**

Zhu, Song and Liu's Theorem 2 gives HOC's finite-sample rate, but only when the diagonal exceeds a threshold depending on class rarity. At K = 4 with this project's prevalences:

| Condition | Posts | Share | Required diagonal | Achievable? |
|---|---|---|---|---|
| Depression | 117,331 | ~0.81 | 0.27 | **yes, easily** |
| Eating disorder | 14,577 | ~0.10 | 1.00 | no |
| Schizophrenia | 8,712 | ~0.06 | 1.56 | no |
| Bipolar | 5,780 | ~0.04 | **2.25** | **impossible** |

**Depression is the only one of the four for which the leading no-clean-data estimator has a finite-sample guarantee at all.** The other three require a diagonal probability above 1.

**Caveat that must be stated, not hidden.** That threshold comes from their **Theorem 2** (finite-sample rate), derived under a tractability assumption that off-diagonal entries are uniform. Their **Theorem 1** (uniqueness in the infinite-data limit) carries no such condition. So the honest claim is *"HOC has no finite-sample rate for three of four conditions,"* **not** *"HOC provably fails."* It might still work in practice, which is exactly why D-030 runs it and measures.

**UNVERIFIED.** This arithmetic was produced during this session from their formula. **Plug the actual counts in and check it independently.** It is currently the strongest single result in the project, which means it is also the one that will hurt most if it is wrong.

**A further refinement.** The regime boundary is **not just sample size**. Zhu, Wang and Liu (2022) show AG's News has 30,000 examples per class, perfectly balanced, and HOC still errs by 0.133 because BERT text features are not clusterable enough. So the boundary is set by **sample size AND feature quality together**. For the rare conditions here, both are against the project at once. This is a sharper version of the original thesis than the one it started with.

---

## D-032 · Three-arm experiment as the core evaluation
**Date:** 2026-07-17 · **Status:** ACTIVE · **Category:** C2/C3

**Decision.** The headline evaluation compares three arms:

| Arm | Description |
|---|---|
| **(a)** | Noise-blind: standard conformal, ignoring label noise |
| **(b)** | Sesia's Algorithm 2 with a plug-in estimate (from HOC or cleanlab) |
| **(c)** | This project: Sesia's Algorithm 2 with the clinically elicited region |

**Reasoning.** Arm (b) is essential. Beating (a) is table stakes and proves nothing anyone will care about. The claim reduces to (c) beating (b) on risk-coverage for bipolar and schizophrenia.

**Cost.** Low. Sesia's code is public at github.com/msesia/conformal-label-noise; HOC's at github.com/UCSC-REAL/HOC.

**The insurance, and why the project is safe either way.** Add a **noise-heterogeneity sweep**: vary how unequal the true noise rates are synthetically and find where (c) overtakes (b), then locate the real data on that axis. Head-to-head, (c) > (b) is a coin flip that can be lost. With the sweep, losing is still a result: *"the coupling pays off above heterogeneity level h*, and Reddit proxy noise sits below it"* is a publishable, honest finding.

**The same logic applies to the feasibility threshold.** If C2 works, the project has a method. If it degenerates to always abstaining on bipolar, the project has **the first characterisation of where this family of methods stops working on real mental-health proxy labels**. A project whose headline survives its own negative result is well designed, and that should be said in the defence.

---

## D-034 · Clusterability diagnostic results: the prediction was falsified
**Date:** 2026-07-24 · **Status:** ACTIVE · **Category:** C1

**The prediction.** Before running, the stated expectation was: 2-NN agreement
would be high for depression and noticeably low for bipolar, and low bipolar
agreement would be evidence that HOC's clusterability condition fails on this
data. It was explicitly framed as a one-sided test: high bipolar agreement would
falsify the clusterability-failure argument.

**The result.** Run on MentalBERT embeddings from the Milestone 0 fine-tuned
checkpoint. Reddit training split, 111,892 posts. |E| = 15,000, G = 20 rounds,
negative cosine similarity, seed 0.

| Condition | n/round | per-neighbour agreement | sd | chance | lift over chance |
|---|---|---|---|---|---|
| depression | 12,173 | 99.54% | 0.07% | 81.2% | 1.23x |
| eating_disorder | 1,438 | 98.78% | 0.39% | 9.6% | 10.30x |
| schizophrenia | 822 | 94.49% | 0.74% | 5.5% | 17.22x |
| bipolar | 565 | **86.70%** | 1.10% | 3.8% | **23.00x** |

**The prediction was wrong.** Bipolar agreement is 86.70%, inside and near the
top of the 78 to 88 percent range Zhu, Song and Liu report for noisy CIFAR-10
(their Table 3), a setting where HOC works. Bipolar also has the **highest** lift
over chance of all four conditions at 23x, meaning bipolar posts are the most
distinctive class in this embedding space relative to their base rate, not the
least.

**Same-author contamination ruled out.** Nearest neighbours share the centre
post's author in under 0.1% of neighbour slots across all conditions (bipolar
0.08%, depression 0.01%). Authors average roughly 1.1 posts each in the training
split. The observed clustering is not an artifact of near-duplicate posts by the
same person.

**Neighbour label distribution given centre label** (row-normalised):

| centre \ neighbour | bipolar | depression | eating_dis. | schizophrenia |
|---|---|---|---|---|
| bipolar | 0.8669 | **0.0995** | 0.0027 | 0.0310 |
| depression | 0.0029 | 0.9954 | 0.0006 | 0.0011 |
| eating_disorder | 0.0012 | 0.0101 | 0.9878 | 0.0008 |
| schizophrenia | 0.0205 | **0.0324** | 0.0020 | 0.9450 |

**One positive finding.** For both bipolar and schizophrenia centres, the
dominant off-diagonal neighbour class is depression (9.95% and 3.24%
respectively). This is directionally consistent with the clinical elicitation
argument (bipolar-to-depression is the expected dominant confusion flow). It is
weak evidence: it concerns embedding neighbourhoods rather than label noise
directly, and the depression class is also the largest by a wide margin. Record
it as consistent-with, not as support.

**Two reasons the falsification is weaker than it first appears.**

1. **Circular measurement.** The embeddings come from a model fine-tuned to
   separate these four noisy labels. Agreement on noisy labels was measured in a
   space optimised to separate noisy labels. Depression at 99.54% against Zhu et
   al.'s 78 to 88 percent is the tell: the gap is the training objective showing
   through. Zhu et al. also use noisy-label-trained extractors, but measure
   feasibility against **true** labels, which breaks the circularity. That option
   is not available here.

2. **Structural blindness to the failure mode of interest.** Posts are
   partitioned by their *noisy* label. A truly-bipolar person who posted in
   r/depression sits in the depression row, indistinguishable from truly-depressed
   posts. The bipolar row therefore measures only the self-identified subset,
   those whose proxy label happened to be correct. The phase-predominance cases
   the argument depends on are filed elsewhere by construction.

**Integrity note, recorded deliberately.** Reason (2) is true a priori and should
have been identified when the test was designed. It was articulated only after
the result came back against the hypothesis. That is the structural shape of
motivated reasoning, and it is logged as such so that a reader can discount it
appropriately. The point stands or falls on its own merits, not on the timing of
its appearance.

**Consequence.** The claim that clusterability fails on this data currently has
**no support from this project's own data.** It rests entirely on Zhu, Wang and
Liu's (2022) published Table 2 results on other BERT corpora. That remains real
published evidence from the method's own authors, but it is borrowed rather than
measured.

**What this promotes.** Stage 2 becomes the load-bearing diagnostic: HOC's
stability across random seeds, and HOC-versus-cleanlab disagreement. Neither
carries the circularity confound. **If HOC returns a stable and clinically
plausible matrix on this data, D-030's premise is in trouble and C1 needs
rethinking again.**

**Links.** Tests the assumption behind D-030. Supersedes the framing in open
item 2.

---

## D-035 · Base-embedding control run
**Date:** 2026-07-24 · **Status:** ACTIVE · **Category:** C1

**Decision.** Repeat the D-034 diagnostic using embeddings from **base
MentalBERT** (`mental/mental-bert-base-uncased`, no fine-tuning on this
project's four labels), as a control against the fine-tuned run.

**What this does and does not establish.** It does **not** produce "the real
clusterability number." Clusterability is defined over true labels, which are
unavailable, so it cannot be measured directly by any run. What the control
provides is the **delta**: the difference between fine-tuned and base agreement
quantifies how much of the apparent structure in D-034 is an artifact of the
training objective rather than intrinsic to the text.

**Interpretation rule set in advance, to avoid post-hoc reasoning.**

- If base bipolar agreement stays high (say above 70%), the clusterability-failure
  argument is genuinely weak and should be **dropped rather than defended**. The
  C1 justification then rests solely on Zhu, Wang and Liu (2022) plus whatever
  Stage 2 shows.
- If base bipolar agreement drops sharply (say below 40%), the fine-tuned number
  was substantially artifact, and the honest report is that neither number
  resolves clusterability but the space is far less separable than D-034 implied.

**What this does NOT change.** HOC itself should still be run on the
**fine-tuned** embeddings, because that matches HOC's own protocol: Zhu, Song and
Liu take the feature extractor from a model trained to near-100% training
accuracy on the noisy labels. Running HOC on base embeddings would handicap it
unfairly and would not be a fair test of the method.

**Links.** Controls for the confound identified in D-034.

---

# Part 6 · Writing

## D-013 · Follow the "Jazzify" thesis structure
**Date:** undated · **Status:** ACTIVE · **Category:** Writing

**Decision.** Structure follows the supervisor's own past thesis, cross-checked against three IIT/Westminster exemplars (Ammar W1761196, Hashim w1957407, AutoDistil-KG w1954098).

**Conventions confirmed.** Literature Review: Concept Map → Problem Domain → Technical Review → Datasets → Benchmarking → gap-pointing Summary, with the Existing Work table in an Appendix (Citation / Summary / Limitation / Contribution). Methodology: Saunders Research Onion table → Development Methodology → Project Management → Resources → Risks, with system architecture deferred to a separate Design chapter. Research Objectives table mapped to Learning Outcomes and Research Questions.

**Note.** Exemplars use a Layer/Choice/Justification table for the Research Onion with no diagram required. The diagram (Figure_3_1_Research_Onion.png) is optional supporting material.

---

## D-014 · Adopt supervisor's prose rules
**Date:** undated · **Status:** ACTIVE · **Category:** Writing

**Decision.** All chapters follow Sachindu Jayasinghe's stated rules.

1. **Do not ascribe intent or deliberateness to prior authors** when describing limitations. Avoid "deliberately," "by design," "by its authors' explicit design," "agnostic," "stop short." State plainly what a method does or does not do. (E.g. "does not model a noise distribution and is applied uniformly," not "deliberately noise-distribution-agnostic by design.") This applies even when the underlying factual contrast is correct.
2. **Do not open statements with combative negations** such as "No existing system…," "However, no identified work…". Lead with what the project does; let the "not yet combined" observation land plainly at the end.
3. **Do not praise the project's own honesty.** Remove "honestly," "remaining honest about," "we are transparent that," "made explicit rather than elided." Use neutral verbs: "acknowledging," "the analysis reports…"
4. **Unifying rule:** describe what prior work does and where your scope differs, without ascribing intent, without leading on a negation, without narrating your own virtue.
5. **UK English throughout** (characterise, modelling, labelling). Chapter 1 drifted into mixed US spellings; monitor in all chapters.
6. **No em dashes in deliverables.** Use colons, parentheses, semicolons or commas. En dashes retained in page/number ranges.

**Note.** Rule 2 independently required the rewrite that D-020 forced anyway. The false claim was also a badly-phrased one.

---

# Part 7 · Open items

| # | Item | Blocking | Owner |
|---|---|---|---|
| 1 | Run HOC and `cleanlab` on the real data | D-030 (C1 is currently an argument, not a measurement) | Soshan |
| 2 | Measure 2-NN *noisy*-label agreement per condition in MentalBERT space (**amended 2026-07-24: a one-sided falsification test, not confirmatory; see expanded note below**) | The clusterability diagnostic | Soshan |
| 3 | Verify the `ρ = (Pᵀ)⁻¹ρ̃` derivation | D-028, D-031 | Soshan + supervisor |
| 4 | Verify the prevalence arithmetic in D-018 against Zhu et al.'s Theorem 2 | The strongest result in the project | Soshan |
| 5 | Decide prune vs no-prune | D-031 | Soshan + supervisor |
| 6 | Source every elicited range to a specific clinical study | The main attack on C2 | Soshan |
| 7 | Read Bortolotti et al. (arXiv:2501.18060) in full | D-029 | Soshan |
| 8 | Read Ding et al. (arXiv:2306.09335) in full | One leg of C3's convergence table is from a snippet | Soshan |
| 9 | Aich et al. access outcome | D-009 | External |
| 10 | Chapter 2 (Literature Review) draft | Next chapter | Soshan |
| 11 | Re-verify novelty before camera-ready | D-017; the noisy-conformal area published 4+ items in 2024–25 | Soshan |

---

## What the 2-NN clusterability statistic can and cannot show (open item 2, amended)

**Date:** 2026-07-24 · **Amends:** open item 2 · **Category:** C1

**Amendment.** The 2-NN *noisy*-label agreement statistic is confounded between the
noise level and clusterability failure and cannot distinguish them. It functions as
a **one-sided test**: high agreement would falsify the clusterability-failure
argument, whereas low agreement is consistent with it but does not establish it. The
**seed-instability of the noise estimators** (running HOC and cleanlab across seeds
and showing they disagree or wander) is the primary diagnostic for C1's failure
claim; the clusterability statistic is descriptive support, not the proof.

**Why this changes what open item 2 is for.** Item 2 was originally read as "measure
this and low bipolar agreement proves clusterability fails." That over-reads it. Two
different generative facts produce the same low agreement: (a) the feature geometry
is genuinely not clusterable for that condition, or (b) the proxy labels are simply
noisier for that condition, so neighbours in a perfectly clusterable space still
disagree on the *noisy* label. Measured against noisy labels, the statistic cannot
separate (a) from (b). So a low number is consistent with the argument but is not by
itself evidence for it; only a high number is decisive, and only in the falsifying
direction.

**How it is reported.** The statistic is reported per condition (not pooled) as
descriptive context, with the one-sided caveat and the noisy-vs-true-label caveat
stated in-line on every report. Both caveats are baked into
`src/noise/clusterability.py` (`NOISY_LABEL_CAVEAT`) so they cannot be dropped when
the numbers are copied into the thesis. Zhu, Song and Liu (2021, Table 3) report
roughly 78 to 88 percent feasible 2-NN tuples on noisy CIFAR-10 against **true**
labels; that is not comparable to a **noisy**-label agreement number.

**Consequences.** C1's load-bearing evidence shifts to estimator seed-instability and
cross-estimator disagreement (open item 1), with the arithmetic in D-018 (no
finite-sample rate for three of four conditions) as the theory. Clusterability
agreement is retained as a cheap, honest, one-sided check that can only ever *hurt*
the argument if it comes back high for a rare condition.

**Links.** Depends on D-030 (C1 is a diagnostic). Feeds open item 1.

---

## Elicitation sources still needed (open item 6, expanded)

The elicited set needs each range tied to a specific study. The *directions* are clinically predictable, which is the answer to "your bipolar threshold is just your own belief":

- **Bipolar to depression should dominate the off-diagonal**, for two independent documented reasons: **diagnostic delay** (bipolar is characteristically misdiagnosed as unipolar depression for years, so a truly-bipolar person may sincerely self-identify as depressed) and **phase predominance** (depressive phases occupy far more time-in-illness than manic ones).
- **Schizophrenia's diagonal should be lower**, because impaired insight is a core clinical feature, making self-identification less reliable than for mood disorders.
- **Depression's diagonal should be highest**: high base rate, high public awareness, lower stigma relative to psychosis.

**WARNING.** These three claims were stated from general knowledge during this session and are **exactly the kind of thing that sounds right and turns out subtly wrong.** Each needs a citation to a specific study. Where the literature is thin, **the range goes wide**, and the envelope honestly propagates that width into wider prediction sets. That propagation is a feature: vagueness becomes visible instead of hidden.

---

## Key references, verified by reading the full paper

| Reference | Role |
|---|---|
| Sesia, M., Wang, Y.X.R. and Tong, X. (2025) 'Adaptive conformal classification with noisy labels', *JRSSB* 87(3), 796–815, doi:10.1093/jrsssb/qkae114 | C2's foundation. Theorem 4 is the slot being filled. |
| Penso, C., Goldberger, J. and Fetaya, E. (2025) 'Conformal prediction of classifiers with many classes based on noisy labels', COPA, PMLR 266, 1–14 | The elicitable parameterisation (D-028). States that for a general noise matrix "all finite sample correction terms are not effective." |
| Zhu, Z., Song, Y. and Liu, Y. (2021) 'Clusterability as an alternative to anchor points when learning with noisy labels', ICML, PMLR 139, 12912–12923 | Killed D-024. Theorem 2 gives D-018's arithmetic. |
| Zhu, Z., Wang, J. and Liu, Y. (2022) 'Beyond images: label noise transition matrix estimation for tasks with lower-quality features', ICML, PMLR 162 | Rescued the project. Evidence for D-030. |

## Key references, cited but NOT yet read in full

| Reference | Risk |
|---|---|
| Ding et al. (2023, NeurIPS), 'Class-conditional conformal prediction with many classes', arXiv:2306.09335 | One leg of C3's convergence table. |
| Bortolotti et al. (2025), arXiv:2501.18060 | Bears on D-029. |
| Einbinder et al. (2024), *JMLR* 25(328), 1–66 | Context-setting only; lower risk. |
| Liu, Cheng and Zhang (2023, ICML), 'Identifiability of label noise transition matrix', PMLR 202, 21475–21496 | Underpins D-018. In the verified survey but not read cover to cover. |

---

## Reversal scoreboard

| Claims made from abstracts or snippets | 4 |
| Of those, materially wrong | **4** |
| Claims made after reading the full paper | 5 |
| Of those, materially wrong | **0** |

**This table is the argument for D-033.** Keep it updated. If it ever shows a wrong claim made after reading a full paper, that is worth knowing too.
