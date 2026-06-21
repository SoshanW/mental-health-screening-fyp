# CHAPTER 01: INTRODUCTION

> **Template note:** This chapter follows the structure and section format used in the
> supervisor-provided exemplar thesis ("Jazzify", S. D. N. Jayasinghe), adapted to this
> project's domain. Citations in the Existing Work table (Section 1.5) and throughout this
> chapter have been verified (title, authors, venue, year, and DOI/arXiv ID) against primary
> or publisher sources during the literature survey, **with the following items flagged for a
> final pre-submission spot-check**: (i) the SDCNL Springer LNCS volume number (the ICANN 2021
> proceedings span LNCS 12891–12895 across parts; confirm the exact part for chapter `_35`
> before final submission); (ii) page ranges for conference proceedings that are inconsistently
> indexed (NeurIPS, ICLR papers have no canonical page numbers; cite arXiv/OpenReview);
> (iii) Ernala et al. (2019) and Harrigian & Dredze (2022), which were carried over from the
> author's own prior reading and sit outside the formally verified survey batch. All
> factual/dataset claims (post counts, dataset availability) were independently verified during
> project planning; see the project proposal for the verification record.
>
> **Novelty-timing caveat:** The novelty claim in Sections 1.6–1.7 rests on a thorough literature
> search current to mid-2026. The CLPsych 2026 workshop (the field's most likely venue for a
> directly competing paper) is held at ACL 2026 in early July 2026, after this draft; its
> proceedings were not yet public at the time of writing. The novelty claim should be
> re-verified against the CLPsych 2026 and ACL 2026 proceedings once released, and again
> immediately before final submission.

---

## 1.1 Chapter Overview

This research investigates trustworthy multi-condition mental-health screening from
social-media text, addressing two compounding weaknesses shared across the field: the
unexamined, condition-dependent unreliability of proxy diagnostic labels, and the
absence of calibrated, principled abstention in existing screening systems. This chapter
provides an overview of the research domain, its goals and objectives, and the evidence
needed to identify the research need and the study's originality. The author acknowledges
the challenges encountered during the research process, including a significant dataset
disruption mid-planning, and outlines the steps taken to overcome them.

---

## 1.2 Problem Domain

### 1.2.1 Multi-Condition Mental-Health Screening from Social Media

Automated mental-health screening from social-media text is a well-established and
active subfield of Natural Language Processing (NLP) applied to healthcare. The premise
is straightforward: people disclose symptoms, diagnoses, and lived experience of mental
illness in public posts, and these disclosures can in principle be used to train
classifiers that flag posts, users, or communities exhibiting signs of conditions such as
depression, anxiety, bipolar disorder, schizophrenia, eating disorders, and suicidality
(Coppersmith et al., 2015; Low et al., 2020).

Common technical approaches in this space include:

- **Self-report and subreddit-membership proxy labelling**: the dominant labelling
  strategy in the field, originating with the mining of self-reported diagnoses on
  Twitter (Coppersmith et al., 2015) and now most commonly realised on Reddit by treating
  membership in a condition-specific community (e.g. r/depression) as a stand-in label for
  the condition itself (Low et al., 2020; Ji et al., 2021).
- **Domain-pretrained transformer classifiers**: models such as MentalBERT (Ji et al.,
  2022), pretrained on mental-health-related corpora and fine-tuned for downstream
  screening tasks.
- **Multi-condition classification**: extending binary depression/non-depression
  classifiers to distinguish among several conditions simultaneously, as in the SWMH
  dataset spanning depression, anxiety, bipolar disorder, suicidality, and a general
  off-topic class (Ji et al., 2021).

When applied carefully, these techniques can support large-scale, low-cost population
screening that would be infeasible through clinical interview alone. However, as
detailed in Section 1.3, the field's reliance on subreddit membership as a ground-truth
label, combined with an absence of principled uncertainty handling, significantly limits
the trustworthiness of the resulting systems (Chancellor and De Choudhury, 2020).

---

## 1.3 Problem Definition

Multi-condition mental-health screening from social media is built almost entirely on
**proxy labels**: a post is labelled with a condition because it was made in a
condition-associated subreddit, not because the poster holds a verified clinical
diagnosis (Coppersmith et al., 2015; Ernala et al., 2019; Harrigian and Dredze, 2022).
This is a long-standing and openly acknowledged limitation of the field: a systematic
review of seventy-five studies found that the use of unvalidated proxy and self-report
labels as ground truth was pervasive, and that only thirty-two of the seventy-five
studies (approximately 42%) reported enough methodological detail to be reproducible
(Chancellor and De Choudhury, 2020). Because such datasets typically contain no
guaranteed clinical negatives, the resulting systems distinguish people _likely to belong
to a condition-associated community_ from the general population, rather than clinically
diagnosed from non-diagnosed individuals (Ernala et al., 2019).

Two specific, compounding weaknesses follow from this and remain inadequately addressed
in current systems:

**First, proxy-label unreliability is not uniform across conditions.** Self-identification
is comparatively reliable for common, well-understood conditions such as depression, but
markedly less reliable for conditions such as bipolar disorder and schizophrenia, which
are more frequently self-misattributed, more widely discussed online relative to their
clinical prevalence, and carry differing degrees of stigma (Mitchell et al., 2015;
Chancellor and De Choudhury, 2020). Despite this, existing noise-aware approaches to
mental-health label correction treat label noise as **undifferentiated**. SDCNL, the
closest such method, performs unsupervised label correction for a single binary task
(suicide versus depression) using a procedure that is, by its authors' explicit design,
agnostic to any noise distribution: it applies one correction process uniformly rather
than modelling noise as a function of the condition itself (Haque et al., 2021). The
nearest multi-condition noise-aware work, an ensemble of "negatively correlated noisy
learners" spanning depression, anorexia, self-harm, and suicide, treats noise as a source
of beneficial _ensemble diversity_ rather than as an estimable, condition-specific
property of the labels (Ragheb et al., 2023).

**Second, existing screening systems lack calibrated uncertainty and principled
abstention.** Modern neural classifiers are well documented to be overconfident, with predicted
probabilities that systematically exceed empirical accuracy (Guo et al.,
2017); this confidence is known to degrade further, with post-hoc calibration proving
insufficient, under distribution shift (Ovadia et al., 2019). Systematic reviews of the
field repeatedly cite a lack of external clinical validation as a central weakness
(Chancellor and De Choudhury, 2020), yet rarely quantify _how_ confidence and calibration
degrade when a system trained on social-media data is evaluated against real clinical
data. No existing multi-condition system combines calibrated, per-condition risk
estimation with a principled mechanism to abstain from prediction when confidence is
insufficient.

### 1.3.1 Problem Statement

Multi-condition mental-health screening systems are trained on proxy labels whose
reliability varies systematically by condition, yet existing systems treat this noise as
uniform or ignore it entirely; the same systems provide no calibrated, condition-aware
mechanism for declining to predict when uncertain, creating a risk of confidently
incorrect outputs in precisely the conditions, and the precise circumstances, where
errors are most consequential.

---

## 1.4 Motivation

The motivation for this research is the gap between how trustworthy mental-health
screening systems _appear_ and how trustworthy they _actually are_ (Chancellor and
De Choudhury, 2020). A system that silently absorbs structured, condition-dependent label
noise, and that cannot recognise or communicate its own uncertainty, risks systematically
under-serving exactly the conditions (bipolar disorder, schizophrenia) where
misclassification carries the greatest potential for harm (for example, mistaking a
bipolar depressive episode for unipolar depression and thereby contributing to an
inappropriate care pathway).

This research is further motivated by the observation that the machine-learning
subfields best equipped to address these weaknesses are mature, well-validated, and
computationally inexpensive, yet have barely been brought to bear on multi-condition
mental-health screening in a coupled, coherent way. Learning under class-conditional
label noise rests on a substantial theoretical foundation: from the unbiased-estimator
treatment of class-conditional noise (Natarajan et al., 2013), through transition-matrix
loss correction for deep networks (Patrini et al., 2017), to the model-agnostic joint
estimation of label errors in Confident Learning (Northcutt et al., 2021). Model
calibration is similarly well established, with temperature scaling and the Expected
Calibration Error (ECE) now standard tools (Guo et al., 2017), as is the study of how
that calibration fails under shift (Ovadia et al., 2019). Principled abstention is
supported by selective classification with guaranteed risk (Geifman and El-Yaniv, 2017),
distribution-free conformal prediction (Angelopoulos and Bates, 2023), and the
learning-to-defer framework that casts abstention as deferral to a human expert (Mozannar
and Sontag, 2020). There is a clear opportunity to apply this principled, already-trusted
machine-learning methodology to a real and openly acknowledged weakness in an applied
healthcare-adjacent domain, rather than to develop a new model architecture in isolation.

---

## 1.5 Existing Work

_(The table below presents the headline prior art that defines the research gap; the full
treatment, spanning the broader noisy-label, calibration, conformal-prediction, and
mental-health-NLP literatures, is given in Chapter 2. Verification status is noted per row;
items flagged in the chapter template note above still require a final pre-submission check.)_

| Citation                                                                                                                                                                                                                                                                                                                                                        | Summary                                                                                                                                                                                                                                                                                                                                                                          | Limitation                                                                                                                                                                                                                                                                                                                                                                              | Contribution                                                                                                                                                                                                                                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Northcutt, Jiang & Chuang (2021)**. _Confident Learning: Estimating Uncertainty in Dataset Labels_, JAIR 70, pp. 1373–1411, doi:10.1613/jair.1.12125. _(Verified: title, venue, year, DOI.)_                                                                                                                                                                  | Introduces Confident Learning, a model-agnostic framework that estimates the joint distribution between noisy observed labels and latent true labels from a model's out-of-sample predicted probabilities, enabling identification and pruning of label errors.                                                                                                                  | Assumes class-conditional noise; in its general form it does not encode _domain knowledge_ about why noise rates should differ across classes (e.g. clinical knowledge of self-identification reliability); it estimates noise structure from data alone, with no clinically-seeded prior.                                                                                              | Establishes the standard, theoretically grounded machinery for class-conditional noise-transition estimation that this project adapts and extends with a condition-dependent, clinically-seeded prior (C1).                                                                                                                |
| **Liu, Cheng & Zhang (2023)**. _Identifiability of Label Noise Transition Matrix_, ICML 2023, PMLR 202, pp. 21475–21496. _(Verified: title, venue, year.)_                                                                                                                                                                                                      | Formally analyses when a noise-transition matrix is identifiable, showing that a class-conditional **T** is **generally not identifiable from noisy labels alone** without additional structural assumptions, and characterising what extra information restores identifiability.                                                                                                | Largely theoretical; the identifiability conditions can be difficult to verify in practice and depend on representation quality.                                                                                                                                                                                                                                                        | The formal precedent that **anchors and disciplines this project's identifiability analysis** (C3): it implies the project's clinical seeding must supply precisely the "extra structure" identifiability requires, and frames RQ2 as a genuine open question rather than an assumed property.                             |
| **Haque, Reddi & Giallanza (2021)**. _Deep Learning for Suicide and Depression Identification with Unsupervised Label Correction_ ("SDCNL"), ICANN 2021 (Springer LNCS, _volume to confirm, see template note_), doi:10.1007/978-3-030-86383-8_35; arXiv:2102.09427. _(Verified: title, authors, year, DOI, arXiv ID; LNCS volume number pending final check.)_ | Proposes SDCNL, distinguishing **suicide risk versus depression** (a binary task, using r/depression and r/SuicideWatch) from noisy, web-scraped Reddit labels via an unsupervised label-correction method (dimensionality reduction plus clustering on transformer embeddings) that, by the authors' own design, does **not** require any prior noise-distribution information. | Binary task only (not extended to bipolar disorder, schizophrenia, or eating disorders). More significantly, the correction is **deliberately noise-distribution-agnostic**: it treats label noise as undifferentiated by design, with no mechanism to encode that different conditions carry structurally different self-identification reliability, and no calibration or abstention. | The **direct prior-art baseline** this project is evaluated against, and the precise point of contrast: where SDCNL deliberately avoids any noise-distribution prior, this project deliberately injects a clinically-seeded, condition-dependent one and couples it to calibrated abstention.                              |
| **Ragheb, Azé, Bringay & Servajean (2023)**. _Negatively Correlated Noisy Learners for At-Risk User Detection on Social Networks: A Study on Depression, Anorexia, Self-Harm, and Suicide_, IEEE TKDE 35(1), pp. 770–783, doi:10.1109/TKDE.2021.3085500. _(Verified: title, authors, venue, DOI.)_                                                              | Proposes NCNL, a deep ensemble of multiple "noisy" base learners trained with negative-correlation learning for early at-risk user detection across four conditions, backbone-independent and applicable to transformer encoders.                                                                                                                                                | "Noise" here denotes **ensemble regularisation/diversity**, not an estimated, identifiable, condition-dependent label-noise structure; provides **no** transition-matrix estimation, **no** calibration (ECE), and **no** conformal/selective abstention.                                                                                                                               | The **closest multi-condition, noise-embracing prior art**, and the second key point of contrast: it handles multiple conditions but does not model proxy-label noise as an estimable per-condition property, nor couple it to calibrated abstention (C1–C3).                                                              |
| **Guo, Pleiss, Sun & Weinberger (2017)**. _On Calibration of Modern Neural Networks_, ICML 2017, PMLR 70, pp. 1321–1330; arXiv:1706.04599. _(Verified: full citation.)_                                                                                                                                                                                         | Demonstrates that modern neural networks are poorly calibrated despite high accuracy, introduces/popularises the Expected Calibration Error (ECE), and shows temperature scaling is a simple, effective post-hoc fix.                                                                                                                                                            | Calibration is studied on standard image/vision benchmarks; the paper does not address a multi-condition, proxy-labelled, healthcare-adjacent setting, nor calibration under deliberate cross-source distribution shift; a single global temperature cannot repair _per-class_ miscalibration.                                                                                          | Provides the calibration metric (ECE) and correction method (temperature scaling) this project adopts and extends to a _per-condition_ form as the foundation of its calibrated-abstention contribution (C2).                                                                                                              |
| **Ovadia et al. (2019)**. _Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift_, NeurIPS 32; arXiv:1906.02530. _(Verified: title, venue, year, arXiv ID.)_                                                                                                                                                            | A large-scale benchmark of predictive-uncertainty methods under dataset shift, finding that traditional post-hoc calibration falls short as shift increases, while some model-averaging methods are more robust.                                                                                                                                                                 | Evaluated on vision and tabular tasks, not on NLP or a clinical mental-health setting.                                                                                                                                                                                                                                                                                                  | The **methodological template for this project's headline shift evaluation** (C3): it establishes both the expected phenomenon (calibration degrades under shift) and the metric protocol (ECE, Brier, NLL versus shift intensity).                                                                                        |
| **Angelopoulos & Bates (2023)**. _A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification_, Foundations and Trends in Machine Learning 16(4); arXiv:2107.07511. _(Verified: journal venue/volume and arXiv ID; page range pending check.)_                                                                               | Provides a distribution-free framework for producing prediction sets with a guaranteed marginal coverage rate, usable with any underlying model without retraining, plus extensions to shift and abstention.                                                                                                                                                                     | The general framework gives only _marginal_ (not class-conditional) coverage, relies on exchangeability that breaks under shift, and provides no noise-aware mechanism for setting thresholds; abstention is driven by model confidence, not label reliability.                                                                                                                         | Supplies the **formal coverage-guaranteed abstention mechanism** that this project couples to its condition-dependent noise estimate, rather than leaving abstention decoupled from label reliability (C2), with its exchangeability caveat directly relevant to C3.                                                       |
| **Wang, Zhang & Lim (2021)**. _Show or Suppress? Managing Input Uncertainty in Machine Learning Model Explanations_, Artificial Intelligence 294, 103456, doi:10.1016/j.artint.2021.103456; arXiv:2101.09498. _(Verified: now upgraded to the journal version; title, journal, volume, article number, DOI.)_                                                   | Frames _measured input features_ as carrying uncertainty and studies how that uncertainty should be reflected (shown or suppressed) in feature-attribution explanations, evaluated via simulation and human-subjects studies.                                                                                                                                                    | Addresses **measurement uncertainty in input features for explanation purposes**; does not address proxy _label_ uncertainty, is not applied to mental health, and does not connect uncertainty to a prediction-time abstention mechanism.                                                                                                                                              | The closest prior work treating "things the model is told" as uncertain rather than ground truth; establishes that the broad framing has precedent, which this project differentiates by applying it to _condition-dependent proxy labels_ with a _coupled abstention_ mechanism, in a domain this paper does not address. |

_Table 1.1: Table of Existing Work_

---

## 1.6 Research Gap

The reviewed literature establishes several things independently. Class-conditional label
noise can be formally modelled and corrected, via unbiased estimators (Natarajan et al.,
2013), transition-matrix loss correction (Patrini et al., 2017), and model-agnostic joint
estimation (Northcutt et al., 2021); and in the mental-health domain specifically, proxy-label
noise can be corrected without noise-distribution information (Haque et al., 2021) or
absorbed as ensemble diversity across conditions (Ragheb et al., 2023). Model calibration
can be formally measured and improved (Guo et al., 2017), though it degrades under
distribution shift (Ovadia et al., 2019). Abstention can be made statistically rigorous,
through selective classification (Geifman and El-Yaniv, 2017), conformal prediction
(Angelopoulos and Bates, 2023), or learning to defer (Mozannar and Sontag, 2020).
Separately, the identifiability literature shows that a class-conditional transition
matrix is generally _not_ identifiable from noisy labels alone without additional
structure (Liu, Cheng and Zhang, 2023; Xia et al., 2019).

However, no identified work combines these into a single system in which **(a)**
label-noise correction is modelled as _condition-dependent_, grounded in clinical
knowledge of differing self-identification reliability across disorders, and **(b)** the
resulting per-condition noise estimate _directly informs_ a calibrated abstention
threshold, so that the system becomes demonstrably more cautious exactly where its
training signal was least trustworthy. The two nearest mental-health efforts each occupy
one part of this space and stop short: SDCNL corrects noise but only for a single binary
pair and deliberately without a noise-distribution prior (Haque et al., 2021), while NCNL
spans multiple conditions but treats noise as ensemble regularisation rather than an
estimable per-condition property, with no calibration or abstention (Ragheb et al., 2023).
Nor does existing work formally characterise the conditions under which a _condition-dependent_
noise-transition matrix is identifiable for this setting, or test whether rarer, noisier
conditions fall outside those conditions in practice, a question made non-trivial precisely
because plain class-conditional identifiability is not guaranteed (Liu, Cheng and Zhang,
2023).

The gap, precisely stated: **no existing multi-condition mental-health screening system
models proxy-label noise as condition-dependent and clinically seeded, couples that
estimate into a calibrated, condition-aware abstention mechanism, and formally
characterises where that coupling can and cannot be trusted.**

---

## 1.7 Contribution to the Body of Knowledge

### 1.7.1 Contribution to the Problem Domain (Trustworthy Mental-Health Screening)

This research contributes to the applied problem of mental-health screening by
demonstrating a screening approach that is explicit and honest about the reliability of
its own training signal, rather than treating proxy labels as ground truth by default
(cf. Chancellor and De Choudhury, 2020). By distinguishing conditions whose
self-identification is reliable from those that are not, and by abstaining from prediction
when its supervision was least trustworthy, this work aims to reduce the risk of
confidently incorrect screening outcomes for exactly the conditions (bipolar disorder,
schizophrenia) most vulnerable to misclassification. This research also contributes to
the responsible-AI conversation in digital mental health by providing a quantified,
reproducible measurement of how screening reliability degrades when systems trained on
social-media data are evaluated against real clinical interview data (cf. Ovadia et al.,
2019; Gratch et al., 2014).

### 1.7.2 Contribution to the Research Domain (Trustworthy and Noise-Aware Machine Learning)

This research contributes to the machine-learning research domain by coupling two
previously independent lines of work, namely class-conditional noise modelling (Northcutt et al.,
2021; Patrini et al., 2017) and calibrated selective prediction (Guo et al., 2017;
Geifman and El-Yaniv, 2017; Angelopoulos and Bates, 2023), into a single mechanism in
which label-uncertainty propagates directly into prediction-time caution. It further
contributes an identifiability analysis appropriate to this coupling. Because a plain
class-conditional transition matrix is generally not identifiable from noisy labels alone
(Liu, Cheng and Zhang, 2023), this project's contribution is to state precisely what
additional structure its clinical seeding supplies, to characterise the conditions under
which the condition-dependent matrix is thereby identifiable, and to test empirically
where that identifiability breaks down for rare, high-noise conditions. Where those
conditions are not met, the contribution is honestly scoped down from exact
identifiability to high-variance estimation consistency, a distinction made explicit
rather than elided.

This combination (condition-dependent, clinically-seeded noise modelling, noise-coupled
calibrated abstention, and an identifiability analysis tested against its own theoretical
prediction) has not, to the best of a thorough literature search current to mid-2026 (and
subject to the re-verification noted in the chapter template note), been previously
assembled. It is intended to be of interest beyond the specific mental-health application,
to any domain in which proxy labels of varying reliability are used to supervise a model
expected to abstain responsibly.

---

## 1.8 Research Challenge

Several significant challenges are anticipated. First, ground-truth clinical validation
is available only for depression (via the DAIC-WoZ corpus; Gratch et al., 2014), whose
own construct validity has been questioned on two distinct grounds: its contested
relationship to the underlying depression construct (Patapati et al., 2025) and the
finding that many models inadvertently exploit the interviewer's prompts rather than
participant language, inflating reported performance (Burdisso et al., 2024). No
clinically validated anchor of comparable scale is available for bipolar disorder or
schizophrenia, so noise-rate estimates for those conditions must be partly evaluated
through sensitivity analysis rather than direct clinical ground truth, and the
Reddit-to-clinical evaluation must explicitly control for the prompt-leakage confound.

Second, the conditions for which condition-dependent noise modelling is most needed
(bipolar disorder, schizophrenia) are also, in the available data, the rarest: a verified
roughly 20:1 imbalance exists between depression and bipolar-disorder posts in the
combined dataset, meaning the noise-transition matrix may be estimated with high variance,
or be formally unidentifiable (Liu, Cheng and Zhang, 2023), for precisely the conditions
of greatest interest. This is not merely an engineering inconvenience but a direct
empirical test of the identifiability theory, and the project treats it as such.

Third, mid-project, the originally planned nine-condition SMHD corpus and its companion
RSDD dataset became permanently undistributable due to a change in Reddit's API terms of
service, requiring a substantial, verified replanning of the dataset pipeline (see Chapter
3). These challenges necessitate a research design that is explicit about its own limits:
a formal identifiability analysis, honest scoping of clinical claims to the conditions
that can actually be validated, and an evaluation protocol designed to produce a
defensible result whether or not the central hypotheses are confirmed.

---

## 1.9 Research Questions

**RQ1:** To what extent does proxy-label reliability for self-reported mental-health
conditions vary systematically by condition, and can this variation be estimated from
data using a condition-dependent, clinically-seeded noise-transition model?

**RQ2:** Under what conditions is a condition-dependent noise-transition matrix
identifiable, given that the class-conditional case is not identifiable from noisy labels
alone (Liu, Cheng and Zhang, 2023), and do empirically rare, high-noise conditions
(e.g. bipolar disorder, schizophrenia) fall outside those conditions in practice?

**RQ3:** Does coupling a condition-dependent noise estimate directly into a calibrated
abstention mechanism reduce selective risk, relative to calibrated abstention with a
uniform, noise-unaware threshold, particularly for high-noise conditions?

**RQ4:** How does the calibration and risk-coverage behaviour of the resulting system
degrade when evaluated under a distribution shift from social-media data to real
clinical interview data?

---

## 1.10 Research Aim

The aim of this research is to design, develop, and rigorously evaluate a multi-condition
mental-health screening system that models proxy-label noise as condition-dependent and
clinically seeded, couples that noise estimate into a calibrated, principled abstention
mechanism, and formally characterises, both theoretically and empirically, the limits of
that approach across conditions of differing data availability and label reliability.

The primary aim of this research is to investigate whether explicitly modelling
_which_ training labels deserve trust, and propagating that judgement into _when_ the
system is willing to answer, produces a measurably more trustworthy screening tool than
existing approaches that treat proxy labels as uniformly reliable and abstention as
decoupled from label quality. By focusing on the coupling between label-noise awareness
and calibrated abstention, this research intends to contribute both a practical
screening artefact and a transferable methodological pattern for trustworthy machine
learning under heterogeneous label reliability.

---

## 1.11 Research Objectives

| Research Objectives                 | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Research Questions |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| **Problem Definition**              | RO1: To identify a problem within trustworthy mental-health NLP by reading existing literature and confirming the gap is unaddressed.                                                                                                                                                                                                                                                                                                                                              | RQ1, RQ2           |
| **Literature Review**               | RO2: To identify existing approaches to label-noise correction, calibration, and abstention in mental-health and general ML literature. RO3: To identify the precise, narrow gap left unaddressed by this prior work.                                                                                                                                                                                                                                                              | RQ1, RQ2, RQ3, RQ4 |
| **Data Acquisition & Verification** | RO4: To identify, acquire, and directly verify (integrity, label correctness, scale) public datasets suitable for multi-condition proxy-labelled screening. RO5: To identify and, where feasible, acquire a clinically-labelled anchor dataset for validation.                                                                                                                                                                                                                     | RQ1, RQ4           |
| **Methodology / Design**            | RO6: To design a condition-dependent, clinically-seeded noise-transition model. RO7: To design a noise-coupled calibrated abstention mechanism. RO8: To derive the identifiability conditions for the condition-dependent noise-transition matrix, taking as the starting point that the plain class-conditional case is not identifiable from noisy labels alone (Liu, Cheng and Zhang, 2023), and to state precisely what additional structure the clinical seeding contributes. | RQ1, RQ2, RQ3      |
| **Implementation**                  | RO9: To implement the baseline multi-condition classifier, the noise-transition model, and the calibrated abstention layer as a reproducible pipeline.                                                                                                                                                                                                                                                                                                                             | RQ1, RQ3           |
| **Testing (Quantitative)**          | RO10: To design and execute a baseline-and-ablation evaluation isolating the contribution of condition-dependent noise modelling and noise-coupled abstention, reporting per-condition and pooled metrics.                                                                                                                                                                                                                                                                         | RQ3, RQ4           |
| **Evaluation**                      | RO11: To evaluate calibration and risk-coverage degradation under Reddit-to-clinical distribution shift, controlling for known clinical-corpus confounds such as interviewer-prompt leakage (Burdisso et al., 2024). RO12: To empirically test the theoretical identifiability prediction against observed estimation behaviour per condition.                                                                                                                                     | RQ2, RQ4           |
| **Documentation**                   | RO13: To provide adequate documentation of the steps, decisions, and verification performed throughout the research and development process, including an honest account of dataset and scope changes.                                                                                                                                                                                                                                                                             | N/A                |

_Table 1.2: Table of Research Objectives_

---

## 1.12 Chapter Summary

This chapter has introduced the research domain of multi-condition mental-health
screening from social media, and established that the field's reliance on unvalidated,
uniformly-trusted proxy labels (Chancellor and De Choudhury, 2020), combined with the
absence of calibrated, principled abstention, constitutes a significant and openly
acknowledged weakness. A review of existing work in label-noise correction (Natarajan et
al., 2013; Patrini et al., 2017; Northcutt et al., 2021; Haque et al., 2021; Ragheb et
al., 2023), model calibration (Guo et al., 2017; Ovadia et al., 2019), and
conformal/selective prediction (Geifman and El-Yaniv, 2017; Angelopoulos and Bates, 2023),
alongside the identifiability literature (Liu, Cheng and Zhang, 2023), identified a
precise, narrow gap: no existing system models proxy-label noise as condition-dependent
and clinically seeded, couples that estimate into calibrated abstention, and formally
characterises where that coupling can be trusted. The research questions, aim, and
objectives presented in this chapter are designed to address that gap directly, while
remaining honest about the data-availability and identifiability challenges the project
has already encountered and accounted for. The following chapter presents a comprehensive
literature review situating this work within the broader fields of noisy-label learning,
model calibration, and trustworthy NLP for mental health.

---

## References (Chapter 1)

Angelopoulos, A.N. and Bates, S. (2023) 'A gentle introduction to conformal prediction and distribution-free uncertainty quantification', _Foundations and Trends in Machine Learning_, 16(4). Available at: https://arxiv.org/abs/2107.07511.

Burdisso, S., Villatoro-Tello, E., Madikeri, S. and Motlicek, P. (2024) 'DAIC-WOZ: on the validity of using the therapist's prompts in automatic depression detection from clinical interviews', in _Findings of the Association for Computational Linguistics: ACL 2024_. Available at: https://arxiv.org/abs/2404.14463.

Chancellor, S. and De Choudhury, M. (2020) 'Methods in predictive techniques for mental health status on social media: a critical review', _npj Digital Medicine_, 3, 43. doi:10.1038/s41746-020-0233-7.

Coppersmith, G., Dredze, M., Harman, C. and Hollingshead, K. (2015) 'From ADHD to SAD: analyzing the language of mental health on Twitter through self-reported diagnoses', in _Proceedings of the 2nd Workshop on Computational Linguistics and Clinical Psychology (CLPsych)_, pp. 1–10. doi:10.3115/v1/W15-1201.

Ernala, S.K., Birnbaum, M.L., Candan, K.A., Rizvi, A.F., Sterling, W.A., Kane, J.M. and De Choudhury, M. (2019) 'Methodological gaps in predicting mental health states from social media: triangulating diagnostic signals', in _Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems_. doi:10.1145/3290605.3300364. _(Carried from author's prior reading; outside the verified survey batch; confirm before final submission.)_

Geifman, Y. and El-Yaniv, R. (2017) 'Selective classification for deep neural networks', in _Advances in Neural Information Processing Systems 30 (NeurIPS)_. Available at: https://arxiv.org/abs/1705.08500.

Gratch, J., Artstein, R., Lucas, G., Stratou, G., Scherer, S., Nazarian, A., Wood, R., Boberg, J., DeVault, D., Marsella, S., Traum, D., Rizzo, S. and Morency, L.-P. (2014) 'The Distress Analysis Interview Corpus of human and computer interviews', in _Proceedings of the 9th International Conference on Language Resources and Evaluation (LREC)_, pp. 3123–3128.

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017) 'On calibration of modern neural networks', in _Proceedings of the 34th International Conference on Machine Learning (ICML)_, PMLR 70, pp. 1321–1330. Available at: https://arxiv.org/abs/1706.04599.

Haque, A., Reddi, V. and Giallanza, T. (2021) 'Deep learning for suicide and depression identification with unsupervised label correction', in _Artificial Neural Networks and Machine Learning – ICANN 2021_. Lecture Notes in Computer Science. Cham: Springer, pp. 436–447. doi:10.1007/978-3-030-86383-8_35. Available at: https://arxiv.org/abs/2102.09427. _(LNCS volume number pending final confirmation; see template note.)_

Harrigian, K. and Dredze, M. (2022) _(Carried from author's prior reading; full bibliographic details outside the verified survey batch; confirm exact title, venue, and year before final submission.)_

Ji, S., Li, X., Huang, Z. and Cambria, E. (2021) 'Suicidal ideation and mental disorder detection with attentive relation networks', _Neural Computing and Applications_, 34, pp. 10309–10319. doi:10.1007/s00521-021-06208-y.

Ji, S., Zhang, T., Ansari, L., Fu, J., Tiwari, P. and Cambria, E. (2022) 'MentalBERT: publicly available pretrained language models for mental healthcare', in _Proceedings of the 13th Language Resources and Evaluation Conference (LREC)_, pp. 7184–7190. Available at: https://arxiv.org/abs/2110.15621.

Liu, Y., Cheng, H. and Zhang, K. (2023) 'Identifiability of label noise transition matrix', in _Proceedings of the 40th International Conference on Machine Learning (ICML)_, PMLR 202, pp. 21475–21496.

Low, D.M., Rumker, L., Talkar, T., Torous, J., Cecchi, G. and Ghosh, S.S. (2020) 'Natural language processing reveals vulnerable mental health support groups and heightened health anxiety on Reddit during COVID-19: observational study', _Journal of Medical Internet Research_, 22(10), e22635. doi:10.2196/22635.

Mitchell, M., Hollingshead, K. and Coppersmith, G. (2015) 'Quantifying the language of schizophrenia in social media', in _Proceedings of the 2nd Workshop on Computational Linguistics and Clinical Psychology (CLPsych)_, pp. 11–20.

Mozannar, H. and Sontag, D. (2020) 'Consistent estimators for learning to defer to an expert', in _Proceedings of the 37th International Conference on Machine Learning (ICML)_, PMLR 119, pp. 7076–7087. Available at: https://arxiv.org/abs/2006.01862.

Natarajan, N., Dhillon, I.S., Ravikumar, P. and Tewari, A. (2013) 'Learning with noisy labels', in _Advances in Neural Information Processing Systems 26 (NeurIPS)_, pp. 1196–1204.

Northcutt, C.G., Jiang, L. and Chuang, I.L. (2021) 'Confident learning: estimating uncertainty in dataset labels', _Journal of Artificial Intelligence Research_, 70, pp. 1373–1411. doi:10.1613/jair.1.12125.

Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J.V., Lakshminarayanan, B. and Snoek, J. (2019) 'Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift', in _Advances in Neural Information Processing Systems 32 (NeurIPS)_. Available at: https://arxiv.org/abs/1906.02530.

Patapati, [initials to confirm] et al. (2025) _(DAIC-WoZ construct-validity critique, ICMI 2025; carried from project planning; confirm full author list, title, and pages before final submission.)_

Patrini, G., Rozza, A., Menon, A.K., Nock, R. and Qu, L. (2017) 'Making deep neural networks robust to label noise: a loss correction approach', in _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, pp. 2233–2241. doi:10.1109/CVPR.2017.240.

Ragheb, W., Azé, J., Bringay, S. and Servajean, M. (2023) 'Negatively correlated noisy learners for at-risk user detection on social networks: a study on depression, anorexia, self-harm, and suicide', _IEEE Transactions on Knowledge and Data Engineering_, 35(1), pp. 770–783. doi:10.1109/TKDE.2021.3085500.

Wang, D., Zhang, W. and Lim, B.Y. (2021) 'Show or suppress? Managing input uncertainty in machine learning model explanations', _Artificial Intelligence_, 294, 103456. doi:10.1016/j.artint.2021.103456. Available at: https://arxiv.org/abs/2101.09498.

Xia, X., Liu, T., Wang, N., Han, B., Gong, C., Niu, G. and Sugiyama, M. (2019) 'Are anchor points really indispensable in label-noise learning?', in _Advances in Neural Information Processing Systems 32 (NeurIPS)_, pp. 6835–6846.
