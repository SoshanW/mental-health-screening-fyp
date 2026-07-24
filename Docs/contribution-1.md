# Mental Health NLP FYP – Label Noise, Noise Matrix Estimation, and the Motivation for a Clinically-Informed Approach

---

# Table of Contents

1. Introduction
2. Original Research Structure
3. The Noise Transition Matrix
4. Zhu, Song & Liu (2021): Theoretical Identifiability
5. Zhu, Wang & Liu (2022): Practical Failure on BERT Text
6. Why Piece 1 Changed
7. The Three Routes to Obtaining a Noise Matrix
   - Route 1 – Clean Data
   - Route 2 – Anchor Points
   - Route 3 – Clusterability (HOC)
8. Understanding Embeddings
9. Understanding HOC (Higher-Order Consensus)
10. Why HOC Fails in Mental Health
11. HOC's Informative Matrix Assumption
12. The Evidence Against Existing Estimators
13. The Final Research Narrative

---

# 1. Introduction

The core research problem is:

> Mental-health NLP datasets generally use proxy labels (e.g., subreddit membership or self-reported diagnoses) rather than clinically verified diagnoses.

These proxy labels are noisy.

Many modern methods require knowledge of the label-noise transition matrix:

\[
T_{ij}=P(\text{Observed Label}=j \mid \text{True Label}=i)
\]

Estimating this matrix is therefore a central challenge.

---

# 2. Original Research Structure

Originally the project was organised into two pieces.

## Piece 1

Estimate the label-noise transition matrix using Confident Learning, optionally strengthened with a clinically informed prior.

## Piece 2

Use that estimated matrix to build a clinically informed, calibrated, abstaining mental-health screening model.

The implicit argument behind Piece 1 was:

> "A clinical prior is required because the data alone cannot estimate the transition matrix."

The literature later showed this argument is incomplete.

---

# 3. The Noise Transition Matrix

Suppose we have two conditions:

- Depression
- Bipolar

The transition matrix might look like

\[
T=
\begin{bmatrix}
0.90 & 0.10\\
0.25 & 0.75
\end{bmatrix}
\]

Meaning:

True depression:

- 90% appear as depression
- 10% appear as bipolar

True bipolar:

- 75% appear as bipolar
- 25% appear as depression

Every noise-aware learning method ultimately requires this matrix.

---

# 4. Zhu, Song & Liu (2021)

## What they proved

They showed that the transition matrix can be identifiable from noisy labels alone.

No clean labels are theoretically required.

No anchor points are theoretically required.

No expert priors are theoretically required.

They prove uniqueness under a set of assumptions.

---

## Important distinction

They prove:

> The matrix can theoretically be recovered.

They do NOT prove:

> Existing estimators recover it accurately on modern NLP datasets.

This is the difference between

**Identifiability**

and

**Practical estimation quality.**

---

# 5. Zhu, Wang & Liu (2022)

The follow-up paper asked:

> Even if the matrix is theoretically identifiable, do existing estimators recover it accurately?

They evaluated:

- Higher-Order Consensus (HOC)
- Confident Learning
- T-Revision
- Their proposed estimator

on BERT-based text.

The answer was largely:

No.

---

## AG's News

Properties:

- 4 classes
- BERT embeddings
- 120k documents

Very similar to the FYP setting.

Noise level:

≈0.178

HOC estimation error:

≈0.133

Therefore

0.133 / 0.178 ≈ 75%

The estimation error is roughly 75% as large as the quantity being estimated.

This is already poor.

---

## Jigsaw

Properties:

- Binary
- Highly imbalanced

Noise:

≈11%

HOC error:

14.25

Identity estimator ("assume no noise"):

11.1

Meaning:

Doing nothing performs better than HOC.

---

Confident Learning:

20.17

At this noise level for binary classification, this is approximately equivalent to random guessing.

---

## Why this matters

The original Piece 1 relied on Confident Learning.

The literature shows that Confident Learning performs poorly on BERT text.

Therefore:

Piece 1 can no longer simply say

> "We estimate the matrix."

---

# 6. Why Piece 1 Changed

Originally:

```
Estimate matrix
↓

Use matrix
```

Now:

```
Evaluate existing estimators
↓

Demonstrate failure
↓

Motivate new method
↓

Introduce clinically-informed approach
```

Piece 1 becomes diagnostic rather than algorithmic.

Instead of assuming existing estimators fail,

you experimentally show whether they fail on mental-health data.

That evidence then motivates Piece 2.

This is scientifically much stronger.

---

# 7. The Three Routes to Obtaining a Noise Matrix

Every published method obtains the transition matrix through one of three routes.

---

## Route 1 — Clean Data

### Idea

Use examples where the true diagnosis is known.

Example:

| Proxy Label | Clinical Diagnosis |
|-------------|--------------------|
| Depression | Depression |
| Depression | Bipolar |

From this we directly estimate

\[
P(\text{Observed}|\text{True})
\]

### Why it works

The clinical diagnosis provides the ground truth.

### Why it closes

Most mental-health datasets only contain proxy labels.

Examples:

- Reddit
- SWMH
- SMHD

DAIC-WoZ is clinically labelled but only for depression.

It cannot estimate a full multi-condition matrix.

Therefore:

Route 1 closes.

---

## Route 2 — Anchor Points

### Idea

Find examples that almost certainly belong to one class.

Mathematically,

\[
P(Y=c|x)\approx1
\]

Example:

An unmistakable handwritten digit "7".

These anchor examples reveal the noise process.

---

### Why it closes

Mental-health conditions share symptoms.

Example:

"I cannot sleep."

Could indicate

- depression
- bipolar
- anxiety

"I feel hopeless."

Could occur in

- depression
- bipolar depressive episodes

Therefore there are very few pure anchor points.

Route 2 closes.

---

## Route 3 — Clusterability (HOC)

This is the most complicated route.

It relies on embeddings.

---

# 8. Understanding Embeddings

Computers cannot understand language directly.

Everything must become numbers.

---

## One-hot encoding

Originally,

words became vectors like

```
happy → [1,0,0]

sad → [0,1,0]
```

Problem:

The computer thinks every word is unrelated.

---

## Word2Vec

Researchers instead learned vectors automatically.

Words appearing in similar contexts obtain similar vectors.

Example:

```
happy

[2.1,1.0,-0.3]

joyful

[2.2,0.9,-0.4]
```

These vectors become close together.

This is based on the idea

> "You shall know a word by the company it keeps."

---

## BERT

Word2Vec gives one vector per word.

BERT gives one vector per word **in context.**

Example:

```
bank
```

Money sentence:

One vector.

River sentence:

Different vector.

BERT therefore produces contextual embeddings.

---

## Sentence embeddings

Eventually the entire sentence becomes one vector.

Example

"I feel hopeless today."

↓

```
[0.84,
-1.13,
0.52,
...
]
```

Typically this vector has 768 dimensions.

Each Reddit post therefore becomes one point in a very high-dimensional space.

---

## MentalBERT

MentalBERT is BERT further pre-trained on mental-health language.

It produces embeddings specialised for mental-health text.

---

# 9. Understanding HOC (Higher-Order Consensus)

HOC estimates the transition matrix using nearest neighbours.

---

## Nearest neighbours

Imagine every Reddit post is a point.

```
      ●

   ●     ●

      X

  ●         ●
```

The closest points are the nearest neighbours.

---

## HOC assumption

HOC assumes

> A post's nearest neighbours share its true class.

Example

If a post is surrounded by depression posts,

HOC assumes it is probably truly depression.

It repeats this over many examples to infer the transition matrix.

---

# 10. Why HOC Fails in Mental Health

This is the key insight.

The problem is NOT simply that embeddings are imperfect.

The problem is clinical.

---

Suppose someone truly has bipolar disorder.

During a depressive episode they write

> "I cannot get out of bed."

> "Nothing matters anymore."

Their language is genuinely similar to depression.

MentalBERT therefore places them close to depression posts.

Their nearest neighbours become depression posts.

Their subreddit may also be

r/depression.

Therefore

similar language

≠

same diagnosis.

This violates HOC's clusterability assumption.

---

## The elegant argument

> The reason the clusterability condition fails is the same clinical fact that creates the label noise in the first place.

A truly bipolar person in a depressive phase writes like a depressed person.

Therefore

- nearest neighbours are depression posts
- subreddit choice is often depression

The estimator's assumption

and

the phenomenon under study

are in direct conflict.

This is why mental-health NLP requires a different route to estimating the transition matrix.

---

# 11. HOC's Informative Matrix Assumption

HOC also requires an informative transition matrix.

Meaning

every diagonal entry must be the largest.

Example

Good case

\[
\begin{bmatrix}
0.90&0.10\\
0.30&0.70
\end{bmatrix}
\]

True bipolar

↓

Most still appear as bipolar.

---

Suppose diagnostic delay becomes severe.

\[
\begin{bmatrix}
0.90&0.10\\
0.70&0.30
\end{bmatrix}
\]

Now

True bipolar

↓

Observed depression

is more common.

The diagonal is no longer dominant.

HOC's uniqueness theorem no longer applies.

A clinically elicited matrix can represent such a row.

HOC cannot identify one.

---

# 12. The Evidence Against Existing Estimators

One important point:

The researchers know the true matrix.

Why?

Because they synthetically inject noise.

They therefore know

T

exactly.

They compare

Estimated matrix

vs

True matrix.

That is how they calculate estimation error.

---

In your project,

the true matrix is unknown.

There are no clinician labels.

Therefore

even if HOC estimates a matrix,

you cannot measure how wrong it is.

You never know

whether the estimate is

- excellent,
- mediocre,
- completely random.

This makes blindly trusting existing estimators very risky.

---

# 13. Final Research Narrative

The strongest research story becomes

## Piece 1

Evaluate existing transition-matrix estimators on proxy-labelled mental-health text.

Measure:

- stability
- agreement
- calibration
- downstream performance

Determine whether existing estimators are trustworthy.

---

## Piece 2

Introduce a clinically informed route to estimating or constraining the transition matrix.

The motivation is evidence-based rather than assumption-based.

---

# Final Conclusion

The project no longer argues:

> "We need a clinical prior because estimation is impossible."

Instead it argues:

1. Existing theory shows the matrix is identifiable under certain assumptions.

2. Practical studies show existing estimators perform poorly on BERT text.

3. Mental-health data violates key assumptions because symptom overlap, diagnostic delay, and phase-dependent expression create clinically meaningful language overlap.

4. The same clinical phenomena that generate proxy-label noise also invalidate embedding-based clusterability assumptions.

5. Existing estimators therefore cannot simply be trusted in this domain.

6. A clinically informed route to the transition matrix is consequently well motivated.

---

# The Central Insight

The single most important sentence from these notes is:

> **"The reason the clusterability condition fails is the same clinical fact that creates the label noise in the first place. A truly bipolar person in a depressive phase writes like a depressed person, so their nearest neighbours in embedding space are depression posts, and their subreddit choice is often r/depression too. The estimator's assumption and the phenomenon under study are in direct conflict. That is why this domain needs a different route to the matrix, and why the clinical literature is the natural place to find it."**