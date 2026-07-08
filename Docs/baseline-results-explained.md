# Baseline results, explained from scratch

_A plain-English walkthrough of what the MentalBERT baseline produced and what every
number means. No prior ML vocabulary assumed — each technical term is defined the first
time it appears, then used normally. Companion to the factual record in
[`baseline-results.md`](baseline-results.md)._

---

## 1. What we actually did (the 10,000-ft view)

We took **MentalBERT** — a language model that has already read a huge amount of
mental-health-related text and "understands" how such text tends to read — and we
**fine-tuned** it. Fine-tuning means: start from that pre-trained model and keep
training it a little more on *our* specific task, so it adapts from "general
understanding of text" to "sorting posts into our conditions."

Our specific task is **multi-class classification**: given the text of one Reddit post,
predict **which one** of several mental-health conditions the post belongs to. "Multi-
class" just means there are more than two possible answers (here: *bipolar*,
*depression*, *eating_disorder*, *schizophrenia*). Each post gets exactly one predicted
label.

Where do the "true" labels come from? From the **subreddit** the post was written in
(e.g. a post from r/bipolar is labelled `bipolar`). This is called a **proxy label** —
a stand-in for a real clinical diagnosis. It is *not* a doctor's diagnosis; it's a
convenient approximation. Remember this — it matters a lot later (§9).

---

## 2. Train / validation / test — why we split the data

You cannot fairly judge a student using the exact questions they studied. Same for a
model. So we split our ~140,000 posts into three piles:

- **Training set** (`train`, 111,892 posts) — the model *learns* from these.
- **Validation set** (`val`, 13,967 posts) — a "practice exam" checked during training
  to watch progress and catch problems.
- **Test set** (`test`, 14,039 posts) — the **final exam**. The model never trains on
  these. Every headline number below is measured on the test set, so it reflects
  performance on posts the model has genuinely never seen.

One subtlety we enforced: **author-grouped splitting**. All posts by a given author go
into *one* pile only. If the same person appeared in both train and test, the model
could "recognise the author" instead of learning the condition — that's called
**leakage**, and it makes scores look better than they really are. We prevented it.

---

## 3. The single most important idea: class imbalance

Count how many test posts belong to each condition (this count is called the
**support**):

| Condition | Test posts (support) | Share |
|---|---|---|
| depression | 11,351 | **80.9%** |
| eating_disorder | 1,421 | 10.1% |
| schizophrenia | 786 | 5.6% |
| bipolar | 481 | 3.4% |

Depression is **~24× more common than bipolar**. This is called **class imbalance**,
and it's the reason we can't just trust a single overall score. Here's the trap:

> A lazy model that **ignores the text and always guesses "depression"** would still be
> right 80.9% of the time — just because depression is so common.

So "right 80.9% of the time" can mean *"learned nothing, always says depression."* We
need scores that reward getting the **rare** classes right too. That's what the rest of
this document builds toward.

---

## 4. Accuracy (and why it's not enough)

**Accuracy** = (posts predicted correctly) ÷ (all posts). It's the most intuitive score.

Our model's test accuracy is **0.9605** — it labels **96.05%** of test posts correctly.

That sounds great, and it *is* good — but from §3 you know the "always depression"
cheat already scores ~80.9%. So accuracy alone can't tell us whether the model actually
handles the rare conditions. We need to look **per condition**. Enter the confusion
matrix.

---

## 5. The confusion matrix — the raw truth table

A **confusion matrix** is a grid that shows, for every true condition, what the model
*actually predicted*. **Rows = the truth. Columns = the model's guess.** Here is ours
(test set):

```
                      pred_bipolar  pred_depression  pred_eating_disorder  pred_schizophrenia
true_bipolar                   342              116                     4                  19
true_depression                 57            11192                    55                  47
true_eating_disorder             4               79                  1334                   5
true_schizophrenia              27              133                     9                 617
```

How to read it:

- **The diagonal (top-left to bottom-right) is the "got it right" cells.** E.g.
  `true_bipolar → pred_bipolar = 342`: 342 bipolar posts were correctly called bipolar.
- **Everything off the diagonal is a mistake.** Read a mistake as "true X, but predicted
  Y." E.g. `true_bipolar → pred_depression = 116`: 116 posts that were really from
  bipolar got mislabelled as depression.
- **Each row adds up to that condition's support.** Bipolar row: 342+116+4+19 = **481**
  (matches the table in §3).

Just by eye, one pattern jumps out: **the big off-diagonal numbers are almost all in the
`pred_depression` column** (116, 55, 79... wait, 133). That is, when the model is wrong,
it usually wrongly says *depression*. We'll quantify that next — it's the heart of the
result.

---

## 6. Precision, Recall, and F1 — scoring each condition separately

For any single condition, three numbers describe how well the model did. Let's use
**bipolar** as the running example (from its row/column in the matrix above).

### Recall — "of the real ones, how many did we catch?"
**Recall** = (correctly found) ÷ (all that truly were this class).
For bipolar: 342 caught ÷ 481 truly bipolar = **0.711**, i.e. **71.1%**.
> Meaning: of every 100 genuinely-bipolar posts, the model correctly flagged ~71 and
> **missed ~29** (mostly calling them depression). Low recall = "misses a lot."

### Precision — "when we say bipolar, how often are we right?"
**Precision** = (correctly found) ÷ (everything we *labelled* this class).
The model predicted "bipolar" for 342+57+3+27 = 429 posts total (that's the
`pred_bipolar` **column** summed). Of those, 342 were truly bipolar.
So precision = 342 ÷ 429 = **0.797**, i.e. **79.7%**.
> Meaning: when the model shouts "bipolar!", it's correct ~80% of the time and
> **falsely accuses ~20%** (those were really other conditions). Low precision = "cries
> wolf."

### Why you need both
Precision and recall pull against each other. A model could get 100% precision by only
ever predicting bipolar on the one post it's absolutely sure about (never wrong, but
misses everyone else → terrible recall). Or 100% recall by labelling *everything*
bipolar (catches all real ones, but is wrong constantly → terrible precision). You need
a single number that's only high when **both** are high.

### F1 — the balanced combination
**F1** is the **harmonic mean** of precision and recall — a special kind of average that
stays low unless *both* inputs are high (it punishes imbalance between them).

$$F1 = 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}$$

For bipolar: 2 × (0.797 × 0.711) ÷ (0.797 + 0.711) = **0.752**.

**F1 is the go-to per-class score.** Higher = better; 1.0 is perfect. Here are all four:

| Condition | Precision | Recall | **F1** | Support | Plain reading |
|---|---|---|---|---|---|
| depression | 0.972 | 0.986 | **0.979** | 11,351 | Nearly perfect — easy & abundant |
| eating_disorder | 0.952 | 0.939 | **0.945** | 1,421 | Very strong |
| schizophrenia | 0.897 | 0.785 | **0.837** | 786 | Good, but misses ~21% |
| **bipolar** | 0.797 | 0.711 | **0.752** | 481 | **Weakest** — misses ~29% |

---

## 7. "Macro" vs "weighted" — averaging the four F1s into one number

You asked specifically about **macro**. It's a *way of averaging* the per-class scores
into a single summary. There are two common ways, and the choice completely changes the
story under imbalance:

### Macro average — treat every condition as equally important
**Macro-F1** = the plain average of the four F1 scores, giving **each condition equal
weight regardless of how many posts it has**:

$$\text{macro-F1} = \frac{0.979 + 0.945 + 0.837 + 0.752}{4} = \textbf{0.878}$$

Bipolar (481 posts) counts just as much as depression (11,351 posts). So macro-F1 is
**harsh about rare classes** — if the model neglects bipolar, macro-F1 drops a lot even
if depression is perfect. This is exactly why we use it as our **headline number**: it
refuses to let the model hide behind the easy majority class.

### Weighted average — bigger classes count more
**Weighted-F1** averages the F1s but weights each by its support (post count). Depression
(80.9% of posts) dominates, so weighted-F1 = **0.960**, very close to depression's own
0.979. It mostly reflects the majority class.

### The contrast that proves the point
- Weighted-F1 = 0.960 (flattered by the easy majority)
- **Macro-F1 = 0.878** (honest about the hard minorities)
- The "always depression" cheat from §3 would score **macro-F1 ≈ 0.22** (depression
  F1 ≈ 0.89, the other three = 0).

Our macro-F1 of **0.878 ≫ 0.22** is the real evidence that the model **genuinely learned
the minority conditions** instead of collapsing to "always depression." That's the thing
we most needed to confirm, and it passed.

**Macro = "average that treats all classes equally." Weighted = "average that favours
common classes." We report macro because we care about the rare conditions.**

---

## 8. Reading the story the numbers tell

Putting §5–§7 together, one clear narrative emerges:

1. **Depression is near-perfect (F1 0.979)** and everything else tends to get confused
   *with depression*. Look back at the confusion matrix column `pred_depression`: 116
   bipolar, 133 schizophrenia, 79 eating-disorder posts all leaked into it. Depression
   is both the **majority** class and clinically a kind of **catch-all** — many
   conditions involve depressive language.
2. **Bipolar is the hardest (F1 0.752, recall 0.711).** ~1 in 4 bipolar posts get called
   depression. This is *clinically sensible*: bipolar disorder includes depressive
   episodes that read almost identically to plain depression. The model isn't being
   dumb — the categories genuinely overlap in text.
3. **Schizophrenia is next-hardest (F1 0.837).** Same pattern, smaller.

This asymmetric confusion — **specific minority classes systematically bleeding into
depression** — is not just a curiosity. It is the exact phenomenon the *next* phase of
the project (the "condition-dependent noise model") is designed to measure and correct.
In other words, this baseline's mistakes are the raw material for the real research.

---

## 9. Three honest caveats (so you don't over-claim this)

1. **These are proxy labels, not diagnoses.** "Correct" here means "matched the
   subreddit the post came from," not "matched a doctor." A high score partly reflects
   that different subreddits use distinctive vocabulary, which is easier than real
   diagnosis. So **0.878 macro-F1 is a strong *reference point*, not a claim of clinical
   accuracy.**
2. **This is "in-distribution."** Train and test are both Reddit posts from the same
   world. The genuinely hard test — coming later — is the **shift** to real clinical
   interview data (DAIC-WoZ), which reads very differently. Performance there will
   (expectedly) be lower; measuring that drop is a core goal of the project.
3. **Only 4 of 6 conditions are present.** *anxiety* and *suicidality* have no data on
   disk yet, so they weren't trained or tested. The pipeline already supports them; they
   just need data.

---

## 10. Bonus finding: 1 epoch was as good as 3

An **epoch** = one full pass of the model through the entire training set. We ran a quick
**1-epoch** version and the full **3-epoch** version. The scores were essentially
identical (macro-F1 0.880 vs 0.878 — the tiny difference is noise).

Why does that matter? Normally more epochs = more learning. Here, extra passes **did not
help**. The interpretation: the model already extracted everything the *proxy labels* can
teach in a single pass. The ceiling on performance isn't the model's effort or size —
it's the **quality (noisiness) of the labels themselves**. To push past this ceiling you
have to *model the label noise*, not train longer. That is precisely the motivation for
the next research phase.

---

## 11. One-paragraph summary

We fine-tuned MentalBERT to sort Reddit posts into four mental-health conditions and
tested it on 14,039 unseen posts. It scored **96% accuracy**, but because depression is
~81% of the data, the fairer score is **macro-F1 = 0.878** — an average of each
condition's F1 (a balanced precision/recall score) that weights all conditions equally.
That number confirms the model really learned the rare conditions, not just the common
one. It's strongest on depression (F1 0.98) and weakest on bipolar (F1 0.75), and its
mistakes overwhelmingly involve confusing other conditions *with* depression — a
clinically sensible, systematic pattern that the project's next phase will explicitly
model. Caveat: these are subreddit-based *proxy* labels on Reddit text, so this is a
reference baseline, not a measure of clinical diagnostic accuracy.
