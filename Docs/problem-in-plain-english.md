# The Problem: Why Mental-Health Screening on Social Media Can't Be Trusted Yet

_A plain-language explanation of the problem this final-year project addresses._

> **Status:** This project's direction has been reviewed and confirmed by my supervisor. The problem framing below is the agreed basis for the work.

---

## Start with the everyday picture

For years, researchers have built systems that read people's social-media posts and try to
guess whether the person shows signs of a mental-health condition — depression, bipolar
disorder, schizophrenia, an eating disorder, and so on. The idea sounds useful: if a
computer could flag people who might be struggling, maybe help could reach them sooner.

These systems are built using machine learning, which means they learn by example. You
show the system thousands of posts that are "labelled" with the right answer, and it
gradually learns the patterns. So the entire thing depends on one question: **where do
those labels come from, and are they actually correct?**

This is where the whole field has a quiet, shared weakness.

## The labels are guesses dressed up as facts

When a dataset says a post is from someone "with depression," what it almost always really
means is: _this post was written in the depression section of Reddit_ (a "subreddit"). The
label isn't a doctor's diagnosis. It's just which online group the post came from.

That's a much weaker thing than it sounds. Someone posting in the bipolar subreddit might
be:

- a person who actually has bipolar disorder,
- a worried friend or partner of someone who has it,
- someone who _thinks_ they have it but has never been diagnosed,
- someone who will be diagnosed years later, or never at all.

So the "label" is really a **proxy** — a stand-in for the truth, not the truth itself. The
field knows this. It's even admitted in the footnotes of major papers, where researchers
acknowledge their systems really distinguish "people likely to be diagnosed" from "the
general population," rather than genuinely sick from genuinely healthy. But most systems
are still built as if the labels were rock-solid facts. They quietly absorb all the
mistakes baked into those labels and carry them forward.

## The crucial twist: the guesses are worse for some conditions than others

Here's the insight at the heart of this project, and it's one the field has mostly ignored.

The "guess" isn't equally unreliable across all conditions.

If someone says "I have depression," that's usually a fairly safe bet. Depression is
common, widely understood, and people are generally accurate when they describe it.

But "I have bipolar disorder" or "I have schizophrenia"? Far shakier. These conditions are
talked about constantly online, frequently confused with other things, often
self-diagnosed incorrectly, and carry more stigma. So the labels for these conditions are
_noisier_ — less trustworthy — than the labels for depression.

This isn't just a hunch. When I downloaded and inspected the actual dataset for this
project, the imbalance was stark: there were about **117,000 depression posts but only
about 5,800 bipolar posts** — roughly twenty times more depression data. Depression
dominates these platforms. The rarer conditions are exactly the ones with less data _and_
shakier labels — a double problem.

(One real complication along the way: the dataset originally planned for this project,
which covered nine conditions, turned out to be permanently unavailable partway through
planning, because of a change to Reddit's data-access rules. A replacement dataset was
found, checked by hand post by post, and confirmed to cover the same conditions at a
comparable scale — including the exact figures above. The companion "Solution" document
covers this in more detail.)

And today's systems are blind to all of this. They treat a shaky bipolar label and a solid
depression label exactly the same way. A system trained like this doesn't just make
mistakes — it makes _structured_ mistakes that are worse for precisely the conditions where
getting it wrong matters most.

## The second half of the problem: false confidence

Even setting the labels aside, there's a separate danger.

When one of these systems makes a prediction, it doesn't tell you how unsure it is. Ask it
"does this person have depression?" and it answers something like "87% yes" — and it gives
that confident-sounding number even when the post it's looking at is nothing like anything
it was trained on. The system has no honest sense of its own uncertainty.

Researchers have a name for this failure: the systems are **overconfident**, and their
confidence falls apart the moment they're shown data that's even slightly different from
their training data. This is why review after review of the field complains about "lack of
external validation" — systems look great on the Reddit data they were trained on, then
collapse on real clinical data, and they collapse _silently_, with no warning, because
they never learned to say "I'm not sure."

In a real setting, this is genuinely dangerous. Mistaking a bipolar depressive episode for
ordinary depression, with full confidence, could push someone toward the wrong kind of
care. A tool meant to help people needs to know when to step back and say "this one needs
a human," and almost none of them can.

## Putting the two halves together

So the problem this project tackles has two interlocking parts:

1. **The training labels are noisy in a structured, condition-dependent way** — and no
   existing system models that structure. They treat unreliable labels as if they were
   facts, and treat all conditions as equally reliable when they aren't.

2. **The systems are falsely confident** — they can't tell you when they don't know, and
   they fall apart on new data without warning.

The result is a field that is busy and technically impressive on the surface, but
fundamentally untrustworthy underneath: **confident predictions, built on labels that were
never true to begin with, with no ability to admit doubt.**

That gap — between how trustworthy these tools _look_ and how trustworthy they _actually
are_ — is the problem this project sets out to address. The companion document, "The
Solution," explains how.

---

_Note: This project is a research study and a screening/decision-support concept, not a
diagnostic tool. It is designed to study and improve the trustworthiness of these systems,
and any real use would be to support — never replace — a human professional's judgment._
