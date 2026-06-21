# The Whole Project, In Depth — But In Plain Terms

*A complete, plain-language walkthrough of the project: the problem, the solution, the three upgraded contributions, how they fit together, the data, and how it all gets tested. No machine-learning background assumed. For the technical version, see "Technical Contribution — In Depth."*

> **Status:** This direction has been reviewed and confirmed by my supervisor. We are agreed and proceeding.

---

## Part 1 — The problem, in one breath

People build systems that read social-media posts and try to guess whether someone shows
signs of a mental-health condition. To learn, these systems need examples labelled with the
"right answer." But the labels they use aren't doctors' diagnoses — they're just *which
online group the post came from* (someone posted in the depression forum, so the post gets
labelled "depression"). That's a guess, not a fact. The field calls it a **proxy label**.

Two things go wrong because of this, and they're the whole reason this project exists:

1. **The guesses are unreliable in an uneven way.** "I have depression" is usually accurate.
   "I have bipolar disorder" or "I have schizophrenia" is far shakier — those conditions are
   more often self-misdiagnosed, more talked about, more stigmatised. So the labels are
   noisier for some conditions than others, and today's systems are blind to that.

2. **The systems are falsely confident.** They give confident answers even when they're
   looking at something unfamiliar, and they have no way to say "I'm not sure." On new data
   they fall apart silently, with no warning.

The result: tools that look trustworthy but aren't — confident answers built on labels that
were never solid, with no ability to admit doubt.

---

## Part 2 — The solution at a glance

This project builds a screening system that is **honest in ways the existing ones aren't.**
It has three parts, and — this is the important upgrade from earlier versions — the three
parts are now **wired together into one machine**, not three separate gadgets sitting next to
each other.

The single idea connecting all three:

> **How unsure we are about the labels should flow through into how cautious the system is
> about its answers.**

Everything below is a version of that one sentence.

---

## Part 3 — The three parts, explained

### Part 3a — Learn which labels to trust less (and prove when that's even possible)

**The everyday version:** imagine a teacher who knows some students have neat handwriting and
others have messy scrawl. A good teacher reads the messy ones more carefully and holds their
conclusions more loosely. This system does the same: it's told up front that bipolar and
schizophrenia labels are "messier" than depression labels, and it adjusts how much it trusts
each one.

**The upgrade (this is new and makes the work much stronger):** instead of just *showing* that
this adjustment sometimes works, the project also *explains, in advance, when it will and
won't work* — and then runs an experiment to confirm that explanation.

Here's the intuition for why it sometimes can't work. To figure out how unreliable a
condition's labels are, the system needs enough clear examples of that condition to learn
from. But the conditions with the messiest labels (bipolar, schizophrenia) are also the
*rarest* in the data — there are about twenty times more depression posts than bipolar posts.
So for exactly the conditions where this adjustment matters most, there's the least data to
make the adjustment reliably. The project works out *where* that breaking point is, in
principle, and then checks whether the real data breaks down at the predicted spot.

This "predict the breaking point, then confirm it" move is the kind of rigour that separates a
top-tier project from an ordinary one. It's the difference between "I tried it and here's what
happened" and "here's the rule that says what *should* happen, and here's proof it does."

**A small optional add-on, considered carefully:** during planning, someone suggested also
checking whether a post actually *describes* textbook symptoms of the condition it's labelled
with — not just which forum it was posted in. That's a genuinely good instinct. The original
phrasing of the suggestion mentioned a specific AI technique that, on closer inspection, turned
out not to actually fit this kind of task (it's a technique for teaching a chatbot to generate
better responses, not for cleaning up a dataset) — so rather than force in a tool that doesn't
fit, the *idea* was kept and built the right way: as one more small clue feeding into the
existing "how much should I trust this label" judgement from Part 3a, never as a hard rule
that throws posts away. Used as a strict filter, a tool like this would quietly introduce its
own new bias (favouring posts that use textbook clinical language over posts that describe the
same experience in plain words) — exactly the kind of problem this whole project exists to
avoid. So it's kept as a small, optional bonus signal, tested with the same rigour as
everything else, not a core requirement.

### Part 3b — Know when to say "I don't know" — and tie that to Part 3a

**The everyday version:** a trustworthy tool should refuse to answer when it isn't sure,
rather than confidently guessing. This part gives the system two abilities: (1) making its
confidence numbers *mean* something (so "70% sure" really does mean right 70% of the time),
and (2) letting it say "uncertain — a human should look at this" when its confidence is too
low, with a mathematical guarantee about how often it's right when it *does* answer.

**The upgrade — and this is the heart of the whole project now:** the system's willingness to
answer is *directly controlled by* what Part 3a learned. For a condition whose labels Part 3a
flagged as messy (bipolar, schizophrenia), the system demands *more* confidence before it
commits to an answer. For a condition with reliable labels (depression), it relaxes a bit.

So the two parts aren't two separate features any more — they're one connected mechanism. The
system **refuses to answer most often exactly on the conditions where its training was least
trustworthy.** That's a genuinely original idea, and it's testable: if it works, the system
should make fewer confident mistakes on the hard conditions than a system that treats every
condition the same.

This connection is what an examiner means by a "coherent contribution" rather than "two things
bolted together" — and it's the single biggest improvement to the project.

### Part 3c — Find and report exactly where the method breaks

**The everyday version:** the project is honest about the fact that its own approach has limits.
The conditions where Part 3a matters most are the ones with the least reliable data to check
against — so there's a real chance the method works well for depression (where it's needed
least) and struggles for bipolar/schizophrenia (where it's needed most).

Rather than hide this, the project *measures it directly* and reports where the method starts
to fail — and checks whether that real-world breaking point matches the one Part 3a predicted
in theory. If the prediction and the reality line up, that's a satisfying, rigorous result. If
they don't, that's interesting too — it means the story is more complicated than expected,
which is itself worth reporting.

Either way, a clearly documented "here's how far this kind of approach can be trusted, and
here's where it stops working" is a real contribution to the field, not a failure.

---

## Part 4 — How the three parts form one argument

Put simply:

- Part 3a works out **how trustworthy each condition's labels are** — and proves when that's
  even knowable.
- Part 3b uses that to decide **when the system should refuse to answer** — demanding more
  certainty where the labels were shakier.
- Part 3c **tests the limits** of the whole thing and checks the theory against reality.

One sentence ties them: *uncertainty about the labels flows into caution about the answers, and
we know in advance where that chain is strong and where it snaps.*

That's not three projects. It's one honest system with a spine running through it.

---

## Part 5 — How it's proven (the testing that earns marks)

A good project doesn't just claim it works — it proves it against fair competition. This one
compares itself against several strong alternatives:

- a standard system that trusts all labels equally (the usual approach),
- a system that cleans up label noise but treats all conditions the same,
- a system that's well-calibrated but doesn't model noise,
- a system that can abstain but with one fixed threshold (not tied to label trustworthiness),
- a system that only fixes the rare-condition data shortage but ignores noise.

By turning each piece on and off in every combination, the project can say precisely what each
part contributes — not just "our system is better," but "*this specific piece* caused *this
specific improvement*." That precision is exactly what earns high marks.

Everything is measured both overall *and* per condition (because an overall number can hide the
fact that the system is great on depression and broken on bipolar), and crucially, measured for
how much it degrades when moved from easy Reddit data to harder real clinical data — the exact
honesty test the field keeps asking for.

---

## Part 6 — The data

The project uses real, public Reddit mental-health data (two public datasets combined),
covering depression, anxiety, bipolar disorder, suicidality, schizophrenia, and eating
disorders. It's been downloaded and checked by hand — confirmed clean, with exact post counts
per condition.

A wrinkle that turned into evidence: depression posts outnumber bipolar posts about 20 to 1.
That's a real challenge to handle in the code, but it's also proof of the project's core point —
the conditions that are rarer in the data are exactly the ones the method most needs to be
careful about. (A small, depression-only set of real clinical data is used as a limited
real-world reality check.)

Two honest limits, stated plainly:
- The only real clinical data available covers depression, so the system's "reality check" is
  strongest for depression and weaker for the rare conditions — handled by carefully reasoning
  about a range of possibilities rather than claiming certainty.
- The labels themselves can never be fully verified as true diagnoses — but that's the whole
  point of the project, not a flaw to apologise for.

---

## Part 7 — Why this is a strong final-year project

It doesn't just take an existing AI model and point it at a new dataset — the move examiners are
trained to be unimpressed by. Instead it:

- identifies a real, admitted weakness shared across an entire field,
- brings serious, principled tools to bear on it in a combination nobody has assembled,
- ties those tools into one coherent mechanism rather than a pile of features,
- adds a layer of *theory* that predicts where the method works and fails,
- and is designed so that it produces a genuine, defensible result whether its central bets
  succeed or fail.

That last point is what makes it a sound bet rather than a gamble: there's no single outcome it
depends on to be worth writing up.

This direction has been confirmed with my supervisor, and we're moving forward.

---

*Note: This is a research project and a screening / decision-support concept — not a diagnostic
tool. It's designed to support, and explicitly defer to, human professional judgment, never to
replace it.*