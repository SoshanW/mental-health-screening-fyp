# The Solution: A More Honest Mental-Health Screening System

_A plain-language explanation of what this final-year project builds and why it works._

> **Status:** This solution and its overall direction have been reviewed and confirmed by my
> supervisor. We are agreed and good to proceed.

---

## The one-line idea

Instead of building yet another system that confidently guesses from untrustworthy labels,
this project builds one that is **honest in two specific ways**: it knows which of its
training labels to trust less, and it knows when to admit it isn't sure.

(If you haven't read the companion document "The Problem" yet, read that first — it explains
why those two things are broken in every existing system.)

The solution has three parts. The first two are the actual fixes; the third is an honest
test of how far the fixes can go.

---

## Part 1: Teach the system that some labels are shakier than others

**The problem this fixes:** existing systems treat every label as equally trustworthy, even
though a "depression" label is reliable and a "bipolar" or "schizophrenia" label is far
noisier.

**The fix, in plain terms:** instead of feeding the system labels as if they were perfect
facts, this project builds in a model of _how wrong each label is likely to be_ — and
crucially, allows that wrongness to be **different for each condition**.

Think of it like a teacher grading essays with the knowledge that some students' handwriting
is reliable and others' is a mess. A good teacher doesn't read every paper with equal trust
in what they think they're seeing — they mentally adjust, reading the messy handwriting more
carefully and holding their conclusions more loosely. This project does the equivalent: it
explicitly tells the system "treat the bipolar and schizophrenia labels with more suspicion
than the depression labels, because we know they're noisier."

There's an existing technique in machine learning for estimating how often labels are wrong
(it's the engine this part builds on), but the new contribution here is making that estimate
**condition-specific** and grounding it in what's actually known about how reliably people
self-identify each condition.

**The honest test built into this part:** the project doesn't just assume this helps. It
directly asks — does building in this condition-specific knowledge actually make the system
better than a simpler approach that just cleans up all the labels uniformly? If yes, that's
the contribution. If it turns out _not_ to help, that's still a genuine, reportable finding —
it tells the field something true that wasn't known before. Either way, the project produces
a real answer.

**An upgrade worth knowing about:** rather than just trying this and reporting what happened,
the project also works out, in advance and on paper, _when_ this kind of label-trustworthiness
estimate can even be calculated reliably in the first place — and then checks whether the real
data behaves the way that prediction says it should (this is explained fully in Part 3 below,
since the two are closely linked).

**A small optional addition, considered and scoped carefully:** a suggestion was raised
during planning to also check whether a post's actual content mentions textbook clinical
symptoms for its labelled condition, as one more clue about how much to trust that label.
That instinct is sound, and it has been folded in as an optional extra signal feeding into
this same noise-estimation process — never as a hard rule that throws posts out, since doing
that would create a new, different kind of bias rather than removing one. It is treated as a
small, time-permitting addition, tested the same rigorous way as everything else, not as a
core requirement of the project.

---

## Part 2: Teach the system to say "I don't know" — and connect it directly to Part 1

**The problem this fixes:** existing systems give confident answers even when they're staring
at something they've never seen, and they fall apart silently on new data.

**The fix, in plain terms:** this project adds two things on top of the system.

First, **calibration** — making the system's confidence numbers actually mean something. Right
now when a system says "70% sure," that number is often meaningless. A calibrated system is
one where, when it says "70% sure," it really is right about 70% of the time. There's a
well-established method for this, and the project measures honesty-of-confidence directly
(using a standard score that captures the gap between how confident the system is and how
often it's actually right).

Second, **principled abstention** — giving the system permission to refuse. When its
confidence is too low, instead of guessing, it says "uncertain — a human should look at
this." And it does this with a mathematical guarantee about how often it will be right when
it _does_ choose to answer. This is the difference between a tool that bluffs and a tool that
knows its own limits.

**The upgrade that ties Part 1 and Part 2 together (the heart of the whole project):**
on their own, "know which labels are shaky" and "know when to refuse to answer" are two good
ideas sitting side by side. The project goes one step further and **wires them into a single
mechanism**: the system's willingness to answer on a given condition is _directly set by_ how
noisy Part 1 found that condition's labels to be. For a condition flagged as shaky (bipolar,
schizophrenia), the system demands more confidence before committing to an answer. For a
condition with reliable labels (depression), it relaxes a little. The result is a system that
**refuses to answer most often exactly on the conditions where its training was least
trustworthy** — not two features, one connected chain of reasoning running from "how much do I
trust this label" straight through to "how willing am I to act on it."

**The headline test for this part:** the real proof is what happens when the system is moved
from the easy data it was trained on (Reddit posts) to harder, real-world clinical data. The
project measures exactly how much its honesty and accuracy degrade across that jump — instead
of just hoping it holds up. And because Part 1 and Part 2 are now connected, there's a sharper
test available too: does the _connected_ version handle that jump more gracefully than a
version where abstention isn't tied to label trustworthiness? That comparison is what proves
the connection is actually doing useful work, not just sounding good on paper.

---

## Part 3: Find and report the method's own breaking point

**Why this part exists:** there's a catch, and the project confronts it head-on instead of
hiding it.

The conditions where the condition-specific noise modelling (Part 1) matters _most_ — bipolar,
schizophrenia — are also the conditions with the _least_ reliable data to check the work
against. So there's a real possibility that the method works beautifully for depression (where
it's needed least) and struggles for the rare conditions (where it's needed most).

Rather than pretend this isn't a risk, the project measures it directly: it studies how well
the approach holds up as the available reliable data shrinks, and reports honestly where it
starts to fail.

This is what turns a potential weakness into a strength. If the method has a breaking point,
finding and clearly documenting that breaking point is itself a valuable contribution — it
tells future researchers exactly how far this kind of approach can be trusted and where new
ideas are needed.

---

## How the three parts fit together

These aren't three separate mini-projects bolted together. They form one coherent argument:

> Today's mental-health screening tools are dishonest about two things — where their training
> data comes from, and how sure they really are. This project builds a system that is honest
> about both (Parts 1 and 2), and is also honest about its own limits (Part 3).

The end result isn't a polished product anyone would deploy tomorrow. It's a **research
artifact** — a working, tested system — whose real output is knowledge: a clearer, more honest
picture of how trustworthy this whole category of tool can actually be, and a concrete
demonstration of a better way to build one.

---

## The data, briefly

The project runs on real, publicly available Reddit mental-health data (combining two public
datasets), covering depression, anxiety, bipolar disorder, suicidality, schizophrenia, and
eating disorders. The data has been downloaded and checked by hand to confirm it's clean and
usable. One real finding from that check — that depression posts outnumber bipolar posts about
twenty to one — turned out to be supporting evidence for the project's whole argument, since it
shows just how unevenly these conditions appear in the wild. (A small piece of clinical data,
focused on depression, is used as a limited real-world check.)

The whole thing is designed to run on modest computing resources, because the contribution is
the _ideas and the honest evaluation_, not raw computing scale.

---

## Why this is a strong project

It doesn't just apply an existing AI model to a new dataset — the thing examiners are trained
to be unimpressed by. Instead it identifies a specific, real, admitted weakness shared across
an entire field, and brings principled tools to bear on it in a combination nobody has put
together before. And it's designed so that it produces a genuine, defensible result whether its
central bets succeed _or_ fail — which makes it a sound bet rather than a gamble.

This direction has been reviewed and confirmed with my supervisor, and we are agreed to move
forward with it.

---

_Note: This project is a research study and a screening/decision-support concept, not a
diagnostic tool. It is designed to support — and explicitly defer to — human clinical
judgment, never to replace it._
