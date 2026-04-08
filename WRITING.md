# Writing Guide — Learning Curve

A reference for whenever you're unsure whether to write, what to write, or how to write it.
Return here when you feel stuck. Most doubts are answered by one of the three moves.

---

## The Three Moves

Every short post is built from exactly three moves. No exceptions.

---

### Move 1: The Confusion

State the specific friction point — not the topic, the *precise thing* that didn't make sense.

**Too broad (skip):**
> I didn't understand KL divergence.

**Right level:**
> I understood that KL(P||Q) ≠ KL(Q||P), but I couldn't see why the direction of measurement
> would matter *in practice*. It felt like a mathematical footnote.

**Another example:**
> Batch norm is said to "reduce internal covariate shift" — but I couldn't connect that phrase
> to why it speeds up training. The words made sense; the mechanism didn't.

**The test:** Could someone with your background read the confusion and immediately know
exactly what you mean — not just the topic, but the *sticking point*? If yes, it's specific enough.

---

### Move 2: What I Thought Was True

Write out your wrong mental model. This is often the most valuable part of the post.

**Why it matters:** The resolution only makes sense against what you believed before.
Readers who share your wrong model will experience the same "click" you did.

**Example (KL divergence):**
> I thought the asymmetry was a quirk — like how subtraction isn't commutative.
> Interesting mathematically, but no practical consequence.
> Both directions "measured the same thing," just from different sides.

**Example (batch norm):**
> I thought batch norm normalised activations so the network could train with a higher
> learning rate without exploding. I was ignoring everything after the normalisation step
> (the learnable scale and shift parameters). I thought those were just there for "flexibility."

---

### Move 3: The Resolution + Takeaway

What actually makes sense, and the one sentence you'd say to yourself 6 months ago.

**Example (KL divergence):**
> KL(P||Q) — where P is the true distribution — heavily penalises places where P is
> high but Q is low. You're forced to cover P's mass. KL(Q||P) penalises the reverse.
> In variational inference, minimising KL(Q||P) lets Q be zero-seeking — it ignores
> modes of P it can't explain, rather than spreading thin mass across all of them.
> The direction isn't a quirk. It's a design decision about which errors you prefer to make.
>
> **Takeaway:** KL direction = whose "mistakes" you're punishing.

**Example (batch norm):**
> The scale and shift (γ, β) are learned per-feature. After normalising to mean=0, std=1,
> the network *re-learns* the optimal scale for each feature. The normalisation step
> makes the loss landscape smoother; the learnable parameters ensure you don't lose
> representational power. They're not afterthoughts — they're what keeps the layer useful.
>
> **Takeaway:** Batch norm's job isn't to fix the scale — it's to make the scale
> learnable from a stable starting point.

---

## The Level Test

**Write to yourself from 6 months ago.**

That version of you had your general ML/CS background but hadn't encountered *this specific thing*.
They can handle technical language. They don't need basics re-explained.
They do need the specific nuance explained clearly.

| Fails the test | Passes the test |
|---|---|
| "Gradient descent updates weights to minimise loss" — too basic | "I assumed Adam's β₂ was just variance smoothing, but its interaction with the warmup schedule changes the effective learning rate non-linearly in the first N steps" |
| "KL divergence measures the difference between two distributions" — too generic | "I thought both KL directions punished the same errors — just from opposite sides" |
| A formal definition from a textbook | The specific sentence that made something click |

---

## What Makes Something Worth Posting

Ask one question: **Would I have found this useful 6 months ago?**

**High value:**
- A concept from an OMSCS course the lecture made opaque — and you found the intuition elsewhere
- A library or API that behaves differently from what the docs suggest
- A result that surprised you, and why the naive expectation was wrong
- A mental model you held for months that turned out to be subtly incorrect

**Low value:**
- "Here's how to install X" — documentation, not insight
- A notebook export with no commentary on the surprising parts
- "Today I learned Y" without the wrong-model → resolution arc

---

## OMSCS Posts: Handle with Care

You're a practitioner doing grad school. That's rare. Honour both sides.

**Do:**
- Tag posts with course codes (`CS7641`, `CS7642`, `ISYE6501`, etc.) — these are among the most-searched ML phrases
- Write about where theory and practice diverge — that's your unique angle
- Write about *why* an assignment approach is designed the way it is, not just what it does

**Don't:**
- Share solutions or code for graded assignments — this violates the honour code
- Skip the practitioner lens — "here's how this maps to something I've built at work" adds value no student-only blog can

---

## Notebook Posts vs. Short Posts

| | Notebook post | Short post |
|---|---|---|
| **Length** | Long — code + explanation | Short — 200–400 words |
| **Source** | `notebooks/` → automation pipeline | `_drafts/` → written directly |
| **Best for** | Experiments, data exploration, algorithm comparisons | Conceptual confusions, mental model corrections, OMSCS insights |
| **Rule of thumb** | The *process* matters (what you tried, what happened) | The *moment* matters (one thing that clicked) |

---

## The Draft Workflow

```
INBOX  →  _drafts/  →  _posts/
```

1. **Capture:** Add a one-liner to `_drafts/INBOX.md` the moment you feel the confusion.
   On your phone: open a GitHub Issue labelled "confusion". Don't elaborate yet.

2. **Draft:** Copy `_drafts/_template.md` to `_drafts/your-topic.md`.
   Fill in Move 1 and Move 2 immediately — they don't require a resolution.
   Leave Move 3 blank.

3. **Resolve:** Work through it (course material, paper, experiment, ask someone).
   Return to the draft and fill in Move 3.

4. **Publish:** Rename to `_posts/YYYY-MM-DD-your-topic.md`. Add the date to front matter.
   Push to `dev`. Done.

A draft can sit for weeks. That's fine. The goal is to **never lose a confusion**.

---

## Tag Taxonomy

Use at most 2–3 tags per post. Be consistent — these are how future-you finds things.

| Tag | Use for |
|---|---|
| `probability` | Distributions, inference, statistics |
| `optimization` | Gradient methods, convergence, schedulers |
| `linear-algebra` | Matrix ops, decompositions, geometric intuition |
| `deep-learning` | Neural nets, training, architectures |
| `ml-theory` | Bias-variance, PAC learning, generalisation bounds |
| `implementation` | Code behaviour, library quirks, debugging |
| `OMSCS` | Course-specific insights — pair with course code tag |
| `paper-notes` | Insight from a specific paper |
| `intuition` | Mental model corrections, "aha" moments |
| `mlops` | Deployment, pipelines, experiment tracking |

---

*Keep this file. Re-read the examples when you feel a post isn't working.*
