# Writing Guide - Learning Curve

A reference for whenever you're unsure whether to write, what to write, or how to write it.
Return here when you feel stuck. There are two post formats: Resolution and Exploration.

---

## Two Post Formats

| | Resolution | Exploration |
|---|---|---|
| **For** | A confusion you've worked through | A curiosity you're still inside |
| **Ends with** | A takeaway - one clear sentence | Open questions - genuinely unresolved |
| **Trigger** | Something clicked | Something has your attention |
| **Template** | `_drafts/_template.md` | `_drafts/_exploration_template.md` |
| **Issue label** | `confusion` → `drafting` → `ready` | `exploration` |

---

## Post Format 1: The Resolution

For confusions you've resolved. Three moves, in order.

---

### Move 1: The Confusion

State the specific friction point - not the topic, the *precise thing* that didn't make sense.

**Too broad (skip):**
> I didn't understand KL divergence.

**Right level:**
> I understood that KL(P||Q) != KL(Q||P), but I couldn't see why the direction of measurement
> would matter *in practice*. It felt like a mathematical footnote.

**Hobby example:**
> I understood that wet-on-wet watercolor was supposed to give soft edges.
> But every time I tried it, the paint bloomed into blotches I didn't intend.
> I couldn't tell if the problem was the water ratio, the paper, or my timing.

**The test:** Could someone with your background read the confusion and immediately know
exactly what you mean - not just the topic, but the *sticking point*? If yes, it's specific enough.

---

### Move 2: What I Thought Was True

Write out your wrong mental model. This is often the most valuable part of the post.

**Why it matters:** The resolution only makes sense against what you believed before.
Readers who share your wrong model will experience the same "click" you did.

**Example (KL divergence):**
> I thought the asymmetry was a quirk - like how subtraction isn't commutative.
> Interesting mathematically, but no practical consequence.
> Both directions "measured the same thing," just from different sides.

**Example (watercolors):**
> I thought wet-on-wet meant: wet the paper, then paint.
> I was treating the wetness as a fixed state rather than something that changes
> as the paper dries - so my timing was always off.

---

### Move 3: The Resolution + Takeaway

What actually makes sense, and the one sentence you'd say to yourself 6 months ago.

**Example (KL divergence):**
> KL(P||Q) - where P is the true distribution - heavily penalises places where P is
> high but Q is low. You're forced to cover P's mass. KL(Q||P) penalises the reverse.
> In variational inference, minimising KL(Q||P) lets Q be zero-seeking - it ignores
> modes of P it can't explain, rather than spreading thin mass across all of them.
> The direction isn't a quirk. It's a design decision about which errors you prefer to make.
>
> **Takeaway:** KL direction = whose "mistakes" you're punishing.

**Example (watercolors):**
> The paper goes through stages: very wet, damp, slightly damp, dry. Each stage produces
> a different bloom behaviour. Wet-on-wet only works predictably if both the paper and
> the brush charge are at the same moisture level. The fix was working faster and
> re-wetting intentionally rather than assuming the paper was still wet from before.
>
> **Takeaway:** Wet-on-wet is about matching moisture levels, not just "wet paper."

---

## Post Format 2: The Exploration

For things you're curious about but haven't resolved. Three moves, different shape.
Publish when the snapshot is interesting enough on its own - you don't need the answer first.
An Exploration can later become a Resolution if it resolves.

---

### Move 1: What I'm Curious About

The question or pull. Not a confusion with a wrong model - just something that has your attention.
State it precisely: "I keep wondering whether X..." or "I can't find a clear answer on Y..."

**ML example:**
> "Emergent capabilities" gets discussed constantly but I can't find a crisp definition
> that distinguishes genuine emergence from just "we didn't test at smaller scales."

**Hobby example:**
> Does watercolor paper weight (300gsm vs 140gsm) actually change how wet-on-wet behaves,
> or is the only real difference whether the paper buckles?

---

### Move 2: What I've Found So Far

Current state of understanding. Partial is fine - this is a snapshot, not a conclusion.
What have you read, tried, or asked? What partial answers exist?

**ML example:**
> The Wei et al. (2022) paper defines emergence as a capability absent below a threshold
> and present above it. But the Schaeffer et al. (2023) response argues the "sharpness"
> depends entirely on metric choice - so the phenomenon might be real or a measurement artefact.

**Hobby example:**
> 140gsm buckles badly without stretching. 300gsm stays flat. But I haven't found a clear
> answer on whether the sizing (the coating that controls absorbency) differs between weights,
> or whether it's purely structural.

---

### Move 3: What I'm Still Wondering

The open questions. This is the honest part: you don't have it figured out.
Leave these genuinely open - don't fake uncertainty you've already resolved.

**ML example:**
> Whether there's a domain-agnostic way to detect emergence that doesn't depend on
> the metric you chose to measure.

**Hobby example:**
> Whether stretching 140gsm actually replicates the paint behaviour of 300gsm,
> or just the flatness.

---

## The Level Test

**Write to yourself from 6 months ago.**

That version of you had your background in this domain but hadn't encountered
*this specific nuance*. They can handle technical or craft-specific language.
They don't need basics re-explained. They do need the specific nuance explained clearly.

| Fails the test | Passes the test |
|---|---|
| "Gradient descent updates weights to minimise loss" - too basic | "I assumed Adam's beta_2 was just variance smoothing, but its interaction with the warmup schedule changes the effective learning rate non-linearly in the first N steps" |
| "Watercolor bleeds on wet paper" - obvious | "I thought the bloom was caused by too much paint - it was actually caused by the brush charge being wetter than the paper" |
| A formal definition from a textbook or manual | The specific sentence that made something click |

---

## What Makes Something Worth Posting

Ask one question: **Would I have found this useful 6 months ago?**

**High value:**
- A concept from an OMSCS course the lecture made opaque - and you found the intuition elsewhere
- A library or API that behaves differently from what the docs suggest
- A result in a hobby or craft that surprised you, and why the naive expectation was wrong
- A mental model you held for months that turned out to be subtly incorrect
- A question you keep returning to across different contexts (good Exploration candidate)

**Low value:**
- "Here's how to install X" - documentation, not insight
- A notebook export with no commentary on the surprising parts
- "Today I learned Y" without the wrong-model → resolution arc

---

## OMSCS Posts: Handle with Care

You're a practitioner doing grad school. That's rare. Honour both sides.

**Do:**
- Tag posts with course codes (`CS7641`, `CS7642`, `ISYE6501`, etc.) - these are among the most-searched ML phrases
- Write about where theory and practice diverge - that's your unique angle
- Write about *why* an assignment approach is designed the way it is, not just what it does

**Don't:**
- Share solutions or code for graded assignments - this violates the honour code
- Skip the practitioner lens - "here's how this maps to something I've built at work" adds value no student-only blog can

---

## Post Formats at a Glance

| | Resolution | Exploration | Notebook |
|---|---|---|---|
| **Length** | 200-400 words | 200-400 words | Long - code + explanation |
| **Source** | GitHub Issues → `_posts/` | GitHub Issues → `_posts/` | `notebooks/` → automation |
| **Ends with** | A takeaway | Open questions | Observations + code |
| **Trigger** | Something resolved | Something has attention | Experiment completed |

---

## Posting Frequency

**Don't set a cadence. Set a trigger.**

Two triggers, not one:

- **Resolution trigger:** Something clicked. Write within 48 hours - after that, the confusion
  feels obvious in hindsight and you lose the ability to explain it to someone still confused.
- **Exploration trigger:** Something has your sustained attention. Publish when the snapshot
  of where you currently are is interesting on its own.

A "post every week" commitment produces mediocre posts written just to hit the number.
The constraint for Resolutions isn't time - it's having the resolution.
The constraint for Explorations is finding the question precise enough to be useful.

---

## The Draft Workflow: GitHub Issues

All short posts (Resolution and Exploration) live in GitHub Issues until publish time.
You don't need a laptop until the last step.

```
GitHub Issue (phone)  →  _posts/ (laptop, when ready)
```

**Labels:**

| Label | Meaning |
|---|---|
| `confusion` | Resolution post captured - specific confusion, resolution pending |
| `drafting` | Moves 1 + 2 written, working on resolution |
| `ready` | Complete and ready to publish (either format) |
| `exploration` | Curiosity post - publish when snapshot is interesting, no resolution needed |

**Resolution workflow:**

1. **Capture (phone):** Feel confusion → open Issue → one sentence → label `confusion`
2. **Draft:** Fill in moves 1 + 2 in the issue body. Label `drafting`. Leave move 3 blank.
3. **Iterate:** Add comments as you think more. Edit the body as understanding evolves.
4. **Resolve:** Fill in move 3. Label `ready`.
5. **Publish (laptop):** Copy to `_posts/YYYY-MM-DD-slug.md`. Push to `dev`. Close issue.

**Resolution issue body template:**

```markdown
## The Confusion


## What I Thought Was True


## The Resolution
<!-- leave blank until resolved -->

## The Takeaway
<!-- leave blank until resolved -->

---
*Source:*
```

**Exploration workflow:**

1. **Capture (phone):** Notice a curiosity → open Issue → state the question precisely → label `exploration`
2. **Build:** Add what you're finding in comments. Edit the body as the snapshot evolves.
3. **Publish when ready:** When the snapshot is worth sharing - even without an answer.
4. **Promote if it resolves:** If the curiosity becomes a resolved confusion, relabel `confusion` → work through moves → publish as a Resolution.

**Exploration issue body template:**

```markdown
## What I'm Curious About


## What I've Found So Far


## What I'm Still Wondering

---
*Domain:*
*Related to:*
```

**Notebook posts** still use the automation pipeline (`notebooks/` → GitHub Actions → `_posts/`).

An issue can sit open for weeks. The goal is to **never lose a curiosity or confusion**.

---

## Tag Taxonomy

Use at most 2-3 tags per post. Be consistent - these are how future-you finds things.

**ML and technical:**

| Tag | Use for |
|---|---|
| `probability` | Distributions, inference, statistics |
| `optimization` | Gradient methods, convergence, schedulers |
| `linear-algebra` | Matrix ops, decompositions, geometric intuition |
| `deep-learning` | Neural nets, training, architectures |
| `ml-theory` | Bias-variance, PAC learning, generalisation bounds |
| `implementation` | Code behaviour, library quirks, debugging |
| `OMSCS` | Course-specific insights - pair with course code tag |
| `paper-notes` | Insight from a specific paper |
| `mlops` | Deployment, pipelines, experiment tracking |

**Craft and hobbies:**

| Tag | Use for |
|---|---|
| `hardware` | Arduino, electronics, physical computing |
| `creative` | Painting, music, visual arts |
| `craft` | Cycle maintenance, physical skills, making things |

**Format:**

| Tag | Use for |
|---|---|
| `intuition` | Mental model corrections, "aha" moments (Resolution) |
| `exploration` | Posts using the Exploration format |

---

*Keep this file. Re-read the examples when you feel a post isn't working.*
