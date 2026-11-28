---
layout: post
title: "Training steadied the answers, but decisions still flipped"
date: 2026-11-29
tags: [deep-learning, experiment]
series: can-ai-know-when-its-wrong
permalink: /review/trained-layouts/
description: "AI models trained on 20,000 customers read the same numbers in five new layouts. Training steadied each customer's number. It didn't stop a new layout from flipping lending decisions."
toc: true
# Review copy: not listed on the home page or in the feed, hidden from search engines and the
# sitemap, with Hypothesis for comments. To launch, move it to _posts/ and drop these four.
noindex: true
sitemap: false
hypothesis: true
comments: false
---

> **Draft for review, version 1 (29 November), not yet published.** Thank you for reading. This is a short follow-up to [Why AI can write code but struggle to read a table](/2026/11/27/reading-tables/), the fifth post in the series. That post asked AI models answering cold; this one asks the versions I trained on 20,000 customers in [I gave AI the same data as a bank's model](/2026/11/25/fair-fight/), and checks the models built for tables too.
>
> **To comment, no account needed:** reply to my message and quote the sentence you mean. If you already use [Hypothesis](https://web.hypothes.is/), you can also select text and click **Annotate**.
>
> I'd most like to know: does the title say what the post found? Does the section on the customers near the cut-off explain why decisions flipped while the averages barely moved? Is it clear why the table models and the AI models aren't one test? Would you act on the bank advice? And where did you get lost or bored?

*In the last post, AI models answering cold changed their answers when the same numbers came in a different layout. So I asked the versions I'd trained on 20,000 customers. Training steadied each customer's number. It didn't stop a new layout from flipping lending decisions.*

Here is a credit-card customer from Taiwan in 2005. Their credit limit is NT$100,000. Their bill hovers around NT$80,000 every month, they pay between NT$1,100 and NT$8,000 of it, and two summer months are marked two months late.

```text
Credit limit: NT$100,000

Month (most recent first) | Bill amount | Amount paid | Repayment status
Sep 2005 | NT$80,402 | NT$8,000 | not late (code 0)
Aug 2005 | NT$86,137 | NT$1,100 | not late (code 0)
Jul 2005 | NT$84,998 | NT$3,000 | 2 months late (code 2)
... three more months
```

In [the last post](/2026/11/27/reading-tables/), I asked Kev-4B, an open AI model built for decisions, how likely this customer was to miss next month's payment. Reading the table above, it said 54%. Given exactly the same numbers in the format programs use to pass data to each other, it said 42%. Not one number had changed.

I didn't pick this customer for drama. Sort all 2,000 test customers by how far Kev-4B's answer moved between the two, and they're the one in the middle, the 1,000th.

Then I asked the version of Kev-4B that I trained on 20,000 other customers in [post 3](/2026/11/25/fair-fight/):

| Kev-4B | Original table | Same numbers as JSON |
|---|---:|---:|
| Answering cold (post 5) | 54% | 42% |
| Trained on 20,000 customers (post 3) | 16% | 18% |

A 12-point move became a 2-point one. (The trained numbers are lower across the board because training also removed most of Kev-4B's habit of overstating risk. The customer paid.)

So does training teach an AI model to read a table? It matters to anyone who trains a model and later lets an engineer tidy up its prompt. And post 5 made a claim I hadn't tested: that models built for tables "don't have this problem by design". Do they?

{% include chart.html src="/public/images/trained-layouts/post5b_overview.png" alt="How far each customer's answer moved, on average, when the same numbers came in another layout, in percentage points." words="Table models, rearranged 14 ways: Kumo-Tabular 0.0 to 0.4 and TabICLv2 0.0 to 0.5, each less than a change of random seed (0.41 and 0.63). AI models trained on the table, in five text layouts: Kev-4B 0.8 to 3.7, Kev-0.8B 1.0 to 4.4, Qwen3.5-4B 1.0 to 2.4, Laya 1.3 to 26.4. The same AI models answering cold in post 5: Kev-4B 2.8 to 15.1, Kev-0.8B 2.6 to 5.5, Qwen3.5-4B 4.7 to 9.2, Laya 6.4 to 25.8." eager=true %}

## Two kinds of model, the same 20,000 customers

- **Built for tables:** Kumo-Tabular and TabICLv2, from post 3. They read the 20,000 labelled customers as examples and predict, with no training run. They never see text, only the 19 numbers in columns, so I rearranged those 14 ways: the column order (months reversed, grouped by month, each bill beside the next month's payment, and five random orders), the order of the 20,000 example rows (five random orders), and the amounts in thousands of NT$.
- **AI models trained on the table:** Kev-4B, Kev-0.8B, Qwen3.5-4B and Laya, each fine-tuned in post 3 on the same 20,000 customers, written as the original table only. Each read the 2,000 test customers in post 5's five layouts: a Markdown table, CSV, JSON, sentences, and a realigned table that puts each bill beside the payment made the following month. Their cold answers come from post 5.

Both groups learned from the same 20,000 customers, so that part is like for like. But the table models never see text, so their rearrangements are a different kind of change from the AI models' layouts: the chart above puts the groups side by side, but it isn't one test.

Each run is compared with the same model on the original table, cold with cold and trained with trained. For lending decisions I used the lender from [post 2](/2026/11/24/why-99-percent-accuracy/): each customer who pays earns 1 unit, each default costs 4, and the cut-off is chosen for the original table on 2,000 separate customers, then kept for the new layout. Nothing was tuned on the test customers.

## The table models didn't care how the table was arranged

Both table models use randomness, so first I changed only the random seed, the starting point for those random choices. That alone moved Kumo-Tabular's answers by up to 0.41 percentage points per customer, and TabICLv2's by up to 0.63.

- **Every column order moved them less than that:** 0.30 to 0.36 points for Kumo-Tabular, 0.43 to 0.54 for TabICLv2.
- **Reordering the example rows or writing the amounts in thousands moved them 0.09 points at most.**
- **Their ranking (AUC) didn't change beyond chance in any of the 28 runs.** The [methods page](/methods/) explains each measure.

They're steady by construction. By default, each model runs eight copies of itself, each copy sees the columns in a different order, and the answers are averaged. The order I hand them is just one more shuffle.

To see what that averaging hides, I ran every arrangement again with one copy and the shuffle turned off. TabICLv2 then moved 1.1 to 1.4 points per customer with a new column order. That's small, and its ranking moved by 0.004 at most, but with one copy TabICLv2 gives the same answer every time, so the order caused all of it. Kumo-Tabular stayed within its own seed noise even then.

So post 5's line holds, with a footnote: the table models are steady because they average over column orders, not because column order can't reach them.

## Training steadied each customer's number

{% include chart.html src="/public/images/trained-layouts/post5b_answers.png" alt="Each trained AI model's average move per customer in each of five layouts, answering cold and after training." words="Kev-4B: JSON 15.1 points cold to 2.0 trained, the realigned table 13.7 to 2.8, sentences 8.1 to 3.7. Qwen3.5-4B: 4.7 to 9.2 cold, 1.0 to 2.4 trained. Kev-0.8B: smaller trained on four layouts, but the realigned table went from 3.8 to 4.4. Laya: CSV from 25.7 to 1.3, JSON from 25.8 to 26.4, sentences from 10.1 to 11.7." %}

- **Kev-4B's answers moved 0.8 to 3.7 points per customer trained,** against 2.8 to 15.1 cold.
- **Qwen's moved 1.0 to 2.4,** against 4.7 to 9.2.
- **Kev-0.8B steadied on four layouts of five.** The realigned table was the exception: 4.4 points trained, 3.8 cold.
- **Ranking and honesty held for these three.** The only change in ranking beyond chance was Kev-4B with sentences, from 0.792 to 0.784, and no change in honesty ([calibration error](https://scikit-learn.org/stable/modules/calibration.html)) was beyond chance.

Trained answers moved less in 14 of the 15 Kev and Qwen rows, and the movement is the layout, not noise: run again on the original table, trained Kev moved 0.08 to 0.10 points per customer, and Qwen gave identical answers. Still, even the steadiest trained AI moved more than the table models: 0.8 points at the least, against at most 0.54 for any rearrangement. Laya, the exception, gets its own section below.

## But a new layout still flipped 2 to 10 in 100 decisions

Steadier answers did mean fewer flipped decisions on average. Across the five layouts, trained Kev-4B flipped 5 in 100 lending decisions, against 23 cold; Qwen 5 against 10; Kev-0.8B 5 against 6. But not everywhere, and nowhere near zero.

{% include chart.html src="/public/images/trained-layouts/post5b_decisions.png" alt="The share of lending decisions that flipped with each layout, cold and trained, at the cut-off chosen for the original table." words="Trained Kev-4B flipped 2 to 9.6 in 100 (cold 5.5 to 48). Trained Kev-0.8B about 2 to 8 (cold 0.7 to 25), more than cold on four of five layouts: with JSON, 7.4 against 0.7. Trained Qwen3.5-4B 2.8 to 9.4 (cold 7.8 to 13): with JSON, 9.4 against 7.8. Trained Laya 4.4 to 65 (cold 22 to 63). Run again on the original table, the trained models flipped at most 0.25 in 100; rearranging the table models flipped at most 2.1, and a new seed up to 2.75." %}

- **Trained Kev and Qwen still flipped about 2 to 10 in 100 decisions,** depending on the layout. Run again on the original table, they flipped at most 0.25 in 100.
- **In 6 of the 15 rows, trained flipped more than cold.** Kev-0.8B on four of five layouts: with JSON, 7.4 in 100 against 0.7 cold. Qwen with JSON, 9.4 against 7.8. Kev-4B with sentences, 9.6 against 9.5.
- **For scale, the table models:** rearranging them flipped at most 2.1 in 100 decisions, and a new seed up to 2.75. Trained Kev and Qwen with Markdown and CSV, about 2 to 3.5 in 100, are at that level. JSON, sentences and the realigned table mostly flipped more.

The lender's profit barely moved: at most 13 units per 1,000 customers lost and 15 gained, against about 290. The customers who flip sit near the cut-off, where approving and declining are worth about the same to the lender, so swapping them costs little on average. But each one is a customer who got a different answer because the layout changed.

## Why: the layout nudged the customers near the cut-off one way

How can answers barely move on average and still flip 1 in 10 decisions? Look at the customers near the line.

{% include chart.html src="/public/images/trained-layouts/post5b_nudge.png" alt="Each dot is one test customer near the cut-off: the trained model's answer to the original table against its answer to the new layout." words="Kev-4B with sentences, cut-off 12%: the average moved up 0.1 points, but the customers within 5 points of the cut-off rose 1.8, so 9.5 in 100 decisions flipped to decline and 0.1 to approve. Qwen3.5-4B with JSON, cut-off 20%: the average moved down 1.8 points and the customers near the cut-off down 4.1, so 8.6 in 100 flipped to approve and 0.8 to decline." %}

- **Kev-4B with sentences moved 0.1 points on average: nothing.** But that average hides two moves. Sentences pulled most answers up, by 1.4 to 3.4 points on average depending on how risky the customer looked, and pulled down the riskiest 11%, those it had put above 50%, by 14 points. The customers within 5 points of its cut-off rose 1.8, so 9.5 in 100 decisions flipped to decline and only 0.1 to approve.
- **Qwen with JSON went the other way:** down 1.8 points on average and 4.1 near the cut-off. It flipped 8.6 in 100 to approve and 0.8 to decline, and approval went from 63% to 71%.
- **Random wobble would flip decisions both ways about equally.** Of the seven trained Kev and Qwen runs that flipped 5 in 100 or more, five flipped more than 80% of them the same way.

Training also crowded Kev-4B's customers near its line. Its cut-off fell from 40% cold to 12% trained, and 44% of its customers sat within 5 points of it, against 19% cold, so a 2-point nudge catches many of them. That isn't general: within 5 points of the cut-off, Kev-0.8B has 21.6% of its customers trained against 22.5% cold, Qwen 18.2% against 14.4%, and Laya 22.8% against 39.4%.

I added these near-the-cut-off measures after seeing the results, to explain them. None of them needs outcomes.

## Laya: training fixed one layout and broke on another

{% include chart.html src="/public/images/trained-layouts/post5b_laya.png" alt="Each dot is one test customer: Laya's answer to the original table against its answer to CSV or JSON, cold and trained." words="Cold with CSV: answers scattered upward, an average move of 25.7 points, rank correlation 0.75, ranking 0.666 to 0.630. Trained with CSV: on the no-change line, 1.3 points, rank correlation 0.99, ranking 0.778 to 0.776. Cold with JSON: answers bunched between 30% and 55%, 25.8 points, rank correlation 0.08, ranking 0.666 to 0.478. Trained with JSON: still bunched near 40%, 26.4 points, rank correlation 0.09, ranking 0.778 to 0.591." %}

- **CSV went from 25.7 points per customer cold to 1.3 trained,** nearly on the line. Markdown (6.4 to 4.8) and the realigned table (12.8 to 5.6) improved too.
- **Sentences didn't:** 10.1 points cold, 11.7 trained. Its average answer rose from 21% to 32%, and the lender, keeping its cut-off, would approve 7 in 100 customers instead of 65.
- **JSON broke it:** 26.4 points per customer, an average answer of 41% instead of 21%, and almost no link to its answers to the original table (a rank correlation of 0.09, where 1 means the same order). Its ranking fell from 0.778 to 0.591, and the lender would approve nobody.

Laya is the smallest model here, and its JSON summaries are nearly twice as long as the layout it trained on: a median of 304 tokens against at most 183, still within its limit. Why it breaks, I didn't test.

## My predictions, written before the runs

Before any of these runs, I [wrote down](https://github.com/nishzsche/can-ai-know-when-its-wrong/blob/main/PREDICTIONS.md) what I expected:

| My guess | What happened |
|---|---|
| Rearranging the table models would move them no more than changing their seed | ✓ At their default settings. With one copy and the shuffle off, TabICLv2 did respond to column order |
| The trained AI models would be much sturdier than when cold | Half right. Each customer's number: ✓ for Kev-4B and Qwen, and Kev-0.8B on four layouts of five. Decisions: only partly. Laya: steadier with three layouts, not with sentences or JSON |
| Laya would move most | ✓ Up to 26.4 points per customer, with JSON |

## What I'd tell a bank

- **Training doesn't make the format safe to change.** Fix the input format in the model's documentation, and treat any change to it as a model change: revalidate before it goes live. Even trained, Kev and Qwen flipped up to 1 in 10 decisions with a new layout, and Laya far more.
- **Watch decisions, not just ranking.** Kev's and Qwen's rankings held almost everywhere while their decisions flipped. Score the same customers in both layouts, count the decisions that change, and look at the customers near the cut-off, where the average can hide a move. None of it needs outcomes.
- **Keep a table model's default settings.** Averaging over column orders is what makes it steady; a single copy is the order-sensitive version.

## What this test can't tell you

- **One layout learned.** Each AI model trained on the original table only. Training on several layouts is the obvious next experiment.
- **The cut-off wasn't re-chosen.** The new layouts ran on test customers only, and nothing is tuned on them. So a flipped decision means swapping in a new layout and keeping the cut-off, as a team might when it tidies a prompt.
- **Two kinds of change.** The overview shows how each kind of model reacts to the changes it can see, not which is sturdier.
- **One training run per model.** In post 3, retraining with a new seed changed how honest a model was; it might change this too.
- **I didn't test why Laya breaks on JSON.**
- **These test customers have now been used in six posts.** 6,000 customers that no model has seen are kept aside for one final replication.

The caveats every part of the series shares, from the dataset's age to what "default" means, are on the [methods page](/methods/).

## How I ran it

Everything ran in one session on a rented NVIDIA L4 GPU: about 2.5 hours, about $1.10. The table models used the makers' default settings and post 3's seed, and a test undoes every arrangement to check that the original table comes back exactly. The trained AI models are post 3's, with Kev and Qwen at full size through [post 4](/2026/11/27/shrinking-ai-models/)'s code. Each answered all 2,000 test customers in each layout, plus the original table once more to measure its run-to-run noise. Laya's longest summary was 320 tokens, within its 512-token limit.

Changes in ranking and honesty were tested in pairs against the original table, allowing for 20 comparisons for the trained models and 28 for the table models (Holm). All the measures were fixed before the runs except the near-the-cut-off ones, which I added while writing.

## What's next

Every number, the code and every prediction are on [GitHub](https://github.com/nishzsche/can-ai-know-when-its-wrong). This series started with [whether AI can know when it's wrong](/2026/11/23/can-ai-know-when-its-wrong/), then [why 99% accuracy can still be a disaster](/2026/11/24/why-99-percent-accuracy/), [giving AI the same data as a bank's model](/2026/11/25/fair-fight/), [shrinking the models](/2026/11/27/shrinking-ai-models/) and [why AI can write code but struggle to read a table](/2026/11/27/reading-tables/).

**Training steadied the answers, but a new layout is still a new model.**
