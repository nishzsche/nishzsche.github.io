---
layout: post
title: "I shrank AI models to save money. They stayed honest, and got slower."
date: 2026-11-26
tags: [deep-learning, mlops, experiment]
series: can-ai-know-when-its-wrong
permalink: /review/shrinking-ai-models/
description: "I squeezed three trained AI models down to 8, 4, 3 and 2 bits. Their percentages held up longer than I expected. Their speed didn't."
# Review copy: not listed on the home page or in the feed, hidden from search engines and the
# sitemap, with Hypothesis for comments. To launch, move it to _posts/ and drop these four.
noindex: true
sitemap: false
hypothesis: true
comments: false
---

> **Draft for review, version 1 (26 November), not yet published.** Thank you for reading. This is the fourth post in the series, after [Can AI know when it's wrong?](/2026/11/23/can-ai-know-when-its-wrong/), [Why 99% accuracy can still be a disaster](/2026/11/24/why-99-percent-accuracy/) and [I gave AI the same data as a bank's model](/2026/11/25/fair-fight/). It takes the three models trained in the last post and shrinks them to save memory, then checks whether their percentages still come true.
>
> **To comment, no account needed:** reply to my message and quote the sentence you mean. If you already use [Hypothesis](https://web.hypothes.is/), you can also select text and click **Annotate**.
>
> I'd most like to know: does the headline ("they stayed honest, and got slower") feel fair given the details, does the section on customers moving while the scores held make sense and matter to you, would you act on the bank advice, and where did you get lost or bored?

*Three trained models, squeezed to 8, 4, 3 and 2 bits. Their percentages held up longer than I expected, their speed didn't, and the summary scores hid how far each customer's number moved.*

Save a photo at a lower quality setting and, for a while, you can't tell the difference. Then, past some point, it turns into blocks.

AI models get compressed the same way. A model is billions of numbers, each normally stored in 16 bits, enough for about 65,000 different values. Shrinking a model rounds those numbers more coarsely, like rounding every price in a shop to the nearest ten cents, then the nearest dollar, then the nearest ten dollars. It's called [quantization](https://huggingface.co/docs/transformers/main/en/quantization/overview), and it lets a model fit on smaller, cheaper hardware. The usual check afterwards is whether the model still gets the answers right.

This series checks something else: whether a model's percentages come true. In [my last post](/2026/11/25/fair-fight/), I trained three AI models on 20,000 credit-card customers whose outcomes were known, and they ranked customers about as well as the models banks use. This time I shrank those three, and asked the same 2,000 test customers the same question: how likely is this customer to miss next month's payment?

| Size | Values each number can take | GPU memory for the two 4B models |
|---|---|---|
| Full size, 16 bits | about 65,000 | 7.9 GB |
| 8 bits | 256 | 4.5–4.6 GB |
| 4 bits | 16 | 3.1 GB |
| 3 bits | 8 | 2.7 GB |
| 2 bits | 4 | 2.2 GB |

Each small block of numbers gets its own scale, so even at 2 bits the four values differ from block to block. Some parts of each model stay full size, which is why 4 bits saves less than three-quarters of the memory.

![Each trained model's ranking (AUC) at each size. Kev-4B, Kev-0.8B and Qwen3.5-4B stay between 0.780 and 0.792 from full size down to 4 bits, with either tool, all within 0.005 of their full-size scores and above the scorecard's 0.777. At 3 bits, Kev-0.8B (0.786) and Qwen (0.779) hold and Kev-4B slips to 0.776. At 2 bits, Kev-4B falls to 0.60, Kev-0.8B to 0.45 and Qwen to 0.43, the last two worse than a coin flip.](/public/images/shrinking-ai-models/post4_ranking.png){: data-dark-src="/public/images/shrinking-ai-models/post4_ranking-dark.png"}

## Three models, six sizes

The three are the ones I trained last time: Kev-4B and Kev-0.8B, two open models built for decisions, and Qwen3.5-4B, a general open model. I left out Laya, the fourth: at 421 million parameters it already runs on a laptop's processor, so shrinking it doesn't answer a real question.

I used two tools:

- **bitsandbytes** at 8 bits and at 4 bits (a format called NF4), with the settings most people get when they ask the popular Hugging Face library to load a model "in 8-bit" or "in 4-bit".
- **HQQ** at 4, 3 and 2 bits, so that one method goes all the way down, and its 4-bit version shows how much the tool matters at the same size.

Neither tool needs any data to shrink a model, so nothing was tuned on the test customers. Each shrunk model is compared only with its own full-size version, run through the same server on the same GPU. And all three models learned from the same 20,000 customers as the scorecard I use as a reference.

## Ranking held to 3 bits, then fell off a cliff

To measure ranking I use the same score as before: pick one customer who defaulted and one who paid, and ask how often the model rates the defaulter as riskier ([AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve); 0.5 is a coin toss).

The chart above tells it:

- **Down to 4 bits, with either tool, nothing moved.** All three models stayed within 0.005 of their full-size scores, too close to tell apart.
- **At 3 bits, Kev-4B slipped.** It dropped from 0.792 to 0.776, just under the scorecard. That was the only drop above 2 bits that held up statistically, even after correcting for making 15 comparisons. Kev-0.8B and Qwen held.
- **At 2 bits, all three collapsed.** Kev-4B fell to 0.60, Kev-0.8B to 0.45 and Qwen to 0.43. The last two are worse than a coin toss: they'd do better with their answers reversed.

## Honesty held too, but drifted in directions nobody chose

Ranking is half the job. The other half is whether a model's percentages come true. I measure that as before: the average gap, in percentage points, between the risk a model promised and what happened ([calibration error](https://scikit-learn.org/stable/modules/calibration.html)). With 2,000 customers, a perfectly honest model stays below about 2.5 points in 95 samples out of 100.

![Calibration error at each size, in points. At full size Kev-0.8B is 4.4 points off, Kev-4B 3.8 and Qwen3.5-4B 2.4. At 8 bits and 4-bit NF4, each stays within half a point of that. At 4-bit HQQ, Kev-0.8B improves to 2.3 and Qwen worsens to 3.2. At 3 bits, Kev-4B is 4.8, Kev-0.8B 3.1 and Qwen 1.5. At 2 bits all three are off the scale: Kev-0.8B 15, Kev-4B 20 and Qwen 46 points off.](/public/images/shrinking-ai-models/post4_honesty.png){: data-dark-src="/public/images/shrinking-ai-models/post4_honesty-dark.png"}

- **8 bits and 4-bit NF4 changed almost nothing.** Every model stayed within half a point of its full-size honesty.
- **4-bit HQQ moved two models in opposite directions.** Kev-0.8B went from 4.4 points off to 2.3: more honest. Qwen went from 2.4 to 3.2: less. Shrinking nudged Kev-0.8B's average answer up from 18.1% to 19.9%, toward the 22.1% who actually defaulted (the last post found Kev understates risk), and nudged Qwen's down from 22.1% to 19.2%. Nobody chose those directions. They're side effects of rounding. For scale, retraining Kev-0.8B from a different random start moved it by 1.2 to 1.5 points in the last post.
- **3 bits was mixed:** Kev-4B got worse (4.8), Kev-0.8B and Qwen better (3.1 and 1.5).
- **2 bits broke all three:** 15, 20 and 46 points off. Qwen's average answer was 67%, against 22% who defaulted.

One twist at 2 bits. Apply the same correction as in the last post, learned from 2,000 separate customers, and the broken models come out about 1 point off: perfectly honest. That's because they say about 22% to almost everyone. A model that can't tell customers apart can still promise the right average. Honest and useless, which is why this series always measures ranking and honesty separately. (Down to 3 bits, the same correction left every version 1.4 to 3.3 points off, as at full size.)

## The scores held while the customers moved

So far, so reassuring, down to 3 bits. But those are summary scores, and summaries can hide a lot. Here is each test customer's answer from Kev-0.8B, full size against shrunk:

![Each dot is one test customer: Kev-0.8B's full-size answer against its shrunk answer, in four panels. At 4-bit NF4 the dots hug the no-change line: the average customer moved 0.9 points and 5 in 1,000 moved 5 points or more. At 4-bit HQQ, 3.0 points, and 8 in 100 moved 5 or more. At 3 bits, 6.6 points, and 62 in 100 moved 5 or more, with low answers pushed up and high answers pulled down, yet ranking 0.786 and 3.1 points off, as good as full size (0.790, 4.4 points off). At 2 bits the dots form a flat band between about 20% and 55% whatever the full-size answer, 26.6 points on average.](/public/images/shrinking-ai-models/post4_drift.png){: data-dark-src="/public/images/shrinking-ai-models/post4_drift-dark.png"}

- **At 4-bit NF4, the answers barely moved:** less than a point for the average customer.
- **At 3 bits, 6 in 10 customers' answers moved by 5 points or more.** Yet Kev-0.8B's ranking (0.786) and honesty (3.1 points off) looked as good as at full size. Its safest customers were pushed up and its riskiest pulled down. That squeezed the answers toward the middle, but it mostly kept their order and kept the averages roughly right.
- **At 2 bits, the answer hardly depends on the customer at all.**

The other two models show the same pattern, less strongly: at 3 bits, 31 in 100 of Kev-4B's customers and 26 in 100 of Qwen's moved 5 points or more.

A dashboard of summary scores would have passed the 3-bit Kev-0.8B. But each customer's number feeds a decision.

## What it does to a lending decision

To see what that means for a lender, I used the setup from [post 2](/2026/11/24/why-99-percent-accuracy/). A default costs four times what a paying customer earns, and the lender approves everyone below a cut-off chosen on 2,000 customers whose outcomes are known.

| Version | Approve/decline decisions that differ from full size |
|---|---|
| The same full-size model, run a second way | 0.2% (Kev), 1.4% (Qwen) |
| 8 bits | 1.3–3.3% |
| 4 bits, NF4 | 1.7–3.0% |
| 4 bits, HQQ | 1.8–4.9% |
| 3 bits | 6.2–9.8% |
| 2 bits | 48–72% |

If the lender re-chooses each version's cut-off on the customers with known outcomes, every version down to 3 bits earns about the same: 281 to 304 per 1,000 customers, against 287 to 300 at full size. At 2 bits, it earns nothing (−10 to 73).

But a lender who swaps in the shrunk model and keeps the old cut-off ends up with a different lending policy. The 3-bit Kev-4B would approve 43 in 100 applicants instead of 62, and earn 253 per 1,000 instead of 294. The 4-bit HQQ Qwen, whose answers ran lower, would approve 70 instead of 63.

## Smaller, but slower

Now the reason to shrink a model at all: cost.

![GPU memory at peak and median time per answer at each size. Memory falls with every step: the 4B models go from 8.1 to 8.3 GB at full size to 3.2 to 3.6 GB at 4 bits and 2.5 to 2.7 GB at 2 bits, and Kev-0.8B from 1.6 GB to 0.9 GB at 4 bits. Time per answer rises at every shrunk size: at full size, 62 ms for Kev-0.8B, 111 ms for Kev-4B and 120 ms for Qwen; at 4-bit NF4, 81, 139 and 152 ms; at 8 bits up to 289 ms; at HQQ 3 bits up to 467 ms for Kev-4B.](/public/images/shrinking-ai-models/post4_cost.png){: data-dark-src="/public/images/shrinking-ai-models/post4_cost-dark.png"}

- **Memory fell as promised.** At 4 bits the two 4B models needed 3.2 to 3.6 GB at peak instead of 8.1 to 8.3, about 40%.
- **Every shrunk version was slower.**
  - 4-bit NF4 was a quarter to a third slower: Kev-4B went from 111 to 139 milliseconds per answer, Qwen from 120 to 152 and Kev-0.8B from 62 to 81.
  - 8 bits took 1.4 to 3.5 times as long.
  - HQQ took up to about 4 times as long: 426 milliseconds for Kev-4B at 4 bits, 467 at 3.

A rented GPU bills by the minute, so per decision, slower means more expensive. Kev-4B cost 1.4 cents per 1,000 decisions at full size, and 5.2 cents at 4-bit HQQ. All small amounts, but the shrunk model costs more to run, not less.

It gets worse in practice. Kev's own server has a shortcut, called CUDA graphs, that records the GPU's sequence of steps once and replays it for every request. It only works with full-size numbers. With it on, the last post's full-size Kev-0.8B answered in 24 milliseconds. Shrunk to 4 bits, it can't use the shortcut, and took 81: more than three times as long. (To compare sizes fairly, I measured every version here with the shortcut off.)

Why slower? Each answer is one pass over a summary a few hundred words long, which the GPU does as one big block of arithmetic. Most likely the limit there is how fast the GPU can calculate, not how fast it can read its memory. Shrunk numbers have to be unpacked back to full size before each multiplication, which adds work. Shrinking pays off when memory is the limit: writing long answers word by word, serving many customers at once, or fitting a model onto a smaller, cheaper chip. On a GPU with room to spare, as here, it bought nothing but slower answers. Cheaper AI doesn't always mean cheaper bills.

## I got most of my predictions wrong, again

Before any of these runs, I [wrote down](https://github.com/nishzsche/can-ai-know-when-its-wrong/blob/main/PREDICTIONS.md) what I expected:

| My guess | What happened |
|---|---|
| Honesty would get clearly worse already at 8 bits | ✗ At 8 bits every model stayed within half a point of full size. The first clear break came at 2 bits |
| Ranking would hold at 4 bits for all three | ✓ Within 0.005, with either tool |
| 4-bit NF4 would be faster than full size | ✗ A quarter to a third slower, for all three |
| Qwen would hold up best, its percentages changing least | ✗ Mostly. Kev-4B changed least at 8 and 4 bits (0.3 to 1.1 points per customer). Qwen changed least at 3 bits (4.0 points), then broke hardest at 2 (49) |

## What I'd tell a bank

- **A shrunk model is a new model, so validate it like one.** Look at individual scores and cut-offs, not just summary scores: at 3 bits, the summaries held while most of one model's customers moved by 5 points or more.
- **8 bits and the common 4-bit format were safe here.** Each moved the average customer by less than a point. Below 4 bits, test hard. 2 bits was broken.
- **The tool matters as much as the size.** At the same 4 bits, HQQ moved the answers 1.5 to 3.2 times as much as NF4.
- **Shrink to fit hardware, not to save time.** Measure speed and cost on your own setup. Here, every shrunk version was slower.
- **Record the size with every score.** Each prediction here carries its precision and library versions, because the same model at a different size gives different answers.

## What this test can't tell you

- **Three models, one dataset, one way of serving:** a bank in Taiwan in 2005, one NVIDIA L4 GPU, one customer at a time. Speeds will differ on other hardware, with many requests at once, and with code built specifically for shrunk models.
- **Only methods that need no data.** Methods that tune their rounding on sample data, such as GPTQ and AWQ, often do better at 3 and 4 bits. I didn't test them.
- **One training run per model.** Retraining alone moved honesty by up to 1.8 points in the last post, as much as most of the shrinking did here.
- **Shrunk after training.** Training a model that's already shrunk, or in low precision from the start, could behave differently.
- **Small gaps are hard to see:** with 2,000 test customers, ranking differences under about 0.02 can't be told apart. These test customers have also now been used in four posts. 6,000 customers that no model has seen are kept aside for one final replication.
- **"Default" is the dataset's label:** missing next month's payment, not the 90-days-overdue definition banks use for capital.

## How I ran it

Each model was merged with its training update first and then shrunk, so the result is one model, as you'd deploy it. The part of each model that turns its reading into a probability stays full size. Both tools' versions are pinned and recorded in every prediction. Kev doesn't support shrinking, so I wrapped its loader. Before any shrunk run, the full-size version had to reproduce the last post's predictions through the new code, and it did, within 0.002 on average. That's 72,000 predictions in all, in about 6 hours of GPU time for about $2.55. The measures were fixed before the runs, except the lending-decision counts, which I added while writing.

## What's next

Every number, the code and every prediction are on [GitHub](https://github.com/nishzsche/can-ai-know-when-its-wrong). This series started with [whether AI can know when it's wrong](/2026/11/23/can-ai-know-when-its-wrong/), [why 99% accuracy can still be a disaster](/2026/11/24/why-99-percent-accuracy/) and [giving AI the same data as a bank's model](/2026/11/25/fair-fight/).

Next: why AI can write code but struggle to read a table. I gave six AI models the same customers as a table, as sentences, as CSV and as JSON. For one of them, a different layout moved its answers by 26 points per customer on average, about four times what shrinking to 3 bits did to any model here. And the sequel I promised in the first post, teaching a small model to say "I don't know", is still coming.
