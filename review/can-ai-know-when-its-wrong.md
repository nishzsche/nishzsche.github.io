---
layout: post
title: "Can AI actually know when it's wrong?"
date: 2026-11-23
tags: [probability]
permalink: /review/can-ai-know-when-its-wrong/
# Review copy: not listed on the home page or in the feed, hidden from search engines,
# with Hypothesis loaded for comments. To launch, move it to _posts/ and drop these three.
noindex: true
hypothesis: true
---

> **Draft for review, not yet published.** Thank you for reading. To comment, select any text and click **Annotate**, or open the sidebar with the arrow at the top right. Commenting needs a free [Hypothesis](https://web.hypothes.is/) account. I'd most like to know: where did you get lost or bored, does the twist in finding 3 land, and does anything feel overstated?

*I gave eight models the same 2,000 real credit-card customers and checked whether their percentages meant what they said.*

When a weather app says there's a 70% chance of rain, you don't expect rain every time. You expect it on about seven of every ten days the app says that. If it rains on all ten, or on two, the app wasn't wrong on any single day. It was wrong about how sure it was.

Banks are held to exactly this standard. Every lending decision rests on a number: the probability that this customer won't pay back. Banks that build their own risk models are [expected to show](https://www.bis.org/basel_framework/chapter/CRE/36.htm) that those probabilities hold up against what actually happens. If a model says 10% and 30% of those customers default, there's a problem, however well it ranked them.

AI models hand out percentages now too. A [new kind of model](https://typesafe.ai/blog/introducing-system-one-models-and-jev), built only for yes-or-no decisions, returns an answer with a confidence attached, and any chatbot will give you a probability if you ask. So I wanted to know: when an AI says "30% likely", does it mean 30%? Statisticians call this property [calibration](https://scikit-learn.org/stable/modules/calibration.html).

I spent years building credit-risk models, where an honest probability matters more than a clever one. So I ran the test a bank would run.

## The experiment

I used a [well-known public dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients): 30,000 credit-card customers of a bank in Taiwan in 2005, and whether each one missed their next payment. About 22% did. I split the customers into four groups:

- **Test (2,000):** every model is scored on these same customers, and no model or setting was ever adjusted using them.
- **Train (20,000):** for teaching the two classic models.
- **Tune (2,000):** customers with known outcomes, used only to correct the AI models' percentages ("recalibration", below).
- **Reserve (6,000):** never used.

Each contestant saw the same information about each customer: the credit limit and six months of bills, payments and repayment status. Nothing else. I left out age, sex, marital status and education on purpose. A lending model shouldn't use them, and leaving them out keeps the comparison clean. For the AI models, that became a short text summary:

```
Credit card account summary
Credit limit: NT$160,000

Month (most recent first) | Bill amount | Amount paid | Repayment status
Sep 2005 | NT$30,265 | NT$131 | paid on time (code -1)
Aug 2005 | -NT$131 (credit) | NT$396 | paid on time (code -1)
Jul 2005 | -NT$527 (credit) | NT$396 | not late (code -2)
...
```

followed by one question: *will this customer miss next month's payment?*

Eight contestants answered. Together they cover the main ways a business can get a probability out of AI today: buy a model built for decisions, run an open one yourself, read one out of a general-purpose model, or simply ask a chatbot. Two classic models set the bar.

| Contestant | What it is | Why it's here |
|---|---|---|
| **[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)** (TypeSafe) | A commercial "decision model": ask it a yes-or-no question and it answers with a confidence | The newest kind of AI built for exactly this job, and sold on its confidence |
| **[Kev-4B, Kev-0.8B](https://github.com/jaredpalmer/kev)** | Open-source decision models, in a larger and a smaller size, free to run on your own hardware | What a bank could run in-house, where customer data can't leave the building |
| **[Laya](https://huggingface.co/convaiinnovations/laya)** | A much smaller open decision model | Can a tiny, fast model be honest? |
| **[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)** | A small, free, general-purpose language model. I read its probability straight from its internals: how much weight it put on "yes" versus "no" ([token log-probabilities](https://cookbook.openai.com/examples/using_logprobs)) | The do-it-yourself route, with any open model |
| **[GPT-6.2 Sol](https://openrouter.ai/openai/gpt-6.2-sol)** | The model ChatGPT starts new chats on, asked for a probability from 0 to 100 | What most people would actually do: ask ChatGPT |
| **[Scorecard](https://en.wikipedia.org/wiki/Credit_scorecards)** | A [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression), trained on the 20,000 train customers | The kind of model banks have used for decades |
| **[Gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting)** | The standard modern machine-learning model, trained on the same 20,000 | The modern bar |

The six AI models answered cold, with no examples ([zero-shot](https://en.wikipedia.org/wiki/Zero-shot_learning)).

That difference matters, and I'll repeat it wherever I compare them: the classic models learned from 20,000 labelled examples, and the AI models got none. But that *is* the practical question. Do you need the new AI, or the old model and some data?

Each AI model answered three ways:
1. **Out of the box.**
2. **With one extra sentence** stating the [base rate](https://en.wikipedia.org/wiki/Base_rate): "In this portfolio, about 22.1% of customers miss next month's payment."
3. **After [recalibration](https://scikit-learn.org/stable/modules/calibration.html):** like fixing a bathroom scale that always reads 3 kg heavy. You work out the correction on people whose weight you know, then apply it to everyone else. Here the correction was learned from the 2,000 tune customers and applied, unchanged, to the test customers.

## What "honest" looks like

Here's the whole test in one picture, a [reliability diagram](https://scikit-learn.org/stable/modules/calibration.html#calibration-curves). For each model, I sorted the customers from lowest predicted risk to highest and split them into ten equal groups. Then I plotted what the model promised for each group against what actually happened.

![Promised vs actual default rate](/public/images/can-ai-know-when-its-wrong/1_promised_vs_actual.png)

How to read it:
- **Each dot is a tenth of the customers.** Take the top-right dot in Jev's panel: the 200 customers Jev rated riskiest. Jev promised that about 80% of them would default; 63% did.
- **On the diagonal, the model is honest.** When it says 30%, 30% default.
- **Below the diagonal, it's too pessimistic.** It promised more defaults than happened, as with that Jev dot. Above the diagonal, it's too optimistic.
- **A flat line means it can't tell customers apart.** Kev-4B's promises range from 24% to 67%, but what actually happened only ranges from 14% to 30%.
- **The grey line is the scorecard,** and it hugs the diagonal.

I'll sum up the gap as one number: the average distance between promised and actual, known as *expected calibration error* ([ECE](https://arxiv.org/abs/1706.04599)). The scorecard's is 0.014, so its percentages are off by about 1.4 points on average.

## What I found

### 1. Most AI models think customers are riskier than they are

Five of the six overstated the risk, and two of them more than doubled it:
- **Jev** said 48% of these customers would default, on average, and **Kev-4B** said 49%. The real figure was 22%.
- **What a bank would see:** acting on Jev's numbers, it would find 70% of its customers in the "40% or more" risk band. In that group, 26% actually defaulted.
- **The two close ones:** only Laya and GPT-6.2 Sol were near the truth out of the box, both about 5 points off.

![What a bank would see](/public/images/can-ai-know-when-its-wrong/4_risk_bands.png)

### 2. One sentence fixes most of it

Telling the models the portfolio's default rate improved calibration for five of the six:
- **Kev-4B** went from 27 points off to 3. It also got much better at telling risky customers from safe ones, which I didn't expect: the sentence changed *which* customers it flagged.
- **Kev-0.8B** went the wrong way: its error tripled, from 9 points to 27.

Context helps, but only if you check that it does.

![The honesty arc](/public/images/can-ai-know-when-its-wrong/2_honesty_arc.png)

### 3. The most "honest" model had seen the answers

GPT-6.2 Sol looked like the star: about 5 points off out of the box and 3 with the base rate. So I checked whether it had seen this data before. It had.

- **It could recite 31 customers from the original data file, word for word**, in standard [memorization tests](https://arxiv.org/abs/2404.06209). Only 3 of them are in my test set, and leaving them out changes nothing.
- **It recognises the dataset.** Shown any one of my summaries and asked where it came from, it named the dataset every time, 53 out of 53, and gave its exact default rate: 22.12%.
- **Hiding the obvious clues didn't help.** I removed the currency, the dates and the raw codes, and it still recognised the dataset 50 times out of 50, from the shape of the data alone.
- **Then I told it a false rate,** "about 10%", and it moved about 70% of the way there. So it does listen to what it's told. When told the true rate, it barely moved, because it already knew.

Sol's percentages are honest here most likely because it knows this population, not because it judged each customer well from scratch. On a bank's own customers, there's no famous dataset to remember. (In fairness to Sol, I can't fully rule out that it would guess about 25% for any credit-card portfolio.)

### 4. The old scorecard still ranks customers best

Honest percentages are half the job. The other half is putting risky customers above safe ones, measured by a score called [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve), where 0.5 is a coin flip and 1.0 is perfect.
- **The classic models:** the scorecard scored 0.777, and gradient boosting 0.789.
- **The AI models:** none beat the scorecard, though the best came within the margin of error ([bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29) 95% intervals). Kev-4B with the base rate scored 0.757, and Sol 0.754.

![Ranking ability](/public/images/can-ai-know-when-its-wrong/3_ranking_auc_ks.png)

### 5. You can buy honesty back, with labels

Recalibration on the 2,000 tune customers brought every AI model's error down to between 0.3 and 4 points, close to the scorecard's 1.4. It doesn't change how well a model ranks, only what its numbers mean.

The catch: labelled outcomes are exactly what zero-shot AI was supposed to free you from needing. And with 20,000 of them, a scorecard ranks better than any of the AI models.

### 6. What it costs

- **Sol:** about 74 cents per 1,000 decisions, at around two seconds each.
- **Jev, through its API:** about 2 cents per 1,000.
- **The open models, on one rented GPU:** under 2 cents per 1,000, with answers in milliseconds.
- **The scorecard:** essentially free to run.
- **The whole experiment,** every check included, cost about 14 dollars in API calls plus a few more for GPU time.

![Cost vs ranking ability](/public/images/can-ai-know-when-its-wrong/5_cost_vs_auc.png)

## I got most of my own predictions wrong

Before running anything, I wrote down my guesses and committed them to the repository, so I couldn't quietly revise them later.

| My guess | What happened |
|---|---|
| Jev would be the most honest | ✗ Jev was among the least honest out of the box |
| Laya would be the least honest | ✗ Laya was among the most honest |
| Telling the base rate would help most models | ✓ It helped five of six |
| At least one AI model would beat the scorecard at ranking | ✗ None did |

One out of four. Which is rather the point: confident guesses need checking.

Looking back, the evidence was in front of me. The 20-customer test run I did before guessing had Jev averaging 55% and Laya 23%, on customers who defaulted 20% of the time. I went with reputation over a small sample, and the small sample was right. The next section explains what was going on: Jev wasn't so much dishonest as starting from the wrong assumption, Laya looked honest partly because its answers barely move, and the scorecard had learned something about this data that no AI model knew.

## Why the results look like this

I broke the test customers down by their most recent repayment status and compared what actually happened with what each model predicted.

| Most recent status | Customers | Actually defaulted | Jev | Kev-4B | Laya | GPT-6.2 Sol | Scorecard |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paid on time (code −1) | 405 | 14% | 30% | 37% | 12% | 13% | 15% |
| Not late (code 0) | 971 | 12% | 51% | 54% | 20% | 18% | 13% |
| 1 month late | 245 | 37% | 61% | 51% | 27% | 51% | 36% |
| 2 months late | 170 | 72% | 75% | 58% | 34% | 74% | 66% |

- **One group causes most of the overstatement.** Half the customers are "not late (code 0)": they carry a balance and pay part of it, on time. They're the safest group in the data, and only 12% defaulted. Jev predicted 51% for them and Kev-4B 54%, which accounts for about three quarters of both models' overstatement. To a lender, paying part of a balance on time is ordinary; it's how card issuers earn interest. The models seem to read a small payment against a large bill as a customer in trouble.
- **Some models start near even odds.** Kev-4B gives 80% of customers between 30% and 64%, centred near 49%, and Jev averages 48%. My best explanation is that decision models learn from tasks where "yes" isn't rare, so with no base rate they start close to 50/50. [Kev's published training data](https://github.com/jaredpalmer/kev) fits this: general text-classification sets, and nothing like this data. Jev's training data isn't published. This is exactly what the base-rate sentence corrects.
- **Honest on average isn't the same as useful.** Laya's percentages are close to right on average, but they barely move with risk. It gives customers two months late 34%, when 72% of them defaulted. Calibration and ranking are separate skills, and Laya has more of the first.
- **Sol tracks reality group by group:** 13% for on-time payers (14% actual) and 74% for two months late (72% actual). That fits a model that knows this data, as finding 3 suggests.
- **Small models can misread context.** Told that "about 22% of customers miss a payment", Kev-0.8B raised its average from 31% to 49%. It seems to treat the sentence as a warning sign, not a statistic.
- **The ceiling is low.** Even gradient boosting, trained on 20,000 customers, reaches only 0.789. Six months of payment history can only tell you so much. The best AI models' 0.75 is about 88% of the way from a coin flip to the best trained model.
- **The scorecard learned what the AI models had to guess.** From 20,000 labelled examples, it learned what this data's repayment codes really mean, starting with the most important lesson: customers paying part of their balance on time are the safest, not the riskiest.

## What would change with different data

**The portfolio's default rate changes who looks honest.** These customers defaulted at 22%. I reweighted the same customers to imitate portfolios where 5% or 50% default: the same kinds of defaulters and non-defaulters, in different proportions. Then I measured calibration error again, in points off:

| If the default rate were | 5% | 22% (actual) | 50% |
|---|---:|---:|---:|
| Jev | 41 | 26 | 11 |
| Kev-4B | 43 | 27 | 5 |
| Kev-0.8B | 23 | 9 | 15 |
| Laya | 14 | 5 | 28 |
| Qwen3.5-4B | 28 | 14 | 10 |
| GPT-6.2 Sol | 17 | 5 | 17 |
| Scorecard (trained at 22%) | 13 | 1.4 | 22 |

- **In a low-risk book, everyone overstates.** In a typical prime card portfolio, where far fewer than 22% default, every model looks worse, and the models that start near even odds look far worse.
- **In a high-risk book, the ranking flips.** At 50%, Kev-4B becomes the most honest model and Laya the least.
- **The scorecard is no exception.** It's honest at 22% because that's the rate it was trained on.
- **Ranking ability doesn't move.** AUC compares defaulters with non-defaulters, so the mix doesn't affect it. Honesty does.

So "is this model honest?" only has an answer for a particular portfolio, and you need some of your own labelled outcomes to find it. Other things that would likely change the picture:
- **Data the models haven't seen.** On a bank's own customers, Sol can't recognise the portfolio, and I'd expect its out-of-the-box honesty to fall back towards the others'. I haven't tested this.
- **A different era.** 2005 was the height of Taiwan's credit-card debt crisis, [as the dataset's authors note](https://doi.org/10.1016/j.eswa.2007.12.020), which is why so many customers defaulted. A normal year means a lower default rate: see the table.
- **Richer data.** Income, credit-bureau scores or a longer history would raise the ceiling for the trained models. Whether they help the AI models depends on whether a longer summary stays easy to read.
- **Different wording.** I used one fixed template for every AI model. Other wordings could shift these numbers, and that's worth testing before trusting any one result.

## What I'd tell a bank

- **Ask "when it says 30%, how often is it right?", not just "how accurate is it?"** A model can rank customers well and still give you numbers that are double the truth.
- **Don't take a zero-shot percentage at face value.** Hold back a labelled sample from your own portfolio and recalibrate. Here, 2,000 customers was enough. Honesty depends on your default rate, so a test on someone else's data won't tell you.
- **Tell the model your base rate.** It's free, and it helped five of six models. But test it, because it broke one.
- **Test on data the model can't have seen.** Famous public datasets flatter large language models, which can be remarkably good at recognising them.
- **For ranking alone, the old scorecard is hard to beat.** The case for AI here isn't better risk ranking, at least not yet.

## What this test can't tell you

- **One dataset:** one bank, one country, 2005. The section above shows how much the default rate alone could change.
- **A famous dataset:** AI models have seen it in training. I tested the ones I could, but the decision models can't be probed this way.
- **No demographic fields, on purpose:** every contestant lost them equally.
- **Partly undocumented repayment codes:** so I showed the AI models the raw codes rather than guess their meaning.
- **Not like for like:** the AI models answered cold while the classic models were trained. That's the real-world trade-off, but it isn't an even contest.
- **Models change:** these results are for the versions I tested in November 2026, all recorded.

## What's next

Dishonest percentages are one way a model can mislead you. A single headline number is another, so next I'll look at why 99% accuracy can still be a disaster. After that comes the sequel to this post: training my own small decision model and rewarding it for knowing when it doesn't know.

---

## Appendix: method and how to reproduce

Code, every prediction and every check: [github.com/nishzsche/can-ai-know-when-its-wrong](https://github.com/nishzsche/can-ai-know-when-its-wrong).

**Data**
- **Source:** [UCI "Default of Credit Card Clients"](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients): Yeh, I. C., & Lien, C. H. (2009), *Expert Systems with Applications* 36(2), 2473–2480, [doi:10.1016/j.eswa.2007.12.020](https://doi.org/10.1016/j.eswa.2007.12.020). Licence: CC BY 4.0.
- **Splits:** 30,000 customers, split with a fixed seed and stratified on the outcome: train 20,000, tune 2,000, test 2,000, and a reserve of 6,000 that is never used. The default rate is 22.1% in each.
- **Inputs:** 19 fields (credit limit, and six months each of bill amount, amount paid and repayment status). Sex, marital status, age and education are dropped before any rendering or training.

**The question**
- **One template, unit-tested and versioned (v2),** renders every customer for every AI model. Repayment codes are shown raw, with only their documented meaning, because the common interpretation of the undocumented codes is contradicted by the data.
- **The base-rate condition** puts one sentence in front of the question. Recalibration picks [Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling) or [isotonic regression](https://en.wikipedia.org/wiki/Isotonic_regression) per model by 5-fold [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29) on tune, then applies it to test. Nothing is ever tuned on test.

**Contestants and versions**

| Contestant | Version and setup |
|---|---|
| Jev | `typesafe/jev-1.14-20261014` via [OpenRouter](https://openrouter.ai)'s System One endpoint |
| Kev-4B, Kev-0.8B | [`jaredpalmer/kev`](https://github.com/jaredpalmer/kev) v1.0, served on an NVIDIA L4 |
| Laya | `convaiinnovations/laya` at revision `fd4f9dbc`, on the L4 |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` at `851bf6e8`, served by [vLLM](https://github.com/vllm-project/vllm) with thinking off. P(yes) comes from the first answer token's top-20 [logprobs](https://cookbook.openai.com/examples/using_logprobs), with every spelling of each option combined |
| GPT-6.2 Sol | `openai/gpt-6.2-sol` via OpenRouter: reasoning effort medium, seed 0, strict JSON answer |
| Scorecard | [Weight-of-evidence binning](https://en.wikipedia.org/wiki/Credit_scorecards) plus logistic regression, all 19 fields |
| Gradient boosting | [LightGBM](https://github.com/microsoft/LightGBM), 157 rounds; isotonic-calibrated on tune |

**Metrics**
- **Ranking:** [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) and [KS](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test).
- **Calibration:** calibration error ([ECE](https://arxiv.org/abs/1706.04599), 10 equal-width bins), [Brier score](https://en.wikipedia.org/wiki/Brier_score), and [calibration-in-the-large and calibration slope](https://doi.org/10.1186/s12916-019-1466-7).
- **Intervals:** [bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29) 95% intervals (1,000 resamples) for AUC, ECE and calibration-in-the-large.
- **Records:** every prediction is saved with its probability, model version, latency and cost.

**Checks on my own results**
- **Memorization tests** ([tabmemcheck](https://github.com/interpretml/LLM-Tabular-Memorization-Checker), from [Bordt et al., 2024](https://arxiv.org/abs/2404.06209), plus a longer, deterministic version with room for the model's reasoning): GPT-6.2 Sol reproduces 31 customers verbatim, 3 of them in test. Without those 3, no metric for any model moves by more than 0.0006. Qwen3.5-4B showed no memorization. The decision models can't be probed this way.
- **Recognition:** Sol names the dataset for 53 of 53 summaries and states its default rate (22.12%). With the currency, dates and raw codes removed, it still does, 50 of 50.
- **False base rate:** on 500 test customers told "about 10.0%", Sol's average fell from 0.258 to 0.175, a pass-through of 0.69 (95% interval 0.64–0.73), with its ranking unchanged.
- **Qwen option order:** swapping which letter means "yes" left its average probability almost unchanged (−0.004), and the two orders' answers correlated at 0.91, so there's no letter bias.
- **A bug in a library:** jevper, an open-source adapter used to prompt Qwen, mis-read its probabilities when the model spelled an option more than one way. I computed them myself instead and [reported the bug upstream](https://github.com/zhulinchng/jevper/issues/1).

**Reproduce**

```bash
uv run python data/load.py && uv run python data/splits.py
uv run python data/render.py --split test
uv run python baselines/scorecard.py && uv run python baselines/gbm.py
uv run --group runners python runners/<contestant>.py --split test   # needs API keys / a GPU
uv run python analysis/recalibrate.py
uv run python analysis/summarize.py && uv run python analysis/figures.py
```
