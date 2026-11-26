---
layout: post
title: "Why 99% accuracy can still be a disaster"
date: 2026-11-24 12:55:00 +0530
tags: [probability]
description: "A model that says nobody will default is right 78% of the time on real credit data. A bank that used it would lose money."
image: /public/images/why-99-percent-accuracy/social-card.png
image_alt: "Why 99% accuracy can still be a disaster. Profit per 1,000 customers: approving everyone (77.9% accurate) loses 105; Qwen3.5-4B (81.9% accurate) earns 17; the scorecard (81.8% accurate) earns 284."
toc: true
icon: /public/drawings/20261027-155301.jpg
---

*On my credit data, a model that says "nobody will default" is right 78% of the time. A bank that used it would lose money.*

Picture a card network where one payment in every 1,000 is fraud. I can build a fraud detector in one line: it says "not fraud" every time. It is 99.9% accurate. It has never caught a fraudster, and it never will.

Accuracy counts how often a model agrees with what happened. When one answer is far more common than the other, agreeing with the majority is easy and worthless. The rarer the thing you care about, the better a model that ignores it will score.

In my [last post](/2026/11/23/can-ai-know-when-its-wrong/) I gave eight models the same 2,000 credit-card customers and asked how likely each one was to miss next month's payment. That post asked whether their percentages came true. This one asks something simpler: if you had judged them by the headline score most people quote, how would you have done?

## The model that does nothing

22.1% of the test customers missed their payment. So a "model" that says nobody will default is right 77.9% of the time. That's the floor: every model has to be judged against 77.9%, not against zero.

Here is how the eight did, counting a predicted risk of 50% or more as "will default", the usual cut-off:

| Model | Accuracy | Defaulters caught |
|---|---|---|
| Say "won't default" to everyone | 77.9% | 0% |
| Gradient boosting* | 82.5% | 36% |
| Qwen3.5-4B | 81.9% | 44% |
| Scorecard* | 81.8% | 34% |
| GPT-6.2 Sol | 81.7% | 46% |
| Kev-0.8B | 80.8% | 40% |
| Laya | 76.5% | 13% |
| Jev | 60.9% | 71% |
| Kev-4B | 55.2% | 64% |

\*Trained on 20,000 customers whose outcomes were known. The AI models answered cold, with no examples.

{% include chart.html src="/public/images/why-99-percent-accuracy/post2_accuracy.png" alt="Each model's accuracy at the usual 50% cut-off, beside the share of the 442 defaulters it caught." words="Doing nothing, saying “won't default” to everyone, is 77.9% accurate and catches none. Gradient boosting is the most accurate at 82.5%, 4.6 points above doing nothing. Laya (76.5%), Jev (60.9%) and Kev-4B (55.2%) are less accurate than doing nothing, yet Jev and Kev-4B catch the most defaulters, 71% and 64%." %}

Three things stand out:

- **The best model beats doing nothing by less than 5 points.**
- **Laya is less accurate than doing nothing.**
- **By accuracy, Qwen ties the scorecard** (81.9% against 81.8%), and Jev and Kev-4B look like failures, even though they catch the most defaulters.

So which model is best? Accuracy can't say, because it's answering the wrong question.

## Three things accuracy hides

**Which mistakes it makes.** Accuracy counts approving a customer who then defaults and turning away one who would have paid as the same mistake. To a lender they're not the same at all. Of the 2,000 test customers, 442 defaulted and 1,558 paid. Here's where each of them landed, at the same 50% cut-off:

| At the 50% cut-off | Defaulters caught | Defaulters missed | Payers turned away | Payers approved |
|---|---|---|---|---|
| Say "won't default" to everyone | 0 | 442 | 0 | 1,558 |
| Scorecard* | 151 | 291 | 74 | 1,484 |
| Jev | 316 | 126 | 656 | 902 |

This four-way count is called a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix), and accuracy only adds up two of its columns: defaulters caught plus payers approved. The other two are where the money goes. The scorecard misses two defaulters in three but turns away few good customers. Jev catches 71% of the defaulters, but only by turning away 656 people who would have paid.

{% include chart.html src="/public/images/why-99-percent-accuracy/post2_confusion.png" alt="Where the 2,000 test customers landed at the 50% cut-off, for doing nothing, the scorecard and Jev." words="Of the 442 who defaulted, doing nothing misses all 442, the scorecard misses 291 and Jev 126. Of the 1,558 who paid, doing nothing turns away none, the scorecard 74 and Jev 656. Accuracy adds up only the right calls." %}

**Whether it can rank.** Pick one customer who defaulted and one who paid, at random. How often does the model rate the defaulter as riskier? That's a model's ranking power, known as [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve), and 50% is a coin toss. The scorecard gets it right 78 times in 100, gradient boosting 79, Kev-0.8B 75 and Qwen 74. No cut-off is involved.

**Whether its percentages mean anything.** That was the last post. Qwen's average prediction was 35% for a group where 22% defaulted: it ranks decently, but its numbers run high.

Accuracy folds all three into one number at one arbitrary cut-off. So a model can be more accurate and worth less.

## What a lender actually does with a model

A lender doesn't care how often a model is right. It cares what its decisions earn. So let's make some decisions.

Say each customer who pays earns the bank 1 unit, and each customer who defaults costs it 4 units. These numbers are illustrative, and I check other ratios below. Then a customer is worth approving only if their chance of default is under 20%: at exactly 20%, the 0.8 units they're expected to earn match the 0.8 units they're expected to cost.

First, the model that does nothing. It says nobody defaults, so it approves everyone. Per 1,000 customers that earns 779 units and loses 221 × 4 = 884: a loss of 105. **The 78%-accurate model loses money.** It would at any cost ratio above about 3.5 to 1.

A bank can use each real model in two ways:

1. **Take it at its word:** approve everyone it rates under 20%.
2. **Use only its ranking:** approve its safest customers, choosing how far down the list to go by trying cut-offs on 2,000 other customers whose outcomes are known. Nothing is chosen using the test customers.

| Model | Accuracy | Approved at its word | Profit per 1,000, at its word | Profit per 1,000, cut-off from known outcomes |
|---|---|---|---|---|
| Gradient boosting* | 82.5% | 67% | 303 | 304 |
| Scorecard* | 81.8% | 71% | 284 | 285 |
| GPT-6.2 Sol | 81.7% | 61% | 294 | 286 |
| Kev-0.8B | 80.8% | 22% | 99 | 306 |
| Qwen3.5-4B | 81.9% | 5% | 17 | 252 |
| Laya | 76.5% | 69% | 148 | 149 |
| Jev | 60.9% | 8% | 14 | 226 |
| Kev-4B | 55.2% | 2% | 2 | 49 |
| Approve everyone | 77.9% | 100% | loses 105 | |

\*Trained on 20,000 customers whose outcomes were known; the AI models answered cold.

{% include chart.html src="/public/images/why-99-percent-accuracy/post2_profit.png" alt="Profit per 1,000 customers for each model, sorted by accuracy." words="Gradient boosting, the scorecard and GPT-6.2 Sol earn about 300 either way. Qwen earns 17 on its own percentages and 252 with a learned cut-off, Kev-0.8B 99 and 306, Jev 14 and 226. Approving everyone loses 105." %}

- **Qwen is as accurate as the scorecard.** Taken at its word, it approves 5 customers in 100 and earns 17 units per 1,000, against the scorecard's 284. Its percentages run about 1.6 times too high, so almost everyone looks too risky to lend to. The confusion matrix at the 20% cut-off shows it plainly: Qwen catches 428 of the 442 defaulters, but turns away 1,468 of the 1,558 customers who would have paid. The scorecard, at the same cut-off, catches 272 and turns away 309.
- **Kev-0.8B is the most striking.** It ranks well enough that, with a cut-off learned from known outcomes, it earns 306, the most of any model in the table. Taken at its word, it earns 99. Two-thirds of its value is lost to percentages that don't mean what they say.
- **Jev looks hopeless by accuracy (61%)**, yet used for its ranking alone it earns 226. That's more than Laya, which is 16 points more accurate.
- **The honest models earn about the same either way.** Gradient boosting, the scorecard and Sol make roughly as much taking their own word as using a cut-off from known outcomes. That's what honest percentages buy you: you can use the number as it comes. (Sol carries the caveat from the last post: it recognizes this dataset and knows its default rate.)

The comparison isn't even. The scorecard and gradient boosting learned from 20,000 customers with known outcomes, and the AI models didn't. The "cut-off from known outcomes" column needs labelled customers too: 2,000 of them.

## When mistakes cost more

Raise the cost of a default to 6 units and approving everyone loses 547 per 1,000. The trained models, taken at their word, still earn about 190. The best AI model at its word, Sol, earns 110, and Kev-0.8B approves nobody. At 10 to 1, only the trained models make real money: every AI model taken at its word either loses money or approves almost nobody.

{% include chart.html src="/public/images/why-99-percent-accuracy/post2_cost_ratio.png" alt="Profit per 1,000 customers when a lender takes each model at its word, as the cost of one default rises from 1 to 10 times what a paying customer earns." words="Approving everyone drops below zero at about 3.5 and loses 547 at 6. Gradient boosting and the scorecard stay profitable throughout: about 190 at 6, and 92 and 62 at 10. GPT-6.2 Sol earns 110 at 6 but loses money by 10, as does Laya. Jev, Qwen and both Kev models approve only a few customers in 100, or none, once a default costs 5 or more." %}

The costlier a bad approval, the more every decision rests on the low-risk end of the list, and on the percentages there being honest.

## So what should you look at?

- **Accuracy, only next to the do-nothing score.** 82% sounds good until you learn that 78% comes free.
- **Ranking**, to know whether the model can tell risky from safe at all.
- **Honesty**, to know whether you can use its numbers directly: to set a cut-off, a price or a reserve.
- **Your own costs**, to set the cut-off. Here the right cut-off was 20%, not 50%. The 50% that accuracy assumes is a convention, not a decision.

A model that ranks well but states the wrong percentages is still usable, if you have known outcomes to correct it with. That's the expensive part that answering cold was supposed to save.

## What this doesn't show

- **The costs are illustrative.** A real lender earns different amounts from different customers, recovers part of some defaults, and sets credit limits rather than saying yes or no.
- **One dataset:** one bank, 2005, where 22% defaulted. Where defaults are rarer, doing nothing scores even higher, and accuracy says even less.
- **Known outcomes for the cut-offs** came from 2,000 separate customers. Nothing was chosen on the test set.

## What's next

Most of the gap in that table comes from AI models answering cold against models that learned from 20,000 customers. So I'm running the fair fight: every AI model gets those same 20,000 customers, by training where it can be trained and by seeing examples where it can't. Two newcomers join them: AI models built specifically for tables, NVIDIA's [Kumo-Tabular](https://huggingface.co/nvidia/Kumo-Tabular) and [TabICLv2](https://github.com/soda-inria/tabicl). They read all 20,000 customers at once and predict the rest, with no training at all. That's [the next post](/2026/11/25/fair-fight/).

The numbers here come from `analysis/accuracy.py` in the [project repo](https://github.com/nishzsche/can-ai-know-when-its-wrong), using the same predictions as the last post.
