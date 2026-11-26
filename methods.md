---
layout: page
title: Methods
description: "How the experiments in the series Can AI know when it's wrong? are measured, and the caveats every part shares."
---

Every part of [Can AI know when it's wrong?](/archive/#can-ai-know-when-its-wrong) uses the same data and the same yardsticks. This page explains them once, so each post can get to the point. The code, every prediction and the full write-ups are on [GitHub](https://github.com/nishzsche/can-ai-know-when-its-wrong).

## The data

- **One public dataset:** [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), 30,000 credit-card customers of a bank in Taiwan in 2005, and whether each one missed their next payment. About 22% did.
- **Fixed splits:** 20,000 customers to learn from, 2,000 to tune on (to choose a correction or a cut-off), and 2,000 to test on, each with the same 22% default rate. The other 6,000 are held in reserve. Every model is scored on the same 2,000 test customers, and nothing is ever adjusted using them.
- **What every model sees:** the credit limit and six months of bills, payments and repayment status. Age, sex, marital status and education are left out on purpose. The AI models get this as a short account summary, written the same way for every model, followed by one question: *will this customer miss next month's payment?*

## The yardsticks

**Honesty.** When a model says 30%, do 30% of those customers default? I sort the test customers by the model's predicted risk, split them into ten equal groups, and compare what the model promised for each group with what actually happened. A chart of one against the other is a [reliability diagram](https://scikit-learn.org/stable/modules/calibration.html#calibration-curves): on the diagonal, a model is honest.

**Calibration error.** The gap from that chart as one number: the average distance between promised and actual, in percentage points (the [expected calibration error](https://arxiv.org/abs/1706.04599), or ECE). The scorecard's is about 1.4 points. Even a perfectly honest model is 1 to 3 points off by chance on 2,000 customers, so smaller gaps are noise; charts shade that range.

**Ranking.** Pick one customer who defaulted and one who paid: how often does the model rate the defaulter as riskier? That's [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve), where 0.5 is a coin flip and 1.0 is perfect. With 2,000 test customers, ranking gaps under about 0.02 can't be told apart.

**KS.** The widest gap, at any cut-off, between the share of defaulters and the share of payers a model puts below it (the [Kolmogorov-Smirnov statistic](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)). Lenders have long used it alongside AUC.

**Accuracy and profit.** Accuracy counts how often a model's yes-or-no, at a cut-off such as 50%, matches what happened. Profit puts a price on the mistakes: in the examples, each customer who pays earns a lender 1 unit and each default loses 4.

**Running cost.** Per 1,000 decisions: open models at the rates of a rented GPU (an NVIDIA L4), closed models at their API prices. The trained baselines cost next to nothing to run.

## The models

- **The baselines:** a [scorecard](https://en.wikipedia.org/wiki/Credit_scorecards) (a logistic regression, the kind of model banks have used for decades) and gradient boosting, both trained on the 20,000 customers.
- **The AI models:** in the first parts they answer cold, with no examples. From the third part on, they get the same 20,000 customers, by training where they can be trained and by seeing examples where they can't. Each post lists the models it tested.

## The correction

Where a post says a model was corrected, it means a correction learned on the 2,000 tune customers and applied the same way to every model: [Platt scaling or isotonic regression](https://scikit-learn.org/stable/modules/calibration.html), whichever cross-validation picks. It changes a model's percentages, not how it ranks customers.

## How I keep myself honest

- **Guesses first:** before each run I write down what I expect in [PREDICTIONS.md](https://github.com/nishzsche/can-ai-know-when-its-wrong/blob/main/PREDICTIONS.md), committed to the repository so I can't quietly revise them, and each post reports how they did.
- **Everything recorded:** every prediction is saved with the model's version, the condition, how long it took and what it cost.
- **Checks:** whether a model has memorized the dataset, whether different wording changes the results, and re-runs where they're affordable. The details are in each post and its write-up.

## Caveats every part shares

- **One dataset, one moment:** one bank in one country, in 2005, where 22% defaulted. Honesty depends on that rate: where 5% default, a model calibrated here would overstate the risk.
- **The same months:** the test customers come from the same months as the training customers. Banks validate on a later period, because patterns drift.
- **A famous dataset:** some AI models have seen it in training. GPT-6.2 Sol recognizes it and knows its default rate.
- **"Default" is the dataset's label:** missing next month's payment, not the 90-days-overdue definition banks use for capital.
- **Wording matters:** every AI model got the same template. Changing the wording moved results by up to 13 points.
- **Fair lending isn't tested:** leaving out age, sex, marital status and education keeps them out of the inputs, not out of the outcomes.
- **Models change:** the results are for the versions tested in November 2026, all recorded.

Each post's "What this doesn't show" adds the caveats that are specific to it.
