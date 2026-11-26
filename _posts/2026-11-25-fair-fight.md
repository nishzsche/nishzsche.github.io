---
layout: post
title: "I gave AI the same data as a bank's model. It caught up."
date: 2026-11-25 13:02:00 +0530
tags: [probability, deep-learning, experiment]
series: can-ai-know-when-its-wrong
description: "The biggest objection to my first experiment was that it wasn't a fair fight. So every AI model got the same 20,000 customers the bank's models learned from."
image: /public/images/fair-fight/social-card.png
image_alt: "I gave AI the same data as a bank's model. It caught up. Ranking (AUC) answering cold, then after training on 20,000 customers: Kev-4B 0.594 to 0.792, Laya 0.666 to 0.778, Kev-0.8B 0.750 to 0.790, Qwen3.5-4B 0.744 to 0.780, against gradient boosting's 0.789 and the scorecard's 0.777."
toc: true
icon: /public/drawings/20261001-125135.jpg
icon_caption: "Inktober #1, Apple: one law for every apple, one ceiling for every model."
---

*The biggest objection to my first experiment was that it wasn't a fair fight. So I made it one.*

If one student studies last year's exam papers and another walks in cold, their scores tell you more about the papers than about the students.

That was the fairest objection to [my first experiment](/2026/11/23/can-ai-know-when-its-wrong/). I asked eight models how likely 2,000 real credit-card customers were to miss next month's payment, and checked whether their percentages came true. Most AI models put the risk too high, the worst at about twice the real rate, and none ranked customers better than the scorecard banks have used for decades. But the scorecard, and a modern machine-learning model called gradient boosting, had studied 20,000 customers whose outcomes were known. The AI models had studied nobody. As one reader put it, generic models that haven't been trained on the task don't make a fair comparison.

So I gave every AI model the same 20,000 customers, and asked the same 2,000 the same question.

{% include chart.html src="/public/images/fair-fight/m6_catch_up.png" alt="Each AI model's ranking ability (AUC) answering cold and after learning from the 20,000 customers." words="Kev-4B goes from 0.594 to 0.792, Kev-0.8B from 0.750 to 0.790, Qwen from 0.744 to 0.780 and Laya from 0.666 to 0.778, around gradient boosting's 0.789 and above the scorecard's 0.777. Kumo-Tabular and TabICLv2 reach 0.794 and have no cold version. Jev and GPT-6.2 Sol, shown 20 examples, end at 0.743 and 0.762, around the 20-neighbours average of 0.764." eager=true %}

## Three ways to learn from 20,000 customers

Not every AI model can learn from your data in the same way, and that turned out to be part of the story.

| How it used the 20,000 customers | Models |
|---|---|
| **Trained on them** (fine-tuned): the model's internal numbers adjusted to fit these customers | Kev-4B and Kev-0.8B (one pass, their maker's recipe); Laya (its maker's recipe: four passes, ending with its own calibration step); Qwen3.5-4B (one pass; it has no maker recipe, so common settings) |
| **Read all of them as examples,** with no training | Kumo-Tabular (NVIDIA), TabICLv2 |
| **Shown its 20 most similar customers** and what happened to them, in the prompt | Jev, GPT-6.2 Sol |

The middle row is new: AI built only for tables of numbers. It has practised on many other tables to learn how to learn from a new one, so you hand it your labelled rows and it predicts the rest in one go. These are called [tabular foundation models](https://aimultiple.com/tabular). A recent industry benchmark ranks them above gradient boosting on accuracy-style scores, which, as [my last post](/2026/11/24/why-99-percent-accuracy/) showed, can hide a lot. It doesn't report whether their percentages come true, so I checked.

The bottom row is the only option for the two closed models. Jev's makers say it isn't fine-tuned with customer data, and OpenAI never offered fine-tuning for GPT-6.2. So they can only look at examples, and a prompt only holds so much: each saw 20 customers per question, not 20,000. To see whether they add anything beyond those examples, I also scored the simplest possible reference: average the outcomes of the same 20 look-alike customers.

**The rules didn't change:** the same customers for every model, nothing adjusted using the test set, no age, sex, marital status or education anywhere, and no search for better settings. The scorecard and gradient boosting are unchanged from the first post. Gradient boosting got the only tuning in the contest: its number of training rounds was picked by cross-checking within the 20,000.

## Train it, and it catches up

To measure ranking I use the same score as before: pick one customer who defaulted and one who paid, and ask how often the model rates the defaulter as riskier ([AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve); 0.5 is a coin toss). Gradient boosting rates the defaulter riskier 79 times in 100 (0.789), the scorecard 78 (0.777).

Every model that actually learned from the data now lands next to them:

| Model | Answering cold | After learning from the 20,000 |
|---|---|---|
| Kumo-Tabular | (needs examples) | **0.794** |
| TabICLv2 | (needs examples) | **0.794** |
| Kev-4B | 0.594 | **0.792** |
| Kev-0.8B | 0.750 | **0.790** |
| *Gradient boosting* | | *0.789* |
| Qwen3.5-4B | 0.744 | 0.780 |
| Laya | 0.666 | 0.778 |
| *Scorecard* | | *0.777* |

Kev-4B is the most dramatic: from 59 times in 100 to 79, from last place to the leading group, 0.002 behind the top. Part of its cold score was my fault: in the first post, simply re-aligning the table's columns had already lifted it to 0.720.

How close is close? I compared every pair on the same customers with a resampling test, corrected for comparing eight models at once.

- **Against gradient boosting, none of the six can be told apart.** Even allowing for comparing eight models, the test rules out a gap of more than about 0.02 either way, a little under 0.03 for Qwen and Laya.
- **Against the scorecard, three rank slightly better,** by 0.015 to 0.017: Kev-4B, Kumo-Tabular and TabICLv2. After the correction, Kev-4B's edge holds and the two tabular models' is borderline.

{% include chart.html src="/public/images/fair-fight/m6_how_close.png" alt="Each model's ranking minus gradient boosting's, and minus the scorecard's, with ranges widened for comparing eight models." words="Against gradient boosting, the ranges for Kumo-Tabular, TabICLv2, Kev-4B, Kev-0.8B, Qwen and Laya all cross zero; GPT-6.2 Sol (0.026 lower) and Jev (0.046 lower) fall below it. Against the scorecard, Kev-4B's range (0.015 higher) just clears zero, the two tabular models (0.017 higher) only just miss, Qwen, Laya, Kev-0.8B and Sol cross it, and Jev falls below." %}

So the AI models caught up with gradient boosting, and the best of them edged past the scorecard. With 2,000 test customers, gaps smaller than about 0.02 are hard to see, and at a bank's scale a gap of 0.01 can still matter.

## Honest, after the same correction

Ranking is half the job. The other half, the subject of the first post, is whether a model's percentages come true. I measure it the same way: the average gap, in percentage points, between the risk a model promised and what happened ([calibration error](https://scikit-learn.org/stable/modules/calibration.html)). With 2,000 customers, a perfectly honest model stays below about 2.5 points in 95 samples out of 100, so anything under that is within chance.

Raw output first: Laya is 1.7 points off on average, Kumo-Tabular 2.0 and Qwen 2.4, as honest as gradient boosting (1.7). That order isn't solid, though. I retrained Kev-0.8B and Laya twice more, changing only the random starting point, and Laya came out 2.2 and 3.5 points off: how honest a freshly trained model is depends partly on the luck of the run. And raw numbers aren't a fair comparison here anyway. Laya's maker recipe ends with its own calibration step, which I kept. Kev's maker recipe ends with one too, and I skipped it: I served Kev straight out of training. So I compare honesty the way the first post compared recalibration: every model gets the same correction, learned from 2,000 separate customers and checked on the test customers.

{% include chart.html src="/public/images/fair-fight/m6_honesty.png" alt="Calibration error for each model, raw and after the same correction learned from 2,000 separate customers." words="After the correction, Laya is 0.9 points off, GPT-6.2 Sol 1.2 (it recognizes this dataset), Kev-0.8B 1.4, TabICLv2 1.6, Qwen 1.7, Jev 1.9 and Kumo-Tabular 2.0, all within the range a perfectly honest model reaches by chance, like the scorecard and gradient boosting. Kev-4B stays outside it at 3.3. Jev's raw error was 10.6." %}

After the same correction, every model is within chance of perfectly honest except one: Laya is 0.9 points off, Sol 1.2 (which, as before, recognizes this dataset and knows its default rate), Kev-0.8B 1.4, TabICLv2 1.6, Qwen 1.7, Jev 1.9 and Kumo-Tabular 2.0, against the scorecard's 1.4 and gradient boosting's 2.0. The exception is Kev-4B, still 3.3 points off. The correction also steadies the re-runs: Kev-0.8B's came out 1.5 and 1.3 points off, Laya's 1.8 and 2.5, the last right at the edge of chance.

The Kev models show why the check matters after training too. Straight out of training they understate risk: about 18 to 19% against the 22% who defaulted, and worst in the middle, where Kev-0.8B promised 11% to a group in which 21% defaulted. It isn't a fluke of one run: Kev-0.8B's two re-runs averaged 19% and 20%. That's the worse direction for a lender, because provisions and prices come out too low. Their percentages are too spread out, which is exactly what their skipped maker step corrects. A single adjustment of that kind, learned from the 2,000 calibration customers, brings Kev-0.8B's average back to 21% and its error from 4.5 points to 1.7. Kev-4B's average recovers too, but its error barely moves (3.6 to 3.3). Training is not calibration.

{% include chart.html src="/public/images/fair-fight/m6_kev_calibration.png" alt="Promised against actual default rate in ten risk groups for Kev-0.8B and Kev-4B, straight out of training and after the same correction." words="Straight out of training, both sit above the diagonal in the middle groups, promising too little: Kev-0.8B promised 11% to a group in which 21% defaulted. After the correction, Kev-0.8B follows the diagonal (1.4 points off), while Kev-4B still overstates risk for its safest groups and understates it in the middle (3.3 points off)." %}

## The AI built for tables gets there with no training

The two tabular models match the best of the field without a training run: they read the 20,000 customers once and predict. Their percentages are close to honest straight out of the box (2.0 and 2.5 points off), and they're cheap. On a rented GPU, TabICLv2's predictions cost about 0.03 cents per 1,000 decisions, and Kumo-Tabular's about a third of a cent.

Fine-tuning the language models wasn't expensive either, a dollar of GPU time at most each. But each run took between 40 minutes and almost two and a half hours, plus setting up and checking. The tabular models took seconds to a minute.

{% include chart.html src="/public/images/fair-fight/m6_cost.png" alt="Ranking (AUC) against running cost per 1,000 decisions, on a log scale." words="The tabular models, the fine-tuned models and the baselines all cost under 2 cents per 1,000 and rank between 0.777 and 0.794; TabICLv2 is the cheapest AI model at about 0.03 cents and ranks 0.794. Jev, shown 20 examples, costs about 21 cents and ranks 0.743. GPT-6.2 Sol, shown 20 examples, costs about 10 dollars and ranks 0.762, just below simply averaging the 20 neighbours (0.764), which costs next to nothing." %}

One caution: no training run doesn't mean nothing to validate. These models read your 20,000 labelled customers every time they score, so the training data is part of the model, and it needs checking like any other.

## The closed models can only look, and that isn't enough

GPT-6.2 Sol and Jev can't learn from your data except through the prompt. Each saw its 20 most similar customers and their outcomes, then answered the usual question.

- **Jev ranked better** (0.712 to 0.743) and got more honest (26.5 to 10.6 points off), but it still overstates risk: about 33% against 22%.
- **Sol's ranking barely moved** (0.754 to 0.762, within chance), though its percentages improved (5.2 to 3.5 points off).

Neither beat simply averaging the outcomes of those same 20 customers, which ranks at 0.764 and is 2.3 points off. Sol matched it, and Jev ranked worse. Sol's answers track that average closely: whatever extra it reads from the account summaries is small, and not clearly beyond chance. (The usual caveat applies to Sol: it recognizes this dataset and knows its default rate.)

{% include chart.html src="/public/images/fair-fight/m6_sol_neighbours.png" alt="Each test customer's answer from GPT-6.2 Sol and from Jev, against how many of the 20 look-alike customers in its prompt defaulted." words="Sol's answers follow the share who defaulted closely (correlation 0.94): its middle answer is 10% when 2 of the 20 defaulted, 23% when 5 did and 57% when 10 did. Jev's answers start higher: its middle answer is 20% when only 1 of the 20 defaulted, where counting gives 5%." %}

It was also by far the most expensive option. The 20 examples made each of Sol's prompts much longer, so it cost about 10 dollars per 1,000 decisions (the far right of the cost chart above), 14 times its cost when answering cold. The fine-tuned models cost under 2 cents per 1,000. And every decision sent 21 customers' account records to an outside vendor: a privacy and vendor-risk question before it's a modelling one.

## A ceiling, not a winner

The most interesting result is what didn't happen. Eight very different models, from a decades-old scorecard to a 4-billion-parameter language model to NVIDIA's table model, rank customers within 0.02 of each other (0.777 to 0.794). And in the first experiment, a scorecard trained on just 2,000 customers, a tenth of the data, scored 0.758: ten times the data bought about 0.02. With a credit limit and six months of bills, payments and repayment codes, there's only so much anyone can tell. Here, more model didn't buy any signal this test could detect.

{% include chart.html src="/public/images/fair-fight/m6_ceiling.png" alt="Ranking for every model that learned from the 20,000 customers, on the full scale from a coin flip (0.5) to perfect (1.0)." words="Kumo-Tabular and TabICLv2 reach 0.794, Kev-4B 0.792, Kev-0.8B 0.790, gradient boosting 0.789, Qwen 0.780, Laya 0.778 and the scorecard 0.777: all within 0.02 of each other, and none past 0.80. A scorecard trained on a tenth of the data, 2,000 customers, reaches 0.758." %}

## I got most of my predictions wrong, again

Before any of these runs, I wrote down what I expected, judged on raw output:

| My guess | What happened |
|---|---|
| Qwen would be the most honest after training | ✗ Laya was (1.7 points off), with Kumo-Tabular (2.0) and Qwen (2.4) close behind. But Laya's re-runs were 2.2 and 3.5, so that order isn't stable |
| At least one trained AI model would beat gradient boosting at ranking | ✓ On the numbers, four did (0.790 to 0.794 against 0.789), though none by more than chance, and Kev-0.8B's two re-runs scored 0.786 and 0.789, level with it or just below |
| Jev and Sol, shown examples, would beat their 20 neighbours | ✗ Neither did: Sol matched them, and Jev ranked worse |
| Kev-4B would be the least honest after training | ✗ Jev, shown examples, was (10.6 points off), then Kev-0.8B (4.5; its re-runs were 3.0 and 3.3), then Kev-4B (3.6) |

## What I'd tell a bank

- **If everything ties, start with the model you can explain.** A scorecard tells a declined customer, and a regulator, exactly why, and validators already know how to check it. On this data the AI options bought at most a sliver of extra ranking (0.015 over the scorecard), which won't pay for losing that.
- **If you want AI, try the kind built for tables before fine-tuning a language model.** It ranked with the best, was close to honest out of the box, needed no training run and cost almost nothing.
- **Check calibration after training, and keep checking it in production.** Kev learned to rank and still understated risk. Training is not calibration.
- **A closed model can only look at your data.** A prompt full of look-alike customers added little or nothing measurable over counting them, at many times the cost, and every decision ships customer records to a vendor whose model can change without notice.

## What this test can't tell you

- **"Default" is the dataset's label:** missing next month's payment, not the 90-days-overdue definition banks use for capital.
- **One dataset, one moment:** one bank in Taiwan, 2005, where 22% defaulted. The test customers come from the same months as the training customers, so this says nothing about how calibration holds up as the economy or the customer mix shifts. Banks test that on later periods, and monitor it.
- **Few re-runs:** I retrained two models, Kev-0.8B and Laya, twice more with a different random starting point. Their ranking moved by less than 0.01, but their raw honesty moved by up to 1.8 points, more than some of the gaps between models. Kev-4B and Qwen were trained once, so their raw honesty is just as uncertain.
- **Small gaps are hard to see:** with 2,000 test customers, ranking differences under about 0.02 can't be told apart.
- **No tuning:** each model used its maker's settings, or common ones for Qwen. More effort might move a model; the baselines got only a little.
- **Fair lending isn't tested:** leaving out age, sex, marital status and education keeps them out of the inputs, not out of the outcomes. A fair-lending review would check whether approval rates differ by group; I haven't.
- **Famous data:** Sol recognizes this dataset. I haven't checked whether the tabular models met it while practising, and Kumo-Tabular reads the original column names.
- **Choices I made:** Jev and Sol's 20 examples were picked by a simple similarity measure, and the language models trained on the same account summaries as in the first post. Different choices could help or hurt.

## What's next

Every number, the code, [the trained models' weights](https://github.com/nishzsche/can-ai-know-when-its-wrong/releases/tag/m6-weights), the training logs and every prediction are on [GitHub](https://github.com/nishzsche/can-ai-know-when-its-wrong). This series started with [whether AI can know when it's wrong](/2026/11/23/can-ai-know-when-its-wrong/) and [why 99% accuracy can still be a disaster](/2026/11/24/why-99-percent-accuracy/). Next: does shrinking a model to save money make it less honest? I'll take the models trained here and squeeze them to a quarter of their size. And the sequel I promised in the first post, teaching a small model to say "I don't know", is still coming.
