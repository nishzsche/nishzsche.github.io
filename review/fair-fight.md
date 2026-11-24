---
layout: post
title: "I gave AI the same data as a bank's model. It caught up."
date: 2026-11-24
tags: [probability]
permalink: /review/fair-fight/
description: "Post 1's biggest objection, answered: every AI model gets the 20,000 labelled customers the traditional models learned from."
# Review copy: not listed on the home page or in the feed, hidden from search engines and the
# sitemap, with Hypothesis for comments. To launch, move it to _posts/ and drop these four.
noindex: true
sitemap: false
hypothesis: true
comments: false
---

> **Draft for review, version 1 (24 November), not yet published.** Thank you for reading. This is the third post in the series, after [Can AI know when it's wrong?](/2026/11/23/can-ai-know-when-its-wrong/) and [Why 99% accuracy can still be a disaster](/2026/11/24/why-99-percent-accuracy/). It answers the biggest objection to the first: every AI model now learns from the same 20,000 customers the traditional models did. A panel of AI reviewers has already checked the statistics and every number; I'd like a human read now. One check is still running (re-training two models with different random seeds), so one limitation line may change.
>
> **To comment, no account needed:** reply to my message and quote the sentence you mean. If you already use [Hypothesis](https://web.hypothes.is/), you can also select text and click **Annotate**.
>
> I'd most like to know: does the headline ("it caught up") feel fair given the details, does comparing honesty "after the same correction" make sense or feel like moving the goalposts, would you act on the bank advice, and where did you get lost or bored?

*The biggest objection to my first experiment was that it wasn't a fair fight. So I made it one.*

If one student studies last year's exam papers and another walks in cold, their scores tell you more about the papers than about the students.

That was the fairest objection to [my first experiment](/2026/11/23/can-ai-know-when-its-wrong/). I asked eight models how likely 2,000 real credit-card customers were to miss next month's payment, and checked whether their percentages came true. Most AI models put the risk too high, the worst at about twice the real rate, and none ranked customers better than the scorecard banks have used for decades. But the scorecard, and a modern machine-learning model called gradient boosting, had studied 20,000 customers whose outcomes were known. The AI models had studied nobody. As one reader put it, generic models that haven't been trained on the task don't make a fair comparison.

So I gave every AI model the same 20,000 customers, and asked the same 2,000 the same question.

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

![Each AI model's ranking ability (AUC) answering cold and after learning from the 20,000 customers. Kev-4B goes from 0.594 to 0.792, Kev-0.8B from 0.750 to 0.790, Qwen from 0.744 to 0.780 and Laya from 0.666 to 0.778, around gradient boosting's 0.789 and above the scorecard's 0.777. Kumo-Tabular and TabICLv2 reach 0.794 and have no cold version. Jev and GPT-6.2 Sol, shown 20 examples, end at 0.743 and 0.762, around the 20-neighbours average of 0.764.](/public/images/fair-fight/m6_catch_up.png){: data-dark-src="/public/images/fair-fight/m6_catch_up-dark.png"}

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

- **Against gradient boosting, none of the six can be told apart.** The test rules out a gap of more than about 0.02 either way.
- **Against the scorecard, three rank slightly better,** by 0.015 to 0.017: Kev-4B, Kumo-Tabular and TabICLv2. After the correction, Kev-4B's edge holds and the two tabular models' is borderline.

So the AI models caught up with gradient boosting, and the best of them edged past the scorecard. With 2,000 test customers, gaps smaller than about 0.02 are hard to see, and at a bank's scale a gap of 0.01 can still matter.

## Honest, after the same correction

Ranking is half the job. The other half, the subject of the first post, is whether a model's percentages come true. I measure it the same way: the average gap, in percentage points, between the risk a model promised and what happened ([calibration error](https://scikit-learn.org/stable/modules/calibration.html)). With 2,000 customers, a perfectly honest model stays below about 2.5 points in 95 samples out of 100, so anything under that is within chance.

Raw output first: Laya is 1.7 points off on average, Kumo-Tabular 2.0 and Qwen 2.4, as honest as gradient boosting (1.7). But raw numbers aren't a fair comparison here. Laya's maker recipe ends with its own calibration step, which I kept. Kev's maker recipe ends with one too, and I skipped it: I served Kev straight out of training. So I compare honesty the way the first post compared recalibration: every model gets the same correction, learned from 2,000 separate customers and checked on the test customers.

![Calibration error for each model, raw and after the same correction learned from 2,000 separate customers. After the correction, Laya is 0.9 points off, GPT-6.2 Sol 1.2, Kev-0.8B 1.4, TabICLv2 1.6, Qwen 1.7, Jev 1.9 and Kumo-Tabular 2.0, all within the range a perfectly honest model reaches by chance, like the scorecard and gradient boosting. Kev-4B stays outside it at 3.3. Jev's raw error was 10.6.](/public/images/fair-fight/m6_honesty.png){: data-dark-src="/public/images/fair-fight/m6_honesty-dark.png"}

After the same correction, every model is within chance of perfectly honest except one: Laya is 0.9 points off, Sol 1.2, Kev-0.8B 1.4, TabICLv2 1.6, Qwen 1.7, Jev 1.9 and Kumo-Tabular 2.0, against the scorecard's 1.4 and gradient boosting's 2.0. The exception is Kev-4B, still 3.3 points off.

The Kev models show why the check matters after training too. Straight out of training they understate risk: about 18 to 19% against the 22% who defaulted. That's the worse direction for a lender, because provisions and prices come out too low. Their percentages are too spread out, which is exactly what their skipped maker step corrects. A single adjustment of that kind, learned from the 2,000 calibration customers, brings Kev-0.8B's average back to 21% and its error from 4.5 points to 1.7. Kev-4B's average recovers too, but its error barely moves (3.6 to 3.3). Training is not calibration.

## The AI built for tables gets there with no training

The two tabular models match the best of the field without a training run: they read the 20,000 customers once and predict. Their percentages are close to honest straight out of the box (2.0 and 2.5 points off), and they're cheap. On a rented GPU, TabICLv2's predictions cost about 0.03 cents per 1,000 decisions, and Kumo-Tabular's about a third of a cent.

Fine-tuning the language models wasn't expensive either, a dollar of GPU time at most each. But each run took between 40 minutes and almost two and a half hours, plus setting up and checking. The tabular models took seconds to a minute.

One caution: no training run doesn't mean nothing to validate. These models read your 20,000 labelled customers every time they score, so the training data is part of the model, and it needs checking like any other.

## The closed models can only look, and that isn't enough

GPT-6.2 Sol and Jev can't learn from your data except through the prompt. Each saw its 20 most similar customers and their outcomes, then answered the usual question.

- **Jev ranked better** (0.712 to 0.743) and got more honest (26.5 to 10.6 points off), but it still overstates risk: about 33% against 22%.
- **Sol's ranking barely moved** (0.754 to 0.762, within chance), though its percentages improved (5.2 to 3.5 points off).

Neither beat simply averaging the outcomes of those same 20 customers, which ranks at 0.764 and is 2.3 points off. Sol matched it, and Jev ranked worse. Sol's answers track that average closely: whatever extra it reads from the account summaries is small, and not clearly beyond chance. (The usual caveat applies to Sol: it recognizes this dataset and knows its default rate.)

It was also by far the most expensive option. The 20 examples made each of Sol's prompts much longer, so it cost about 10 dollars per 1,000 decisions, 14 times its cost when answering cold. The fine-tuned models cost under 2 cents per 1,000. And every decision sent 21 customers' account records to an outside vendor: a privacy and vendor-risk question before it's a modelling one.

## A ceiling, not a winner

The most interesting result is what didn't happen. Eight very different models, from a decades-old scorecard to a 4-billion-parameter language model to NVIDIA's table model, rank customers within 0.02 of each other (0.777 to 0.794). And in the first experiment, a scorecard trained on just 2,000 customers, a tenth of the data, scored 0.758: ten times the data bought about 0.02. With a credit limit and six months of bills, payments and repayment codes, there's only so much anyone can tell. Here, more model didn't buy any signal this test could detect.

## I got most of my predictions wrong, again

Before any of these runs, I wrote down what I expected, judged on raw output:

| My guess | What happened |
|---|---|
| Qwen would be the most honest after training | ✗ Laya was (1.7 points off), with Kumo-Tabular (2.0) and Qwen (2.4) close behind, all within chance of perfectly honest |
| At least one trained AI model would beat gradient boosting at ranking | ✓ On the numbers, four did (0.790 to 0.794 against 0.789), though none by more than chance |
| Jev and Sol, shown examples, would beat their 20 neighbours | ✗ Neither did: Sol matched them, and Jev ranked worse |
| Kev-4B would be the least honest after training | ✗ Jev, shown examples, was (10.6 points off), then Kev-0.8B (4.5), then Kev-4B (3.6) |

## What I'd tell a bank

- **If everything ties, start with the model you can explain.** A scorecard tells a declined customer, and a regulator, exactly why, and validators already know how to check it. On this data the AI options bought at most a sliver of extra ranking (0.015 over the scorecard), which won't pay for losing that.
- **If you want AI, try the kind built for tables before fine-tuning a language model.** It ranked with the best, was close to honest out of the box, needed no training run and cost almost nothing.
- **Check calibration after training, and keep checking it in production.** Kev learned to rank and still understated risk. Training is not calibration.
- **A closed model can only look at your data.** A prompt full of look-alike customers added little or nothing measurable over counting them, at many times the cost, and every decision ships customer records to a vendor whose model can change without notice.

## What this test can't tell you

- **"Default" is the dataset's label:** missing next month's payment, not the 90-days-overdue definition banks use for capital.
- **One dataset, one moment:** one bank in Taiwan, 2005, where 22% defaulted. The test customers come from the same months as the training customers, so this says nothing about how calibration holds up as the economy or the customer mix shifts. Banks test that on later periods, and monitor it.
- **One run each:** each model was trained once, and I haven't yet measured how much a re-run moves the numbers.
- **Small gaps are hard to see:** with 2,000 test customers, ranking differences under about 0.02 can't be told apart.
- **No tuning:** each model used its maker's settings, or common ones for Qwen. More effort might move a model; the baselines got only a little.
- **Fair lending isn't tested:** leaving out age, sex, marital status and education keeps them out of the inputs, not out of the outcomes. A fair-lending review would check whether approval rates differ by group; I haven't.
- **Famous data:** Sol recognizes this dataset. I haven't checked whether the tabular models met it while practising, and Kumo-Tabular reads the original column names.
- **Choices I made:** Jev and Sol's 20 examples were picked by a simple similarity measure, and the language models trained on the same account summaries as in the first post. Different choices could help or hurt.

## What's next

Every number, the code, [the trained models' weights](https://github.com/nishzsche/can-ai-know-when-its-wrong/releases/tag/m6-weights), the training logs and every prediction are on [GitHub](https://github.com/nishzsche/can-ai-know-when-its-wrong). This series started with [whether AI can know when it's wrong](/2026/11/23/can-ai-know-when-its-wrong/) and [why 99% accuracy can still be a disaster](/2026/11/24/why-99-percent-accuracy/). Next: does shrinking a model to save money make it less honest? I'll take the models trained here and squeeze them to a quarter of their size. And the sequel I promised in the first post, teaching a small model to say "I don't know", is still coming.
