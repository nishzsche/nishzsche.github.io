---
layout: post
title: "Why AI can write code but struggle to read a table"
date: 2026-11-27
tags: [deep-learning, experiment]
series: can-ai-know-when-its-wrong
permalink: /review/reading-tables/
description: "Six AI models read the same 2,000 credit-card customers as a table, as sentences, as CSV and as JSON. Not one number changed, but the answers did."
toc: true
# Review copy: not listed on the home page or in the feed, hidden from search engines and the
# sitemap, with Hypothesis for comments. To launch, move it to _posts/ and drop these four.
noindex: true
sitemap: false
hypothesis: true
comments: false
---

> **Draft for review, version 1 (27 November), not yet published.** Thank you for reading. This is the fifth post in the series, after [I asked AI how risky customers were](/2026/11/23/can-ai-know-when-its-wrong/), [Why 99% accuracy can still be a disaster](/2026/11/24/why-99-percent-accuracy/), [I gave AI the same data as a bank's model](/2026/11/25/fair-fight/) and [I shrank AI models](/2026/11/27/shrinking-ai-models/). It gives the six AI models from the first post the same customers, with the same numbers written five other ways, and checks whether their answers change.
>
> **To comment, no account needed:** reply to my message and quote the sentence you mean. If you already use [Hypothesis](https://web.hypothes.is/), you can also select text and click **Annotate**.
>
> I'd most like to know: does the title pull you in, or would one that gives the result ("I reformatted a table. AI changed its answers.") do better? Does the section on answers moving while the ranking held make sense? Would you act on the bank advice? And where did you get lost or bored?

*Six AI models read the same 2,000 credit-card customers as a table, as sentences, as CSV and as JSON. Not one number changed. The frontier chatbot's answers barely moved. The small ones moved more than shrinking ever moved them, and one went from its best ranking to worse than a coin flip.*

Ask an AI chatbot to write a program that reads a spreadsheet, and it will do it in seconds. The program will read every number exactly, whatever the layout: to a program, 46,990 is 46,990 whether it sits in a table, a CSV file or JSON.

Now give the chatbot the spreadsheet itself and ask it a question. It doesn't write a program this time. It reads the file the way it reads everything, as text, a few characters at a time. Does the layout change what it concludes?

A person wouldn't expect it to. Show an underwriter a customer's last six months as a table, as a spreadsheet export or read aloud over the phone, and they should reach the same view. The numbers are the numbers.

This series asks whether an AI model's percentages can be trusted. In [the first post](/2026/11/23/can-ai-know-when-its-wrong/) I gave six AI models 2,000 real credit-card customers, each written as a small table, and asked how likely each one was to miss next month's payment. They answered cold: no examples, no training. For this post I asked them again, with the same numbers laid out five other ways: as a Markdown table, as CSV, as JSON, as sentences, and as a table with each bill beside the payment that usually settles it.

{% include chart.html src="/public/images/reading-tables/post5_ranking.png" alt="Each AI model's ranking (AUC) of the same 2,000 customers with the original table and five other layouts of the same numbers." words="Laya ranged from 0.716 with sentences to 0.478 with JSON, worse than a coin flip; it was 0.666 with the original table. Kev-4B ranged from 0.594 with the original table to 0.729 with JSON. Qwen3.5-4B, Jev, Kev-0.8B and GPT-6.2 Sol stayed between 0.69 and 0.76, and most of their changes were within chance. The scorecard, trained on 20,000 customers, is at 0.777." eager=true %}

## Six ways to write the same customer

Here is one test customer as every model saw them in the first post, which I'll call the original table:

```text
Credit card account summary
Credit limit: NT$50,000

Month (most recent first) | Bill amount | Amount paid | Repayment status
Sep 2005 | NT$46,990 | NT$2,000 | not late (code 0)
Aug 2005 | NT$48,233 | NT$2,019 | not late (code 0)
... four more months
```

And here is the September row in four of the new layouts:

```text
Markdown:   | Sep 2005 | NT$46,990 | NT$2,000 | not late (code 0) |
CSV:        Sep 2005,"NT$46,990","NT$2,000",not late (code 0)
JSON:       {"month": "Sep 2005", "bill_amount": "NT$46,990", "amount_paid": "NT$2,000", "repayment_status": "not late (code 0)"}
Sentences:  In Sep 2005, the bill amount was NT$46,990, the amount paid was NT$2,000, and the repayment status was not late (code 0).
```

The Markdown version is the original table with a line of dashes under the header and a bar at each end of every row: the smallest change of the lot. Every value is copied character for character from the original, and a test reads all 4,000 tune and test customers back from each layout to check that nothing changed but the layout.

The fifth layout comes from a check in the first post. In this data, a month's payment usually settles the previous month's bill, but the original table puts each payment beside the same month's bill. The realigned table puts each bill beside the payment made the following month, labelled as such ("NT$2,000 (paid in Sep 2005)").

The six models are the same as in the first post: Jev, a commercial model built for decisions; Kev-4B, Kev-0.8B and Laya, three open models built for decisions; Qwen3.5-4B, a general open model; and GPT-6.2 Sol, a frontier chatbot. All six answered cold, as before. The scorecard on the chart learned from 20,000 labelled customers, so it's a reference, not a fair opponent.

## The same numbers, a different ranking

Ranking is measured as in every post (the [methods page](/methods/) explains each measure): pick one customer who defaulted and one who paid, and ask how often the model rates the defaulter as riskier ([AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve); 0.5 is a coin flip). To tell a real change from luck, I compared each layout with the original table on the same customers, allowing for 30 comparisons (six models, five layouts).

- **Laya's ranking depended on the layout more than on anything else I've tried.** With sentences it ranked at 0.716, its best in the series without training. With JSON it fell to 0.478: worse than a coin flip, so it would have done better with its answers reversed. Its range across layouts, 0.24, is bigger than the gap between the best and worst of the six models with the original table (0.16).
- **Kev-4B went the other way.** It ranked worst of the six in the first post, at 0.594. Every layout except Markdown did better, and JSON took it to 0.729, a little ahead of the realigned table (0.720). One sentence telling it the default rate did even more in the first post (0.757).
- **The other four barely moved.** Qwen lost 0.030 with the realigned table and Jev about 0.02 with sentences and the realigned table. Kev-0.8B and Sol showed no change bigger than chance in any layout.

In all, 11 of the 30 changes were bigger than chance, and all but three of those belonged to the two models at the top of the chart.

## Even where the ranking held, the answers moved

Ranking is a summary. As the [last post](/2026/11/27/shrinking-ai-models/) found, a summary score can hold while each customer's number moves. So I measured how far each customer's answer moved between the original table and each layout.

Two models need a baseline for this, because they don't give the same answer twice: asked the same table again, Jev's answers moved 1.8 points per customer and Sol's 2.5. Kev gives exactly the same answer every time, and Laya almost does, so nearly all of their movement came from the layout.

{% include chart.html src="/public/images/reading-tables/post5_moved.png" alt="How far each model's answer moved per customer, in percentage points, in each of the five layouts." words="Laya: 6.4 to 26 (CSV and JSON). Kev-4B: 2.8 to 15 (JSON). Qwen3.5-4B: 4.7 to 9.2 (JSON). Jev: 3.1 to 7.5 (realigned table), against 1.8 when the original table is asked again. Kev-0.8B: 2.6 to 5.5 (CSV). GPT-6.2 Sol: 2.6 to 2.9 in all five, against 2.5 when asked again. For comparison, shrinking a trained model to 3 bits in post 4 moved answers by 6.6 points at most." %}

- **Laya's answers moved 26 points per customer with CSV and JSON.** That's four times what shrinking to 3 bits did to any model in the last post.
- **Even the smallest change mattered to Laya.** The line of dashes and the bars at each end (Markdown) moved its answers by 6.4 points per customer, about as much as shrinking to 3 bits.
- **Kev-0.8B's ranking held in every layout, but its answers didn't:** up to 5.5 points per customer, and 57 in 100 customers moved by 5 points or more with CSV.
- **Sol moved 2.6 to 2.9 points in every layout,** hardly more than the 2.5 it moves when asked the same table again. The layout made no difference to it that I could detect. One caveat, which applies to all of Sol's results in this series: it recognizes this dataset and knows its default rate, so it may be answering from memory as much as from the summary.

Here is what that looks like customer by customer:

{% include chart.html src="/public/images/reading-tables/post5_customers.png" alt="Each dot is one test customer: the answer to the original table against the answer to a new layout, in four panels." words="Laya with JSON: answers bunched between 30% and 55% whatever the original answer, an average move of 25.8 points, 96 in 100 customers moved 5 points or more, ranking 0.666 to 0.478, rank correlation 0.08. Laya with sentences: most answers pushed down, 10.1 points, 63 in 100 moved 5 or more, ranking 0.666 to 0.716. Kev-4B with JSON: most answers pulled down, 15.1 points, 83 in 100, ranking 0.594 to 0.729. GPT-6.2 Sol with JSON: answers close to the no-change line, 2.6 points, 17 in 100, ranking 0.754 to 0.749." %}

Laya's JSON answers have almost nothing to do with its answers to the original table: knowing one tells you next to nothing about the other (rank correlation 0.08, where 1 would be the same order). Kev-4B's JSON answers were mostly pulled down, and its order improved. Sol's stayed on the line.

## The percentages moved too

The question this series keeps asking is whether a model's percentages come true. With the original table, 22.1% of these customers defaulted.

{% include chart.html src="/public/images/reading-tables/post5_promised.png" alt="Each AI model's average answer for the same 2,000 customers in each layout, against the 22.1% who defaulted." words="Laya: 20% with the original table, 14% with sentences, 44% with CSV. Kev-4B: 49% with the original table, from 34% with JSON to 54% with sentences. Qwen3.5-4B: 36%, from 27% to 37%. Jev: 48%, from 47% to 53%. Kev-0.8B: 31%, from 25% to 36%. GPT-6.2 Sol: 25% to 26% in every layout." %}

- **Laya promised 20% with the original table and 44% with CSV,** for the same customers. In the first post Laya tied with Sol as the most honest model out of the box, 5.2 points off on average ([calibration error](https://scikit-learn.org/stable/modules/calibration.html)). With CSV it was 25 points off, about as far as Jev and Kev-4B with the original table.
- **For Kev-4B and Qwen, JSON helped.** Kev-4B went from 26.7 points off to 12.4, and Qwen from 14.0 to 7.1.
- **Kev-0.8B swung both ways:** 4.9 points off with sentences, 13.8 with CSV, against 9.0 with the original table.
- **Sol's average stayed within a point of 26%.**

In all, 17 of the 30 changes in honesty were bigger than chance. So "which model is most honest" depends on how you write the question down.

## What it does to a lending decision

I used the lender from [post 2](/2026/11/24/why-99-percent-accuracy/): each paying customer earns 1 unit, each default costs 4, and the lender approves everyone below a cut-off chosen on a separate set of 2,000 customers whose outcomes are known. Suppose the lender changes the layout and keeps the cut-off it chose for the original table, as an engineering team might when it tidies up a prompt:

| Model | Asked the same table again | Markdown | CSV | JSON | Sentences | Realigned table |
|---|---:|---:|---:|---:|---:|---:|
| Laya | 0.5% | 22% | 38% | 63% | 27% | 44% |
| Kev-4B | 0% | 5.5% | 11% | 48% | 9.5% | 42% |
| Qwen3.5-4B | 2.0% | 9.8% | 9.4% | 7.8% | 9.5% | 13% |
| Jev | 3.5% | 5.1% | 5.4% | 16% | 8.7% | 16% |
| Kev-0.8B | 0% | 0.7% | 25% | 0.7% | 2.5% | 2.1% |
| GPT-6.2 Sol | 3.5% | 4.3% | 4.0% | 3.9% | 3.9% | 4.3% |

*The share of approve-or-decline decisions that change, at the cut-off chosen for the original table. "Asked again" is measured on 200 customers.*

- **With JSON, Laya's lender would have approved nobody,** against 63 in 100 with the original table.
- **Kev-0.8B, whose ranking never moved, would have approved 41 in 100 customers with CSV instead of 66,** earning 196 units per 1,000 customers instead of 306.
- **A change can help, too.** Kev-4B with JSON changed half its decisions, approved 70 in 100 instead of 27, and earned 254 per 1,000 instead of 49. Its answers to the original table were simply that poor.
- **Sol changed 4 in 100 decisions in every layout,** close to the 3.5 it changes when asked the same table again.

## Why: writing code and reading a table are different jobs

A program that reads a CSV file splits each line at the commas, turns "NT$46,990" into the number 46990, and knows it's a bill because it's in the second column. A language model does none of that. It reads text in small pieces called tokens: Kev and Qwen read "NT$46,990" one digit at a time, as eight pieces, and Laya as five. Nothing in the pieces says which number is a bill and which a payment, except where they sit, and the header that names the columns is up to six lines away.

{% include chart.html src="/public/images/reading-tables/post5_pieces.png" alt="One row of the original table, 'Sep 2005 | NT$46,990 | NT$2,000 | not late (code 0)', read by a program and by two kinds of model." words="A program that splits it at the bars reads four values: the month, the bill 46990, the payment 2000 and the status. Kev and Qwen read it as 31 pieces, including each digit separately: 4, 6, comma, 9, 9, 0. Laya reads it as 21 pieces, with the amounts split as 46, comma, 990." %}

So each layout asks the model to do a slightly different job: JSON repeats every label beside its value; CSV and Markdown give the labels once; sentences put each label in a phrase. A model can only do the job its training prepared it for, and these models were trained on very different text.

My results show that the job matters; they don't show why. Two clues, which I haven't tested:

- **Saying what each number is helped the model that read the original table worst.** JSON, which names every number right beside it, and the realigned table, which spells out which payment goes with which bill, gave Kev-4B its two best rankings. JSON also made both Kev-4B and Qwen more honest.
- **But no layout was best for everyone.** JSON was Kev-4B's best layout and Laya's worst. Laya, the smallest open model at 421 million numbers, did best with sentences. JSON is also Laya's longest input, 303 tokens on average against 164 for the original table, though still within its limit.

Others have found the same sensitivity in other tasks. Changing only the formatting of a prompt moved one open model's accuracy by up to 76 points ([Sclar et al., ICLR 2024](https://arxiv.org/abs/2310.11324)). Models given tables struggled with tasks as simple as counting the rows, and the table format changed the results ([Sui et al., WSDM 2024](https://arxiv.org/abs/2305.13062)). And researchers who turn table rows into text for a language model already compare several ways of writing them ([TabLLM, AISTATS 2023](https://proceedings.mlr.press/v206/hegselmann23a.html)).

Models built for tables don't have this problem by design: they read 46,990 as a number. In the last post but one, two of them ranked these customers at 0.794, against at most 0.757 for any AI model in any layout here, though they also had 20,000 labelled customers as examples, so that isn't a like-for-like comparison.

## My predictions, written before the runs

Before any of these runs, I [wrote down](https://github.com/nishzsche/can-ai-know-when-its-wrong/blob/main/PREDICTIONS.md) what I expected:

| My guess | What happened |
|---|---|
| Sentences would move the answers most | ✗ JSON did: 10.4 points per customer on average across the six models, against 6.1 for sentences, 7.7 for CSV and 3.7 for Markdown. Without Laya, JSON still moved them most |
| No new layout would change a model's ranking as much as the realigned table did (Kev-4B, 0.594 to 0.720) | ✗ JSON did, twice: Kev-4B up to 0.729, and Laya down to 0.478 |
| A small open model, Kev-0.8B or Laya, would be the most sensitive | Half right. Laya, by far. But Kev-0.8B's ranking was one of the steadiest, though its percentages moved |

## What I'd tell a bank

- **The layout is part of the model.** Version the prompt template with the model, and treat any change to it, even a line of dashes, as a model change: revalidate before it goes live. Here, adding one line and some bars changed 22 in 100 of one model's decisions.
- **Test a few layouts before trusting any one.** It needs no outcomes: score the same customers in two layouts and compare each customer's number. A model whose answers move with the layout isn't reading the numbers the way a person would, even when its scores look good.
- **If you choose a layout, choose it on data set aside for tuning, never on the test set.** I didn't choose one: this series' headline results stay on the original table, which was fixed before any run, so no model gets its best layout picked after the fact.
- **When a decision rests on numbers, let something read them as numbers.** A scorecard, a table model, or code that computes the figures first. Save the language model for the text that only it can read.

## What this test can't tell you

- **Zero-shot only.** The models trained in the last post but one learned from 20,000 customers in the original table's layout. They may be more sensitive to a new layout, or less. I didn't test them.
- **One run per layout.** Jev and Sol vary from run to run, and their noise was measured on 200 customers, not all 2,000.
- **Sol knows this dataset.** That may be why its answers didn't move.
- **Six layouts of one summary, with one question.** Other layouts (HTML, a different column order, an image of the table) and other wordings could do better or worse.
- **No best layout.** These are test customers, so picking each model's best layout from these results would be tuning on the test set.
- **I didn't test why.** The clues above are clues.
- **These test customers have now been used in five posts.** 6,000 customers that no model has seen are kept aside for one final replication.

The caveats every part of the series shares, from the dataset's age to what "default" means, are on the [methods page](/methods/).

## How I ran it

The four new layouts are built by one script from the original summaries, copying every value character for character, and a test checks all 4,000 tune and test customers against the original values. Laya can read at most 512 tokens; its longest summary in any layout was 320, and its runner stops if a summary is ever cut short. Each model answered with its own settings from the first post, out of the box, on all 2,000 test customers in each layout. The realigned table's answers are the first post's layout check, re-scored here.

The four open models ran on a rented NVIDIA L4 GPU in the same session as the last post's runs, about 43 minutes of answering, under 40 cents. Jev and Sol ran through OpenRouter, for $0.18 and $7.59. The measures fixed in advance were the average move, ranking, calibration error and the change per customer. I added three while writing, after the runs: the paired tests against the original table (Holm-corrected over 30), the share of customers moving 5 points or more, and the lending decisions.

## What's next

Every number, the code and every prediction are on [GitHub](https://github.com/nishzsche/can-ai-know-when-its-wrong). This series started with [whether AI can know when it's wrong](/2026/11/23/can-ai-know-when-its-wrong/), then [why 99% accuracy can still be a disaster](/2026/11/24/why-99-percent-accuracy/), [giving AI the same data as a bank's model](/2026/11/25/fair-fight/) and [shrinking the models](/2026/11/27/shrinking-ai-models/).

**Change the layout and you've changed the model. Test it like one.**
