# Instamart Customer Intelligence & Support Copilot

## Overview
One repository, four connected Parts, all on synthetic data (amounts in INR, all model calls go through the offline `mock_llm.py`; no network, no API key).

* **Part A** audits and cleans the raw exports, stores them in SQLite (`data/instamart.db`) and answers seven SQL questions plus a pandas cross-check.
* **Part B** builds RFM customer segments (K-means, PCA) and a lapse-prediction model.
* **Part C** triages support tickets with input guardrails, redaction, retries, output validation, a token budget and segment-aware routing.
* **Part D** is a TF-IDF policy assistant and a resumable ReAct refund agent.

`generate_data.py` and `mock_llm.py` are the supplied files and are unmodified.

## Reproduce
Run from the repository root, in this order:

1. `python -m venv venv`
2. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python part_a.py`
6. `python validate.py`
7. `python run_sql.py`
8. `python part_b.py`
9. `python part_c.py`
10. `python part_d.py`

Optional shortcut after step 3: `python run_all.py` runs steps 4-10 in order and stops at the first failure. The numbered commands above are the reference procedure.

---

## Part A: Data Foundation

### Audit (shape, data types, missing values)
| Table | Shape | Columns with missing values (count, %) |
| --- | --- | --- |
| customers | 1,200 x 5 | age 61 (5.08%), gender 30 (2.50%) |
| products | 40 x 4 | none |
| orders | 13,084 x 9 | order_amount 53 blank (0.41%), delivery_minutes 2,541 (19.42%) |
| tickets | 164 x 5 | none (read with `keep_default_na=False`) |

Data types: ids, city, dates, status, text and `order_amount` are strings in the raw files (`order_amount` also holds non-numeric text); `age` and `delivery_minutes` are float; `quantity`, `discount_pct`, `price_inr` are integers.

### Audit figures (computed by `part_a.py`)
| Figure | Value |
| --- | --- |
| raw orders rows | 13,084 |
| exact duplicate rows | 119 |
| pending rows | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| blank age | 61 |
| age outside 10-100 | 6 |
| distinct raw city spellings | 24 |
| distinct non-blank raw gender spellings | 10 |
| blank gender | 30 |

### Cleaning rules and how many values each changed
Customers:
1. City: strip spaces and title-case. 476 rows changed; exactly 6 cities remain (Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184).
2. Gender mapped to male / female / unknown. 950 rows changed (spelling or case), including 30 blanks set to unknown. Result: female 610, male 560, unknown 30.
3. Age: 6 values outside 10-100 set to missing, then all 67 missing ages (61 blank + 6) filled with the city median (`groupby` + `transform`). 0 missing; all ages are between 18 and 60.
4. `signup_date` converted to datetime.

Orders, in this order (counts are measured on the data as it stands when each rule runs, so some differ slightly from the raw audit figures):
1. `order_ts` converted to datetime.
2. Exact duplicates removed: 119 rows.
3. Keep only the latest `order_ts` per `order_id`: 656 older (pending) records dropped; 12,309 rows left.
4. `order_amount` made numeric; 112 missing amounts filled with `round(price_inr x quantity x (1 - discount_pct/100), 2)` (the raw audit shows 120 non-numeric entries; 8 of them sat on rows removed by rules 2-3).
5. Negative `delivery_minutes` set to missing: 105 values (106 in the raw file, one on a removed row). Then 774 missing delivery times of completed orders filled with the median of the order's product category among completed orders (`groupby` + `transform`). Cancelled orders keep a missing delivery time (1,199), because a cancelled order was never delivered and imputing would invent data.

Result: 12,309 orders (11,110 completed, 1,199 cancelled, 0 pending). Revenue of completed orders is INR 2,998,668.80.

### Storage and validation
Clean tables are written to `data/clean/*.csv` and to `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as `YYYY-MM-DD HH:MM:SS`). `validate.py` asserts 16 checkpoints (row and status counts, uniqueness of `order_id`, missing-value counts, revenue, city and gender counts, age range, DB tables and DB totals) and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run by `run_sql.py`, which reads the file)
| Query | Result |
| --- | --- |
| Q1 WHERE then HAVING | 30 customers |
| Q2 three-table join | 30 rows; sum INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products above the average price |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 CTE | 550 customers above the average total spend of INR 2,552.06 |
| Q6 window functions | 1,175 rows; 8 rows where RANK() differs from ROW_NUMBER() |
| Q7 running total | 183 daily rows; final cumulative revenue INR 2,998,668.80 |

Q6 top customer per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10).

RANK() differs from ROW_NUMBER() because tied customers get the same rank but ROW_NUMBER() always numbers rows one after another, so every row after a tie is numbered differently.

### pandas cross-check
`pd.pivot_table(..., margins=True)` grand total is 2,998,668.80, equal to the SQL total.

| City | Sum | Mean | Count |
| --- | --- | --- | --- |
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

### Exploratory analysis (completed orders)
| Statistic | Value |
| --- | --- |
| mean / median / skewness | 269.91 / 135.00 / 4.61 |
| Q1 / Q3 / IQR | 64.80 / 297.00 / 232.20 |
| fences (lower / upper) | -283.50 / 645.30 |
| outliers below / above | 0 / 897 |

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
| --- | --- | --- | --- | --- |
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

**Narrative.** The data are six months (April to September 2025) of completed grocery, dairy, snack, personal-care and electronics orders in six cities. Order values are strongly right-skewed: the mean (269.91) is about twice the median (135.00) and skewness is 4.61. 897 orders lie above the upper fence of 645.30 and none below the lower fence; the expensive electronics items (299 to 1,999 INR) are the likely source, so these look like genuine large baskets rather than errors and were kept. The pair with the largest absolute correlation is order_amount with discount_pct (-0.095); order_amount with quantity is +0.087. Both are very weak, and a correlation of this size does not establish causation. Delivery time is essentially unrelated to the other variables.

---

## Part B: Customer Segments and Lapse Prediction

### RFM table (snapshot 2025-10-01 00:00:00)
1,175 customers. Recency: median 19, min 0, max 180. Frequency: mean 9.46, max 33. Monetary: mean 2,552.06, max 14,766.55.

### K-means (scaled R, F, M; `n_init=10`, `random_state=42`)
| K | Inertia | Silhouette |
| --- | --- | --- |
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K = 4.** The elbow is at K=4 (inertia drops by 525 from K=3 but only by 126 from K=4 to K=5), K=4 has the highest silhouette (0.526), and four groups map onto four clear campaign types the business can act on. Inertia always falls as K grows because each extra centroid can only move points closer to a centre, so K=8 beats K=2 by construction; that fact alone cannot pick K, because K = number of customers would give inertia 0 and no useful segments.

### Segment profile (cluster numbers are only labels with no ordering)
Names are assigned once per cluster from its mean profile (Premium = highest mean monetary; At Risk = highest mean recency among the rest; Loyal Regulars = highest mean frequency among the rest).

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary | Recommended action |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Premium | 138 | 24.12 | 6.77 | 7,129.70 | VIP perks, early access, proactive retention outreach |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Subscription and bundle offers to protect frequency |
| 3 | At Risk | 182 | 109.04 | 2.20 | 685.99 | Time-limited win-back discount |
| 0 | Occasional Buyers | 436 | 28.86 | 3.77 | 1,057.62 | Nudges and recommendations to raise order frequency |

PCA (2 components) explained variance ratio: 0.634 and 0.218, total 0.852. `outputs/customer_segments.csv` has columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium.

### Lapse problem
Cutoff 2025-09-01. Cohort: 1,156 customers; 443 lapsed (38.32%). Split (`test_size=0.2`, stratified, `random_state=42`): 924 training, 232 test rows. Baseline "never lapsed" on the test set: accuracy 61.6%, lapsed-class F1 0.000.

Features: recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure. A code assertion confirms no order on or after the cutoff entered the features and that the feature matrix has no missing values.

### Leakage control
No encoder, scaler or resampler saw test rows: the `ColumnTransformer` (StandardScaler + OneHotEncoder with `drop="first"`) is fitted on the 924 training rows and only applied to the test rows, and oversampling touches the training rows only. No data dated 2025-09-01 or later entered the features. The segment labels from the RFM step were not used as features. The scaler's learned mean of frequency (8.11) differs from the full-cohort mean (8.03) because the scaler only saw the training rows.

### Tuning and test results (lapsed class)
| Model | Best parameters | Best CV F1 | Test precision | Test recall | Test F1 | Confusion [[TN, FP], [FN, TP]] |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic regression (C in 0.01, 0.1, 1, 10, 100) | C=100 | 0.651 | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| Random forest (depth 3, 5, 8; leaf 5, 10, 20) | depth 8, leaf 10 | 0.686 | 0.640 | 0.640 | 0.640 | [[111, 32], [32, 57]] |

**Selected model: random forest.** It was chosen on cross-validated F1 (0.686 against 0.651) before looking at the test set; the two models are statistically indistinguishable on the 232 test rows (F1 0.640 against 0.648), and both clear the 0.55 requirement. Missing a lapsing customer (a false negative) loses that customer's future orders, while wrongly targeting a loyal customer (a false positive) only wastes a small discount, so recall matters at least as much as precision and F1 is a sensible balance. The forest also handles non-linear effects of frequency and recency without extra feature work.

### Class-balance experiment (training set only)
Training rows kept as they are, plus random duplicates of lapsed rows (with replacement, `random_state=42`) until both classes have 570 rows. Logistic regression (C=100) refitted and evaluated on the unchanged test set:

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| un-balanced | 0.644 | 0.652 | 0.648 |
| balanced | 0.589 | 0.854 | 0.697 |

Recall moved most (+0.202); precision fell (-0.055). For the retention team this means the model flags 76 of 89 true lapsers instead of 58, at the price of 53 false alarms instead of 32, which is acceptable when a discount is cheap compared with a lost customer.

### Interpretation
Top five random-forest features: monetary (0.278), frequency (0.277), recency (0.111), avg_deliv (0.062), avg_disc (0.053).
* monetary: how much a customer has spent so far is the strongest signal of staying power.
* frequency: how often they order is just as important.
* recency: days since the last order shows whether the habit is already fading.
* avg_deliv: slow average deliveries may push customers away.
* avg_disc: reliance on discounts may indicate price-driven, less loyal buyers.

Logistic-regression coefficients (standardised features): frequency -1.698 (more orders lowers lapse risk, as expected); recency +0.040 (longer since the last order raises risk, but the effect is tiny once frequency is included); cancel_rate -0.124 (small and counter-intuitive, so treated as noise rather than as a finding).

### Persisted model and retention list
The selected pipeline (preprocessor + random forest) is saved to `outputs/lapse_model.joblib`; a code assertion shows the reloaded model reproduces the original test predictions exactly (`np.array_equal`). `outputs/retention_list.csv` holds the 10 Premium test-set customers with the highest predicted lapse probability; 1 of them truly lapsed. The retention list is illustrative and not a leakage-free evaluation, because the Premium flag comes from whole-window RFM that includes the September window used to define the label.

---

## Part C: Support Ticket Triage

### Guardrails and redaction
* Input guardrails: 4 empty tickets (`skipped_empty`) and 4 long tickets (`flagged_long`, over 400 estimated tokens) were never sent to the model; 156 tickets were eligible.
* Redaction (email, then phone, then order id): 55 emails, 91 phone numbers, 132 order ids, 278 in total. Re-applying the three patterns to the redacted texts finds 0. An assertion inside the call loop checks that the user message sent to the model equals the redacted text and matches none of the three patterns.

### Retries
Maximum 3 retries per ticket, wait = 0.05 x 2^n x uniform(0.9, 1.15) seconds. Each full run produced exactly 19 TransientLLMErrors, all 19 recovered by a retry; no ticket ended with a permanent error. Every retry line (run, ticket, retry number, wait) is written to `outputs/retry_log.txt` and the final run also prints them. All waits are retry number 0, between 0.0463 and 0.0560 seconds.

### Validation results (one full run, both prompts)
135 `classified_json`, 12 `classified_keyword` (chatty replies parsed with `re.findall(r"[a-z]+", ...)`), 9 `flagged_invalid_output` (non-JSON with no single category word, or a category outside the list).

### Zero-shot versus few-shot
| Run | Classified | Accuracy | Total tokens | Tokens per classified ticket |
| --- | --- | --- | --- | --- |
| Zero-shot | 147 | 85.7% (126 of 147) | 9,712 | 66.1 |
| Few-shot | 147 | 100.0% | 16,576 | 112.8 |

Zero-shot confusion table (rows = true, columns = predicted):

| true \ predicted | billing | delivery | fraud | technical |
| --- | --- | --- | --- | --- |
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

**Is the extra cost worth it?** The few-shot prompt costs 46.7 more tokens per classified ticket (112.8 against 66.1, about 71% more, or 6,864 extra tokens over 147 tickets) and lifts accuracy from 85.7% to 100%. All 21 zero-shot errors were fraud (11) or technical (10) tickets sent to billing, and a fraud ticket in the billing queue misses the 2-hour security target. The extra tokens are worth paying whenever a wrong route is costlier than the added tokens, as with fraud here; for low-stakes, high-volume classification where 85% is acceptable, zero-shot is cheaper.

### Budget demo
With `TOKEN_BUDGET = 6000` (zero-shot) the run stopped calling the model part-way: 59 tickets ended `budget_exhausted` (written to `outputs/triage_budget_demo.csv`). The budget is checked before each call, so the final total (6,049) can slightly pass 6,000.

### Final routing (`outputs/triage_results.csv`, 164 rows, few-shot, budget 100,000)
Status counts: 135 classified_json, 12 classified_keyword, 9 flagged_invalid_output, 4 skipped_empty, 4 flagged_long. Routes: support_queue 102, security_team 34, human_review 17, premium_desk 11.

**Rule-validated:** the 147 tickets with status `classified_json` or `classified_keyword` were validated by rules and routed automatically. **Human review:** the 17 tickets that were empty (4), too long (4) or had invalid output (9); no ticket was guessed.

---

## Part D: Policy Assistant & Refund Agent

### Chunking and retrieval
15 chunks (40 words, overlap 10): refund 4, delivery 3, escalation 3, payment 3, privacy 2. For the query "refund policy late delivery" the top three chunks all come from `refund_policy.txt`. TF measures how often a word occurs in a chunk; IDF measures how rare the word is across all chunks, so rare words weigh more. A limitation of TF-IDF is that it matches words, not meaning: "late" and "delayed" are unrelated to it.

### Grounding evaluation (14 questions)
| Setting | Expected doc in top 3 (of 12) | Answer contains expected fact (of 12) | Fallback on unanswerable (of 2) |
| --- | --- | --- | --- |
| overlap 10 | 12 | 12 | 2 |
| overlap 0 | 12 | 10 | 2 |

With zero overlap two answerable questions break, although the correct document is still retrieved for both. (1) "What delivery time makes a completed order eligible for a full refund?" (expected "more than 45 minutes"): the first refund chunk ends mid-sentence ("...is delivered in more than 45 minutes, the customer"), and the rest of that sentence ("receives a full refund of the order amount.") starts the next chunk. The mock picks the sentence that best matches the question, and the fragment it picks does not contain the rule. (2) "How soon is money returned when a payment fails but is deducted?" (expected "48 hours"): the payment sentence is cut right before "within 48 hours", so the chunk that holds those words is not in the top 3 and the answer stops at "the amount is returned automatically". A 10-word overlap repeats the end of each chunk at the start of the next, so a rule and its condition stay together. That repairs both questions (12 of 12).

### Tools
`get_order` (parameterised SQL on `data/instamart.db`), `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human`, all returning dictionaries (serialised to a JSON string only inside the tool message), described in `TOOLS_SPEC` and dispatched through `TOOL_MAP`. Refund rules: only completed orders; over 45 minutes refunds 100%; 45 minutes or less refunds 0; above INR 1000 is escalated.

### Seven scenarios
The refund amount is read from the `issue_refund` / `escalate_to_human` observation and matches `expected_refund_inr` for every scenario (asserted in code).

| scenario_id | status | refund_inr | llm_calls | tokens_used |
| --- | --- | --- | --- | --- |
| S1 | REFUND | 128.25 | 5 | 882 |
| S2 | REFUND | 135.00 | 5 | 882 |
| S3 | ESCALATE | 1999.00 | 5 | 869 |
| S4 | DENIED | 0.00 | 4 | 593 |
| S5 | DENIED | 0.00 | 3 | 328 |
| S6 | NOT_FOUND | 0.00 | 2 | 66 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 2,982 |

### Stop-safely demonstrations
| Demo | Result |
| --- | --- |
| S1 with `token_limit=200` | FALLBACK_TOKEN_LIMIT (stopped before executing the tool requested by the third call; 2 steps done) |
| S1 with `time_limit_seconds=0` | FALLBACK_TIMEOUT (stopped before the first model call) |

Every fallback returns the completed steps and a hand-off summary naming the tools already run. The token check uses "exceeds" (`>`), the timeout check uses `>=`.

### Crash and resume
Checkpoints in `state/` hold run id, scenario id, message list, tokens used, step number and the list of tools executed. S1 was crashed right after its second tool execution and resumed with `resume_agent`. Tools executed across the crash and the resume, in order: `get_order`, `search_policy`, `calculate_refund`, `issue_refund` (four executions, none repeated). Final status REFUND with 128.25. The demonstrations write to `outputs/refund_ledger_demo.jsonl` so that `outputs/refund_ledger.jsonl` holds only the three real actions of S1-S3 and no duplicate payout.

### ReAct versus plan-and-execute
ReAct calls the model after every observation: 5 calls and 882 tokens for S1, 3 calls and 328 tokens for S5, 2 calls and 66 tokens for S6. A plan-and-execute design would ask the model once for the whole step list (get order, search policy, calculate, issue), so on the happy path it would need only one or two calls and far fewer tokens, because the growing message history is not re-sent after every tool. Its weakness is the unexpected observation. In S6 the order does not exist, and in S4 and S5 the order is cancelled or delivered on time; a fixed plan would carry on towards a refund until an error forced a replan, wasting calls and risking a wrong action. ReAct sees NOT_FOUND or the cancelled status straight away and stops after 2 to 4 calls. Because this workflow moves money, I would choose ReAct: deciding after each observation prevents irreversible payouts based on a bad assumption, and the extra tokens (a few hundred per case) are a small price.

---

## Limitations and Next Steps
* The Premium flag used for the retention list comes from whole-window RFM, which includes September (the window that defines the lapse label), so the retention list is illustrative and not a leakage-free evaluation.
* K-means on three RFM variables ignores category and city behaviour; the lapse model has only 232 test rows, so differences of about 0.01 F1 between models are noise.
* The mock LLM is keyword-based, so Part C and Part D accuracy figures describe the pipeline logic, not a real model.
* Next steps: validate segments with a campaign test, replace TF-IDF with a semantic retriever, add idempotency keys to refund tools, and move SQLite and agent state to a shared database for concurrent use.
