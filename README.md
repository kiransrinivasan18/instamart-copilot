# Instamart Customer Intelligence & Support Copilot

## Overview
This repository contains the code and outputs for the Swiggy Instamart Customer Intelligence & Support Copilot capstone project. It transforms messy, synthetic order exports into a clean SQLite analytical database (Part A), segments customers and predicts lapse likelihood (Part B), builds an LLM-based support ticket triage pipeline with guardrails (Part C), and provides a Policy RAG Assistant and an autonomous ReAct Refund Agent (Part D).

## Reproduce
To reproduce the environment and all outputs from scratch on a clean clone, run these commands sequentially in your terminal:

1. `python -m venv venv`
2. `venv\Scripts\activate` (On Mac/Linux: `source venv/bin/activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python part_a.py`
6. `python run_sql.py`
7. `python part_b.py`
8. `python part_c.py`
9. `python part_d.py`

## Part A: Data Foundation

### Data Audit
* **Orders:** 13,084 raw rows; 119 exact duplicates; 656 pending rows; 12,309 distinct `order_id` values. 120 unparseable `order_amount` entries; 106 negative `delivery_minutes`; 2,541 blank `delivery_minutes`.
* **Customers:** 61 blank ages; 6 ages outside 10–100. 24 raw city spellings; 10 raw gender spellings; 30 blank genders.

### Cleaning Rules Applied
1. **Customers:** Stripped whitespace and title-cased cities resulting in exactly 6 distinct cities (Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184). Standardized gender to male (560), female (610), or unknown (30). Ages outside 10-100 were treated as missing, then filled with the city median (0 missing ages remaining).
2. **Orders:** Kept only the latest `order_ts` record per `order_id` (removed all pending duplicates). Recalculated 120 unparseable amounts using `product_price * qty * (1 - discount)`. Missing delivery times for completed orders were imputed using the median time for that product category. Cancelled orders were left with missing delivery times. Result: 12,309 distinct orders (11,110 completed).

### SQL Analysis Results
* **Q1:** 30 customers.
* **Q2:** 30 rows; Total sums to INR 2,998,668.80.
* **Q3:** 11 products.
* **Q4:** 25 customers.
* **Q5:** 550 customers above the overall average.
* **Q6:** 1,175 rows. Exactly 8 rows have `RANK()` different from `ROW_NUMBER()`. This happens because `RANK()` assigns identical ranks to ties, whereas `ROW_NUMBER()` forces a sequential numbering even if values are identical.
* **Q7:** 183 daily rows; Final cumulative revenue is INR 2,998,668.80.

### Exploratory Analysis
* **Stats:** Mean: 269.91, Median: 135.00, Skewness: 4.61. Q1: 64.80, Q3: 297.00. Upper fence: 645.30 with 897 outliers above it and none below.
* **Narrative:** The data represents customer grocery orders. The distribution is highly right-skewed (mean > median) due to expensive electronics orders acting as natural outliers above the 645.30 upper fence. The strongest correlation in absolute value is between `order_amount` and `discount_pct` (-0.095). A correlation of this size is very weak and does not establish causation.

## Part B: Customer Segments and Lapse Prediction

### Clustering (K-Means)
| K | Inertia | Silhouette |
|---|---------|------------|
| 2 | 1820.8  | 0.465      |
| 3 | 1268.2  | 0.445      |
| 4 | 743.1   | 0.526      |
| 5 | 617.4   | 0.480      |
| 6 | 503.2   | 0.482      |
| 7 | 408.5   | 0.416      |
| 8 | 366.5   | 0.412      |

Selected **K=4** based on the elbow and highest silhouette score. K=8 has a lower inertia because adding more clusters mathematically always reduces the distance to centroids, but that fact alone cannot pick K as it leads to overfitting without business interpretability. PCA explained variance ratio: 0.63427875 and 0.21803762 (Total: 0.852). Segment 1 is the Premium segment with a mean monetary value of 7,129.69.

### Lapse Model
* **Cohort:** 1,156 customers. 443 lapsed (38.32%).
* **Leakage Prevention:** No encoder, scaler, or resampler saw test rows during fitting. No data from 2025-09-01 onwards entered the features. Segment labels from Task 3 were not used as features. The full cohort mean frequency (8.03) differs from the scaler's learned mean (8.12) because the scaler was strictly fitted on the training slice only.
* **Baseline:** Test F1 (predicting all 0) is 0.000.
* **Tuning Results:** 
  * Logistic Regression Best: C=0.1 (CV F1: 0.639).
  * Random Forest Best: max_depth=5, min_samples_leaf=10 (CV F1: 0.670).
* **Model Selection:** The Random Forest was selected (Test F1: 0.71, Precision: 0.67, Recall: 0.74) because its higher F1 score represents a better balance. Missing a lapsing customer (false negative) represents lost future lifetime value, while wrongly targeting a loyal one (false positive) merely wastes a small discount cost.
* **Class Balance Experiment:** Oversampling the lapsed class increased Recall significantly to 0.94 while precision dropped to 0.59 (F1: 0.73). For the retention team, this means the model will flag almost all true lapsers, though they will have to send retention discounts to a few extra non-lapsers.
* **Top Features:** Top Random Forest features were frequency, monetary, recency, cancel_rate, and fav_cat_Electronics.

## Part C: Support Ticket Triage

* **Guardrails:** 4 empty tickets skipped. 4 long tickets skipped. Total redactions: 278 (55 emails, 91 phones, 132 order IDs).
* **Validation (Full Run):** 147 classified (135 JSON, 12 KW, 9 Invalid).

### Zero-Shot vs Few-Shot Comparison
* **Zero-Shot:** Accuracy: 85.7%. Total Tokens: 9,088 (61.8 per classified ticket). Every error was a fraud ticket (11) or technical ticket (10) predicted as billing.
* **Few-Shot:** Accuracy: 100.0%. Total Tokens: 15,796 (107.5 per classified ticket).

**Zero-Shot Confusion Matrix**
| True \ Predicted | billing | delivery | fraud | technical |
|------------------|---------|----------|-------|-----------|
| **billing**      | 38      | 0        | 0     | 0         |
| **delivery**     | 0       | 36       | 0     | 0         |
| **fraud**        | 11      | 0        | 23    | 0         |
| **technical**    | 10      | 0        | 0     | 29        |

* **Cost Analysis:** The few-shot prompt uses extra tokens per ticket, but it is worth paying for because it completely eliminates critical misclassifications, ensuring that high-risk fraud tickets are never incorrectly routed to standard billing queues. 
* **Budget Demo:** Running with a 6,000 token budget successfully halted the pipeline mid-execution, resulting in 53 tickets being marked `budget_exhausted` and routed to human review. Outputs in `triage_results.csv` map all statuses to strict priority queues via rules; anything that wasn't rule-validated went to humans.

## Part D: Policy Assistant & Refund Agent

* **Chunking & Retrieval:** 15 total chunks generated (refund 4, delivery 3, escalation 3, payment 3, privacy 2). TF (Term Frequency) measures how often a word appears in a chunk. IDF (Inverse Document Frequency) measures how rare the word is across all chunks. A limitation of TF-IDF is its inability to understand semantic meaning (e.g., it cannot recognize that "late" and "delayed" mean the same thing).
* **Grounding Evaluation:** 
  * Overlap 10: 12/12 expected docs, 12/12 expected facts, and 2/2 fallbacks.
  * Overlap 0: 10/12 expected facts. Failed on: *"What delivery time makes a completed order eligible for a full refund?"*. This failed because the answer context was split across a chunk boundary, severing the rule from its condition.

### ReAct Agent Scenario Results

| scenario_id | status                  | refund_amount | llm_calls | tokens_used |
|-------------|-------------------------|---------------|-----------|-------------|
| S1          | REFUND                  | 128.25        | 5         | 806         |
| S2          | REFUND                  | 135.00        | 5         | 806         |
| S3          | ESCALATE                | 1000.00       | 5         | 793         |
| S4          | DENIED                  | 0.00          | 4         | 534         |
| S5          | DENIED                  | 0.00          | 3         | 286         |
| S6          | NOT_FOUND               | 0.00          | 2         | 50          |
| S7          | FALLBACK_MAX_ITERATIONS | 0.00          | 6         | 2934        |

* **Safety Fallbacks:** 
  * Token limit demo (200 tokens) ended with: `FALLBACK_TOKEN_LIMIT`.
  * Time limit demo (0 seconds) ended with: `FALLBACK_TIMEOUT`.
* **Crash & Resume:** Simulated a crash during S1. The tools executed across the crash and the resume, in order, were: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`.
* **ReAct vs Plan-and-Execute:** ReAct is highly dynamic; it uses more LLM calls (e.g., 5 for S1, 2 for S6) and tokens because it pauses to observe the result of every tool. A plan-and-execute design uses fewer calls upfront, but struggles with an unexpected observation like `NOT_FOUND` or a cancelled order because its rigid plan assumes a happy path. For a financial refund workflow, ReAct is the clear choice: it safely evaluates real database states and refund rules before making irreversible payment actions.

## Limitations and Next Steps
* The `is_premium` flag used for the retention list was derived post-hoc using the entire window of data (including September). Therefore, the `retention_list.csv` is purely illustrative and not a rigorous, leakage-free evaluation of future performance.
* Next steps: Move SQLite to a cloud data warehouse, migrate the offline LLM mock to an actual API provider, and deploy the Agent state logic to a persistent database to handle concurrent requests.