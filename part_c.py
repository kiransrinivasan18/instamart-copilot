"""Part C: support-ticket triage with guardrails, retries, validation, budget and routing."""
import json
import os
import random
import re
import time

import pandas as pd

import mock_llm

os.makedirs("outputs", exist_ok=True)

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED = {"billing", "delivery", "technical", "fraud"}

# ------------------------------------------------------------------ 1-2. GUARDRAILS + REDACTION
print("--- 1 & 2. INPUT GUARDRAILS AND REDACTION ---")
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]


def redact(text, counts):
    for name, pat, placeholder in PATTERNS:
        counts[name] += len(pat.findall(text))
        text = pat.sub(placeholder, text)
    return text


counts = {n: 0 for n, _, _ in PATTERNS}
clean_texts, base_status = [], []
for text in tickets["text"]:
    redacted = redact(text, counts)          # redaction applied to all 164 tickets
    clean_texts.append(redacted)
    if not text.strip():
        base_status.append("skipped_empty")
    elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
        base_status.append("flagged_long")
    else:
        base_status.append("pending")
tickets["clean_text"], tickets["base_status"] = clean_texts, base_status

print(f"skipped_empty: {base_status.count('skipped_empty')}, flagged_long: {base_status.count('flagged_long')}, "
      f"eligible: {base_status.count('pending')}")
print(f"redactions: emails {counts['email']}, phones {counts['phone']}, order ids {counts['order_id']}, "
      f"total {sum(counts.values())}")
left = sum(len(p.findall(t)) for t in clean_texts for _, p, _ in PATTERNS)
assert left == 0, "personal data left after redaction"
print(f"patterns re-applied to redacted texts find: {left}")

# ------------------------------------------------------------------ 3. PROMPTS
ZERO_SHOT = """CLASSIFY the customer support ticket into exactly one of these categories: billing, delivery, technical, fraud.
Reply with JSON only, in the form {"category": "...", "reason": "..."}"""
FEW_SHOT = ZERO_SHOT + """

EXAMPLES:
Ticket: "Someone hacked my account and used my wallet balance."
{"category": "fraud", "reason": "unauthorized access and stolen balance"}
Ticket: "My delivery for [ORDER_ID] is very late."
{"category": "delivery", "reason": "complaint about delayed arrival"}
"""
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT


# ------------------------------------------------------------------ 5. VALIDATION
def validate_output(text):
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("category") in ALLOWED:
            return data["category"], "classified_json"
    except (json.JSONDecodeError, TypeError):
        pass
    words = re.findall(r"[a-z]+", text.lower())
    found = [c for c in ALLOWED if c in words]
    if len(found) == 1:
        return found[0], "classified_keyword"
    return None, "flagged_invalid_output"


retry_file = open("outputs/retry_log.txt", "w", encoding="utf-8")


# ------------------------------------------------------------------ 4. RESILIENT PIPELINE
def run_pipeline(label, system_prompt, token_budget, show_retries=False):
    """One full run. Every retry gets one log line (written to outputs/retry_log.txt, and printed if show_retries)."""
    mock_llm.reset()
    used, results = 0, []
    stats = {"transient": 0, "recovered": 0, "calls_for_ineligible": 0}
    for _, row in tickets.iterrows():
        tid, status = row["ticket_id"], row["base_status"]
        if status != "pending":                       # guardrail: never sent to the model
            results.append((tid, status, None, 0))
            continue
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["clean_text"]}]
        # privacy assertion on the exact payload that is sent
        sent = messages[-1]["content"]
        assert sent == row["clean_text"]
        assert not any(p.search(sent) for _, p, _ in PATTERNS), "personal data in a prompt"

        final_status, cat, tokens, had_error = "flagged_invalid_output", None, 0, False
        for attempt in range(MAX_RETRIES + 1):
            if used >= token_budget:                  # budget check before every call
                final_status = "budget_exhausted"
                break
            try:
                res = mock_llm.chat(messages)
                tokens = res.usage.total_tokens
                used += tokens
                cat, final_status = validate_output(res.choices[0].message.content)
                if had_error:
                    stats["recovered"] += 1
                break
            except mock_llm.TransientLLMError:
                stats["transient"] += 1
                had_error = True
                if attempt == MAX_RETRIES:
                    final_status = "flagged_invalid_output"      # permanent failure -> human, never guess
                    break
                wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)
                line = f"[{label}] ticket {tid}: retry {attempt}, wait {wait:.4f}s"
                retry_file.write(line + "\n")
                if show_retries:
                    print(line)
                time.sleep(wait)
        results.append((tid, final_status, cat, tokens))
    return results, stats, used


def evaluate(results):
    res = pd.DataFrame(results, columns=["ticket_id", "status", "pred_cat", "tokens"])
    merged = tickets.merge(res, on="ticket_id")
    ok = merged[merged["status"].isin(["classified_json", "classified_keyword"])]
    return {"classified": len(ok), "accuracy": (ok["pred_cat"] == ok["true_category"]).mean(),
            "tokens": int(res["tokens"].sum()), "json": int((merged["status"] == "classified_json").sum()),
            "keyword": int((merged["status"] == "classified_keyword").sum()),
            "invalid": int((merged["status"] == "flagged_invalid_output").sum())}, ok


print("\n--- 3, 4, 5, 6. FULL RUNS (zero-shot vs few-shot) ---")
zs_res, zs_stats, _ = run_pipeline("zero-shot", ZERO_SHOT, 10 ** 9)
fs_res, fs_stats, _ = run_pipeline("few-shot", FEW_SHOT, 10 ** 9)
zs, zs_ok = evaluate(zs_res)
fs, _ = evaluate(fs_res)
for name, r, st in [("Zero-shot", zs, zs_stats), ("Few-shot", fs, fs_stats)]:
    print(f"{name}: classified {r['classified']} ({r['json']} json, {r['keyword']} keyword), invalid {r['invalid']}, "
          f"accuracy {r['accuracy']:.1%}, tokens {r['tokens']}, per classified ticket {r['tokens'] / r['classified']:.1f}")
    print(f"   TransientLLMErrors {st['transient']}, recovered by retry {st['recovered']}")
print("\nZero-shot confusion table (rows = true, columns = predicted):")
print(pd.crosstab(zs_ok["true_category"], zs_ok["pred_cat"]).to_string())
print(f"extra few-shot cost: {(fs['tokens'] / fs['classified']) - (zs['tokens'] / zs['classified']):.1f} tokens per classified ticket")

# ------------------------------------------------------------------ 7. BUDGET DEMO + ROUTING
print("\n--- 7. BUDGET DEMO AND FINAL ROUTING ---")
demo_res, demo_stats, demo_used = run_pipeline("budget-demo", ZERO_SHOT, 6000)
demo = pd.DataFrame(demo_res, columns=["ticket_id", "status", "category", "tokens_used"])
demo.to_csv("outputs/triage_budget_demo.csv", index=False)
n_ex = int((demo["status"] == "budget_exhausted").sum())
assert n_ex >= 1
print(f"budget demo (TOKEN_BUDGET = 6000): {n_ex} tickets ended budget_exhausted; tokens used {demo_used}")

final_res, final_stats, _ = run_pipeline("final-few-shot", FEW_SHOT, 100000, show_retries=True)
retry_file.close()
print(f"final run: {final_stats['transient']} TransientLLMErrors, {final_stats['recovered']} recovered")
final = pd.DataFrame(final_res, columns=["ticket_id", "status", "category", "tokens_used"])

segments = pd.read_csv("outputs/customer_segments.csv")[["customer_id", "segment_name", "is_premium"]]
final = tickets[["ticket_id", "customer_id"]].merge(final, on="ticket_id").merge(segments, on="customer_id", how="left")
final["is_premium"] = final["is_premium"].fillna(False).astype(bool)

HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}


def route(r):
    if r["status"] in HUMAN:
        return "human_review", "NORMAL"
    if r["category"] == "fraud":
        return "security_team", "HIGH"
    if r["is_premium"]:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


final[["route", "priority"]] = final.apply(lambda r: pd.Series(route(r)), axis=1)
final["category"] = final["category"].fillna("")
final["segment_name"] = final["segment_name"].fillna("")
cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
final[cols].to_csv("outputs/triage_results.csv", index=False)
assert len(final) == 164

print(f"triage_results.csv rows: {len(final)}")
print("status counts:", final["status"].value_counts().to_dict())
print("route counts:", final["route"].value_counts().to_dict())
validated = final["status"].isin(["classified_json", "classified_keyword"]).sum()
print(f"rule-validated (classified_json + classified_keyword): {validated}; sent to human review: {(final['route'] == 'human_review').sum()}")
print("\nPART C COMPLETE")
