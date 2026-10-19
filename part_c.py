import pandas as pd
import re
import json
import time
import random
import os
import mock_llm

os.makedirs('outputs', exist_ok=True)
os.makedirs('state', exist_ok=True)

print("--- 1 & 2. INPUT GUARDRAILS & REDACTION ---")
tickets = pd.read_csv('data/raw/tickets.csv', keep_default_na=False)

email_re = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
phone_re = re.compile(r"\b[6-9]\d{9}\b")
order_re = re.compile(r"ORD-\d{6}")

redact_counts = {'email': 0, 'phone': 0, 'order': 0}
clean_texts = []
statuses = []

for idx, row in tickets.iterrows():
    text = row['text']
    # Guardrail: Empty
    if not text.strip():
        clean_texts.append("")
        statuses.append("skipped_empty")
        continue
        
    # Guardrail: Too Long
    if mock_llm.est_tokens(text) > 400:
        clean_texts.append(text)
        statuses.append("flagged_long")
        continue
        
    # Redact Privacy Data
    e_subs = len(email_re.findall(text))
    text = email_re.sub("[EMAIL]", text)
    
    p_subs = len(phone_re.findall(text))
    text = phone_re.sub("[PHONE]", text)
    
    o_subs = len(order_re.findall(text))
    text = order_re.sub("[ORDER_ID]", text)
    
    redact_counts['email'] += e_subs
    redact_counts['phone'] += p_subs
    redact_counts['order'] += o_subs
    
    clean_texts.append(text)
    statuses.append("pending")

tickets['clean_text'] = clean_texts
tickets['base_status'] = statuses

print(f"Redactions - Emails: {redact_counts['email']}, Phones: {redact_counts['phone']}, Orders: {redact_counts['order']}")
print(f"Total Redactions: {sum(redact_counts.values())}")

# Verify no leaks
leftover = sum(1 for t in clean_texts if email_re.search(t) or phone_re.search(t) or order_re.search(t))
assert leftover == 0, "Privacy leak detected! Guardrails failed."

print("\n--- 3, 4, 5. PROMPTS, RETRY & VALIDATION ---")
ZERO_SHOT = """CLASSIFY the customer support ticket into one of these categories: billing, delivery, technical, fraud.
Reply ONLY in JSON format: {"category": "...", "reason": "..."}"""

FEW_SHOT = ZERO_SHOT + """

EXAMPLES:
Ticket: "Someone hacked my account and used my wallet balance."
{"category": "fraud", "reason": "unauthorized access and stolen balance"}

Ticket: "Delivery for [ORDER_ID] is very late."
{"category": "delivery", "reason": "complaint about delayed arrival"}
"""

def validate_output(text):
    allowed = {"billing", "delivery", "technical", "fraud"}
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("category") in allowed:
            return data["category"], "classified_json"
    except:
        pass
        
    # Fallback keyword extraction if JSON fails
    words = re.findall(r"[a-z]+", text.lower())
    found = [w for w in allowed if w in words]
    if len(found) == 1:
        return found[0], "classified_keyword"
        
    return None, "flagged_invalid_output"

def run_pipeline(prompt_sys, budget_limit, is_final=False):
    mock_llm.reset() # Reset transient error counter
    budget = {'used': 0, 'limit': budget_limit}
    results = []
    BASE_DELAY = 0.05
    
    for idx, row in tickets.iterrows():
        tid = row['ticket_id']
        status = row['base_status']
        
        if status != "pending":
            results.append((tid, status, None, 0))
            continue
            
        messages = [
            {"role": "system", "content": prompt_sys},
            {"role": "user", "content": row['clean_text']}
        ]
        
        final_status = "flagged_invalid_output"
        final_cat = None
        tokens = 0
        
        # Max 1 attempt + 3 retries = 4 loops
        for attempt in range(4):
            if budget['used'] >= budget['limit']:
                final_status = "budget_exhausted"
                break
                
            try:
                res = mock_llm.chat(messages)
                tokens = res.usage.total_tokens
                budget['used'] += tokens
                
                cat, val_status = validate_output(res.choices[0].message.content)
                final_status = val_status
                final_cat = cat
                break # Success, exit retry loop
                
            except mock_llm.TransientLLMError:
                if attempt < 3: # Can retry 3 times
                    wait = BASE_DELAY * (2**attempt) * random.uniform(0.9, 1.15)
                    # Log retry only on final full run to keep terminal clean
                    if is_final:
                        print(f"Retry Log - Ticket {tid}: Retry {attempt}, wait {wait:.4f}s")
                    time.sleep(wait)
                else:
                    raise
        
        results.append((tid, final_status, final_cat, tokens))
        
    return results

print("\nRunning Zero-Shot (No Limit)...")
zs_res = run_pipeline(ZERO_SHOT, 999999)
print("Running Few-Shot (No Limit)...")
fs_res = run_pipeline(FEW_SHOT, 999999)

# 6. Evaluate and Compare
def evaluate(res_list):
    df_res = pd.DataFrame(res_list, columns=['ticket_id', 'status', 'pred_cat', 'tokens'])
    df_eval = tickets.merge(df_res, on='ticket_id')
    classified = df_eval[df_eval['status'].isin(['classified_json', 'classified_keyword'])]
    
    acc = (classified['pred_cat'] == classified['true_category']).mean() if not classified.empty else 0
    json_cnt = (df_eval['status'] == 'classified_json').sum()
    kw_cnt = (df_eval['status'] == 'classified_keyword').sum()
    inv_cnt = (df_eval['status'] == 'flagged_invalid_output').sum()
    
    return len(classified), acc, df_res['tokens'].sum(), json_cnt, kw_cnt, inv_cnt, df_eval

zs_cls, zs_acc, zs_tok, z_j, z_k, z_i, zs_df = evaluate(zs_res)
fs_cls, fs_acc, fs_tok, f_j, f_k, f_i, fs_df = evaluate(fs_res)

print(f"\nZero-Shot: {zs_cls} classified ({z_j} JSON, {z_k} KW, {z_i} Invalid). Acc: {zs_acc:.1%}. Total Tokens: {zs_tok}, Per classified ticket: {zs_tok/zs_cls:.1f}")
print(f"Few-Shot:  {fs_cls} classified ({f_j} JSON, {f_k} KW, {f_i} Invalid). Acc: {fs_acc:.1%}. Total Tokens: {fs_tok}, Per classified ticket: {fs_tok/fs_cls:.1f}")

print("\nZero-Shot Confusion Matrix:")
zs_classified = zs_df[zs_df['status'].isin(['classified_json', 'classified_keyword'])]
print(pd.crosstab(zs_classified['true_category'], zs_classified['pred_cat']))

# 7. Budget Demo & Final Routing
print("\n--- 7. BUDGET DEMO & FINAL ROUTING ---")
budget_demo = run_pipeline(ZERO_SHOT, 6000)
demo_df = pd.DataFrame(budget_demo, columns=['ticket_id', 'status', 'category', 'tokens'])
demo_df.to_csv('outputs/triage_budget_demo.csv', index=False)
exhausted = (demo_df['status'] == 'budget_exhausted').sum()
print(f"Budget Demo (6000 tokens): {exhausted} tickets ended as 'budget_exhausted'.")

print("\nGenerating Final Triage Results with Few-Shot (Limit 100,000 tokens)...")
final_res = run_pipeline(FEW_SHOT, 100000, is_final=True)
final_df = pd.DataFrame(final_res, columns=['ticket_id', 'status', 'category', 'tokens_used'])

# Read segments from Part B for routing logic
segments = pd.read_csv('outputs/customer_segments.csv')[['customer_id', 'segment_name', 'is_premium']]

final_df = tickets[['ticket_id', 'customer_id']].merge(final_df, on='ticket_id')
final_df = final_df.merge(segments, on='customer_id', how='left')
final_df['is_premium'] = final_df['is_premium'].fillna(False)

routes = []
priorities = []

for idx, row in final_df.iterrows():
    s = row['status']
    c = row['category']
    p = row['is_premium']
    
    if s in ['skipped_empty', 'flagged_long', 'flagged_invalid_output', 'budget_exhausted']:
        routes.append('human_review')
        priorities.append('NORMAL')
    elif c == 'fraud':
        routes.append('security_team')
        priorities.append('HIGH')
    elif p:
        routes.append('premium_desk')
        priorities.append('HIGH')
    else:
        routes.append('support_queue')
        priorities.append('NORMAL')
        
final_df['route'] = routes
final_df['priority'] = priorities

final_df.loc[final_df['category'].isna(), 'category'] = ""
final_df.loc[final_df['segment_name'].isna(), 'segment_name'] = ""

out_cols = ['ticket_id', 'customer_id', 'status', 'category', 'segment_name', 'priority', 'route', 'tokens_used']
final_df[out_cols].to_csv('outputs/triage_results.csv', index=False)

print("\nPART C COMPLETE! Wrote outputs/triage_results.csv")