import os
import json
import sqlite3
import pandas as pd
import time
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

os.makedirs('outputs', exist_ok=True)
os.makedirs('state', exist_ok=True)

# ---------------------------------------------------------
# TASK 1 & 2: Chunking & Retrieval
# ---------------------------------------------------------
print("--- 1 & 2. CHUNKING & RETRIEVAL ---")
def create_chunks(overlap=10):
    chunks = []
    files = sorted([f for f in os.listdir('data/policies') if f.endswith('.txt')])
    for fn in files:
        with open(os.path.join('data/policies', fn), 'r', encoding='utf-8') as f:
            words = f.read().split()
        
        step = 40 - overlap
        if step <= 0: step = 40
        
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i+40])
            chunks.append({'source': fn, 'text': chunk})
            if i + 40 >= len(words):
                break
    return chunks

chunks_10 = create_chunks(10)
print(f"Total chunks (overlap 10): {len(chunks_10)}")
from collections import Counter
print("Chunks per file:", Counter([c['source'] for c in chunks_10]))

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([c['text'] for c in chunks_10])

def retrieve(query, k=3, custom_vectorizer=None, custom_matrix=None, custom_chunks=None):
    vec = custom_vectorizer or vectorizer
    mat = custom_matrix if custom_matrix is not None else tfidf_matrix
    chk = custom_chunks or chunks_10
    
    q_vec = vec.transform([query])
    sims = cosine_similarity(q_vec, mat)[0]
    
    top_indices = sims.argsort()[-k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'source': chk[idx]['source'],
            'score': float(sims[idx]),
            'text': chk[idx]['text']
        })
    return results

# ---------------------------------------------------------
# TASK 3 & 4: RAG Evaluation
# ---------------------------------------------------------
print("\n--- 3 & 4. RAG EVALUATION ---")
def answer(question, custom_vectorizer=None, custom_matrix=None, custom_chunks=None):
    retrieved = retrieve(question, 3, custom_vectorizer, custom_matrix, custom_chunks)
    context = ""
    for r in retrieved:
        context += f"[{r['source']}]\n{r['text']}\n\n"
        
    messages = [
        {"role": "system", "content": "POLICY ASSISTANT. Answer only from the supplied context. If not found, say 'I could not find the information'."},
        {"role": "user", "content": f"CONTEXT:\n{context}QUESTION:\n{question}"}
    ]
    mock_llm.reset()
    res = mock_llm.chat(messages)
    return res.choices[0].message.content, retrieved

eval_df = pd.read_csv('data/raw/policy_eval.csv')

def run_evaluation(overlap=10):
    chk = create_chunks(overlap)
    vec = TfidfVectorizer()
    mat = vec.fit_transform([c['text'] for c in chk])
    
    stats = {'ans_expected_doc': 0, 'ans_expected_fact': 0, 'unans_fallback': 0}
    failed_questions = []
    
    for _, row in eval_df.iterrows():
        q = row['question']
        exp_doc = row['expected_doc']
        exp_fact = row['expected_fact'] if pd.notna(row['expected_fact']) else ""
        
        ans_text, retrieved = answer(q, vec, mat, chk)
        docs = [r['source'] for r in retrieved]
        
        if exp_doc == "NONE":
            if "I could not find" in ans_text:
                stats['unans_fallback'] += 1
        else:
            if exp_doc in docs:
                stats['ans_expected_doc'] += 1
            if exp_fact in ans_text:
                stats['ans_expected_fact'] += 1
            else:
                failed_questions.append((q, exp_fact, ans_text))
                
    return stats, failed_questions

stats_10, _ = run_evaluation(10)
print(f"Overlap 10 Stats - Docs (Answerable): {stats_10['ans_expected_doc']}/12, Facts (Answerable): {stats_10['ans_expected_fact']}/12, Fallback: {stats_10['unans_fallback']}/2")

stats_0, failed_0 = run_evaluation(0)
print(f"Overlap 0 Stats - Facts: {stats_0['ans_expected_fact']}/12")
if failed_0:
    print(f"Failed with overlap 0: '{failed_0[0][0]}'")

# ---------------------------------------------------------
# TASK 5: Tools Definition
# ---------------------------------------------------------
print("\n--- 5. TOOLS DEFINITION ---")
TOOLS_SPEC = [
    {"name": "get_order", "description": "Fetch order details.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}},
    {"name": "search_policy", "description": "Search policies.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "calculate_refund", "description": "Calculate refund amount.", "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}}},
    {"name": "issue_refund", "description": "Issue automatic refund.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}}},
    {"name": "escalate_to_human", "description": "Escalate high refund.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}}}
]

def get_order(order_id):
    conn = sqlite3.connect('data/instamart.db')
    df = pd.read_sql_query(f"SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = '{order_id}'", conn)
    conn.close()
    if df.empty: return '{"error": "order not found"}'
    return df.iloc[0].to_json()

def search_policy(query):
    return json.dumps({"results": retrieve(query, 3)})

def calculate_refund(order_amount, delivery_minutes):
    amt = float(order_amount)
    deliv = float(delivery_minutes) if pd.notna(delivery_minutes) and str(delivery_minutes).replace('.','',1).isdigit() else 0
    refund = amt if deliv > 45 else 0
    return json.dumps({"refund_inr": refund})

def issue_refund(order_id, refund_inr):
    with open('outputs/refund_ledger.jsonl', 'a') as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(), "tool": "issue_refund", "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return json.dumps({"refund_inr": refund_inr})

def escalate_to_human(order_id, refund_inr):
    with open('outputs/refund_ledger.jsonl', 'a') as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(), "tool": "escalate_to_human", "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return json.dumps({"refund_inr": refund_inr})

TOOL_MAP = {
    "get_order": get_order, "search_policy": search_policy, 
    "calculate_refund": calculate_refund, "issue_refund": issue_refund, 
    "escalate_to_human": escalate_to_human
}

# ---------------------------------------------------------
# TASK 6, 7 & 8: ReAct Agent Loop
# ---------------------------------------------------------
print("\n--- 6, 7 & 8. REACT AGENT ---")
with open('outputs/refund_ledger.jsonl', 'w') as f: pass
with open('outputs/agent_runs.jsonl', 'w') as f: pass

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None, crash_after=None, _state_override=None):
    if _state_override:
        messages = _state_override['messages']
        tokens_used = _state_override['tokens_used']
        step_list = _state_override['steps']
        llm_calls = _state_override['llm_calls']
    else:
        messages = [{"role": "system", "content": "AGENT"}, {"role": "user", "content": user_message}]
        tokens_used, llm_calls, step_list = 0, 0, []
    
    if not run_id: run_id = scenario_id
    start_time = time.monotonic()
    mock_llm.reset()
    
    for i in range(max_iterations):
        if time.monotonic() - start_time >= time_limit_seconds:
            return {"status": "FALLBACK_TIMEOUT", "final_answer": "Handing over due to timeout.", "steps": step_list, "llm_calls": llm_calls, "tokens_used": tokens_used}
            
        res = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += res.usage.total_tokens
        
        if tokens_used >= token_limit:
            return {"status": "FALLBACK_TOKEN_LIMIT", "final_answer": "Handing over due to token limit.", "steps": step_list, "llm_calls": llm_calls, "tokens_used": tokens_used}
            
        msg = res.choices[0].message
        if not msg.tool_calls:
            ans = msg.content
            status = ans.split(":")[0] if ":" in ans else "UNKNOWN"
            return {"status": status, "final_answer": ans, "steps": step_list, "llm_calls": llm_calls, "tokens_used": tokens_used}
            
        tc = msg.tool_calls[0]
        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        
        t_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
        observation = TOOL_MAP[tc.name](**t_args)
        
        step_list.append({"tool": tc.name, "arguments": t_args, "observation": observation})
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": observation})
        
        state = {"run_id": run_id, "scenario_id": scenario_id, "messages": messages, "tokens_used": tokens_used, "steps": step_list, "llm_calls": llm_calls}
        with open(f"state/{run_id}.json", "w") as f: json.dump(state, f)
        
        if crash_after and len(step_list) == crash_after:
            raise Exception("Simulated crash!")
            
    return {"status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Handing over, max iterations reached.", "steps": step_list, "llm_calls": llm_calls, "tokens_used": tokens_used}

def resume_agent(run_id):
    with open(f"state/{run_id}.json", "r") as f: state = json.load(f)
    return run_agent("", state['scenario_id'], run_id=run_id, _state_override=state)

scenarios = pd.read_csv('data/raw/agent_scenarios.csv')
scenario_results = []

for _, row in scenarios.iterrows():
    sid = row['scenario_id']
    ans = run_agent(row['message'], sid)
    
    with open('outputs/agent_runs.jsonl', 'a') as f:
        f.write(json.dumps({"scenario_id": sid, "status": ans['status'], "final_answer": ans['final_answer'], "llm_calls": ans['llm_calls'], "tokens_used": ans['tokens_used'], "steps": ans['steps']}) + "\n")
        
    match = re.search(r"INR ([\d.]+)", ans['final_answer'])
    refund_amt = match.group(1) if match else "0.00"
    scenario_results.append((sid, ans['status'], refund_amt, ans['llm_calls'], ans['tokens_used']))

print("\nScenario Results Table:")
print(pd.DataFrame(scenario_results, columns=['scenario_id', 'status', 'refund_amount', 'llm_calls', 'tokens_used']).to_string(index=False))

print("\nTesting Fallbacks...")
s1_msg = scenarios.iloc[0]['message']
fb_token = run_agent(s1_msg, "S1_tok", token_limit=200)
print(f"Token limit fallback status: {fb_token['status']}")
fb_time = run_agent(s1_msg, "S1_time", time_limit_seconds=0)
print(f"Time limit fallback status: {fb_time['status']}")

print("\nTesting Crash and Resume...")
try:
    run_agent(s1_msg, "S1", run_id="S1_crash", crash_after=2)
except Exception as e:
    print(f"Crashed successfully: {e}")

resume_ans = resume_agent("S1_crash")
print(f"Resumed Run Status: {resume_ans['status']}")
print("Tools executed (Crash + Resume combined):", [s['tool'] for s in resume_ans['steps']])