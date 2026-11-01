"""Part D: policy RAG assistant and ReAct refund agent (all model calls go through mock_llm)."""
import json
import os
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

os.makedirs("outputs", exist_ok=True)
os.makedirs("state", exist_ok=True)

# ------------------------------------------------------------------ 1-2. CHUNK + RETRIEVE
print("--- 1 & 2. CHUNKING AND RETRIEVAL ---")


def create_chunks(chunk_words=40, overlap=10):
    chunks, step = [], chunk_words - overlap
    for fn in sorted(f for f in os.listdir("data/policies") if f.endswith(".txt")):
        with open(os.path.join("data/policies", fn), encoding="utf-8") as f:
            words = f.read().split()
        for start in range(0, len(words), step):
            chunks.append({"source": fn, "text": " ".join(words[start:start + chunk_words])})
            if start + chunk_words >= len(words):
                break
    return chunks


class Index:
    def __init__(self, overlap):
        self.chunks = create_chunks(40, overlap)
        self.vec = TfidfVectorizer()
        self.mat = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        top = np.argsort(-sims, kind="stable")[:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]} for i in top]


INDEX = Index(10)
print(f"total chunks: {len(INDEX.chunks)}; per file: {dict(Counter(c['source'] for c in INDEX.chunks))}")


def retrieve(query, k=3):
    return INDEX.retrieve(query, k)


top3 = retrieve("refund policy late delivery")
print("top-3 sources for 'refund policy late delivery':", [r["source"] for r in top3])
assert all(r["source"] == "refund_policy.txt" for r in top3)


# ------------------------------------------------------------------ 3. AUGMENT + ANSWER
def answer(question, index=None):
    index = index or INDEX
    hits = index.retrieve(question, 3)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    messages = [
        {"role": "system", "content": "POLICY ASSISTANT. Answer only from the supplied context. "
                                      "If the answer is not there, say you could not find the information."},
        {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    mock_llm.reset()
    return mock_llm.chat(messages).choices[0].message.content, [h["source"] for h in hits]


# ------------------------------------------------------------------ 4. EVALUATION
print("\n--- 3 & 4. GROUNDING EVALUATION ---")
eval_df = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)


def evaluate(overlap):
    index = Index(overlap)
    doc_hit = fact_hit = fallback = 0
    n_ans = int((eval_df["expected_doc"] != "NONE").sum())
    failed = []
    for _, r in eval_df.iterrows():
        text, sources = answer(r["question"], index)
        if r["expected_doc"] == "NONE":
            fallback += text.startswith("I could not find")
            continue
        doc_hit += r["expected_doc"] in sources
        if r["expected_fact"] in text:
            fact_hit += 1
        else:
            failed.append((r["question"], r["expected_fact"], sources))
    return {"docs": doc_hit, "facts": fact_hit, "fallback": fallback, "n_ans": n_ans}, failed, index


e10, _, _ = evaluate(10)
e0, failed0, idx0 = evaluate(0)
print("| setting | expected doc in top 3 | answer has expected fact | fallback on unanswerable |")
print(f"| overlap 10 | {e10['docs']}/12 | {e10['facts']}/12 | {e10['fallback']}/2 |")
print(f"| overlap 0  | {e0['docs']}/12 | {e0['facts']}/12 | {e0['fallback']}/2 |")
assert len(failed0) >= 1
for q, fact, srcs in failed0:
    print(f"overlap-0 failure: {q!r} (expected {fact!r}); top-3 sources {srcs}")
    holders = [i for i, c in enumerate(idx0.chunks) if fact in c["text"]]
    print(f"   chunks containing the fact verbatim: {holders if holders else 'none (fact split across a chunk boundary)'}")

# ------------------------------------------------------------------ 5. TOOLS
LEDGER = {"path": "outputs/refund_ledger.jsonl"}
DB = "data/instamart.db"

TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order by id in the Instamart database.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the best matching chunks.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund in INR from the order amount and delivery time.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an automatic refund and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Escalate a refund above INR 1000 to a human support lead.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]


def get_order(order_id):
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                       (order_id,)).fetchone()
    conn.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": retrieve(query, 3)}


def calculate_refund(order_amount, delivery_minutes):
    late = delivery_minutes is not None and float(delivery_minutes) > 45
    return {"refund_inr": round(float(order_amount), 2) if late else 0.0}


def _ledger(tool, order_id, refund_inr):
    with open(LEDGER["path"], "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ------------------------------------------------------------------ 6-8. REACT LOOP
def _fallback(status, reason, steps, llm_calls, tokens_used):
    done = ", ".join(s["tool"] for s in steps) or "no tools"
    summary = (f"HANDOFF to a human agent: stopped because of {reason} after {llm_calls} model calls "
               f"and {len(steps)} tool steps ({done}). Please review the case manually.")
    return {"status": status, "final_answer": summary, "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}


def _checkpoint(run_id, scenario_id, messages, tokens_used, steps, llm_calls):
    state = {"run_id": run_id, "scenario_id": scenario_id, "messages": messages, "tokens_used": tokens_used,
             "llm_calls": llm_calls, "step": len(steps), "tools_executed": [s["tool"] for s in steps], "steps": steps}
    with open(f"state/{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(state, f, default=str)


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after=None, _state=None):
    run_id = run_id or scenario_id
    if _state:
        messages, tokens_used = _state["messages"], _state["tokens_used"]
        steps, llm_calls = _state["steps"], _state["llm_calls"]
    else:
        messages = [{"role": "system", "content": "AGENT: decide refund requests by calling tools."},
                    {"role": "user", "content": user_message}]
        tokens_used, llm_calls, steps = 0, 0, []
    start = time.monotonic()
    mock_llm.reset()

    while True:
        if time.monotonic() - start >= time_limit_seconds:
            return _fallback("FALLBACK_TIMEOUT", "the time limit", steps, llm_calls, tokens_used)
        if llm_calls >= max_iterations:
            return _fallback("FALLBACK_MAX_ITERATIONS", "the maximum number of iterations", steps, llm_calls, tokens_used)

        res = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += res.usage.total_tokens
        if tokens_used > token_limit:            # stop before executing the tool that was just requested
            return _fallback("FALLBACK_TOKEN_LIMIT", "the token limit", steps, llm_calls, tokens_used)

        msg = res.choices[0].message
        if not msg.tool_calls:
            answer_text = msg.content
            status = answer_text.split(":")[0] if ":" in answer_text else "UNKNOWN"
            return {"status": status, "final_answer": answer_text, "steps": steps,
                    "llm_calls": llm_calls, "tokens_used": tokens_used}

        tc = msg.tool_calls[0]
        args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": args}]})
        observation = TOOL_MAP[tc.name](**args)                      # tools return dicts
        steps.append({"tool": tc.name, "arguments": args, "observation": observation})
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                         "content": json.dumps(observation, default=str)})   # JSON string only in the message

        _checkpoint(run_id, scenario_id, messages, tokens_used, steps, llm_calls)
        if crash_after and len(steps) == crash_after:
            raise RuntimeError("Simulated crash!")


def resume_agent(run_id, **kwargs):
    with open(f"state/{run_id}.json", encoding="utf-8") as f:
        state = json.load(f)
    return run_agent("", state["scenario_id"], run_id=run_id, _state=state, **kwargs)


def refund_from(result):
    """Refund amount taken from the issue_refund / escalate_to_human observation (not from free text)."""
    for s in reversed(result["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return float(s["observation"]["refund_inr"])
    return 0.0


print("\n--- 6, 7 & 8. REACT AGENT ---")
open("outputs/refund_ledger.jsonl", "w").close()
open("outputs/agent_runs.jsonl", "w").close()
scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
rows = []
for _, sc in scenarios.iterrows():
    out = run_agent(sc["message"], sc["scenario_id"])
    refund = refund_from(out)
    with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"scenario_id": sc["scenario_id"], "status": out["status"], "final_answer": out["final_answer"],
                            "llm_calls": out["llm_calls"], "tokens_used": out["tokens_used"], "steps": out["steps"]},
                           default=str) + "\n")
    rows.append((sc["scenario_id"], out["status"], f"{refund:.2f}", out["llm_calls"], out["tokens_used"]))
    expected_status = sc["expected_outcome"]
    ok = out["status"].startswith(expected_status) and abs(refund - float(sc["expected_refund_inr"])) < 0.01
    assert ok, f"{sc['scenario_id']} does not match the expected outcome"

print("| scenario_id | status | refund_inr | llm_calls | tokens_used |")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")

# demos write to a separate ledger so the real ledger holds only the S1-S7 actions
LEDGER["path"] = "outputs/refund_ledger_demo.jsonl"
open(LEDGER["path"], "w").close()
s1 = scenarios.iloc[0]["message"]

print("\n--- STOP-SAFELY DEMOS ---")
tok = run_agent(s1, "S1_tok", token_limit=200)
print(f"S1 with token_limit=200 -> {tok['status']} (tokens {tok['tokens_used']}, steps {len(tok['steps'])})")
print(f"   hand-off summary: {tok['final_answer']}")
tim = run_agent(s1, "S1_time", time_limit_seconds=0)
print(f"S1 with time_limit_seconds=0 -> {tim['status']} (llm calls {tim['llm_calls']})")

print("\n--- CRASH AND RESUME ---")
try:
    run_agent(s1, "S1", run_id="S1_crash", crash_after=2)
except RuntimeError as e:
    print(f"crashed as planned: {e}")
resumed = resume_agent("S1_crash")
tools_run = [s["tool"] for s in resumed["steps"]]
print(f"resumed status: {resumed['status']}; refund {refund_from(resumed):.2f}")
print("tools executed across crash and resume:", tools_run)
assert tools_run == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
assert resumed["status"] == "REFUND" and abs(refund_from(resumed) - 128.25) < 0.01
print("\nPART D COMPLETE")
