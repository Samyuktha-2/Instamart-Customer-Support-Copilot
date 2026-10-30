import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
for _p in (SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import mock_llm
from rag import retrieve          # Part D retrieval is reused by the agent

DB_PATH = os.path.join(ROOT, "data", "instamart.db")
LEDGER = os.path.join(ROOT, "outputs", "refund_ledger.jsonl")
STATE_DIR = os.path.join(ROOT, "state")

AGENT_SYSTEM = (
    "You are a refund AGENT for a grocery delivery company. "
    "Use the tools get_order, search_policy, calculate_refund, issue_refund and "
    "escalate_to_human to decide each refund request. When finished, reply with one "
    "line that starts with REFUND:, ESCALATE:, DENIED: or NOT_FOUND:."
)


class SimulatedCrash(Exception):
    pass


# ---------------- tools ----------------
def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    line = {"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool,
            "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def get_order(order_id):
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes "
                          "FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    finally:
        con.close()
    if row is None:
        return {"error": "order not found"}
    oid, status, amount, mins = row
    if mins is not None and mins != mins:      # NaN -> None
        mins = None
    return {"order_id": oid, "status": status,
            "order_amount": None if amount is None else float(amount),
            "delivery_minutes": None if mins is None else float(mins)}


def search_policy(query):
    hits = retrieve(query, k=3)
    for h in hits:
        h["score"] = round(h["score"], 4)
    return {"results": hits}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}


def issue_refund(order_id, refund_inr):
    _ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}


def escalate_to_human(order_id, refund_inr):
    _ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}


def _spec(name, desc, props):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}


TOOLS_SPEC = [
    _spec("get_order", "Look up an order's status, amount and delivery time in the database.",
          {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the policy documents and return the best matching chunks.",
          {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Escalate a refund above INR 1000 to a human lead and record it.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy,
            "calculate_refund": calculate_refund, "issue_refund": issue_refund,
            "escalate_to_human": escalate_to_human}


# ---------------- checkpoint / fallback ----------------
def _checkpoint(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    keys = ("run_id", "scenario_id", "user_message", "messages", "tokens_used",
            "llm_calls", "step", "tools_executed", "steps", "limits")
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump({k: state[k] for k in keys}, f, indent=1)


def _result(state, status, final_answer):
    return {"run_id": state["run_id"], "status": status, "final_answer": final_answer,
            "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "tools_executed": list(state["tools_executed"])}


def _fallback(state, status):
    tools = ", ".join(state["tools_executed"]) or "none"
    summary = (f"HANDOFF: stopped with {status} after {state['llm_calls']} model calls, "
               f"{len(state['steps'])} tool steps ({tools}). "
               f"Request: {state['user_message']} A human should review this case.")
    return _result(state, status, summary)


# ---------------- the ReAct loop ----------------
def _loop(state, crash_after_tool_number=None):
    lim = state["limits"]
    start = time.monotonic()
    while state["llm_calls"] < lim["max_iterations"]:
        if time.monotonic() - start >= lim["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT")
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > lim["token_limit"]:
            return _fallback(state, "FALLBACK_TOKEN_LIMIT")   # tool not executed
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content
            return _result(state, final.split(":")[0].strip(), final)

        calls = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls]
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
        for tc in calls:
            fn = TOOL_MAP.get(tc["name"])
            try:
                obs = fn(**tc["arguments"]) if fn else {"error": f"unknown tool {tc['name']}"}
            except TypeError as e:
                obs = {"error": f"bad arguments: {e}"}
            state["messages"].append({"role": "tool", "tool_call_id": tc["id"],
                                      "name": tc["name"], "content": json.dumps(obs)})
            state["steps"].append({"tool": tc["name"], "arguments": tc["arguments"],
                                   "observation": obs})
            state["tools_executed"].append(tc["name"])
            state["step"] = len(state["steps"])
            _checkpoint(state)                               # after every observation
            if crash_after_tool_number and len(state["tools_executed"]) == crash_after_tool_number:
                raise SimulatedCrash(f"simulated crash after tool #{crash_after_tool_number}")
    return _fallback(state, "FALLBACK_MAX_ITERATIONS")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tool_number=None):
    state = {
        "run_id": run_id or scenario_id, "scenario_id": scenario_id,
        "user_message": user_message,
        "messages": [{"role": "system", "content": AGENT_SYSTEM},
                     {"role": "user", "content": user_message}],
        "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": [],
        "limits": {"max_iterations": max_iterations, "token_limit": token_limit,
                   "time_limit_seconds": time_limit_seconds},
    }
    return _loop(state, crash_after_tool_number)


def resume_agent(run_id):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)
    return _loop(state)