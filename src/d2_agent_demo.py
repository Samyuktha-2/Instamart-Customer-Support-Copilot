import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent
from agent import run_agent, resume_agent, SimulatedCrash, TOOLS_SPEC, TOOL_MAP, ROOT

OUT = os.path.join(ROOT, "outputs")
RUNS = os.path.join(OUT, "agent_runs.jsonl")
DEMO_LEDGER = os.path.join(OUT, "refund_ledger_resume_demo.jsonl")

# clean slate so reruns are reproducible
os.makedirs(OUT, exist_ok=True)
for p in [RUNS, agent.LEDGER, DEMO_LEDGER] + glob.glob(os.path.join(ROOT, "state", "*.json")):
    if os.path.exists(p):
        os.remove(p)


def refund_of(res):
    amt = 0.0
    for s in res["steps"]:
        if s["tool"] == "calculate_refund":
            amt = s["observation"]["refund_inr"]
    return float(amt)


print("===== Task 5: tools =====")
print("tools:", [t["name"] for t in TOOLS_SPEC], "| TOOL_MAP:", sorted(TOOL_MAP))
print("get_order ORD-100048:", agent.get_order("ORD-100048"))
print("get_order ORD-999999:", agent.get_order("ORD-999999"))
print("calculate_refund(300, 50):", agent.calculate_refund(300, 50),
      "| (300, 45):", agent.calculate_refund(300, 45))

sc = pd.read_csv(os.path.join(ROOT, "data/raw/agent_scenarios.csv"), keep_default_na=False)

print("\n===== Task 6: seven scenarios =====")
rows, results = [], {}
with open(RUNS, "a", encoding="utf-8") as f:
    for r in sc.itertuples(index=False):
        res = run_agent(r.message, r.scenario_id)
        results[r.scenario_id] = res
        f.write(json.dumps({"scenario_id": r.scenario_id, "status": res["status"],
                            "final_answer": res["final_answer"], "llm_calls": res["llm_calls"],
                            "tokens_used": res["tokens_used"], "steps": res["steps"]}) + "\n")
        exp = r.expected_outcome
        ok_status = res["status"].startswith("FALLBACK") if exp == "FALLBACK" else res["status"] == exp
        ok_amt = abs(refund_of(res) - float(r.expected_refund_inr)) <= 0.01
        rows.append((r.scenario_id, res["status"], refund_of(res), res["llm_calls"],
                     res["tokens_used"], exp, float(r.expected_refund_inr),
                     "OK" if ok_status and ok_amt else "CHECK"))
print(pd.DataFrame(rows, columns=["scenario", "status", "refund_inr", "model_calls", "tokens",
                                  "expected", "expected_refund", "match"]).to_string(index=False))
s7 = results["S7"]
assert s7["status"] == "FALLBACK_MAX_ITERATIONS" and s7["llm_calls"] == 6
print("S1 model calls:", results["S1"]["llm_calls"], "(ref 5) | S7 tokens:", s7["tokens_used"],
      "(must stay under 6000)")
print("S7 handoff:", s7["final_answer"])

print("\n===== Task 7: stop safely =====")
msg1 = sc.loc[sc["scenario_id"] == "S1", "message"].iloc[0]
d1 = run_agent(msg1, "S1", token_limit=200, run_id="S1_token200")
print(f"S1 token_limit=200 -> {d1['status']} | model calls {d1['llm_calls']} | "
      f"tokens {d1['tokens_used']} | tools executed {d1['tools_executed']}")
d2 = run_agent(msg1, "S1", time_limit_seconds=0, run_id="S1_timeout0")
print(f"S1 time_limit_seconds=0 -> {d2['status']} | model calls {d2['llm_calls']} | "
      f"tools executed {d2['tools_executed']}")
print("handoff text:", d2["final_answer"])

print("\n===== Task 8: crash and resume =====")
main_ledger = agent.LEDGER
agent.LEDGER = DEMO_LEDGER          # keep the main ledger free of a second S1 refund
try:
    run_agent(msg1, "S1", run_id="S1_crash", crash_after_tool_number=2)
except SimulatedCrash as e:
    print("crashed:", e)
with open(os.path.join(ROOT, "state", "S1_crash.json"), encoding="utf-8") as f:
    ck = json.load(f)
print("checkpoint saved: tools so far", ck["tools_executed"], "| step", ck["step"],
      "| tokens", ck["tokens_used"], "| messages", len(ck["messages"]))
res = resume_agent("S1_crash")
agent.LEDGER = main_ledger
print("tools executed across crash + resume:", res["tools_executed"])
print("final:", res["status"], "| refund", refund_of(res), "| total model calls", res["llm_calls"])
assert res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
assert res["status"] == "REFUND" and abs(refund_of(res) - 128.25) <= 0.01
with open(DEMO_LEDGER, encoding="utf-8") as f:
    print("resume-demo ledger lines (must be 1):", sum(1 for _ in f))

print("\nmain ledger:")
with open(agent.LEDGER, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        print("  ", d["tool"], d["order_id"], d["refund_inr"])
print("agent_runs.jsonl lines:", sum(1 for _ in open(RUNS, encoding="utf-8")), "(7 expected)")