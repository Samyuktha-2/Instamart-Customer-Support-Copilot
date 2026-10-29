import json
import os
import random
import re
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mock_llm
from mock_llm import est_tokens, TransientLLMError

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
CATS = ["billing", "delivery", "technical", "fraud"]
HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output",
         "budget_exhausted", "failed_permanent"}

# ---------------- Task 2: redaction (this order) ----------------
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]


def redact(text, counts=None):
    for key, pat, placeholder in PATTERNS:
        text, n = pat.subn(placeholder, text)
        if counts is not None:
            counts[key] += n
    return text


def assert_clean(messages, redacted):
    assert messages[-1]["content"] == redacted, "user message must be exactly the redacted text"
    for m in messages:
        for _, pat, _ in PATTERNS:
            assert not pat.search(m["content"]), "PII found in a prompt"


# ---------------- Task 3: prompts ----------------
ZERO_SHOT = (
    "You CLASSIFY customer support tickets for a grocery delivery app. "
    "Allowed categories: billing, delivery, technical, fraud. "
    'Reply with JSON only, exactly in the form {"category": "...", "reason": "..."}. '
    "Do not add any other text."
)
FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: I was charged twice for one order and want my money back.
{"category": "billing", "reason": "double charge"}
Ticket: Someone made an unauthorized purchase from my account.
{"category": "fraud", "reason": "unauthorized transaction"}
Ticket: The app crashes every time I try to log in.
{"category": "technical", "reason": "app crash"}
Ticket: My groceries arrived two hours late.
{"category": "delivery", "reason": "late arrival"}
"""
for p in (ZERO_SHOT, FEW_SHOT):
    assert "CLASSIFY" in p and all(c in p for c in CATS) and '"category"' in p
assert "EXAMPLES:" not in ZERO_SHOT
assert "\nEXAMPLES:\n" in FEW_SHOT and '"fraud"' in FEW_SHOT


# ---------------- Task 4: retries ----------------
def call_with_retry(messages, ticket_id, stats):
    retry = 0
    while True:
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            stats["transient"] += 1
            if retry >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** retry * random.uniform(0.9, 1.15)
            assert BASE_DELAY * 2 ** retry * 0.9 <= wait <= BASE_DELAY * 2 ** retry * 1.15
            stats["waits"].append((retry, wait))
            print(f"  [retry] ticket={ticket_id} retry={retry} wait={wait:.4f}s")
            time.sleep(wait)
            retry += 1


# ---------------- Task 5: validation ----------------
def validate(content):
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (content or "").lower()))
        found = [c for c in CATS if c in words]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", ""
    if isinstance(obj, dict) and isinstance(obj.get("category"), str) and obj["category"] in CATS:
        return "classified_json", obj["category"]
    return "flagged_invalid_output", ""


# ---------------- pipeline ----------------
def run_pipeline(tickets, system_prompt, label, token_budget=None):
    mock_llm.reset()                      # immediately before each full run
    random.seed(42)
    stats = {"transient": 0, "waits": [], "sent": 0}
    total_tokens = 0
    rows = []
    print(f"\n--- run: {label} (token budget: {token_budget}) ---")
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id,
               "true_category": t.true_category, "category": "", "tokens_used": 0}
        if t.text.strip() == "":
            row["status"] = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            red = redact(t.text)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": red}]
            assert_clean(messages, red)
            stats["sent"] += 1
            try:
                resp = call_with_retry(messages, t.ticket_id, stats)
            except TransientLLMError:
                row["status"] = "failed_permanent"
            else:
                tok = resp.usage.total_tokens
                total_tokens += tok
                row["tokens_used"] = tok
                row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df, stats, total_tokens


def summarise(label, df, stats, total_tokens):
    cl = df[df["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((cl["category"] == cl["true_category"]).sum())
    print(f"[{label}] statuses: {df['status'].value_counts().to_dict()}")
    print(f"[{label}] model calls (tickets sent): {stats['sent']}, TransientLLMErrors: "
          f"{stats['transient']}, retries logged: {len(stats['waits'])}, "
          f"permanent errors: {int((df['status'] == 'failed_permanent').sum())}")
    if stats["waits"]:
        w = [x[1] for x in stats["waits"]]
        print(f"[{label}] retry waits: min {min(w):.4f}s max {max(w):.4f}s "
              f"(retry numbers used: {sorted({x[0] for x in stats['waits']})})")
    if len(cl):
        print(f"[{label}] classified {len(cl)} | accuracy {correct / len(cl) * 100:.1f}% "
              f"({correct}/{len(cl)}) | total tokens {total_tokens} | "
              f"tokens per classified ticket {total_tokens / len(cl):.1f}")
    return cl


# ---------------- routing ----------------
def add_routing(df, seg):
    name = seg.set_index("customer_id")["segment_name"].to_dict()
    prem = {c: str(v).lower() == "true" for c, v in zip(seg["customer_id"], seg["is_premium"])}
    out = df.copy()
    out["segment_name"] = out["customer_id"].map(name).fillna("")
    routes, prios = [], []
    for r in out.itertuples(index=False):
        if r.status in HUMAN:
            routes.append("human_review"); prios.append("NORMAL")
        elif r.category == "fraud":
            routes.append("security_team"); prios.append("HIGH")
        elif prem.get(r.customer_id, False):
            routes.append("premium_desk"); prios.append("HIGH")
        else:
            routes.append("support_queue"); prios.append("NORMAL")
    out["route"], out["priority"] = routes, prios
    return out[["ticket_id", "customer_id", "status", "category", "segment_name",
                "priority", "route", "tokens_used"]]


# =============== main ===============
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
tickets = pd.read_csv(os.path.join(ROOT, "data/raw/tickets.csv"), keep_default_na=False)
seg = pd.read_csv(os.path.join(ROOT, "outputs/customer_segments.csv"))
print("tickets:", len(tickets))

print("\n===== Task 1: input guardrails =====")
empty = tickets["text"].str.strip() == ""
tok = tickets["text"].map(est_tokens)
long_ = (~empty) & (tok > MAX_INPUT_TOKENS)
print(f"empty: {int(empty.sum())} (ref 4) | long: {int(long_.sum())} (ref 4) | "
      f"long est_tokens: {sorted(set(tok[long_]))} (ref 734)")

print("\n===== Task 2: redaction =====")
counts = {"email": 0, "phone": 0, "order_id": 0}
redacted = [redact(t, counts) for t in tickets["text"]]
print(f"emails {counts['email']} (ref 55) | phones {counts['phone']} (ref 91) | "
      f"order ids {counts['order_id']} (ref 132) | total {sum(counts.values())} (ref 278)")
left = sum(len(p.findall(t)) for t in redacted for _, p, _ in PATTERNS)
print("matches left after re-applying the three patterns:", left, "(ref 0)")
assert left == 0

print("\n===== Task 3: prompts =====")
print(f"zero-shot prompt: {est_tokens(ZERO_SHOT)} est. tokens | "
      f"few-shot prompt: {est_tokens(FEW_SHOT)} est. tokens")

print("\n===== Tasks 4-6: full runs, no budget =====")
zdf, zst, ztok = run_pipeline(tickets, ZERO_SHOT, "zero-shot")
zcl = summarise("zero-shot", zdf, zst, ztok)
fdf, fst, ftok = run_pipeline(tickets, FEW_SHOT, "few-shot")
fcl = summarise("few-shot", fdf, fst, ftok)
print("(ref: 135 classified_json / 12 classified_keyword / 9 flagged_invalid_output; "
      "147 classified; zero-shot 85.7% (126/147); few-shot 100%; 19 transient errors)")

print("\nZero-shot confusion table (rows = true, columns = predicted):")
print(pd.crosstab(zcl["true_category"], zcl["category"],
                  rownames=["true"], colnames=["predicted"]).to_string())

print("\n===== Task 7: budget demo and final routing =====")
DEMO_BUDGET = 6000
bdf, bst, btok = run_pipeline(tickets, ZERO_SHOT, "zero-shot budget demo", DEMO_BUDGET)
summarise("budget demo", bdf, bst, btok)
n_ex = int((bdf["status"] == "budget_exhausted").sum())
print(f"budget used: {DEMO_BUDGET} | budget_exhausted tickets: {n_ex} (must be >= 1)")
assert n_ex >= 1
add_routing(bdf, seg).to_csv(os.path.join(ROOT, "outputs/triage_budget_demo.csv"), index=False)

rdf, rst, rtok = run_pipeline(tickets, FEW_SHOT, "few-shot final", 100000)
summarise("final", rdf, rst, rtok)
final = add_routing(rdf, seg)
final.to_csv(os.path.join(ROOT, "outputs/triage_results.csv"), index=False)
print("\ntriage_results.csv rows:", len(final), "(ref 164)")
print("routes:", final["route"].value_counts().to_dict())
print("priorities:", final["priority"].value_counts().to_dict())
print("customers missing from segment file:",
      int((~tickets["customer_id"].isin(seg["customer_id"])).sum()))