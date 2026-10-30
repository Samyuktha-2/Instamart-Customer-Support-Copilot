import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag import ROOT, chunk_docs, get_index, retrieve, answer

print("===== Task 1: chunks =====")
for ov in (10,):
    c = chunk_docs(40, ov)
    print(f"overlap {ov}: {len(c)} chunks (ref 15)", dict(Counter(x["source"] for x in c)))
print("(ref: refund 4, delivery 3, escalation 3, payment 3, privacy 2)")

print("\n===== Task 2: retrieval =====")
for h in retrieve("refund policy late delivery", k=3):
    print(f"{h['source']:24s} score {h['score']:.3f}")
print("(ref: all three from refund_policy.txt)")

ev = pd.read_csv(os.path.join(ROOT, "data/raw/policy_eval.csv"), keep_default_na=False)
print("\npolicy_eval columns:", list(ev.columns), "| rows:", len(ev))
for col in ("question", "expected_doc", "expected_fact"):
    assert col in ev.columns, f"column {col!r} not found; tell me the real names"

NONE = {"", "none", "n/a", "na", "null"}
ans_mask = ~ev["expected_doc"].astype(str).str.strip().str.lower().isin(NONE)
print(f"answerable: {int(ans_mask.sum())} (ref 12) | unanswerable: {int((~ans_mask).sum())} (ref 2)")


def evaluate(size, overlap):
    idx = get_index(size, overlap)
    doc_ok = fact_ok = fb_ok = 0
    failures = []
    for r in ev.itertuples(index=False):
        out = answer(r.question, index=idx)
        if str(r.expected_doc).strip().lower() in NONE:
            if out["answer"].startswith("I could not find"):
                fb_ok += 1
            else:
                failures.append((r.question, "unanswerable but got an answer"))
            continue
        d = r.expected_doc in out["sources"]
        f = r.expected_fact in out["answer"]
        doc_ok += d
        fact_ok += f
        if not (d and f):
            failures.append((r.question,
                             f"doc_in_top3={d}, fact_in_answer={f}, expected_fact={r.expected_fact!r}, "
                             f"got={out['answer'][:90]!r}"))
    n_a, n_u = int(ans_mask.sum()), int((~ans_mask).sum())
    print(f"overlap {overlap}: expected_doc in top-3 {doc_ok}/{n_a} | "
          f"expected_fact in answer {fact_ok}/{n_a} | fallback {fb_ok}/{n_u}")
    for q, why in failures:
        print("   FAIL:", q, "->", why)
    return failures


print("\n===== Task 4: evaluation =====")
evaluate(40, 10)
print("(ref 12/12, 12/12, 2/2)")
print(f"chunks with overlap 0: {len(chunk_docs(40, 0))}")
evaluate(40, 0)
print("(ref: at least one answerable question fails)")