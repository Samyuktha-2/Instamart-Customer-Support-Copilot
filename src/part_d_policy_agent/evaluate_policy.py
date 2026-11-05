
import csv
from pathlib import Path

from policy_rag import build_index, retrieve
from policy_assistant import answer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_FILE = PROJECT_ROOT / "data" / "raw" / "policy_eval.csv"


def evaluate(overlap):
    index = build_index(overlap=overlap)

    with EVAL_FILE.open("r", encoding="utf-8", newline="") as file:
        questions = list(csv.DictReader(file))

    answerable_total = 0
    doc_hits = 0
    fact_hits = 0
    unanswerable_total = 0
    fallback_hits = 0

    failures = []

    for row in questions:
        question = row["question"]
        expected_doc = row["expected_doc"].strip()
        expected_fact = (row["expected_fact"] or "").strip()

        result = answer(question, index=index)
        response = result["answer"]

        if expected_doc and expected_fact:
            answerable_total += 1

            doc_found = expected_doc in result["sources"]
            fact_found = expected_fact in response

            if doc_found:
                doc_hits += 1

            if fact_found:
                fact_hits += 1

            if not doc_found or not fact_found:
                failures.append({
                    "question": question,
                    "answer": response,
                    "expected_doc": expected_doc,
                    "expected_fact": expected_fact,
                    "sources": result["sources"]
                })

        else:
            unanswerable_total += 1

            if response.startswith("I could not find"):
                fallback_hits += 1
            else:
                failures.append({
                    "question": question,
                    "answer": response,
                    "expected": "I could not find..."
                })

    print(f"\nOverlap: {overlap}")
    print(f"Document recall: {doc_hits}/{answerable_total}")
    print(f"Expected fact accuracy: {fact_hits}/{answerable_total}")
    print(f"Unsupported fallback: {fallback_hits}/{unanswerable_total}")

    if failures:
        print("\nFailed questions:")
        for failure in failures:
            print(failure)


if __name__ == "__main__":
    evaluate(overlap=10)
    evaluate(overlap=0)
