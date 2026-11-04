
import re
import sys
from pathlib import Path

import pandas as pd

# Allow importing the supplied mock_llm.py from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm


INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "tickets.csv"
MAX_INPUT_TOKENS = 400

# Apply these patterns in exactly the required order.
PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]


def redact_text(text):
    """Redact PII in the required order and return counts by type."""
    redacted = str(text)
    counts = {}

    for name, pattern, placeholder in PII_PATTERNS:
        redacted, count = pattern.subn(placeholder, redacted)
        counts[name] = count

    return redacted, counts


def main():
    tickets = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)

    expected_columns = {
        "ticket_id",
        "customer_id",
        "order_id",
        "text",
        "true_category",
    }
    assert expected_columns.issubset(tickets.columns), (
        f"Missing columns: {expected_columns - set(tickets.columns)}"
    )
    assert len(tickets) == 164, f"Expected 164 tickets, found {len(tickets)}"

    tickets["redacted_text"] = ""
    tickets["status"] = ""
    tickets["estimated_input_tokens"] = 0

    total_counts = {"email": 0, "phone": 0, "order_id": 0}

    for index, row in tickets.iterrows():
        original_text = row["text"]

        # Input guardrails run before any model call.
        if not original_text.strip():
            tickets.at[index, "status"] = "skipped_empty"
            continue

        estimated_tokens = mock_llm.est_tokens(original_text)
        tickets.at[index, "estimated_input_tokens"] = estimated_tokens

        if estimated_tokens > MAX_INPUT_TOKENS:
            tickets.at[index, "status"] = "flagged_long"
            continue

        redacted_text, counts = redact_text(original_text)
        tickets.at[index, "redacted_text"] = redacted_text

        for name, count in counts.items():
            total_counts[name] += count

        # Prove that the three PII patterns find no matches after redaction.
        for name, pattern, _placeholder in PII_PATTERNS:
            assert pattern.search(redacted_text) is None, (
                f"Unredacted {name} detected in ticket {row['ticket_id']}"
            )

    total_counts["total"] = sum(
        total_counts[name] for name in ("email", "phone", "order_id")
    )

    # Confirm the expected reference counts.
    assert (tickets["status"] == "skipped_empty").sum() == 4
    assert (tickets["status"] == "flagged_long").sum() == 4
    assert total_counts == {
        "email": 55,
        "phone": 91,
        "order_id": 132,
        "total": 278,
    }, f"Unexpected redaction counts: {total_counts}"

    # No model calls are made in this preparation script.
    print(f"Tickets loaded: {len(tickets)}")
    print("Empty tickets:", (tickets["status"] == "skipped_empty").sum())
    print("Long tickets:", (tickets["status"] == "flagged_long").sum())
    print("Eligible tickets:", (tickets["redacted_text"] != "").sum())
    print("Redaction counts:", total_counts)
    print("PII verification: passed (zero matches remain)")
    print("Model calls made: 0")

    # Keep this output for the next implementation steps.
    output_file = PROJECT_ROOT / "outputs" / "tickets_prepared.csv"
    tickets.to_csv(output_file, index=False)
    print(f"Saved: {output_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
