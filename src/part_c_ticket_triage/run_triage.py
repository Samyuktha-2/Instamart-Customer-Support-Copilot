
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm
from prompts import ALLOWED_CATEGORIES, ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05

PII_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"\b[6-9]\d{9}\b"),
    re.compile(r"ORD-\d{6}"),
]


def validate_response(content):
    """Return (status, category); never guess an invalid answer."""
    try:
        parsed = json.loads(content)
        if (
            isinstance(parsed, dict)
            and parsed.get("category") in ALLOWED_CATEGORIES
            and isinstance(parsed.get("reason"), str)
        ):
            return "classified_json", parsed["category"]

        # Valid JSON with an invalid category is flagged, not keyword-parsed.
        return "flagged_invalid_output", ""

    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", str(content).lower())
        found = [word for word in words if word in ALLOWED_CATEGORIES]

        if len(found) == 1:
            return "classified_keyword", found[0]

        return "flagged_invalid_output", ""


def call_with_retries(messages, ticket_id):
    """Retry transient failures up to three times after the initial call."""
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                raise

            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )
            print(
                f"RETRY ticket_id={ticket_id} "
                f"retry={retry_number} wait={wait:.4f}s"
            )
            time.sleep(wait)


def run_pipeline(prompt_style="zero", token_budget=None):
    tickets = pd.read_csv(
        PROJECT_ROOT / "outputs" / "tickets_prepared.csv",
        dtype=str,
        keep_default_na=False,
    )
  
    segments = pd.read_csv(
        PROJECT_ROOT / "outputs" / "customer_segments.csv",
        dtype={"customer_id": str},
    )

    segment_lookup = segments.set_index("customer_id")[
        ["segment_name", "is_premium"]
    ].to_dict("index")

    system_prompt = (
        FEW_SHOT_PROMPT if prompt_style == "few" else ZERO_SHOT_PROMPT
    )

    # Each independent full run starts from a clean mock state.
    mock_llm.reset()

    results = []
    running_tokens = 0
    retry_count = 0
    calls_made = 0

    for _, ticket in tickets.iterrows():
        ticket_id = ticket["ticket_id"]
        original_text = ticket["text"]
        redacted_text = ticket["redacted_text"]
        status = ticket["status"]
        category = ""
        tokens_used = 0

        if status in ("skipped_empty", "flagged_long"):
            results.append({
                "ticket_id": ticket_id,
                "customer_id": ticket["customer_id"],
                "status": status,
                "category": category,
                "true_category": ticket["true_category"],
                "tokens_used": tokens_used,
            })
            continue

        if token_budget is not None and running_tokens >= token_budget:
            status = "budget_exhausted"
            results.append({
                "ticket_id": ticket_id,
                "customer_id": ticket["customer_id"],
                "status": status,
                "category": category,
                "true_category": ticket["true_category"],
                "tokens_used": tokens_used,
            })
            continue

        # Assertions prove that the model receives only redacted ticket text.
        assert redacted_text
        for pattern in PII_PATTERNS:
            assert pattern.search(redacted_text) is None, (
                f"PII remains in ticket {ticket_id}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text},
        ]
        assert messages[-1]["content"] == redacted_text
        assert messages[-1]["content"] == ticket["redacted_text"]

        # Empty/long tickets were skipped before reaching this point.
        assert mock_llm.est_tokens(original_text) <= MAX_INPUT_TOKENS

        try:
            response = call_with_retries(messages, ticket_id)
            calls_made += 1
        except mock_llm.TransientLLMError:
            # Defensive fallback: the supplied mock should recover on retry.
            status = "flagged_invalid_output"
            results.append({
                "ticket_id": ticket_id,
                "customer_id": ticket["customer_id"],
                "status": status,
                "category": category,
                "true_category": ticket["true_category"],
                "tokens_used": tokens_used,
            })
            continue

        tokens_used = response.usage.total_tokens
        running_tokens += tokens_used

        content = response.choices[0].message.content
        status, category = validate_response(content)

        results.append({
            "ticket_id": ticket_id,
            "customer_id": ticket["customer_id"],
            "status": status,
            "category": category,
            "true_category": ticket["true_category"],
            "tokens_used": tokens_used,
        })

    
    result_df = pd.DataFrame(results)
    assert len(result_df) == len(tickets) == 164

    # Attach Part B segment information and apply the required routing rules.
    segment_names = []
    priorities = []
    routes = []

    for _, row in result_df.iterrows():
        customer_id = row["customer_id"]
        status = row["status"]
        category = row["category"]

        segment = segment_lookup.get(customer_id, {})
        segment_name = segment.get("segment_name", "")
        is_premium = str(segment.get("is_premium", False)).strip().lower() == "true"

        if status in {
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted",
        }:
            route = "human_review"
            priority = "NORMAL"
        elif category == "fraud":
            route = "security_team"
            priority = "HIGH"
        elif is_premium:
            route = "premium_desk"
            priority = "HIGH"
        else:
            route = "support_queue"
            priority = "NORMAL"

        segment_names.append(segment_name)
        priorities.append(priority)
        routes.append(route)

    result_df["segment_name"] = segment_names
    result_df["priority"] = priorities
    result_df["route"] = routes

    # Preserve the required final column order.
    result_df = result_df[
        [
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
            "true_category",
        ]
    ]

    classified = result_df[
        result_df["status"].isin(["classified_json", "classified_keyword"])
    ]
    accuracy = (
        (classified["category"] == classified["true_category"]).mean()
        if not classified.empty else 0.0
    )

    print(f"\nPrompt style: {prompt_style}")
    print(f"Tickets processed: {len(result_df)}")
    print(f"Model calls succeeded: {calls_made}")
    print(f"Tokens used: {running_tokens}")
    print(f"Classified tickets: {len(classified)}")
    print(f"Accuracy on classified tickets: {accuracy:.1%}")
    print("\nStatus counts:")
    print(result_df["status"].value_counts().to_string())
    print("\nRouting counts:")
    print(result_df["route"].value_counts().to_string())

    return result_df, running_tokens



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        choices=["zero", "few"],
        default="zero",
        help="Select zero-shot or few-shot classification prompt.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Optional token budget; no call starts once usage reaches it.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path relative to the project root.",
    )
    args = parser.parse_args()

    results, _ = run_pipeline(args.prompt, args.budget)
    
    if args.output:
        output_path = PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_df = results.copy()

        # Keep ground-truth labels out of the final deliverable.
        if output_path.name == "triage_results.csv":
            output_df = output_df.drop(columns=["true_category"], errors="ignore")

        output_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path.relative_to(PROJECT_ROOT)}")

    


if __name__ == "__main__":
    main()
