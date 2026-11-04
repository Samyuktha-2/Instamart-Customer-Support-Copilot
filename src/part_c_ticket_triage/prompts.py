
ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]


ZERO_SHOT_PROMPT = """
CLASSIFY the customer's support ticket into exactly one category.

Allowed categories:
- billing
- delivery
- technical
- fraud

Return a JSON-only reply in exactly this form:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
The reason must briefly explain the classification.
Do not include markdown, code fences, or any text outside the JSON object.
""".strip()


FEW_SHOT_PROMPT = """
CLASSIFY the customer's support ticket into exactly one category.

Allowed categories:
- billing
- delivery
- technical
- fraud

Return a JSON-only reply in exactly this form:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
The reason must briefly explain the classification.
Do not include markdown, code fences, or any text outside the JSON object.

EXAMPLES:
Ticket: "My invoice shows the wrong amount."
{"category": "billing", "reason": "The customer reports an incorrect invoice amount."}

Ticket: "My order has not arrived yet."
{"category": "delivery", "reason": "The customer reports a delayed delivery."}

Ticket: "The app crashes whenever I open it."
{"category": "technical", "reason": "The customer reports an application error."}

Ticket: "There is an unauthorized transaction on my card."
{"category": "fraud", "reason": "The customer reports a potentially unauthorized transaction."}
""".strip()


def validate_prompts():
    for name, prompt in [
        ("zero-shot", ZERO_SHOT_PROMPT),
        ("few-shot", FEW_SHOT_PROMPT),
    ]:
        assert "CLASSIFY" in prompt
        assert all(category in prompt for category in ALLOWED_CATEGORIES)
        assert '{"category": "...", "reason": "..."}' in prompt
        print(f"{name} prompt: basic checks passed")

    assert "EXAMPLES:" not in ZERO_SHOT_PROMPT
    assert "EXAMPLES:" in FEW_SHOT_PROMPT
    assert '"category": "fraud"' in FEW_SHOT_PROMPT
    print("Prompt requirements: passed")


if __name__ == "__main__":
    validate_prompts()
