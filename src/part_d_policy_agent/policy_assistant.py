
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm

from policy_rag import build_index, retrieve


SYSTEM_PROMPT = """
You are a POLICY ASSISTANT.

Answer the user's question using only the supplied policy context.
Do not use outside knowledge or invent information.

If the answer is not available in the supplied context,
respond with a sentence beginning exactly:
I could not find

Provide a concise and direct answer.
"""


def answer(question, index=None):
    """
    Answer a policy question using the top three retrieved chunks.

    Returns:
        {
            "answer": model response,
            "sources": list of source filenames
        }
    """
    if index is None:
        index = build_index(overlap=10)

    results = retrieve(
        query=question,
        k=3,
        index=index
    )

    context_parts = []

    for result in results:
        context_parts.append(
            f"[{result['source']}]\n{result['text']}"
        )

    user_message = (
        "CONTEXT:\n"
        + "\n".join(context_parts)
        + "\nQUESTION:\n"
        + question
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = mock_llm.chat(messages)

    return {
        "answer": response.choices[0].message.content,
        "sources": list(dict.fromkeys(
            result["source"] for result in results
        ))
    }


if __name__ == "__main__":
    index = build_index(overlap=10)

    question = (
        "What happens if a completed order arrives after 45 minutes?"
    )

    result = answer(question, index=index)

    print("Question:", question)
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
