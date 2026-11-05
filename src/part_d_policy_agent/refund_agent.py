
import json
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm

from refund_tools import TOOLS_SPEC, TOOL_MAP


STATE_DIR = PROJECT_ROOT / "state"

SYSTEM_PROMPT = """
You are an AGENT responsible for processing customer refund requests.

Follow the refund policy carefully.

Rules:
1. Only completed orders are eligible for refunds.
2. Orders delivered in more than 45 minutes qualify for a full refund.
3. Orders delivered in 45 minutes or less do not qualify.
4. Refunds above INR 1000 must be escalated to a human support lead.
5. Refunds of INR 1000 or less can be issued automatically.
6. Use the available tools to verify all information.
7. Never invent order details or refund amounts.

Use the tools in the appropriate order.
"""


def _checkpoint_path(run_id):
    return STATE_DIR / f"{run_id}.json"


def _save_checkpoint(state):
    """Save the complete agent state after each tool observation."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    path = _checkpoint_path(state["run_id"])

    with path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=4)

    return path


def _result(state, status, final_answer):
    return {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "status": status,
        "final_answer": final_answer,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "tools_executed": state["tools_executed"]
    }


def _continue_agent(
    state,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    simulate_crash_after_tools=None
):
    """Continue an agent run from its current state."""

    start_time = time.monotonic()

    while state["llm_calls"] < max_iterations:

        # Timeout check before every model call.
        if time.monotonic() - start_time >= time_limit_seconds:
            return _result(
                state,
                "FALLBACK_TIMEOUT",
                "FALLBACK_TIMEOUT: Request handed over for human review."
            )

        response = mock_llm.chat(
            state["messages"],
            tools=TOOLS_SPEC
        )

        state["llm_calls"] += 1

        usage = getattr(response, "usage", None)

        if usage is not None:
            state["tokens_used"] += getattr(
                usage, "total_tokens", 0
            )

        # Token check before processing tool calls.
        if state["tokens_used"] > token_limit:
            return _result(
                state,
                "FALLBACK_TOKEN_LIMIT",
                "FALLBACK_TOKEN_LIMIT: Request handed over for human review."
            )

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        # Final model response.
        if not tool_calls:
            final_answer = message.content or ""

            status = (
                final_answer.split(":", 1)[0].strip()
                if ":" in final_answer
                else "UNKNOWN"
            )

            return _result(state, status, final_answer)

        assistant_tool_calls = []

        for tool_call in tool_calls:
            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            assistant_tool_calls.append({
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": arguments
            })

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls
        })

        # Execute requested tools.
        for tool_call in assistant_tool_calls:

            tool_name = tool_call["name"]
            arguments = tool_call["arguments"]
            tool_call_id = tool_call["id"]

            if tool_name not in TOOL_MAP:
                observation = {
                    "error": f"Unknown tool: {tool_name}"
                }
            else:
                try:
                    observation = TOOL_MAP[tool_name](**arguments)
                except Exception as exc:
                    observation = {"error": str(exc)}

            step = {
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            }

            state["steps"].append(step)

            state["tools_executed"].append({
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            })

            state["messages"].append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": json.dumps(observation)
            })

            state["step_number"] += 1

            # Save immediately after every tool observation.
            _save_checkpoint(state)

            # Simulate a crash only after the checkpoint is saved.
            if (
                simulate_crash_after_tools is not None
                and len(state["tools_executed"])
                == simulate_crash_after_tools
            ):
                raise RuntimeError(
                    f"Simulated crash after tool execution "
                    f"{state['step_number']}"
                )

    return _result(
        state,
        "FALLBACK_MAX_ITERATIONS",
        "FALLBACK_MAX_ITERATIONS: Request handed over for human review."
    )


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    simulate_crash_after_tools=None
):
    """Start a new ReAct agent run."""

    run_id = str(uuid.uuid4())

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "steps": [],
        "tools_executed": []
    }

    return _continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        simulate_crash_after_tools=simulate_crash_after_tools
    )


def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):
    """Load a checkpoint and continue without repeating completed tools."""

    checkpoint_path = _checkpoint_path(run_id)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for run ID: {run_id}"
        )

    with checkpoint_path.open("r", encoding="utf-8") as file:
        state = json.load(file)

    return _continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )


if __name__ == "__main__":
    result = run_agent(
        user_message=(
            "My order ORD-100048 reached me very late. "
            "I would like a refund."
        ),
        scenario_id="S1"
    )

    print(json.dumps(result, indent=4))
