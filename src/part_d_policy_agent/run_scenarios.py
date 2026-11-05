
import csv
import json
from pathlib import Path

from refund_agent import run_agent


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENARIO_FILE = PROJECT_ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "agent_runs.jsonl"


def run_all_scenarios():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with SCENARIO_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        scenarios = list(csv.DictReader(file))

    results = []

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]

        result = run_agent(
            user_message=scenario["message"],
            scenario_id=scenario_id
        )

        record = {
            "scenario_id": scenario_id,
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"]
        }

        with OUTPUT_FILE.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")

        results.append(record)

        print(
            f"{scenario_id}: "
            f"{result['status']} | "
            f"{result['final_answer']} | "
            f"Calls: {result['llm_calls']} | "
            f"Tokens: {result['tokens_used']}"
        )

    return results


if __name__ == "__main__":
    run_all_scenarios()
