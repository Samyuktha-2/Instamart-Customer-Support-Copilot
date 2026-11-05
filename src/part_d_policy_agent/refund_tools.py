
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from policy_rag import build_index, retrieve


DB_PATH = PROJECT_ROOT / "data" / "instamart.db"
LEDGER_PATH = PROJECT_ROOT / "outputs" / "refund_ledger.jsonl"


def get_order(order_id):
    """Retrieve order details from the SQLite database."""
    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.row_factory = sqlite3.Row

            row = connection.execute(
                """
                SELECT order_id, status, order_amount, delivery_minutes
                FROM orders
                WHERE order_id = ?
                """,
                (order_id,)
            ).fetchone()

        if row is None:
            return {"error": "order not found"}

        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "order_amount": float(row["order_amount"]),
            "delivery_minutes": (
                float(row["delivery_minutes"])
                if row["delivery_minutes"] is not None
                else None
            )
        }

    except sqlite3.Error as exc:
        return {"error": str(exc)}


def search_policy(query):
    """Retrieve relevant policy chunks."""
    index = build_index(overlap=10)

    return {
        "results": retrieve(
            query=query,
            k=3,
            index=index
        )
    }


def calculate_refund(order_amount, delivery_minutes):
    """Calculate refund for completed orders delivered late."""
    if order_amount < 0 or delivery_minutes < 0:
        return {"error": "Invalid order amount or delivery time"}

    refund = order_amount if delivery_minutes > 45 else 0.0

    return {
        "refund_inr": round(float(refund), 2)
    }


def _write_ledger(tool_name, order_id, refund_inr):
    """Append a refund or escalation record to the ledger."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def issue_refund(order_id, refund_inr):
    """Record an automatic refund of INR 1000 or less."""
    amount = round(float(refund_inr), 2)

    order = get_order(order_id)

    if "error" in order:
        return order

    if order["status"].lower() != "completed":
        return {"error": "only completed orders can be refunded"}

    if amount <= 0:
        return {"error": "refund amount must be greater than zero"}

    if amount > 1000:
        return {"error": "refund above INR 1000 requires human approval"}

    _write_ledger("issue_refund", order_id, amount)

    return {"refund_inr": amount}


def escalate_to_human(order_id, refund_inr):
    """Record a refund requiring human approval."""
    amount = round(float(refund_inr), 2)

    order = get_order(order_id)

    if "error" in order:
        return order

    if order["status"].lower() != "completed":
        return {"error": "only completed orders can be escalated"}

    if amount <= 1000:
        return {"error": "refund amount does not require escalation"}

    _write_ledger("escalate_to_human", order_id, amount)

    return {"refund_inr": amount}


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Retrieve order status, amount, and delivery time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search the policy documents for relevant rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate the eligible late-delivery refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "number"}
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue an automatic refund of INR 1000 or less.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate refunds above INR 1000 for human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}
