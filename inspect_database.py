
import sqlite3

connection = sqlite3.connect("data/instamart.db")
connection.row_factory = sqlite3.Row

order_ids = [
    "ORD-100048",
    "ORD-100067",
    "ORD-100145",
    "ORD-100001",
    "ORD-100025",
    "ORD-999999",
]

for order_id in order_ids:
    row = connection.execute(
        """
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE order_id = ?
        """,
        (order_id,)
    ).fetchone()

    print(dict(row) if row else {"error": "order not found", "order_id": order_id})

connection.close()
