import re
import sqlite3
import pandas as pd

sql_text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- name:\s*(\S+)\s*$", sql_text, flags=re.M)
queries = {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}

res = {}
with sqlite3.connect("data/instamart.db") as con:
    for name, q in queries.items():
        res[name] = pd.read_sql_query(q, con)

q1, q2, q3, q4, q5, q6, q7 = (res[k] for k in queries)

top = q6[q6["row_num"] == 1].set_index("city")
top_actual = {c: (r["customer_id"], round(r["total_spend"], 2)) for c, r in top.iterrows()}
top_expected = {"Bengaluru": ("C0082", 12856.20), "Chennai": ("C0930", 10151.40),
                "Delhi": ("C0291", 13742.60), "Hyderabad": ("C0743", 12730.05),
                "Mumbai": ("C0693", 14766.55), "Pune": ("C0426", 10377.10)}
mismatch = int((q6["rnk"] != q6["row_num"]).sum())
best = q2.iloc[0]

checks = [
    ("Q1 customers", len(q1), 30),
    ("Q2 rows", len(q2), 30),
    ("Q2 total revenue", round(q2["revenue"].sum(), 2), 2998668.80),
    ("Q2 largest cell", (best["city"], best["category"], best["revenue"]),
     ("Bengaluru", "Electronics", 233909.45)),
    ("Q3 products", len(q3), 11),
    ("Q4 customers", len(q4), 25),
    ("Q5 customers", len(q5), 550),
    ("Q5 average spend", float(q5["avg_total"].iloc[0]), 2552.06),
    ("Q6 top per city", top_actual, top_expected),
    ("Q6 RANK != ROW_NUMBER rows", mismatch, 8),
    ("Q7 daily rows", len(q7), 183),
    ("Q7 last cumulative", float(q7["cumulative_revenue"].iloc[-1]), 2998668.80),
]

print(f"{'check':30s}{'result':>8s}")
for label, got, exp in checks:
    print(f"{label:30s}{'OK' if got == exp else 'CHECK'}   got={got}  expected={exp}")

print("\nQ6 tied rows (RANK != ROW_NUMBER):")
print(q6[q6["rnk"] != q6["row_num"]].to_string(index=False))