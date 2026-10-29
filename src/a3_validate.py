import sqlite3
import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
done = o[o["status"] == "completed"]
canc = o[o["status"] == "cancelled"]

with sqlite3.connect("data/instamart.db") as con:
    db_counts = {t: pd.read_sql_query(f"SELECT COUNT(*) n FROM {t}", con)["n"][0]
                 for t in ("customers", "products", "orders")}

cities = c["city"].value_counts().to_dict()
genders = c["gender"].value_counts().to_dict()

checks = {
    "12,309 order rows": len(o) == 12309,
    "one row per order_id": o["order_id"].is_unique,
    "11,110 completed": len(done) == 11110,
    "1,199 cancelled": len(canc) == 1199,
    "no pending rows": (o["status"] == "pending").sum() == 0,
    "0 missing order_amount": o["order_amount"].isna().sum() == 0,
    "0 missing delivery (completed)": done["delivery_minutes"].isna().sum() == 0,
    "1,199 missing delivery (cancelled)": canc["delivery_minutes"].isna().sum() == 1199,
    "completed revenue 2,998,668.80": round(done["order_amount"].sum(), 2) == 2998668.80,
    "six cities": len(cities) == 6,
    "city counts": cities == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                              "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "gender counts": genders == {"female": 610, "male": 560, "unknown": 30},
    "0 missing ages": c["age"].isna().sum() == 0,
    "ages within 18-60": c["age"].between(18, 60).all(),
    "DB row counts": db_counts == {"customers": 1200, "products": 40, "orders": 12309},
    "order_ts text format": o["order_ts"].str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").all(),
}

failed = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(("PASS  " if v else "FAIL  ") + k)
print("ALL CHECKPOINTS PASSED" if not failed else f"\n{len(failed)} CHECK(S) FAILED")