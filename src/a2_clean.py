import os
import sqlite3
import pandas as pd

RAW, CLEAN = "data/raw/" , "data/clean/"
os.makedirs(CLEAN, exist_ok = True)

customers = pd.read_csv(RAW + "customers.csv")
products = pd.read_csv(RAW + "products.csv")
orders = pd.read_csv(RAW + "orders.csv")

#CUSTOMERS
print("==== CLEAN CUSTOMERS ====")
print("Raw gender spelling", customers["gender"].value_counts(dropna = False).to_dict())

#City: strip + title case
raw_cities = customers["city"].nunique()
customers["city"] = customers["city"].str.strip().str.title()
print(f"(a) city spellings: {raw_cities} -> {customers['city'].nunique()}")

#Gender: Male/ Female/ Unknown
def map_gender(g):
    if pd.isna(g):
        return "unknown"
    g = str(g).strip().lower()
    if g in ("m", "male"):
        return "male"

    if g in ("f","female"):
        return "female"

    return "unknown"

customers["gender"] = customers["gender"].apply(map_gender)
print("(b) gender counts:", customers["gender"].value_counts().to_dict())

#Age
customers["age"] = pd.to_numeric(customers["age"], errors = "coerce")
out_of_range = int(((customers["age"] < 10) | (customers["age"] > 100)).sum())
customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = float("nan")
missing_before_fill = int(customers["age"].isna().sum())
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)
print(f"(c) out-of-range set to missing: {out_of_range}; "
      f"missing ages filled: {missing_before_fill}; "
      f"missing now: {customers['age'].isna().sum()}; "
      f"age range: {customers['age'].min()}-{customers['age'].max()}")

customers["signup_date"] = pd.to_datetime(customers["signup_date"])
print("(d) signup_date dtype:", customers["signup_date"].dtype)

# =============== ORDERS ===============
print("\n===== CLEAN ORDERS =====")
print("raw rows:", len(orders))

# Rule 1: order_ts -> datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Rule 2: remove exact duplicate rows
n = len(orders)
orders = orders.drop_duplicates()
print(f"Rule 2: exact duplicates removed: {n - len(orders)} -> {len(orders)} rows")

# Rule 3: keep the latest order_ts per order_id
n = len(orders)
orders = (orders.sort_values("order_ts", kind="stable")
                .drop_duplicates(subset="order_id", keep="last")
                .sort_values("order_id")
                .reset_index(drop=True))
print(f"Rule 3: superseded records removed: {n - len(orders)} -> {len(orders)} rows")

# Rule 4: order_amount -> numeric, fill from product price
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
price = orders["product_id"].map(products.set_index("product_id")["price_inr"])
missing_amt = orders["order_amount"].isna()
computed = (price * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders.loc[missing_amt, "order_amount"] = computed[missing_amt]
print(f"Rule 4: order_amount values filled from price: {int(missing_amt.sum())}; "
      f"missing now: {orders['order_amount'].isna().sum()}")

# Rule 5: negative delivery -> missing; fill completed with category median
neg = orders["delivery_minutes"] < 0
orders.loc[neg, "delivery_minutes"] = float("nan")
orders["category"] = orders["product_id"].map(products.set_index("product_id")["category"])
done = orders["status"] == "completed"
missing_done = int(orders.loc[done, "delivery_minutes"].isna().sum())
med = orders[done].groupby("category")["delivery_minutes"].transform("median")
orders.loc[done, "delivery_minutes"] = orders.loc[done, "delivery_minutes"].fillna(med)
orders = orders.drop(columns="category")
print(f"Rule 5: negatives set to missing: {int(neg.sum())}; "
      f"completed delivery times filled: {missing_done}; "
      f"cancelled left missing: {int(orders.loc[orders['status']=='cancelled','delivery_minutes'].isna().sum())}")

print("\nstatus counts:", orders["status"].value_counts().to_dict())
print("completed revenue: INR", round(orders.loc[orders['status'] == 'completed', 'order_amount'].sum(), 2))

# =============== STORE ===============
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
customers["signup_date"] = customers["signup_date"].dt.strftime("%Y-%m-%d")

customers.to_csv(CLEAN + "customers.csv", index=False)
products.to_csv(CLEAN + "products.csv", index=False)
orders.to_csv(CLEAN + "orders.csv", index=False)

with sqlite3.connect("data/instamart.db") as con:
    customers.to_sql("customers", con, if_exists="replace", index=False)
    products.to_sql("products", con, if_exists="replace", index=False)
    orders.to_sql("orders", con, if_exists="replace", index=False)
print("\nwrote data/clean/*.csv and data/instamart.db")