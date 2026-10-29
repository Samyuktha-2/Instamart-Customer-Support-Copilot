import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF = pd.Timestamp("2025-09-01")
WINDOW_END = pd.Timestamp("2025-10-01")
NUM = ["recency", "frequency", "monetary", "avg_discount",
       "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]


def build_dataset():
    orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv("data/clean/products.csv")
    orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

    # FEATURES: only orders strictly before the cutoff
    before = orders[orders["order_ts"] < CUTOFF]
    done = before[before["status"] == "completed"]

    g = done.groupby("customer_id").agg(
        last_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean"),
    )
    g["recency"] = (CUTOFF - g["last_ts"]).dt.days

    final = before[before["status"].isin(["completed", "cancelled"])]
    g["cancel_rate"] = (final.groupby("customer_id")["status"]
                        .apply(lambda s: (s == "cancelled").mean()))

    # most frequent category; ties broken alphabetically so it is deterministic
    fav = (done.groupby(["customer_id", "category"]).size().reset_index(name="n")
               .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
               .drop_duplicates("customer_id").set_index("customer_id")["category"])
    g["fav_category"] = fav

    # LABEL: the only place orders on/after the cutoff are used
    window = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < WINDOW_END)
                    & (orders["status"] == "completed")]
    g["lapsed"] = (~g.index.isin(window["customer_id"])).astype(int)

    cust = customers.set_index("customer_id")[["city", "gender", "age", "signup_date"]]
    df = g.join(cust, how="left")
    df["tenure_days"] = (CUTOFF - df["signup_date"]).dt.days
    return df[NUM + CAT + ["lapsed"]]


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT),
    ])