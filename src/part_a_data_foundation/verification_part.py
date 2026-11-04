import pandas as pd

customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")

print("CUSTOMER VALIDATION")
print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())

print(
    "Unique cities:",
    customers["city"].nunique()
)

print(
    "Signup dates missing:",
    customers["signup_date"].isna().sum()
)

print("\nORDER VALIDATION")

print(
    "Unique order IDs:",
    orders["order_id"].nunique()
)

print(
    "Duplicate order IDs:",
    orders["order_id"].duplicated().sum()
)

print(
    "Order timestamps missing:",
    pd.to_datetime(orders["order_ts"]).isna().sum()
)

print(
    "Order amounts missing:",
    orders["order_amount"].isna().sum()
)

print(
    "Completed delivery times missing:",
    orders.loc[
        orders["status"] == "completed",
        "delivery_minutes"
    ].isna().sum()
)

print(
    "Cancelled delivery times missing:",
    orders.loc[
        orders["status"] == "cancelled",
        "delivery_minutes"
    ].isna().sum()
)
