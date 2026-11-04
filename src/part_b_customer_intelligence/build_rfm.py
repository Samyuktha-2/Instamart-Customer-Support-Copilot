
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORDERS_PATH = PROJECT_ROOT / "data" / "clean" / "orders.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    orders = pd.read_csv(ORDERS_PATH)

    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="raise")
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="raise"
    )

    completed = orders.loc[orders["status"].eq("completed")].copy()

    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days

    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ].sort_values("customer_id").reset_index(drop=True)

    assert len(rfm) == 1175, f"Expected 1175 customers, got {len(rfm)}"
    assert rfm["customer_id"].is_unique
    assert rfm[["recency", "frequency", "monetary"]].notna().all().all()

    print("========== RFM VALIDATION ==========")
    print(f"Customers with completed orders: {len(rfm)}")
    print("\nRecency summary:")
    print(rfm["recency"].describe())
    print("\nFrequency summary:")
    print(rfm["frequency"].describe())
    print("\nMonetary summary:")
    print(rfm["monetary"].describe())

    print("\n========== REFERENCE CHECKS ==========")
    print(f"Recency median: {rfm['recency'].median():.0f} (expected 19)")
    print(f"Recency minimum: {rfm['recency'].min()} (expected 0)")
    print(f"Recency maximum: {rfm['recency'].max()} (expected 180)")
    print(f"Frequency mean: {rfm['frequency'].mean():.2f} (expected 9.46)")
    print(f"Frequency maximum: {rfm['frequency'].max()} (expected 33)")
    print(f"Monetary mean: INR {rfm['monetary'].mean():,.2f} (expected INR 2,552.06)")
    print(f"Monetary maximum: INR {rfm['monetary'].max():,.2f} (expected INR 14,766.55)")

    # Check the exact reference values that are not approximate.
    assert rfm["recency"].median() == 19
    assert rfm["recency"].min() == 0
    assert rfm["recency"].max() == 180
    assert rfm["frequency"].max() == 33

    print("\nRFM BUILD COMPLETED")


if __name__ == "__main__":
    main()
