
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CUSTOMERS_PATH = DATA_DIR / "customers.csv"
ORDERS_PATH = DATA_DIR / "orders.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"
OUTPUT_PATH = OUTPUT_DIR / "lapse_cohort.csv"

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load clean Part A datasets.
    customers = pd.read_csv(CUSTOMERS_PATH)
    orders = pd.read_csv(ORDERS_PATH)
    products = pd.read_csv(PRODUCTS_PATH)

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="raise"
    )
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="raise"
    )
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="raise"
    )
    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"], errors="coerce"
    )
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    # Feature engineering must use orders strictly before the cutoff.
    historical = orders.loc[
        orders["order_ts"] < CUTOFF
    ].copy()

    historical_completed = historical.loc[
        historical["status"].eq("completed")
    ].copy()

    historical_final = historical.loc[
        historical["status"].isin(["completed", "cancelled"])
    ].copy()

    # The cohort contains customers with at least one completed
    # order before 1 September 2025.
    eligible_ids = historical_completed["customer_id"].unique()

    cohort = customers.loc[
        customers["customer_id"].isin(eligible_ids)
    ].copy()

    # Completed-order behaviour before the cutoff.
    completed_features = (
        historical_completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
            avg_discount_pct=("discount_pct", "mean"),
            avg_delivery_minutes=("delivery_minutes", "mean"),
        )
        .reset_index()
    )

    completed_features["recency"] = (
        CUTOFF - completed_features["last_order_ts"]
    ).dt.days

    # Cancellation rate uses all final-status orders before cutoff.
    final_counts = (
        historical_final.assign(
            is_cancelled=historical_final["status"].eq("cancelled")
        )
        .groupby("customer_id")
        .agg(
            final_order_count=("order_id", "nunique"),
            cancelled_order_count=("is_cancelled", "sum"),
        )
        .reset_index()
    )

    final_counts["cancellation_rate"] = (
        final_counts["cancelled_order_count"]
        / final_counts["final_order_count"]
    )

    # Favourite category uses completed orders before cutoff only.
    completed_with_category = historical_completed.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    favourite_rows = []

    for customer_id, group in completed_with_category.groupby(
        "customer_id"
    ):
        category_counts = group["category"].dropna().value_counts()

        if category_counts.empty:
            favourite_category = "Unknown"
        else:
            # Deterministic tie-breaking for equally frequent categories.
            highest_count = category_counts.max()
            tied = sorted(
                category_counts[category_counts == highest_count].index
            )
            favourite_category = tied[0]

        favourite_rows.append({
            "customer_id": customer_id,
            "favourite_category": favourite_category,
        })

    favourite_categories = pd.DataFrame(favourite_rows)

    # Label period is September 2025 only.
    # These orders are used to construct the label, never the features.
    september_completed = orders.loc[
        orders["status"].eq("completed")
        & orders["order_ts"].ge(CUTOFF)
        & orders["order_ts"].lt(LABEL_END)
    ]

    september_customer_ids = set(
        september_completed["customer_id"].unique()
    )

    cohort["lapsed"] = (
        ~cohort["customer_id"].isin(september_customer_ids)
    ).astype(int)

    # Assemble features. No orders on or after the cutoff are joined here.
    cohort = cohort.merge(
        completed_features.drop(columns=["last_order_ts"]),
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    cohort = cohort.merge(
        final_counts[["customer_id", "cancellation_rate"]],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    cohort = cohort.merge(
        favourite_categories,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    cohort["tenure_days"] = (
        CUTOFF - cohort["signup_date"]
    ).dt.days

    # Handle any missing feature values without using the label.
    # Missing average delivery is imputed using pre-cutoff completed
    # orders only. Other numeric columns use their historical median.
    numeric_features = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount_pct",
        "cancellation_rate",
        "avg_delivery_minutes",
        "age",
        "tenure_days",
    ]

    for column in numeric_features:
        cohort[column] = pd.to_numeric(
            cohort[column], errors="coerce"
        )

        if cohort[column].isna().any():
            median_value = cohort[column].median()
            if pd.isna(median_value):
                raise ValueError(
                    f"Cannot impute {column}: no valid historical values."
                )
            cohort[column] = cohort[column].fillna(median_value)

    cohort["favourite_category"] = (
        cohort["favourite_category"].fillna("Unknown")
    )
    cohort["city"] = cohort["city"].fillna("Unknown")
    cohort["gender"] = cohort["gender"].fillna("unknown")

    # Save the cohort and its model-ready features for the next tasks.
    output_columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "avg_discount_pct",
        "cancellation_rate",
        "avg_delivery_minutes",
        "favourite_category",
        "city",
        "age",
        "gender",
        "tenure_days",
        "lapsed",
    ]

    cohort = cohort[output_columns].sort_values(
        "customer_id"
    ).reset_index(drop=True)

    cohort.to_csv(OUTPUT_PATH, index=False)

    # Confirm the required cohort and label counts.
    cohort_size = len(cohort)
    lapsed_count = int(cohort["lapsed"].sum())
    lapsed_rate = cohort["lapsed"].mean() * 100

    X = cohort.drop(columns=["customer_id", "lapsed"])
    y = cohort["lapsed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("========== LAPSE COHORT VALIDATION ==========")
    print(f"Cohort customers: {cohort_size} (expected 1156)")
    print(f"Lapsed customers: {lapsed_count} (expected 443)")
    print(f"Lapsed rate: {lapsed_rate:.2f}% (expected 38.32%)")
    print(f"Not lapsed: {int((y == 0).sum())}")
    print(f"Training rows: {len(X_train)} (expected 924)")
    print(f"Test rows: {len(X_test)} (expected 232)")

    print("\n========== LEAKAGE AND MISSING-VALUE CHECKS ==========")
    print("Feature cutoff: strictly before 2025-09-01")
    print("Label window: 2025-09-01 through 2025-09-30")
    print("Segment labels used as features: No")
    print(f"Missing feature values: {int(X.isna().sum().sum())}")

    assert cohort_size == 1156, (
        f"Expected 1156 cohort customers, found {cohort_size}"
    )
    assert lapsed_count == 443, (
        f"Expected 443 lapsed customers, found {lapsed_count}"
    )
    assert len(X_train) == 924 and len(X_test) == 232
    assert X.isna().sum().sum() == 0
    assert cohort["customer_id"].is_unique
    assert cohort["lapsed"].isin([0, 1]).all()
    assert "segment_id" not in X.columns
    assert "segment_name" not in X.columns
    assert "is_premium" not in X.columns

    print("\nLAPSE COHORT VALIDATION PASSED")
    print(f"Output saved to: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
