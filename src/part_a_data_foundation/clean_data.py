
import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

RAW_PATH = Path("data/raw")
CLEAN_PATH = Path("data/clean")

CLEAN_PATH.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# TASK 3: CUSTOMER CLEANING
# --------------------------------------------------

def clean_customers():

    print("\n========== CUSTOMER CLEANING ==========")

    customers = pd.read_csv(RAW_PATH / "customers.csv")

    print("Original shape:", customers.shape)

    # 1. Standardize city names
    customers["city"] = (
        customers["city"]
        .str.strip()
        .str.title()
    )

    print("\nCity distribution:")
    print(customers["city"].value_counts())

    # 2. Standardize gender
    gender_mapping = {
        "m": "male",
        "male": "male",
        "man": "male",
        "men": "male",
        "f": "female",
        "female": "female",
        "woman": "female",
        "women": "female"
    }

    customers["gender"] = (
        customers["gender"]
        .fillna("")
        .str.strip()
        .str.lower()
        .map(gender_mapping)
        .fillna("unknown")
    )

    print("\nGender distribution:")
    print(customers["gender"].value_counts())

    # 3. Convert age to numeric
    customers["age"] = pd.to_numeric(
        customers["age"],
        errors="coerce"
    )

    # Treat ages outside 10-100 as missing
    invalid_age_mask = ~customers["age"].between(
        10, 100,
        inclusive="both"
    )

    invalid_age_count = invalid_age_mask.sum()

    customers.loc[invalid_age_mask, "age"] = np.nan

    # Fill missing ages using city-wise median
    city_median_age = (
        customers.groupby("city")["age"]
        .transform("median")
    )

    missing_age_count = customers["age"].isna().sum()

    customers["age"] = customers["age"].fillna(
        city_median_age
    )

    print("\nInvalid ages converted to missing:", invalid_age_count)
    print("Missing ages filled:", missing_age_count)

    # 4. Convert signup_date to datetime
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"],
        errors="coerce"
    )

    # Save cleaned customers
    customers.to_csv(
        CLEAN_PATH / "customers.csv",
        index=False,
        date_format="%Y-%m-%d"
    )

    print("\nCleaned customer shape:", customers.shape)
    print("Missing ages after cleaning:", customers["age"].isna().sum())

    return customers


# --------------------------------------------------
# PRODUCT LOADING
# --------------------------------------------------

def load_products():

    print("\n========== PRODUCT LOADING ==========")

    products = pd.read_csv(RAW_PATH / "products.csv")

    products.to_csv(
        CLEAN_PATH / "products.csv",
        index=False
    )

    print("Products shape:", products.shape)

    return products


# --------------------------------------------------
# TASK 4: ORDER CLEANING
# --------------------------------------------------

def clean_orders(products):

    print("\n========== ORDER CLEANING ==========")

    orders = pd.read_csv(RAW_PATH / "orders.csv")

    print("Original order shape:", orders.shape)

    # 1. Convert order_ts to datetime
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"],
        errors="coerce"
    )

    # 2. Remove exact duplicate rows
    before_duplicates = len(orders)

    orders = orders.drop_duplicates().copy()

    exact_duplicates_removed = before_duplicates - len(orders)

    print("Exact duplicate rows removed:", exact_duplicates_removed)

    # 3. Keep only the latest record for each order_id
    before_latest = len(orders)

    orders = (
        orders.sort_values(
            by="order_ts",
            ascending=True,
            kind="stable"
        )
        .drop_duplicates(
            subset=["order_id"],
            keep="last"
        )
        .copy()
    )

    duplicate_order_records_removed = before_latest - len(orders)

    print(
        "Older order records removed:",
        duplicate_order_records_removed
    )

    # 4. Convert order_amount to numeric
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"],
        errors="coerce"
    )

    # Merge product prices and categories
    product_details = products[
        ["product_id", "price_inr", "category"]
    ]

    orders = orders.merge(
        product_details,
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    # Calculate missing order amounts
    missing_amount_mask = orders["order_amount"].isna()

    missing_amount_count = missing_amount_mask.sum()

    calculated_amount = (
        orders["price_inr"]
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)

    orders.loc[missing_amount_mask, "order_amount"] = (
        calculated_amount[missing_amount_mask]
    )

    print("Missing order amounts filled:", missing_amount_count)

    # 5. Set negative delivery times to missing
    negative_delivery_mask = orders["delivery_minutes"] < 0

    negative_delivery_count = negative_delivery_mask.sum()

    orders.loc[
        negative_delivery_mask,
        "delivery_minutes"
    ] = np.nan

    print(
        "Negative delivery times converted to missing:",
        negative_delivery_count
    )

    # Fill missing delivery times ONLY for completed orders
    completed_mask = orders["status"].eq("completed")

    completed_category_median = (
        orders.loc[completed_mask]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    )

    completed_missing_mask = (
        completed_mask
        & orders["delivery_minutes"].isna()
    )

    completed_missing_count = completed_missing_mask.sum()

    orders.loc[
        completed_missing_mask,
        "delivery_minutes"
    ] = completed_category_median.loc[
        orders.loc[completed_mask].index
    ].loc[
        orders.loc[completed_mask].index[
            orders.loc[completed_mask, "delivery_minutes"].isna()
        ]
    ]

    print(
        "Missing completed delivery times filled:",
        completed_missing_count
    )

    # Cancelled orders must retain missing delivery times
    cancelled_missing_count = (
        orders.loc[
            orders["status"].eq("cancelled"),
            "delivery_minutes"
        ].isna().sum()
    )

    print(
        "Cancelled orders with missing delivery time:",
        cancelled_missing_count
    )

    # Remove temporary product columns
    orders = orders.drop(
        columns=["price_inr", "category"]
    )

    # Ensure delivery time is numeric
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"],
        errors="coerce"
    )

    # Save cleaned orders
    orders.to_csv(
        CLEAN_PATH / "orders.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S"
    )

    print("\nCleaned order shape:", orders.shape)

    print("\nOrder status distribution:")
    print(orders["status"].value_counts())

    print(
        "\nMissing order amounts:",
        orders["order_amount"].isna().sum()
    )

    print(
        "Missing completed delivery times:",
        orders.loc[
            orders["status"].eq("completed"),
            "delivery_minutes"
        ].isna().sum()
    )

    print(
        "Missing cancelled delivery times:",
        orders.loc[
            orders["status"].eq("cancelled"),
            "delivery_minutes"
        ].isna().sum()
    )

    completed_revenue = orders.loc[
        orders["status"].eq("completed"),
        "order_amount"
    ].sum()

    print(
        f"\nCompleted order revenue: INR {completed_revenue:,.2f}"
    )

    return orders


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    customers = clean_customers()

    products = load_products()

    orders = clean_orders(products)

    print("\n========== CLEANING COMPLETED ==========")

    print("Customers:", len(customers))
    print("Products:", len(products))
    print("Orders:", len(orders))


if __name__ == "__main__":
    main()
