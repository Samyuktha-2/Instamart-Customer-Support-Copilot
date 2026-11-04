
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw")

# Load raw datasets without modifying them
customers = pd.read_csv(RAW_PATH / "customers.csv")
products = pd.read_csv(RAW_PATH / "products.csv")
orders = pd.read_csv(RAW_PATH / "orders.csv")

datasets = {
    "customers": customers,
    "products": products,
    "orders": orders
}


# 1. General dataset audit
for name, df in datasets.items():

    print("\n" + "=" * 70)
    print(f"DATASET: {name.upper()}")
    print("=" * 70)

    print("\nShape:", df.shape)

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Value Audit:")

    missing_count = df.isna().sum()

    missing_percentage = (
        df.isna().mean() * 100
    ).round(2)

    missing_report = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percentage
    })

    print(missing_report)


# 2. Orders-specific audit
print("\n" + "=" * 70)
print("ORDERS AUDIT")
print("=" * 70)

amount_numeric = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

delivery_numeric = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

order_audit = {
    "Raw order rows": len(orders),
    "Exact duplicate rows": orders.duplicated().sum(),
    "Pending rows": orders["status"].eq("pending").sum(),
    "Distinct order IDs": orders["order_id"].nunique(),
    "Non-numeric order amounts": amount_numeric.isna().sum(),
    "Negative delivery minutes": (delivery_numeric < 0).sum(),
    "Blank delivery minutes": (
        orders["delivery_minutes"].isna()
        | orders["delivery_minutes"].astype("string").str.strip().eq("")
    ).sum()
}

for key, value in order_audit.items():
    print(f"{key}: {value}")


# 3. Customers-specific audit
print("\n" + "=" * 70)
print("CUSTOMERS AUDIT")
print("=" * 70)

age_numeric = pd.to_numeric(
    customers["age"],
    errors="coerce"
)

gender_text = customers["gender"].astype("string").str.strip()
city_text = customers["city"].astype("string").str.strip()

customer_audit = {
    "Blank ages": (
        customers["age"].isna()
        | customers["age"].astype("string").str.strip().eq("")
    ).sum(),
    "Ages outside 10-100": ( age_numeric.notna() & ~age_numeric.between(10, 100)
    ).sum(),
    "Distinct raw city spellings": customers["city"].nunique(),
    "Distinct non-blank raw gender spellings": gender_text[
        gender_text.ne("")
    ].nunique(),
    "Blank genders": (
        customers["gender"].isna()
        | gender_text.eq("")
    ).sum()
}

for key, value in customer_audit.items():
    print(f"{key}: {value}")


print("\nRAW DATA AUDIT COMPLETED")
