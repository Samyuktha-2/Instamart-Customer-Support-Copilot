import pandas as pd

RAW = "data/raw/"

customers = pd.read_csv(RAW + "customers.csv")
products = pd.read_csv(RAW + "products.csv")
orders = pd.read_csv(RAW + "orders.csv")

for name,df in [("customers", customers), ("products",products), ("orders", orders)]:
    print(f"\n===={name}====")
    print("shape: ", {df.shape})
    print("\n dtypes: \n", df.dtypes)

    miss = pd.DataFrame({
        "missing" : df.isna().sum(),
        "missing_percentage" : (df.isna().mean() * 100).round(2),
    })
    print("\nMissing Values: \n",miss)


amt_numeric = pd.to_numeric(orders["order_amount"], errors = "coerce")
non_numeric_amt = amt_numeric.isna().sum()

dm = pd.to_numeric(orders["delivery_minutes"], errors="coerce")

gender = customers["gender"].dropna().astype(str).str.strip()
gender = gender[gender != ""]

age = pd.to_numeric(customers["age"], errors="coerce")

audit = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "non-numeric order_amount": int(non_numeric_amt),
    "negative delivery_minutes": int((dm < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "blank age": int(age.isna().sum()),
    "age outside 10-100": int(((age < 10) | (age > 100)).sum()),
    "distinct raw city spellings": int(customers["city"].nunique()),
    "distinct non-blank gender spellings": int(gender.nunique()),
    "blank gender": int(customers["gender"].isna().sum()
                        + (customers["gender"].dropna().astype(str).str.strip() == "").sum()),
}

reference = [13084, 119, 656, 12309, 120, 106, 2541, 61, 6, 24, 10, 30]
print("\n===== AUDIT FIGURES =====")
print(f"{'figure':40s}{'yours':>8s}{'reference':>11s}  match")
for (k, v), r in zip(audit.items(), reference):
    print(f"{k:40s}{v:>8d}{r:>11d}  {'OK' if v == r else 'CHECK'}")