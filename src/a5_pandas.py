import sqlite3
import pandas as pd

with sqlite3.connect("data/instamart.db") as con:
    df = pd.read_sql_query(
        """SELECT o.*, c.city, p.category
           FROM orders o
           JOIN customers c ON o.customer_id = c.customer_id
           JOIN products  p ON o.product_id  = p.product_id
           WHERE o.status = 'completed'""", con)

# ---------- (a) pivot cross-check ----------
pv = pd.pivot_table(df, index="city", columns="category", values="order_amount",
                    aggfunc="sum", margins=True, margins_name="Total")
print("===== (a) pivot_table with margins =====")
print(pv.round(2).to_string())
pivot_total = round(pv.loc["Total", "Total"], 2)
print(f"\npivot grand total: INR {pivot_total:,.2f}")
print("equals SQL total 2,998,668.80:", pivot_total == 2998668.80)

city_stats = df.groupby("city").agg(
    total=("order_amount", "sum"),
    mean=("order_amount", "mean"),
    count=("order_amount", "count"),
).round(2)
print("\nper-city sum / mean / count:")
print(city_stats.to_string())

# ---------- (b) distribution of order_amount ----------
x = df["order_amount"]
q1, med, q3 = x.quantile(0.25), x.median(), x.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\n===== (b) order_amount (completed) =====")
print(f"mean {x.mean():.2f} | median {med:.2f} | skewness {x.skew():.2f}")
print(f"Q1 {q1:.2f} | Q3 {q3:.2f} | IQR {iqr:.2f}")
print(f"lower fence {lo:.2f} -> outliers below: {(x < lo).sum()}")
print(f"upper fence {hi:.2f} -> outliers above: {(x > hi).sum()}")

cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = df[cols].corr()
print("\ncorrelation matrix:")
print(corr.round(3).to_string())

pairs = corr.where(~pd.DataFrame(
    [[i == j for j in range(4)] for i in range(4)], index=cols, columns=cols)).stack()
top_pair = pairs.abs().idxmax()
print(f"\nlargest |correlation|: {top_pair} = {pairs[top_pair]:.3f}")