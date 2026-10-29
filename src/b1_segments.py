import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

os.makedirs("outputs", exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4   # chosen after reading the table below

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = orders[orders["status"] == "completed"]

# ---------- Task 1: RFM ----------
rfm = done.groupby("customer_id").agg(
    last_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
).reset_index()
rfm["recency"] = (SNAPSHOT - rfm["last_ts"]).dt.days
rfm["monetary"] = rfm["monetary"].round(2)
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print("===== Task 1: RFM =====")
print("customers:", len(rfm), "(ref 1175)")
print(f"recency  median {rfm.recency.median():.0f} min {rfm.recency.min()} max {rfm.recency.max()}  (ref 19 / 0 / 180)")
print(f"frequency mean {rfm.frequency.mean():.2f} max {rfm.frequency.max()}  (ref 9.46 / 33)")
print(f"monetary mean {rfm.monetary.mean():.2f} max {rfm.monetary.max():.2f}  (ref 2552.06 / 14766.55)")

# ---------- Task 2: K-means ----------
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
ref = {2: (1820.8, .465), 3: (1268.2, .445), 4: (743.1, .526), 5: (617.4, .481),
       6: (503.2, .482), 7: (408.5, .416), 8: (366.5, .412)}
print("\n===== Task 2: K-means =====")
print(f"{'K':>2} {'inertia':>9} {'silhouette':>11} {'ref inertia':>12} {'ref sil':>8}")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_)
    print(f"{k:>2} {km.inertia_:>9.1f} {sil:>11.3f} {ref[k][0]:>12.1f} {ref[k][1]:>8.3f}")

# ---------- Task 3: profile, name, PCA ----------
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"),
    recency=("recency", "mean"),
    frequency=("frequency", "mean"),
    monetary=("monetary", "mean"),
).round(2)
print(f"\n===== Task 3: profile for K={FINAL_K} =====")
print(prof.to_string())

# Naming by rule, so it does not depend on the arbitrary cluster numbers
premium = prof["monetary"].idxmax()
rest = prof.drop(index=premium)
loyal = rest["frequency"].idxmax()
rest = rest.drop(index=loyal)
at_risk = rest["recency"].idxmax()
names = {premium: "Premium Big-Basket Buyers",
         loyal: "Loyal Frequent Shoppers",
         at_risk: "Slipping Away"}
for sid in prof.index:
    names.setdefault(sid, "Casual Low-Spend")
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium
print("\nnames:", {int(k): v for k, v in names.items()}, "| premium segment id:", int(premium))

pca = PCA(n_components=2).fit(X)
r = pca.explained_variance_ratio_
print(f"\nPCA explained_variance_ratio_: {r[0]:.3f} {r[1]:.3f} total {r.sum():.3f}  (ref 0.634 / 0.218 / 0.852)")

rfm.to_csv("outputs/customer_segments.csv", index=False)
print("wrote outputs/customer_segments.csv", rfm.shape)