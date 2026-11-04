
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORDERS_PATH = PROJECT_ROOT / "data" / "clean" / "orders.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EVALUATION_PATH = OUTPUT_DIR / "kmeans_evaluation.csv"

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

EXPECTED = {
    2: (1820.8, 0.465),
    3: (1268.2, 0.445),
    4: (743.1, 0.526),
    5: (617.4, 0.481),
    6: (503.2, 0.482),
    7: (408.5, 0.416),
    8: (366.5, 0.412),
}


def build_rfm():
    orders = pd.read_csv(ORDERS_PATH)

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="raise"
    )
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="raise"
    )

    completed = orders.loc[
        orders["status"].eq("completed")
    ].copy()

    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (
        SNAPSHOT - rfm["last_order_ts"]
    ).dt.days

    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ].sort_values("customer_id").reset_index(drop=True)

    if len(rfm) != 1175:
        raise ValueError(
            f"Expected 1175 RFM customers, found {len(rfm)}"
        )

    return rfm


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rfm = build_rfm()
    feature_columns = ["recency", "frequency", "monetary"]

    # Fit the scaler on the RFM data used for clustering.
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm[feature_columns])

    results = []

    print("========== K-MEANS EVALUATION ==========")
    print(f"Customers: {len(rfm)}")
    print(f"Features: {', '.join(feature_columns)}")
    print("StandardScaler: applied")
    print("KMeans: n_init=10, random_state=42\n")

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42,
        )

        labels = model.fit_predict(scaled_rfm)
        inertia = model.inertia_
        silhouette = silhouette_score(scaled_rfm, labels)

        expected_inertia, expected_silhouette = EXPECTED[k]

        inertia_difference_pct = (
            abs(inertia - expected_inertia)
            / expected_inertia
            * 100
        )

        silhouette_difference = abs(
            silhouette - expected_silhouette
        )

        inertia_pass = inertia_difference_pct <= 5
        silhouette_pass = silhouette_difference <= 0.05

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette": silhouette,
            "expected_inertia": expected_inertia,
            "expected_silhouette": expected_silhouette,
            "inertia_difference_pct": inertia_difference_pct,
            "silhouette_difference": silhouette_difference,
            "reference_check": (
                "PASS" if inertia_pass and silhouette_pass
                else "CHECK"
            ),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(EVALUATION_PATH, index=False)

    print(
        results_df[
            ["k", "inertia", "silhouette", "reference_check"]
        ].to_string(
            index=False,
            formatters={
                "inertia": "{:.1f}".format,
                "silhouette": "{:.3f}".format,
            },
        )
    )

    print("\n========== REFERENCE COMPARISON ==========")
    for row in results:
        print(
            f"K={row['k']}: "
            f"inertia difference={row['inertia_difference_pct']:.2f}% "
            f"(limit 5%), "
            f"silhouette difference={row['silhouette_difference']:.3f} "
            f"(limit 0.05) -> {row['reference_check']}"
        )

    print(f"\nEvaluation saved to: {EVALUATION_PATH.relative_to(PROJECT_ROOT)}")
    print("\nK-MEANS EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
