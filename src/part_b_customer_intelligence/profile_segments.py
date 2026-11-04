
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORDERS_PATH = PROJECT_ROOT / "data" / "clean" / "orders.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "customer_segments.csv"

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4


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

    return rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ].sort_values("customer_id").reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rfm = build_rfm()
    features = ["recency", "frequency", "monetary"]

    # Standardise the RFM features before clustering.
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm[features])

    # Fit the final K-means model.
    kmeans = KMeans(
        n_clusters=FINAL_K,
        n_init=10,
        random_state=42,
    )
    rfm["segment_id"] = kmeans.fit_predict(scaled_rfm)

    # Profile each cluster using the original, unscaled values.
    profiles = (
        rfm.groupby("segment_id")
        .agg(
            size=("customer_id", "count"),
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
        )
        .reset_index()
    )

    # Premium must be the segment with the highest mean monetary value.
    premium_id = int(
        profiles.loc[profiles["mean_monetary"].idxmax(), "segment_id"]
    )

    # Give the remaining segments names based on their profiles.
    remaining = profiles.loc[
        profiles["segment_id"] != premium_id
    ].copy()

    loyal_id = int(
        remaining.sort_values(
            ["mean_frequency", "mean_recency"],
            ascending=[False, True],
        ).iloc[0]["segment_id"]
    )

    remaining = remaining.loc[
        remaining["segment_id"] != loyal_id
    ]

    at_risk_id = int(
        remaining.sort_values(
            ["mean_recency", "mean_frequency"],
            ascending=[False, True],
        ).iloc[0]["segment_id"]
    )

    occasional_id = int(
        remaining.loc[
            remaining["segment_id"] != at_risk_id,
            "segment_id",
        ].iloc[0]
    )

    segment_names = {
        premium_id: "Premium Customers",
        loyal_id: "Loyal Regulars",
        at_risk_id: "At-Risk Customers",
        occasional_id: "Occasional Customers",
    }

    recommended_actions = {
        "Premium Customers": (
            "Offer exclusive benefits and personalised rewards."
        ),
        "Loyal Regulars": (
            "Encourage repeat orders through loyalty rewards."
        ),
        "At-Risk Customers": (
            "Run a targeted re-engagement campaign."
        ),
        "Occasional Customers": (
            "Encourage a second or more frequent purchase."
        ),
    }

    rfm["segment_name"] = rfm["segment_id"].map(segment_names)
    rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

    # PCA is applied to the scaled RFM matrix.
    pca = PCA(n_components=2)
    pca.fit(scaled_rfm)

    variance_ratios = pca.explained_variance_ratio_

    # Save only the required columns, in the specified order.
    output = rfm[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "segment_id",
            "segment_name",
            "is_premium",
        ]
    ]

    output.to_csv(OUTPUT_PATH, index=False)

    print("========== SEGMENT PROFILES ==========")
    for _, row in profiles.sort_values("segment_id").iterrows():
        segment_id = int(row["segment_id"])
        name = segment_names[segment_id]

        print(f"\nSegment ID: {segment_id}")
        print(f"Name: {name}")
        print(f"Size: {int(row['size'])}")
        print(f"Mean recency: {row['mean_recency']:.2f} days")
        print(f"Mean frequency: {row['mean_frequency']:.2f}")
        print(f"Mean monetary: INR {row['mean_monetary']:,.2f}")
        print(f"Premium: {segment_id == premium_id}")
        print(f"Recommended action: {recommended_actions[name]}")

    print("\n========== PCA RESULTS ==========")
    print(f"PC1 explained variance: {variance_ratios[0]:.3f}")
    print(f"PC2 explained variance: {variance_ratios[1]:.3f}")
    print(f"Total explained variance: {variance_ratios.sum():.3f}")

    print("\n========== VALIDATION ==========")
    assert len(output) == 1175
    assert output["customer_id"].is_unique
    assert output["segment_id"].nunique() == FINAL_K
    assert output["segment_name"].notna().all()
    assert output["is_premium"].sum() == int(
        profiles.loc[
            profiles["segment_id"] == premium_id, "size"
        ].iloc[0]
    )
    assert output.columns.tolist() == [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]

    premium_mean = profiles.loc[
        profiles["segment_id"] == premium_id, "mean_monetary"
    ].iloc[0]
    assert premium_mean == profiles["mean_monetary"].max()

    print(f"Customers assigned: {len(output)}")
    print(f"Segments created: {output['segment_id'].nunique()}")
    print(f"Premium segment ID: {premium_id}")
    print(f"Premium customers: {int(output['is_premium'].sum())}")
    print(f"Output file: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("SEGMENT VALIDATION PASSED")


if __name__ == "__main__":
    main()
