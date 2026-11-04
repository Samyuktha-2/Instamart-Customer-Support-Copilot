
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

COHORT_PATH = OUTPUT_DIR / "lapse_cohort.csv"
SEGMENTS_PATH = OUTPUT_DIR / "customer_segments.csv"
RANDOM_FOREST_PATH = OUTPUT_DIR / "lapse_model_task6.joblib"
BALANCED_LOGISTIC_PATH = OUTPUT_DIR / "lapse_model_task7.joblib"

FINAL_MODEL_PATH = OUTPUT_DIR / "lapse_model.joblib"
RETENTION_PATH = OUTPUT_DIR / "retention_list.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "random_forest_feature_importance.csv"


NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]

CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "favourite_category",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "lapsed"


def get_feature_names(model):
    """Return transformed feature names from the fitted pipeline."""
    preprocessor = model.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(COHORT_PATH)
    segments = pd.read_csv(SEGMENTS_PATH)

    X = cohort[FEATURES].copy()
    y = cohort[TARGET].astype(int)

    # Recreate exactly the same stratified split used in Tasks 5–7.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # ---------------------------------------------------------
    # Task 8(a): Random Forest feature importance
    # ---------------------------------------------------------
    random_forest = joblib.load(RANDOM_FOREST_PATH)

    rf_names = get_feature_names(random_forest)
    rf_classifier = random_forest.named_steps["classifier"]

    rf_importance = pd.DataFrame({
        "feature": rf_names,
        "importance": rf_classifier.feature_importances_,
    }).sort_values("importance", ascending=False)

    rf_importance.to_csv(IMPORTANCE_PATH, index=False)

    print("========== RANDOM FOREST: TOP 5 FEATURES ==========")
    print(rf_importance.head(5).to_string(index=False))

    # ---------------------------------------------------------
    # Task 8(a): Logistic Regression coefficient interpretation
    # ---------------------------------------------------------
    logistic_model = joblib.load(BALANCED_LOGISTIC_PATH)

    logistic_names = get_feature_names(logistic_model)
    logistic_classifier = logistic_model.named_steps["classifier"]

    coefficients = pd.DataFrame({
        "feature": logistic_names,
        "coefficient": logistic_classifier.coef_[0],
    })

    print("\n========== LOGISTIC REGRESSION COEFFICIENTS ==========")
    print("Positive coefficient: associated with higher lapse odds.")
    print("Negative coefficient: associated with lower lapse odds.")
    print("These are associations, not proof of causation.")

    # Report the coefficients of three required numeric features.
    for feature in ["recency", "frequency", "monetary"]:
        transformed_name = f"numeric__{feature}"
        row = coefficients.loc[
            coefficients["feature"] == transformed_name
        ]

        if row.empty:
            raise ValueError(
                f"Could not find coefficient for {transformed_name}"
            )

        value = float(row["coefficient"].iloc[0])
        direction = "higher" if value > 0 else "lower"

        print(
            f"{feature}: coefficient={value:.6f}; "
            f"higher values are associated with {direction} lapse odds."
        )

    # ---------------------------------------------------------
    # Task 8(b): Save, reload, and compare test predictions
    # ---------------------------------------------------------
    # Use the balanced Logistic Regression selected in Task 7.
    joblib.dump(logistic_model, FINAL_MODEL_PATH)
    reloaded_model = joblib.load(FINAL_MODEL_PATH)

    predictions_before = logistic_model.predict(X_test)
    predictions_after = reloaded_model.predict(X_test)

    probabilities_before = logistic_model.predict_proba(X_test)[:, 1]
    probabilities_after = reloaded_model.predict_proba(X_test)[:, 1]

    predictions_identical = np.array_equal(
        predictions_before, predictions_after
    )
    probabilities_identical = np.array_equal(
        probabilities_before, probabilities_after
    )

    print("\n========== SAVE / RELOAD VALIDATION ==========")
    print(f"Final model: {FINAL_MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Test predictions identical: {predictions_identical}")
    print(f"Test probabilities identical: {probabilities_identical}")
    print(
        "Final model test F1: "
        f"{f1_score(y_test, predictions_after, zero_division=0):.4f}"
    )

    assert predictions_identical
    assert probabilities_identical

    # ---------------------------------------------------------
    # Task 8(c): Premium-customer retention list
    # ---------------------------------------------------------
    if "is_premium" not in segments.columns:
        raise ValueError(
            "customer_segments.csv must contain is_premium."
        )

    segments["is_premium"] = (
        segments["is_premium"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    premium_ids = set(
        segments.loc[segments["is_premium"], "customer_id"]
    )

    # Only test-set customers, scored by the final saved model.
    test_results = pd.DataFrame({
        "customer_id": cohort.loc[X_test.index, "customer_id"].values,
        "lapse_probability": probabilities_after,
        "actually_lapsed": y_test.values,
    })

    premium_test = test_results.loc[
        test_results["customer_id"].isin(premium_ids)
    ].copy()

    premium_test = premium_test.sort_values(
        "lapse_probability",
        ascending=False,
    )

    retention_list = premium_test.head(10).copy()
    retention_list = retention_list[
        ["customer_id", "lapse_probability", "actually_lapsed"]
    ]

    retention_list.to_csv(RETENTION_PATH, index=False)

    actual_lapsed_count = int(
        retention_list["actually_lapsed"].sum()
    )

    print("\n========== PREMIUM RETENTION LIST ==========")
    print(f"Premium customers in test set: {len(premium_test)}")
    print(f"Retention list rows: {len(retention_list)}")
    print(retention_list.to_string(index=False))
    print(
        "Actually lapsed customers in retention list: "
        f"{actual_lapsed_count}"
    )
    print(f"Saved to: {RETENTION_PATH.relative_to(PROJECT_ROOT)}")

    # ---------------------------------------------------------
    # Final checks
    # ---------------------------------------------------------
    assert len(X_train) == 924
    assert len(X_test) == 232
    assert len(retention_list) == min(10, len(premium_test))
    assert list(retention_list.columns) == [
        "customer_id",
        "lapse_probability",
        "actually_lapsed",
    ]
    assert retention_list["lapse_probability"].is_monotonic_decreasing
    assert retention_list["customer_id"].is_unique
    assert retention_list["actually_lapsed"].isin([0, 1]).all()

    print("\nIMPORTANT LIMITATION")
    print(
        "The retention list is illustrative and not a leakage-free "
        "evaluation. Premium membership comes from full-window RFM "
        "segmentation, which includes the broader observation window."
    )
    print("\nTASK 8 VALIDATION PASSED")


if __name__ == "__main__":
    main()
