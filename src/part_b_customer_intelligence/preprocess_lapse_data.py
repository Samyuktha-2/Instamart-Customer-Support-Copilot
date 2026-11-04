
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "outputs" / "lapse_cohort.csv"

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

TARGET = "lapsed"


def main():
    cohort = pd.read_csv(INPUT_PATH)

    # Separate features and target. Never include customer_id as a feature.
    X = cohort[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = cohort[TARGET].astype(int)

    # Reproduce the required stratified split exactly.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Define preprocessing. It will be fitted on training data only.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    # Fit only on training features; transform the test set without fitting.
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Establish a baseline that always predicts "not lapsed" (class 0).
    baseline = DummyClassifier(
        strategy="constant",
        constant=0,
    )
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    baseline_accuracy = accuracy_score(
        y_test, baseline_predictions
    )
    baseline_f1 = f1_score(
        y_test,
        baseline_predictions,
        pos_label=1,
        zero_division=0,
    )

    # Inspect the scaler's learned frequency mean.
    fitted_scaler = preprocessor.named_transformers_["numeric"]
    frequency_index = NUMERIC_FEATURES.index("frequency")
    training_frequency_mean = fitted_scaler.mean_[frequency_index]
    full_cohort_frequency_mean = cohort["frequency"].mean()

    print("========== TASK 5: PREPROCESSING VALIDATION ==========")
    print(f"Full cohort rows: {len(cohort)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Training transformed shape: {X_train_transformed.shape}")
    print(f"Test transformed shape: {X_test_transformed.shape}")

    print("\n========== BASELINE RESULTS ==========")
    print("Baseline strategy: Always predict not lapsed (0)")
    print(f"Test accuracy: {baseline_accuracy:.4f}")
    print(f"Test accuracy (%): {baseline_accuracy * 100:.2f}%")
    print(f"Lapsed-class F1: {baseline_f1:.4f}")

    print("\n========== SCALER LEAKAGE CHECK ==========")
    print(
        "Training-set frequency mean learned by scaler: "
        f"{training_frequency_mean:.6f}"
    )
    print(
        "Full-cohort frequency mean (comparison only): "
        f"{full_cohort_frequency_mean:.6f}"
    )
    print("Scaler fitted on training rows only: Yes")
    print("Encoder fitted on training rows only: Yes")
    print("Test rows used to fit preprocessing: No")

    # Validation checks.
    assert len(X_train) == 924
    assert len(X_test) == 232
    assert X_train_transformed.shape[0] == len(X_train)
    assert X_test_transformed.shape[0] == len(X_test)
    assert baseline_f1 == 0.0
    assert 0.60 <= baseline_accuracy <= 0.63

    # Confirm numeric scaler statistics came from the training partition.
    expected_training_mean = X_train["frequency"].mean()
    assert abs(training_frequency_mean - expected_training_mean) < 1e-9

    print("\nTASK 5 PREPROCESSING VALIDATION PASSED")


if __name__ == "__main__":
    main()
