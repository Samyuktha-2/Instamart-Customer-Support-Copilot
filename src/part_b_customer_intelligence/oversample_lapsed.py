
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "outputs" / "lapse_cohort.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

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


def make_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
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

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                C=100,
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])


def evaluate(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "model": name,
        "test_accuracy": accuracy_score(y_test, predictions),
        "lapsed_precision": precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "lapsed_recall": recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "lapsed_f1": f1_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
    }

    print(f"\n========== {name} ==========")
    print(f"Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Precision: {metrics['lapsed_precision']:.4f}")
    print(f"Recall:    {metrics['lapsed_recall']:.4f}")
    print(f"F1 score:  {metrics['lapsed_f1']:.4f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    return metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(INPUT_PATH)

    X = cohort[FEATURES].copy()
    y = cohort[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Preserve all original training rows.
    X_train_original = X_train.copy()
    y_train_original = y_train.copy()

    original_counts = y_train_original.value_counts().to_dict()
    majority_count = int(y_train_original.value_counts().max())
    lapsed_count = int((y_train_original == 1).sum())

    # Add duplicates of lapsed training rows only, with replacement.
    number_to_add = majority_count - lapsed_count

    if number_to_add > 0:
        lapsed_X = X_train_original.loc[y_train_original == 1]
        lapsed_y = y_train_original.loc[y_train_original == 1]

        sampled_indices = lapsed_X.sample(
            n=number_to_add,
            replace=True,
            random_state=42,
        ).index

        X_duplicates = lapsed_X.loc[sampled_indices].copy()
        y_duplicates = lapsed_y.loc[sampled_indices].copy()

        X_balanced = pd.concat(
            [X_train_original, X_duplicates],
            ignore_index=True,
        )
        y_balanced = pd.concat(
            [y_train_original, y_duplicates],
            ignore_index=True,
        )
    else:
        X_balanced = X_train_original.reset_index(drop=True)
        y_balanced = y_train_original.reset_index(drop=True)

    balanced_counts = y_balanced.value_counts().to_dict()

    print("========== TRAINING CLASS COUNTS ==========")
    print(f"Original training counts: {original_counts}")
    print(f"Balanced training counts: {balanced_counts}")
    print(f"Original training rows: {len(X_train_original)}")
    print(f"Additional duplicated lapsed rows: {number_to_add}")
    print(f"Balanced training rows: {len(X_balanced)}")
    print(f"Test rows (unchanged): {len(X_test)}")

    assert original_counts[0] == 570
    assert original_counts[1] == 354
    assert balanced_counts[0] == 570
    assert balanced_counts[1] == 570
    assert len(X_balanced) == 1140
    assert len(X_test) == 232

    # Train a reference model on the original, imbalanced training set.
    original_model = make_pipeline()
    original_model.fit(X_train_original, y_train_original)

    original_metrics = evaluate(
        "LOGISTIC REGRESSION — ORIGINAL TRAINING DATA",
        original_model,
        X_test,
        y_test,
    )

    # Refit the same best Logistic Regression configuration on balanced data.
    balanced_model = make_pipeline()
    balanced_model.fit(X_balanced, y_balanced)

    balanced_metrics = evaluate(
        "LOGISTIC REGRESSION — OVERSAMPLED TRAINING DATA",
        balanced_model,
        X_test,
        y_test,
    )

    comparison = pd.DataFrame([
        original_metrics,
        balanced_metrics,
    ])
    comparison.to_csv(
        OUTPUT_DIR / "lapse_oversampling_comparison.csv",
        index=False,
    )

    # Temporary artifact for Task 8; the final required model is saved there.
    joblib.dump(
        balanced_model,
        OUTPUT_DIR / "lapse_model_task7.joblib",
    )

    print("\n========== TASK 7 COMPARISON ==========")
    for metric in [
        "lapsed_precision",
        "lapsed_recall",
        "lapsed_f1",
    ]:
        before = original_metrics[metric]
        after = balanced_metrics[metric]
        print(
            f"{metric}: before={before:.4f}, "
            f"after={after:.4f}, change={after - before:+.4f}"
        )

    print("\nTest set used for oversampling: No")
    print("Test rows duplicated: No")
    print("Preprocessing fitted on balanced training data only: Yes")
    print("Comparison saved to outputs/lapse_oversampling_comparison.csv")
    print("Balanced model saved to outputs/lapse_model_task7.joblib")
    print("\nTASK 7 OVERSAMPLING VALIDATION PASSED")


if __name__ == "__main__":
    main()
