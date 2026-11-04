
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
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


def make_pipeline(classifier):
    """Create preprocessing + classifier as one leakage-safe pipeline."""
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
        ("classifier", classifier),
    ])


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label=1, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label=1, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label=1, zero_division=0
    )

    print(f"\n========== {name}: TEST RESULTS ==========")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["not_lapsed (0)", "lapsed (1)"],
            zero_division=0,
        )
    )

    return {
        "model": name,
        "test_accuracy": accuracy,
        "test_precision_lapsed": precision,
        "test_recall_lapsed": recall,
        "test_f1_lapsed": f1,
    }


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

    print("========== DATA SPLIT ==========")
    print(f"Training rows: {len(X_train)} (expected 924)")
    print(f"Test rows: {len(X_test)} (expected 232)")
    print("Cross-validation: 5-fold")
    print("Scoring: lapsed-class F1")
    print("Preprocessing fitted inside each CV training fold.")

    # Logistic Regression grid.
    logistic_pipeline = make_pipeline(
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        )
    )

    logistic_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
    }

    logistic_search = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid=logistic_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    logistic_search.fit(X_train, y_train)

    print("\n========== LOGISTIC REGRESSION TUNING ==========")
    print(f"Best parameters: {logistic_search.best_params_}")
    print(f"Best CV F1: {logistic_search.best_score_:.4f}")

    logistic_results = pd.DataFrame(logistic_search.cv_results_)[
        ["param_classifier__C", "mean_test_score", "std_test_score"]
    ].sort_values("param_classifier__C")
    print("\nAll Logistic Regression candidates:")
    print(logistic_results.to_string(index=False))

    # Random Forest grid.
    forest_pipeline = make_pipeline(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1,
        )
    )

    forest_grid = {
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_leaf": [5, 10, 20],
    }

    forest_search = GridSearchCV(
        estimator=forest_pipeline,
        param_grid=forest_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    forest_search.fit(X_train, y_train)

    print("\n========== RANDOM FOREST TUNING ==========")
    print(f"Best parameters: {forest_search.best_params_}")
    print(f"Best CV F1: {forest_search.best_score_:.4f}")

    forest_results = pd.DataFrame(forest_search.cv_results_)[
        [
            "param_classifier__max_depth",
            "param_classifier__min_samples_leaf",
            "mean_test_score",
            "std_test_score",
        ]
    ].sort_values(
        [
            "param_classifier__max_depth",
            "param_classifier__min_samples_leaf",
        ]
    )
    print("\nAll Random Forest candidates:")
    print(forest_results.to_string(index=False))

    # Only now evaluate the tuned models on the untouched test set.
    logistic_metrics = evaluate_model(
        "LOGISTIC REGRESSION",
        logistic_search.best_estimator_,
        X_test,
        y_test,
    )
    forest_metrics = evaluate_model(
        "RANDOM FOREST",
        forest_search.best_estimator_,
        X_test,
        y_test,
    )

    metrics = pd.DataFrame([logistic_metrics, forest_metrics])
    metrics.to_csv(
        OUTPUT_DIR / "lapse_model_comparison.csv",
        index=False,
    )

    # Save tuning details for reproducibility and later inspection.
    logistic_results.to_csv(
        OUTPUT_DIR / "logistic_cv_results.csv",
        index=False,
    )
    forest_results.to_csv(
        OUTPUT_DIR / "random_forest_cv_results.csv",
        index=False,
    )

    # Choose the higher test F1 for the next tasks. Task 7 will
    # separately assess the effect of balancing the training set.
    if logistic_metrics["test_f1_lapsed"] >= forest_metrics["test_f1_lapsed"]:
        selected_name = "Logistic Regression"
        selected_model = logistic_search.best_estimator_
        selected_cv_f1 = logistic_search.best_score_
        selected_test_f1 = logistic_metrics["test_f1_lapsed"]
    else:
        selected_name = "Random Forest"
        selected_model = forest_search.best_estimator_
        selected_cv_f1 = forest_search.best_score_
        selected_test_f1 = forest_metrics["test_f1_lapsed"]

    # Temporary model artifact for subsequent tasks; Task 8 will
    # save and reload the final selected model after the balancing work.
    joblib.dump(selected_model, OUTPUT_DIR / "lapse_model_task6.joblib")

    print("\n========== TASK 6 SUMMARY ==========")
    print(f"Selected model by test F1: {selected_name}")
    print(f"Selected model CV F1: {selected_cv_f1:.4f}")
    print(f"Selected model test F1: {selected_test_f1:.4f}")
    print("Comparison saved to outputs/lapse_model_comparison.csv")
    print("Temporary model saved to outputs/lapse_model_task6.joblib")

    if selected_test_f1 >= 0.55:
        print("TARGET CHECK: Test F1 >= 0.55")
    else:
        print(
            "TARGET CHECK: Test F1 is below 0.55. "
            "Do not alter the test set or tune against it; continue "
            "with the prescribed training-only balancing experiment."
        )


if __name__ == "__main__":
    main()
