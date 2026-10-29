import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score)
from lapse_data import build_dataset, make_preprocessor, NUM, CAT

os.makedirs("outputs", exist_ok=True)
df = build_dataset()
X, y = df[NUM + CAT], df["lapsed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)


def report(name, model, Xte, yte):
    p = model.predict(Xte)
    print(f"{name}: confusion [[TN FP][FN TP]] = {confusion_matrix(yte, p).tolist()}")
    pr = precision_score(yte, p, zero_division=0)
    rc = recall_score(yte, p, zero_division=0)
    f1 = f1_score(yte, p, zero_division=0)
    print(f"   precision {pr:.3f}  recall {rc:.3f}  F1 {f1:.3f}")
    return pr, rc, f1


# ---- (b) logistic regression ----
lr_pipe = Pipeline([("pre", make_preprocessor()),
                    ("clf", LogisticRegression(max_iter=1000))])
lr_gs = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]},
                     cv=5, scoring="f1").fit(X_train, y_train)
print("===== Task 6 =====")
print("LR best params:", lr_gs.best_params_, f"best CV F1 {lr_gs.best_score_:.4f}")

# ---- (c) random forest ----
rf_pipe = Pipeline([("pre", make_preprocessor()),
                    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_gs = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8],
                               "clf__min_samples_leaf": [5, 10, 20]},
                     cv=5, scoring="f1").fit(X_train, y_train)
print("RF best params:", rf_gs.best_params_, f"best CV F1 {rf_gs.best_score_:.4f}")

# ---- (d) test-set evaluation (tuning finished) ----
print("\nTest set, lapsed class:")
lr_m = report("Logistic regression", lr_gs.best_estimator_, X_test, y_test)
rf_m = report("Random forest     ", rf_gs.best_estimator_, X_test, y_test)

best_is_rf = rf_m[2] >= lr_m[2]
best_name = "random forest" if best_is_rf else "logistic regression"
print("\nHigher test F1:", best_name)

# ---- Task 7: oversample training rows only ----
best_lr = lr_gs.best_estimator_
n0, n1 = (y_train == 0).sum(), (y_train == 1).sum()
print(f"\n===== Task 7 =====\ntrain class counts: not lapsed {n0}, lapsed {n1}")
rng_idx = (y_train[y_train == 1].sample(n=n0 - n1, replace=True, random_state=42).index)
X_bal = pd.concat([X_train, X_train.loc[rng_idx]])
y_bal = pd.concat([y_train, y_train.loc[rng_idx]])
print("balanced counts:", y_bal.value_counts().to_dict())

from sklearn.base import clone
lr_bal = clone(best_lr).fit(X_bal, y_bal)   # preprocessor refit on balanced train only
print("Unbalanced LR:")
ub = report("  LR", best_lr, X_test, y_test)
print("Balanced LR:")
bb = report("  LR", lr_bal, X_test, y_test)
print(f"change: precision {bb[0]-ub[0]:+.3f}  recall {bb[1]-ub[1]:+.3f}  F1 {bb[2]-ub[2]:+.3f}")

joblib.dump({"lr": best_lr, "rf": rf_gs.best_estimator_, "best_is_rf": best_is_rf},
            "outputs/_tuned_models_tmp.joblib")