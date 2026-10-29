import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lapse_data import build_dataset, NUM, CAT

df = build_dataset()
X, y = df[NUM + CAT], df["lapsed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

saved = joblib.load("outputs/_tuned_models_tmp.joblib")
lr, rf = saved["lr"], saved["rf"]
selected = rf if saved["best_is_rf"] else lr

# ---- (a) importances and coefficient signs ----
print("===== Task 8a =====")
names = rf.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
print("Top 5 random forest importances:")
print(imp.sort_values(ascending=False).head(5).round(4).to_string())

coef = pd.Series(lr.named_steps["clf"].coef_[0],
                 index=lr.named_steps["pre"].get_feature_names_out())
print("\nLogistic regression coefficients (numeric features):")
print(coef[[n for n in coef.index if n.startswith("num__")]].round(3).to_string())

# ---- (b) persist and reload ----
joblib.dump(selected, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
same_class = np.array_equal(selected.predict(X_test), reloaded.predict(X_test))
same_proba = np.allclose(selected.predict_proba(X_test), reloaded.predict_proba(X_test))
print("\n===== Task 8b =====")
print("reloaded model reproduces test predictions exactly:", same_class, "| probabilities equal:", same_proba)
assert same_class

# ---- (c) retention list ----
seg = pd.read_csv("outputs/customer_segments.csv")
premium_ids = set(seg.loc[seg["is_premium"].astype(bool), "customer_id"])
test = X_test.copy()
test["lapse_probability"] = selected.predict_proba(X_test)[:, 1]
test["actually_lapsed"] = y_test
prem = test[test.index.isin(premium_ids)].sort_values("lapse_probability", ascending=False)
top = prem.head(10).reset_index()[["customer_id", "lapse_probability", "actually_lapsed"]]
top["lapse_probability"] = top["lapse_probability"].round(4)
top.to_csv("outputs/retention_list.csv", index=False)

print("\n===== Task 8c =====")
print("Premium customers in test set:", len(prem))
print(top.to_string(index=False))
print(f"truly lapsed among the {len(top)} listed: {int(top['actually_lapsed'].sum())}")