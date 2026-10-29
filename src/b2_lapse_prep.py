import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from lapse_data import build_dataset, make_preprocessor, NUM, CAT

df = build_dataset()
X, y = df[NUM + CAT], df["lapsed"]

print("===== Task 4: lapse problem =====")
print(f"cohort: {len(df)} (ref 1156)")
print(f"lapsed: {y.sum()} = {y.mean()*100:.2f}% (ref 443 / 38.32%)")
print("tenure_days min:", df["tenure_days"].min(), "(should not be negative)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train {len(X_train)} / test {len(X_test)} (ref 924 / 232)")

pred = [0] * len(y_test)
print(f"baseline accuracy {accuracy_score(y_test, pred)*100:.1f}% (ref 61.6%), "
      f"lapsed F1 {f1_score(y_test, pred, zero_division=0):.2f} (ref 0)")

print("\n===== Task 5: preprocessing =====")
print("missing values in any feature column:", int(X.isna().sum().sum()))
assert X.isna().sum().sum() == 0

pre = make_preprocessor().fit(X_train)          # train rows only
Xt_train, Xt_test = pre.transform(X_train), pre.transform(X_test)
print("transformed shapes:", Xt_train.shape, Xt_test.shape)

i = NUM.index("frequency")
print(f"scaler mean of frequency (train only): {pre.named_transformers_['num'].mean_[i]:.4f}")
print(f"full-cohort mean of frequency:          {X['frequency'].mean():.4f}")