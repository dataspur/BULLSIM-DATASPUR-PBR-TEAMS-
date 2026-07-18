#!/usr/bin/env python3
"""
DataSpur Model — Two-stage XGBoost for Bull Riding
(1) Classifier: will this rider cover the bull? (yes/no)
(2) Regressor: if yes, what score?
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, r2_score, mean_absolute_error

DATA_DIR = Path.home() / "dataspur" / "data"

print(f"DataSpur Model Training — {datetime.now()}")
print("=" * 50)

# Load features
df = pd.read_parquet(DATA_DIR / "features_br.parquet")
print(f"Loaded features: {len(df):,} rows × {df.shape[1]} columns")

# Feature columns (exclude metadata + targets + leak columns)
EXCLUDE = [
    "target_qualified", "target_score", "rider_name", "bull_name",
    "rid", "evt_date", "evt_org", "got_8_seconds", "judge_avg", "judge_range"
]
feature_cols = [c for c in df.columns if c not in EXCLUDE]
print(f"Features: {len(feature_cols)}")

# Fill NaN
df = df.fillna(0)

# Temporal split (80/20 by row order — already chronologically sorted)
split = int(len(df) * 0.8)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

X_train = train[feature_cols]
y_train_qual = train["target_qualified"].astype(int)
y_train_score = train["target_score"]

X_test = test[feature_cols]
y_test_qual = test["target_qualified"].astype(int)
y_test_score = test["target_score"]

print(f"\nTrain: {len(train):,} | Test: {len(test):,}")
print(f"Train qualified: {y_train_qual.sum():,} ({y_train_qual.mean()*100:.1f}%)")
print(f"Test qualified: {y_test_qual.sum():,} ({y_test_qual.mean()*100:.1f}%)")

# ═══════════════════════════════════════════════
# STAGE 1: Qualified Ride Classifier
# ═══════════════════════════════════════════════
print("\n" + "=" * 50)
print("STAGE 1: Qualified Ride Classifier (XGBoost)")
print("=" * 50)

import xgboost as xgb

# Handle class imbalance with scale_pos_weight
scale = (y_train_qual == 0).sum() / max(y_train_qual.sum(), 1)
print(f"Class balance — neg/pos ratio: {scale:.1f}")

clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)
clf.fit(X_train, y_train_qual)

y_pred_qual = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

print("\nClassification Report (Test Set):")
print(classification_report(y_test_qual, y_pred_qual, target_names=["Buck-off", "Qualified"]))

# ═══════════════════════════════════════════════
# STAGE 2: Score Regressor (only on qualified rides)
# ═══════════════════════════════════════════════
print("=" * 50)
print("STAGE 2: Score Regressor (XGBoost — qualified rides only)")
print("=" * 50)

# Train on qualified rides only
train_qual_mask = y_train_qual == 1
test_qual_mask = y_test_qual == 1

X_train_score = X_train[train_qual_mask]
y_train_score_vals = y_train_score[train_qual_mask].dropna()

# Align indices
valid_train = X_train_score.index.intersection(y_train_score_vals.index)
X_train_score = X_train_score.loc[valid_train]
y_train_score_vals = y_train_score_vals.loc[valid_train]

print(f"Training score regressor on {len(X_train_score):,} qualified rides")

reg = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
reg.fit(X_train_score, y_train_score_vals)

# Predict only for rides classified as qualified
X_test_scores = X_test[test_qual_mask]
y_test_scores_true = y_test_score[test_qual_mask].dropna()
valid_test = X_test_scores.index.intersection(y_test_scores_true.index)
X_test_scores = X_test_scores.loc[valid_test]
y_test_scores_true = y_test_scores_true.loc[valid_test]

y_pred_scores = reg.predict(X_test_scores)

print(f"\nScore Regression (Test Set — {len(y_test_scores_true):,} qualified rides):")
print(f"  R²: {r2_score(y_test_scores_true, y_pred_scores):.4f}")
print(f"  MAE: {mean_absolute_error(y_test_scores_true, y_pred_scores):.2f} points")
print(f"  Mean true score: {y_test_scores_true.mean():.2f}")
print(f"  Mean predicted: {y_pred_scores.mean():.2f}")

# ═══════════════════════════════════════════════
# COMBINED: Predict qualifying, then score
# ═══════════════════════════════════════════════
print("\n" + "=" * 50)
print("COMBINED TWO-STAGE EVALUATION")
print("=" * 50)

# For all test rides, predict: will they qualify? If yes, what score?
test_y_qual_pred = clf.predict(X_test)
test_y_score_pred = np.zeros(len(X_test))
test_y_score_pred[:] = np.nan

qual_idx = test_y_qual_pred == 1
if qual_idx.sum() > 0:
    test_y_score_pred[qual_idx] = reg.predict(X_test[qual_idx])

# Compare against actual
results = pd.DataFrame({
    "actual_qual": y_test_qual.values,
    "pred_qual": test_y_qual_pred,
    "actual_score": y_test_score.values,
    "pred_score": test_y_score_pred,
})

# Accuracy metrics
qual_acc = (results["actual_qual"] == results["pred_qual"]).mean()
true_pos = ((results["actual_qual"] == 1) & (results["pred_qual"] == 1)).sum()
false_pos = ((results["actual_qual"] == 0) & (results["pred_qual"] == 1)).sum()
false_neg = ((results["actual_qual"] == 1) & (results["pred_qual"] == 0)).sum()

print(f"Qualification accuracy: {qual_acc*100:.1f}%")
print(f"True positives (correctly predicted qualified): {true_pos:,}")
print(f"False positives (predicted qualified but bucked off): {false_pos:,}")
print(f"False negatives (predicted buck-off but actually qualified): {false_neg:,}")

# Score accuracy on correctly-classified qualified rides
correct_qual = (results["actual_qual"] == 1) & (results["pred_qual"] == 1)
if correct_qual.sum() > 0:
    mae_correct = mean_absolute_error(
        results.loc[correct_qual, "actual_score"].dropna(),
        results.loc[correct_qual, "pred_score"].dropna()
    )
    print(f"Score MAE (correctly predicted qualifiers): {mae_correct:.2f} points")

# ═══════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════
print("\n" + "=" * 50)
print("TOP 15 FEATURES — Qualification Classifier")
print("=" * 50)
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": clf.feature_importances_
}).sort_values("importance", ascending=False)

for i, row in importance.head(15).iterrows():
    bar = "█" * int(row["importance"] / importance["importance"].max() * 40)
    print(f"  {row['feature']:<45s} {bar} {row['importance']:.4f}")

print("\nTOP 15 FEATURES — Score Regressor")
print("=" * 50)
importance_reg = pd.DataFrame({
    "feature": feature_cols,
    "importance": reg.feature_importances_
}).sort_values("importance", ascending=False)

for i, row in importance_reg.head(15).iterrows():
    bar = "█" * int(row["importance"] / importance_reg["importance"].max() * 40)
    print(f"  {row['feature']:<45s} {bar} {row['importance']:.4f}")

# ═══════════════════════════════════════════════
# REGIME BREAKDOWN
# ═══════════════════════════════════════════════
print("\n" + "=" * 50)
print("REGIME BREAKDOWN (Test Set)")
print("=" * 50)

# By organization
for org in ["PBR", "PRCA", "CPRA"]:
    mask = test["evt_org"] == org
    if mask.sum() == 0:
        continue
    sub = results[mask.values]
    acc = (sub["actual_qual"] == sub["pred_qual"]).mean()
    print(f"  {org}: {mask.sum():,} outs | qual acc={acc*100:.1f}% | "
          f"actual qual={sub['actual_qual'].mean()*100:.1f}% | pred qual={sub['pred_qual'].mean()*100:.1f}%")

# By round
for rnd in [1, 2, 3, 4]:
    mask = test["round"] == rnd
    if mask.sum() < 50:
        continue
    sub = results[mask.values]
    acc = (sub["actual_qual"] == sub["pred_qual"]).mean()
    print(f"  Round {rnd}: {mask.sum():,} outs | qual acc={acc*100:.1f}% | "
          f"actual qual={sub['actual_qual'].mean()*100:.1f}%")

print("\nDone.")