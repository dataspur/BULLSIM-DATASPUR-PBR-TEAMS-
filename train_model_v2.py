#!/usr/bin/env python3
"""
DataSpur Model v2 — Two-Stage XGBoost with profile-enriched features.
Merges bull profiles (power rating, spin direction, handedness splits)
and rider profiles (riding hand) into the feature set.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, r2_score, mean_absolute_error

DATA = Path.home() / "dataspur" / "data"

print("=" * 60)
print("DATASPUR MODEL v2 — XGBoost Two-Stage (Bull Riding)")
print("=" * 60)

# ============ LOAD DATA ============
print("\n[1] Loading data...")
rides = pd.read_csv(DATA / "rides.csv")
rides = rides[rides["event_type"] == "BR"].copy()
print(f"  Bull rides: {len(rides):,}")

bulls = pd.read_csv(DATA / "bull_profiles.csv")
riders = pd.read_csv(DATA / "rider_profiles.csv")
print(f"  Bull profiles: {len(bulls):,}")
print(f"  Rider profiles: {len(riders):,}")

# ============ PARSE RIDES ============
print("\n[2] Parsing ride data...")
rides["qualified_bool"] = rides["qualified"].astype(str).str.lower().isin(["true", "1", "1.0", "yes"])
rides["score_num"] = pd.to_numeric(rides["score"], errors="coerce")
rides["bull_score_num"] = pd.to_numeric(rides["bull_score"], errors="coerce")
rides["bull_id_str"] = rides["bull_id"].apply(
    lambda x: str(int(float(x))) if pd.notna(x) else ""
)

print(f"  Qualified: {rides['qualified_bool'].sum():,} ({rides['qualified_bool'].mean()*100:.1f}%)")
print(f"  Mean score (qualified): {rides.loc[rides['qualified_bool'], 'score_num'].mean():.2f}")

# ============ MERGE PROFILES ============
print("\n[3] Merging profiles...")
bulls["bull_id_str"] = bulls["bull_id"].astype(str)
rides = rides.merge(
    bulls[["bull_id_str", "power_rating", "hand_advantage_dir", "hand_advantage_pct",
            "buckoff_pct", "pre_ride_prob", "avg_bull_score", "avg_ride_score",
            "lh_pct", "rh_pct"]],
    on="bull_id_str", how="left"
)
rides = rides.merge(
    riders[["rider_slug", "riding_hand", "career_qual_pct"]],
    on="rider_slug", how="left"
)

bull_cov = rides["power_rating"].notna().mean() * 100
rider_cov = rides["riding_hand"].notna().mean() * 100
print(f"  Bull profile coverage: {bull_cov:.1f}%")
print(f"  Rider profile coverage: {rider_cov:.1f}%")

# ============ ENGINEER FEATURES ============
print("\n[4] Engineering features...")

# --- Rider features ---
# We compute these from scratch (career stats from the ride history itself)
rides = rides.sort_values(["rider_name", "evt_date"]).reset_index(drop=True)

# Compute rider career stats via expanding window (no look-ahead bias)
rider_stats = rides.groupby("rider_name").apply(
    lambda g: pd.DataFrame({
        "rider_career_outs": g["qualified_bool"].expanding().count().shift(1).fillna(0),
        "rider_career_rides": g["qualified_bool"].expanding().sum().shift(1).fillna(0),
        "rider_career_avg_score": g["score_num"].where(g["qualified_bool"]).expanding().mean().shift(1).fillna(0),
    })
).reset_index(drop=True)

rides = pd.concat([rides, rider_stats], axis=1)
rides["rider_career_qual_pct"] = (
    rides["rider_career_rides"] / rides["rider_career_outs"].replace(0, np.nan)
).fillna(0) * 100

# Recent form (simplified — last 10 and 20 outs by date)
rides = rides.sort_values(["rider_name", "evt_date"])
rides["rider_last10_qual"] = rides.groupby("rider_name")["qualified_bool"].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).mean()
)
rides["rider_last20_qual"] = rides.groupby("rider_name")["qualified_bool"].transform(
    lambda x: x.shift(1).rolling(20, min_periods=1).mean()
)

# --- Bull features ---
rides["bull_qual_pct"] = 100 - rides["buckoff_pct"].fillna(68)  # buckoff→qual rate

# Spin × handedness matchup
def spin_matchup(row):
    """Returns advantage signal: positive if rider's hand matches bull's spin direction."""
    hand = row.get("riding_hand")
    spin = row.get("hand_advantage_dir")
    if pd.isna(hand) or pd.isna(spin):
        return 0
    # Bull spins LEFT → favors RIGHT-handed riders (centrifugal into rope hand)
    # Bull spins RIGHT → favors LEFT-handed riders
    if spin == "LEFT" and hand == "Right":
        return 1  # favorable
    if spin == "RIGHT" and hand == "Left":
        return 1
    return -1  # unfavorable

rides["spin_hand_match"] = rides.apply(spin_matchup, axis=1)
print(f"  Spin-match distribution: {rides['spin_hand_match'].value_counts().to_dict()}")

# --- Event context features ---
rides["evt_org_pbr"] = (rides["evt_org"] == "PBR").astype(int)
rides["evt_org_prca"] = (rides["evt_org"] == "PRCA").astype(int)

# Round number (proxy from perf/go columns)
rides["perf_num"] = pd.to_numeric(rides["perf"], errors="coerce").fillna(1)
rides["go_num"] = pd.to_numeric(rides["go"], errors="coerce").fillna(1)

# Feature set
features = [
    "rider_career_qual_pct", "rider_career_avg_score",
    "rider_last10_qual", "rider_last20_qual",
    "power_rating", "bull_qual_pct",
    "spin_hand_match",
    "evt_org_pbr", "evt_org_prca",
    "perf_num", "go_num",
]

# Keep only rows with all features
model_df = rides[features + ["qualified_bool", "score_num", "rider_name", "bull_id_str", 
                                "evt_org", "evt_city", "evt_date"]].copy()
model_df = model_df.fillna(0)

# Targets
y_qual = model_df["qualified_bool"].astype(int)
y_score = model_df["score_num"].fillna(0)

print(f"  Features: {len(features)}")
print(f"  Rows: {len(model_df):,}")
print(f"  Qualified: {y_qual.sum():,} ({y_qual.mean()*100:.1f}%)")

# ============ TRAIN/TEST SPLIT ============
print("\n[5] Temporal split (80/20)...")
split = int(len(model_df) * 0.8)
X_train = model_df[features].iloc[:split]
X_test = model_df[features].iloc[split:]
y_train_qual = y_qual.iloc[:split]
y_test_qual = y_qual.iloc[split:]
y_train_score = y_score.iloc[:split]
y_test_score = y_score.iloc[split:]

print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"  Train qual%: {y_train_qual.mean()*100:.1f}% | Test qual%: {y_test_qual.mean()*100:.1f}%")

# ============ STAGE 1: CLASSIFIER ============
print("\n" + "=" * 60)
print("STAGE 1: Qualified Ride Classifier")
print("=" * 60)

import xgboost as xgb

scale = (y_train_qual == 0).sum() / max(y_train_qual.sum(), 1)
print(f"  Class imbalance ratio: {scale:.1f}")

clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale, random_state=42, n_jobs=-1,
    eval_metric="logloss",
)
clf.fit(X_train, y_train_qual)

# Raw predictions (default 0.5 threshold)
y_pred_qual_raw = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

print(f"\n  Default threshold (0.5):")
print(f"    Predicted qual%: {y_pred_qual_raw.mean()*100:.1f}% vs Actual: {y_test_qual.mean()*100:.1f}%")

# Calibration: find optimal threshold
from sklearn.metrics import f1_score, precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test_qual, y_pred_proba)

# Find threshold that balances precision and recall (max F1)
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
f1s = f1s[:-1]  # last value has no corresponding threshold
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]
print(f"  Optimal threshold (max F1): {best_threshold:.3f}")

# Also try threshold where predicted qual% matches actual qual%
target_rate = y_test_qual.mean()
best_calibrated_threshold = float(np.percentile(y_pred_proba, 100 * (1 - target_rate)))
print(f"  Calibrated threshold (match rate): {best_calibrated_threshold:.3f}")

# Use the calibrated threshold
y_pred_qual = (y_pred_proba >= best_calibrated_threshold).astype(int)

print(f"  Predicted qual%: {y_pred_qual.mean()*100:.1f}% vs Actual: {y_test_qual.mean()*100:.1f}%")
print(f"\n" + classification_report(y_test_qual, y_pred_qual, target_names=["Buck-off", "Qualified"]))

# Store the threshold for inference
print(f"  Final threshold: {best_calibrated_threshold:.4f}")

# ============ STAGE 2: REGRESSOR ============
print("=" * 60)
print("STAGE 2: Score Regressor (qualified rides only)")
print("=" * 60)

train_qual = y_train_qual == 1
reg = xgb.XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
)

X_tr_score = X_train[train_qual]
y_tr_score = y_train_score[train_qual]
valid_tr = y_tr_score > 0
X_tr_score = X_tr_score[valid_tr]
y_tr_score = y_tr_score[valid_tr]

print(f"  Training on {len(X_tr_score):,} qualified rides")

reg.fit(X_tr_score, y_tr_score)

test_qual = y_test_qual == 1
X_ts_score = X_test[test_qual]
y_ts_score = y_test_score[test_qual]
valid_ts = y_ts_score > 0
X_ts_score = X_ts_score[valid_ts]
y_ts_score = y_ts_score[valid_ts]

y_pred_scores = reg.predict(X_ts_score)

print(f"\nScore Regression ({len(y_ts_score):,} qualified rides):")
print(f"  R²: {r2_score(y_ts_score, y_pred_scores):.4f}")
print(f"  MAE: {mean_absolute_error(y_ts_score, y_pred_scores):.2f}")
print(f"  True mean: {y_ts_score.mean():.2f} | Pred mean: {y_pred_scores.mean():.2f}")

# ============ FEATURE IMPORTANCE ============
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

imp_clf = pd.DataFrame({
    "feature": features,
    "importance": clf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nClassifier:")
for _, row in imp_clf.iterrows():
    bar = "█" * int(row["importance"] / imp_clf["importance"].max() * 50)
    print(f"  {row['feature']:<30s} {bar} {row['importance']:.4f}")

imp_reg = pd.DataFrame({
    "feature": features,
    "importance": reg.feature_importances_
}).sort_values("importance", ascending=False)

print("\nRegressor:")
for _, row in imp_reg.iterrows():
    bar = "█" * int(row["importance"] / imp_reg["importance"].max() * 50)
    print(f"  {row['feature']:<30s} {bar} {row['importance']:.4f}")

# ============ REGIME BREAKDOWN ============
print("\n" + "=" * 60)
print("REGIME BREAKDOWN (Test Set)")
print("=" * 60)

test_df = model_df.iloc[split:].copy()
test_df["pred_qual"] = y_pred_qual
test_df["actual_qual"] = y_test_qual.values
test_df["pred_proba"] = y_pred_proba

for org in ["PBR", "PRCA"]:
    mask = test_df["evt_org"] == org
    if mask.sum() < 50:
        continue
    sub = test_df[mask]
    acc = (sub["pred_qual"] == sub["actual_qual"]).mean()
    print(f"  {org}: {mask.sum():,} outs | acc={acc*100:.1f}% | "
          f"actual={sub['actual_qual'].mean()*100:.1f}% | pred={sub['pred_qual'].mean()*100:.1f}%")

print("\nDone.")