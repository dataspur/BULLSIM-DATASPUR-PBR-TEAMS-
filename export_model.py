#!/usr/bin/env python3
"""
Train final XGBoost model on ALL historical data and export for API inference.
Outputs: two .json files (classifier + regressor) + feature reference.
"""
import pandas as pd
import numpy as np
import json, pickle
from pathlib import Path
import xgboost as xgb

DATA = Path.home() / "dataspur" / "data"
OUTPUT = Path.home() / "dataspur" / "backend" / "data"
OUTPUT.mkdir(parents=True, exist_ok=True)

print("Loading data...")
rides = pd.read_csv(DATA / "rides.csv")
rides = rides[rides["event_type"] == "BR"].copy()
bulls = pd.read_csv(DATA / "bull_profiles.csv")
riders_profiles = pd.read_csv(DATA / "rider_profiles.csv")
print(f"  Rides: {len(rides):,}  Bulls: {len(bulls):,}  Riders: {len(riders_profiles):,}")

# Parse
rides["qualified_bool"] = rides["qualified"].astype(str).str.lower().isin(["true", "1", "1.0", "yes"])
rides["score_num"] = pd.to_numeric(rides["score"], errors="coerce")
rides["bull_score_num"] = pd.to_numeric(rides["bull_score"], errors="coerce")
rides["bull_id_str"] = rides["bull_id"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "")

# Merge rider handedness
rides = rides.merge(riders_profiles[["rider_slug", "riding_hand"]], on="rider_slug", how="left")

# Parse dates
rides["evt_date_parsed"] = pd.to_datetime(
    rides["evt_date"].str.extract(r"(\w+),\s*(\d{4})")[0] + " " + 
    rides["evt_date"].str.extract(r"(\w+),\s*(\d{4})")[1],
    format="%b %Y", errors="coerce"
)
rides["evt_date_parsed"] = rides["evt_date_parsed"].fillna(pd.Timestamp("2020-01-01"))
rides = rides.sort_values(["evt_date_parsed", "rid"]).reset_index(drop=True)

print(f"  Date range: {rides['evt_date_parsed'].min().date()} → {rides['evt_date_parsed'].max().date()}")

# Build ALL features (temporal, no leaks)
print("\nBuilding temporal features...")

# Sort by rider for rider features
rides = rides.sort_values(["rider_name", "evt_date_parsed", "rid"])
rides["rider_career_outs"] = rides.groupby("rider_name")["qualified_bool"].transform(
    lambda x: x.shift(1).expanding().count()
).fillna(0)
rides["rider_career_rides"] = rides.groupby("rider_name")["qualified_bool"].transform(
    lambda x: x.shift(1).expanding().sum()
).fillna(0)
rides["rider_career_qual_pct"] = (
    rides["rider_career_rides"] / rides["rider_career_outs"].replace(0, np.nan) * 100
).fillna(0)
rides["rider_career_avg_score"] = rides.groupby("rider_name")["score_num"].transform(
    lambda x: x.where(x > 0).shift(1).expanding().mean()
).fillna(0)
rides["rider_last5_qual"] = rides.groupby("rider_name")["qualified_bool"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
rides["rider_last10_qual"] = rides.groupby("rider_name")["qualified_bool"].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).mean()
)
rides["rider_last20_qual"] = rides.groupby("rider_name")["qualified_bool"].transform(
    lambda x: x.shift(1).rolling(20, min_periods=1).mean()
)

# Sort by bull for bull features
rides = rides.sort_values(["bull_id_str", "evt_date_parsed", "rid"])
rides["bull_career_outs"] = rides.groupby("bull_id_str")["qualified_bool"].transform(
    lambda x: x.shift(1).expanding().count()
).fillna(0)
rides["bull_career_buckoffs"] = rides.groupby("bull_id_str")["qualified_bool"].transform(
    lambda x: (~x).shift(1).expanding().sum()
).fillna(0)
rides["bull_buckoff_pct"] = (
    rides["bull_career_buckoffs"] / rides["bull_career_outs"].replace(0, np.nan) * 100
).fillna(50)
rides["bull_qual_pct"] = 100 - rides["bull_buckoff_pct"]
rides["bull_avg_score"] = rides.groupby("bull_id_str")["bull_score_num"].transform(
    lambda x: x.expanding().mean().shift(1)
).fillna(rides["bull_score_num"].median())
rides["bull_recent_buckoff"] = rides.groupby("bull_id_str")["qualified_bool"].transform(
    lambda x: (~x).shift(1).rolling(10, min_periods=1).mean()
).fillna(0.5)

# Spin x handedness
spin_map = bulls.set_index("bull_id")["hand_advantage_dir"].to_dict()
spin_map_str = {str(k): v for k, v in spin_map.items()}
rides["bull_spin_dir"] = rides["bull_id_str"].map(spin_map_str)

def spin_match(row):
    hand = row.get("riding_hand")
    spin = row.get("bull_spin_dir")
    if pd.isna(hand) or pd.isna(spin):
        return 0
    if (spin == "LEFT" and hand == "Right") or (spin == "RIGHT" and hand == "Left"):
        return 1
    return -1

rides["spin_hand_match"] = rides.apply(spin_match, axis=1)

# Event context
rides["is_pbr"] = (rides["evt_org"] == "PBR").astype(int)
rides["is_prca"] = (rides["evt_org"] == "PRCA").astype(int)
rides["perf_num"] = pd.to_numeric(rides["perf"], errors="coerce").fillna(1)
rides["go_num"] = pd.to_numeric(rides["go"], errors="coerce").fillna(1)
rides["event_qual_sofar"] = rides.groupby("rid")["qualified_bool"].transform(
    lambda x: x.shift(1).expanding().mean()
).fillna(0)

FEATURE_COLS = [
    "rider_career_qual_pct", "rider_career_avg_score",
    "rider_last5_qual", "rider_last10_qual", "rider_last20_qual",
    "bull_qual_pct", "bull_buckoff_pct", "bull_avg_score",
    "bull_recent_buckoff", "bull_career_outs",
    "spin_hand_match",
    "is_pbr", "is_prca", "perf_num", "go_num",
    "event_qual_sofar",
]

print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Spin-match distribution: {rides['spin_hand_match'].value_counts().to_dict()}")

# ============ TRAIN ON ALL DATA ============
print("\nTraining final models on ALL data...")
X_all = rides[FEATURE_COLS].fillna(0).values
y_qual = rides["qualified_bool"].astype(int).values
y_score = rides["score_num"].fillna(0).values

print(f"  Training samples: {len(X_all):,}")
print(f"  Qualification rate: {y_qual.mean()*100:.1f}%")

# Stage 1: Classifier
scale = (y_qual == 0).sum() / max(y_qual.sum(), 1)
clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=min(scale, 5.0), random_state=42, n_jobs=-1,
)
clf.fit(X_all, y_qual)
print(f"  Classifier trained: {clf.n_estimators} trees")

# Find calibrated threshold
y_proba_all = clf.predict_proba(X_all)[:, 1]
cal_thresh = float(np.percentile(y_proba_all, 100 * (1 - y_qual.mean())))
print(f"  Calibrated threshold: {cal_thresh:.4f}")
print(f"  Train accuracy: {(clf.predict(X_all) == y_qual).mean()*100:.1f}%")

# Stage 2: Regressor (qualified rides only)
qual_mask = y_qual == 1
X_score = X_all[qual_mask]
y_sc = y_score[qual_mask]
valid = y_sc > 0
X_score = X_score[valid]
y_sc = y_sc[valid]

reg = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
)
reg.fit(X_score, y_sc)
print(f"  Regressor trained on {len(X_score):,} qualified rides (R²={reg.score(X_score, y_sc):.4f})")

# Feature importance
imp = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": clf.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nClassifier feature importance:")
for _, row in imp.iterrows():
    bar = "█" * int(row["importance"] / imp["importance"].max() * 40)
    print(f"  {row['feature']:<28s} {bar} {row['importance']:.4f}")

# ============ EXPORT ============
print("\nExporting models...")

# Save XGBoost models as JSON
clf.save_model(str(OUTPUT / "clf_model.json"))
reg.save_model(str(OUTPUT / "reg_model.json"))

# Save model metadata
metadata = {
    "feature_cols": FEATURE_COLS,
    "calibrated_threshold": cal_thresh,
    "feature_means": X_all.mean(axis=0).tolist(),
    "feature_stds": X_all.std(axis=0).tolist(),
    "median_bull_score": float(rides["bull_score_num"].median()),
    "n_training_samples": int(len(X_all)),
    "train_qual_rate": float(y_qual.mean()),
    "clf_n_estimators": clf.n_estimators,
    "reg_n_estimators": reg.n_estimators,
}
with open(OUTPUT / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Saved: clf_model.json ({Path(OUTPUT / 'clf_model.json').stat().st_size/1024:.0f}KB)")
print(f"  Saved: reg_model.json ({Path(OUTPUT / 'reg_model.json').stat().st_size/1024:.0f}KB)")
print(f"  Saved: model_metadata.json")

# Also export pre-computed rider/bull stats for the API
print("\nExporting pre-computed temporal stats for API...")

# Get latest stats per rider (most recent date)
rides_latest = rides.sort_values(["rider_name", "evt_date_parsed"])
rider_latest = rides_latest.groupby("rider_name").last().reset_index()
rider_stats_export = rider_latest[[
    "rider_name", "rider_career_outs", "rider_career_rides",
    "rider_career_qual_pct", "rider_career_avg_score",
    "rider_last5_qual", "rider_last10_qual", "rider_last20_qual",
    "riding_hand"
]].copy()
rider_stats_export.to_csv(OUTPUT / "rider_temporal_stats.csv", index=False)
print(f"  Saved: rider_temporal_stats.csv ({len(rider_stats_export)} riders)")

# Get latest stats per bull
rides_bull = rides.sort_values(["bull_id_str", "evt_date_parsed"])
bull_latest = rides_bull.groupby("bull_id_str").last().reset_index()
bull_stats_export = bull_latest[[
    "bull_id_str", "bull_career_outs", "bull_career_buckoffs",
    "bull_buckoff_pct", "bull_qual_pct", "bull_avg_score",
    "bull_recent_buckoff", "bull_spin_dir",
]].copy()
bull_stats_export.to_csv(OUTPUT / "bull_temporal_stats.csv", index=False)
print(f"  Saved: bull_temporal_stats.csv ({len(bull_stats_export)} bulls)")

print("\nDone. API-ready models exported.")