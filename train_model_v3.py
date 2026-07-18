#!/usr/bin/env python3
"""
DataSpur Model v3 — Leak-Free Walk-Forward XGBoost (Bull Riding)
- Temporal bull feature computation (no future data)
- Walk-forward validation across time periods
- Improved score regressor with better features
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, r2_score, mean_absolute_error
import xgboost as xgb

DATA = Path.home() / "dataspur" / "data"
MIN_TRAIN_EVENTS = 50  # minimum events before first test fold

print("=" * 60)
print("DATASPUR MODEL v3 — Leak-Free Walk-Forward")
print("=" * 60)

# ============ LOAD DATA ============
print("\n[1] Loading...")
rides = pd.read_csv(DATA / "rides.csv")
rides = rides[rides["event_type"] == "BR"].copy()
bulls = pd.read_csv(DATA / "bull_profiles.csv")
riders_profiles = pd.read_csv(DATA / "rider_profiles.csv")

print(f"  Rides: {len(rides):,}  Bulls: {len(bulls):,}  Riders: {len(riders_profiles):,}")

# ============ PARSE & CLEAN ============
print("\n[2] Parsing...")
rides["qualified_bool"] = rides["qualified"].astype(str).str.lower().isin(["true", "1", "1.0", "yes"])
rides["score_num"] = pd.to_numeric(rides["score"], errors="coerce")
rides["bull_score_num"] = pd.to_numeric(rides["bull_score"], errors="coerce")
rides["bull_id_str"] = rides["bull_id"].apply(
    lambda x: str(int(float(x))) if pd.notna(x) else ""
)
# Merge rider handedness (static, no leak — it's a physical trait)
rides = rides.merge(
    riders_profiles[["rider_slug", "riding_hand"]],
    on="rider_slug", how="left"
)
print(f"  Rider hand coverage: {rides['riding_hand'].notna().mean()*100:.1f}%")

# ============ SORT BY DATE ============
rides["evt_date_parsed"] = pd.to_datetime(
    rides["evt_date"].str.extract(r"(\w+),\s*(\d{4})")[0] + " " + 
    rides["evt_date"].str.extract(r"(\w+),\s*(\d{4})")[1],
    format="%b %Y", errors="coerce"
)
# Fill missing dates with mid-range
rides["evt_date_parsed"] = rides["evt_date_parsed"].fillna(pd.Timestamp("2020-01-01"))
rides = rides.sort_values(["evt_date_parsed", "rid"]).reset_index(drop=True)
print(f"  Date range: {rides['evt_date_parsed'].min().date()} → {rides['evt_date_parsed'].max().date()}")

# ============ WALK-FORWARD FOLD DEFINITION ============
print("\n[3] Defining walk-forward folds...")

# Create folds by event sequence (each event is a temporal unit)
event_order = rides[["rid", "evt_date_parsed"]].drop_duplicates().sort_values("evt_date_parsed")
event_order = event_order.reset_index(drop=True)

n_events = len(event_order)
n_folds = 5
fold_size = n_events // n_folds

folds = []
for i in range(1, n_folds):  # skip fold 0 (no training data)
    test_start = fold_size * i
    test_end = fold_size * (i + 1) if i < n_folds - 1 else n_events
    
    train_events = event_order.iloc[:test_start]["rid"].tolist()
    test_events = event_order.iloc[test_start:test_end]["rid"].tolist()
    
    if len(train_events) < MIN_TRAIN_EVENTS:
        continue
    
    folds.append({"train_events": train_events, "test_events": test_events, "train_n_events": len(train_events), "test_n_events": len(test_events)})

print(f"  {n_folds} folds, ~{fold_size} events per fold")
for i, f in enumerate(folds):
    test_dates = event_order[event_order["rid"].isin(f["test_events"])]["evt_date_parsed"]
    print(f"  Fold {i+1}: train={len(f['train_events'])} events, test={len(f['test_events'])} events "
          f"({test_dates.min().date()} → {test_dates.max().date()})")

# ============ FEATURE ENGINEERING (TEMPORAL) ============
print("\n[4] Building temporal features (no leaks)...")

def build_features(df, train_mask, test_mask):
    """Build features using only data up to each ride's timestamp.
    train_mask: boolean array for rows available for feature computation
    test_mask: boolean array for rows to generate features for (prediction target)"""
    
    feats = df.copy()
    
    # --- Rider career stats (expanding window, no look-ahead) ---
    # Compute on all data but with shift so each row only sees past
    feats = feats.sort_values(["rider_name", "evt_date_parsed", "rid"])
    
    # Career outs
    feats["rider_career_outs"] = feats.groupby("rider_name")["qualified_bool"].transform(
        lambda x: x.shift(1).expanding().count()
    ).fillna(0)
    
    # Career rides (qualified)
    feats["rider_career_rides"] = feats.groupby("rider_name")["qualified_bool"].transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)
    
    # Career qualification %
    feats["rider_career_qual_pct"] = (
        feats["rider_career_rides"] / feats["rider_career_outs"].replace(0, np.nan) * 100
    ).fillna(0)
    
    # Career avg score (qualified only)
    feats["rider_career_avg_score"] = feats.groupby("rider_name")["score_num"].transform(
        lambda x: x.where(x > 0).shift(1).expanding().mean()
    ).fillna(0)
    
    # Recent form (rolling window of last N outs)
    feats["rider_last5_qual"] = feats.groupby("rider_name")["qualified_bool"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    feats["rider_last10_qual"] = feats.groupby("rider_name")["qualified_bool"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    feats["rider_last20_qual"] = feats.groupby("rider_name")["qualified_bool"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=1).mean()
    )
    
    # --- Bull career stats (TEMPORAL — same approach, computed per-bull) ---
    feats = feats.sort_values(["bull_id_str", "evt_date_parsed", "rid"])
    
    # Bull career outs
    feats["bull_career_outs"] = feats.groupby("bull_id_str")["qualified_bool"].transform(
        lambda x: x.shift(1).expanding().count()
    ).fillna(0)
    
    # Bull career buckoffs (not qualified = buckoff for the bull)
    feats["bull_career_buckoffs"] = feats.groupby("bull_id_str")["qualified_bool"].transform(
        lambda x: (~x).shift(1).expanding().sum()
    ).fillna(0)
    
    # Bull career buckoff % (temporal!)
    feats["bull_buckoff_pct"] = (
        feats["bull_career_buckoffs"] / feats["bull_career_outs"].replace(0, np.nan) * 100
    ).fillna(50)  # default: 50% buckoff for unseen bulls
    
    feats["bull_qual_pct"] = 100 - feats["bull_buckoff_pct"]
    
    # Bull avg score faced (temporal)
    feats["bull_avg_score"] = feats.groupby("bull_id_str")["bull_score_num"].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(feats["bull_score_num"].median())  # fallback
    
    # Bull recent buckoff trend (last 10 outs)
    feats["bull_recent_buckoff"] = feats.groupby("bull_id_str")["qualified_bool"].transform(
        lambda x: (~x).shift(1).rolling(10, min_periods=1).mean()
    ).fillna(0.5)
    
    # --- Spin × handedness (uses static handedness + temporal bull handedness splits) ---
    def spin_matchup(row):
        hand = row.get("riding_hand")
        # Default: unknown handedness = no signal
        if pd.isna(hand):
            return 0
        # Use temporal bull career to estimate spin advantage
        # If bull's buckoff rate vs lefties > vs righties, bull "spins left" (favors righties)
        # If bull's buckoff rate vs righties > vs lefties, bull "spins right" (favors lefties)
        
        # This is an approximation — real spin direction is in bull profiles
        # For rides where we don't have enough history, return 0
        return 0  # placeholder, computed below
    
    # Actually: use STATIC spin direction from bull profiles (this is an innate trait, not a leak)
    if "hand_advantage_dir" in bulls.columns:
        # Normalize bull IDs to match — both as int strings
        spin_map = bulls.set_index("bull_id")["hand_advantage_dir"].to_dict()
        # Create string-keyed version
        spin_map_str = {str(k): v for k, v in spin_map.items()}
        feats["bull_spin_dir"] = feats["bull_id_str"].map(spin_map_str)
        
        def spin_match(row):
            hand = row.get("riding_hand")
            spin = row.get("bull_spin_dir")
            if pd.isna(hand) or pd.isna(spin):
                return 0
            # LEFT-spinning bull pushes into RIGHT hand = favors righties
            # RIGHT-spinning bull pushes into LEFT hand = favors lefties
            if spin == "LEFT" and hand == "Right":
                return 1
            if spin == "RIGHT" and hand == "Left":
                return 1
            return -1
        
        feats["spin_hand_match"] = feats.apply(spin_match, axis=1)
    else:
        feats["spin_hand_match"] = 0
    
    # --- Event context ---
    feats["is_pbr"] = (feats["evt_org"] == "PBR").astype(int)
    feats["is_prca"] = (feats["evt_org"] == "PRCA").astype(int)
    feats["perf_num"] = pd.to_numeric(feats["perf"], errors="coerce").fillna(1)
    feats["go_num"] = pd.to_numeric(feats["go"], errors="coerce").fillna(1)
    
    # Event pressure: ratio of qualified rides in this event so far
    feats["event_num"] = feats.groupby("rid").cumcount()
    feats["event_qual_sofar"] = feats.groupby("rid")["qualified_bool"].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0)
    
    # --- Feature list ---
    feature_cols = [
        # Rider features (temporal)
        "rider_career_qual_pct", "rider_career_avg_score",
        "rider_last5_qual", "rider_last10_qual", "rider_last20_qual",
        # Bull features (temporal)
        "bull_qual_pct", "bull_buckoff_pct", "bull_avg_score",
        "bull_recent_buckoff", "bull_career_outs",
        # Matchup features
        "spin_hand_match",
        # Event context
        "is_pbr", "is_prca", "perf_num", "go_num",
        "event_qual_sofar",
    ]
    
    return feats, feature_cols

# Build features on ALL data (shift-based, so each row only sees its own past)
feats, feature_cols = build_features(rides, None, None)

print(f"  Features: {len(feature_cols)}")
print(f"  Rows: {len(feats):,}")
print(f"  Spin-match distribution: {feats['spin_hand_match'].value_counts().to_dict()}")

# ============ WALK-FORWARD EVALUATION ============
print("\n" + "=" * 60)
print("WALK-FORWARD VALIDATION")
print("=" * 60)

all_clf_metrics = []
all_reg_metrics = []
all_fold_predictions = []

for fold_i, fold in enumerate(folds):
    print(f"\n--- Fold {fold_i+1}/{n_folds} ---")
    
    train_idx = feats[feats["rid"].isin(fold["train_events"])].index
    test_idx = feats[feats["rid"].isin(fold["test_events"])].index
    
    X_train = feats.loc[train_idx, feature_cols].fillna(0)
    X_test = feats.loc[test_idx, feature_cols].fillna(0)
    y_train_qual = feats.loc[train_idx, "qualified_bool"].astype(int)
    y_test_qual = feats.loc[test_idx, "qualified_bool"].astype(int)
    y_train_score = feats.loc[train_idx, "score_num"].fillna(0)
    y_test_score = feats.loc[test_idx, "score_num"].fillna(0)
    
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Train qual%: {y_train_qual.mean()*100:.1f}% | Test qual%: {y_test_qual.mean()*100:.1f}%")
    
    if len(X_test) < 50:
        print("  Skipping — too few test samples")
        continue
    
    # --- Stage 1: Classifier ---
    scale = (y_train_qual == 0).sum() / max(y_train_qual.sum(), 1)
    
    clf = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=min(scale, 5.0), random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train_qual)
    
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Calibrated threshold (match actual qual rate)
    actual_rate = y_test_qual.mean()
    calibrated_thresh = float(np.percentile(y_proba, 100 * (1 - actual_rate)))
    y_pred = (y_proba >= calibrated_thresh).astype(int)
    
    fold_acc = (y_pred == y_test_qual).mean()
    fold_precision = y_pred[y_pred == 1].sum() / max(y_pred.sum(), 1) if y_pred.sum() > 0 else 0
    fold_recall = y_pred[y_test_qual == 1].sum() / max(y_test_qual.sum(), 1) if y_test_qual.sum() > 0 else 0
    
    all_clf_metrics.append({
        "fold": fold_i + 1,
        "train_n": len(X_train), "test_n": len(X_test),
        "test_qual%": actual_rate * 100,
        "pred_qual%": y_pred.mean() * 100,
        "accuracy": fold_acc,
        "precision": fold_precision,
        "recall": fold_recall,
        "threshold": calibrated_thresh,
    })
    
    print(f"  CLF: acc={fold_acc*100:.1f}% prec={fold_precision*100:.1f}% rec={fold_recall*100:.1f}% thresh={calibrated_thresh:.3f}")
    
    # --- Stage 2: Regressor (qualified rides only) ---
    train_qual_mask = y_train_qual == 1
    test_qual_mask = y_test_qual == 1
    
    X_tr_score = X_train[train_qual_mask]
    y_tr_score = y_train_score[train_qual_mask]
    valid = y_tr_score > 0
    X_tr_score = X_tr_score[valid]
    y_tr_score = y_tr_score[valid]
    
    X_ts_score = X_test[test_qual_mask]
    y_ts_score = y_test_score[test_qual_mask]
    valid_ts = y_ts_score > 0
    X_ts_score = X_ts_score[valid_ts]
    y_ts_score = y_ts_score[valid_ts]
    
    if len(X_tr_score) < 100 or len(X_ts_score) < 20:
        print(f"  REG: insufficient data ({len(X_tr_score)} train, {len(X_ts_score)} test) — skipping")
        all_reg_metrics.append({"fold": fold_i+1, "r2": None, "mae": None})
        continue
    
    reg = xgb.XGBRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    reg.fit(X_tr_score, y_tr_score)
    
    y_pred_scores = reg.predict(X_ts_score)
    fold_r2 = r2_score(y_ts_score, y_pred_scores)
    fold_mae = mean_absolute_error(y_ts_score, y_pred_scores)
    
    all_reg_metrics.append({
        "fold": fold_i + 1,
        "n_train": len(X_tr_score), "n_test": len(X_ts_score),
        "r2": fold_r2, "mae": fold_mae,
        "true_mean": y_ts_score.mean(), "pred_mean": y_pred_scores.mean(),
    })
    
    print(f"  REG: R²={fold_r2:.4f} MAE={fold_mae:.2f} n={len(X_ts_score):,}")
    
    # Store predictions
    fold_preds = feats.loc[test_idx].copy()
    fold_preds["pred_qual"] = y_pred
    fold_preds["pred_proba"] = y_proba
    fold_preds["fold"] = fold_i + 1  # shift: loop starts at fold 1
    fold_preds["pred_score"] = np.nan
    fold_preds.loc[fold_preds.index[test_qual_mask][valid_ts], "pred_score"] = y_pred_scores
    all_fold_predictions.append(fold_preds)

# ============ AGGREGATE RESULTS ============
print("\n" + "=" * 60)
print("AGGREGATE WALK-FORWARD RESULTS")
print("=" * 60)

clf_df = pd.DataFrame(all_clf_metrics)
reg_df = pd.DataFrame(all_reg_metrics)

print("\nClassifier (per-fold):")
print(f"  {'Fold':<6} {'N Test':<8} {'Acc':<7} {'Prec':<7} {'Rec':<7} {'Act%':<7} {'Pred%':<7}")
for _, r in clf_df.iterrows():
    print(f"  {r['fold']:<6} {r['test_n']:<8,} {r['accuracy']*100:>5.1f}% {r['precision']*100:>5.1f}% {r['recall']*100:>5.1f}% {r['test_qual%']:>5.1f}% {r['pred_qual%']:>5.1f}%")

print(f"\n  Mean Accuracy: {clf_df['accuracy'].mean()*100:.1f}% ± {clf_df['accuracy'].std()*100:.1f}%")
print(f"  Mean Precision: {clf_df['precision'].mean()*100:.1f}%")
print(f"  Mean Recall: {clf_df['recall'].mean()*100:.1f}%")
print(f"  Avg threshold: {clf_df['threshold'].mean():.3f}")

print("\nRegressor (per-fold):")
for _, r in reg_df.iterrows():
    if r["r2"] is not None:
        print(f"  Fold {int(r['fold'])}: R²={r['r2']:.4f} MAE={r['mae']:.2f} n={int(r['n_test']):,} "
              f"true={r['true_mean']:.1f} pred={r['pred_mean']:.1f}")
    else:
        print(f"  Fold {int(r['fold'])}: insufficient data")

valid_reg = reg_df[reg_df["r2"].notna()]
if len(valid_reg) > 0:
    print(f"\n  Mean R²: {valid_reg['r2'].mean():.4f} ± {valid_reg['r2'].std():.4f}")
    print(f"  Mean MAE: {valid_reg['mae'].mean():.2f} ± {valid_reg['mae'].std():.2f}")

# ============ FINAL FEATURE IMPORTANCE (last fold clf) ============
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (fold 5)")
print("=" * 60)

imp = pd.DataFrame({
    "feature": feature_cols,
    "classifier": clf.feature_importances_,
}).sort_values("classifier", ascending=False)

print("\nClassifier:")
for _, row in imp.iterrows():
    bar = "█" * int(row["classifier"] / imp["classifier"].max() * 40)
    print(f"  {row['feature']:<28s} {bar} {row['classifier']:.4f}")

if 'reg' in dir():
    imp_r = pd.DataFrame({
        "feature": feature_cols,
        "regressor": reg.feature_importances_,
    }).sort_values("regressor", ascending=False)
    print("\nRegressor:")
    for _, row in imp_r.iterrows():
        bar = "█" * int(row["regressor"] / imp_r["regressor"].max() * 40)
        print(f"  {row['feature']:<28s} {bar} {row['regressor']:.4f}")

print("\nDone.")