#!/usr/bin/env python3
"""
DataSpur Feature Pipeline — Bull Riding
Builds ML-ready features from Probullstats data.
Two-stage model: (1) qualified ride classifier, (2) score regressor.
"""
import csv
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

DATA_DIR = Path.home() / "dataspur" / "data"
RIDES_PATH = DATA_DIR / "rides.csv"

# ── LOAD ──────────────────────────────────────────
def load_rides():
    df = pd.read_csv(RIDES_PATH)
    # Bull riding only (user wants separate models per event)
    df = df[df["event_type"] == "BR"].copy()
    print(f"Loaded {len(df)} bull riding outs")
    
    # Parse qualified: may be boolean, string, or numeric
    if "qualified" in df.columns:
        df["qualified"] = df["qualified"].astype(str).str.strip().map({
            "True": True, "true": True, "1": True, "1.0": True, "yes": True,
            "False": False, "false": False, "0": False, "0.0": False, "": False, "nan": False,
        }).fillna(False)
    for col in ["score", "bull_score", "ride_time", "stat_avg_bull_score",
                "stat_records", "stat_official_outs", "stat_qualified_rides", "stat_qualified_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Parse date
    df["evt_date"] = df["evt_date"].fillna("")
    
    # Parse judge scores (e.g., "22.50|22.00")
    def parse_judge_scores(js):
        if pd.isna(js) or not js:
            return None, None
        parts = str(js).split("|")
        try:
            j1 = float(parts[0]) if len(parts) > 0 else None
            j2 = float(parts[1]) if len(parts) > 1 else None
            return j1, j2
        except:
            return None, None
    
    judge_data = df["judge_scores"].apply(parse_judge_scores)
    df["judge1"] = judge_data.apply(lambda x: x[0])
    df["judge2"] = judge_data.apply(lambda x: x[1])
    
    # Sort chronologically for walk-forward
    df = df.sort_values(["evt_date", "rid", "perf", "go"]).reset_index(drop=True)
    
    return df


# ── RIDER FEATURES ────────────────────────────────
def build_rider_features(df):
    """Career and recent-form features for each rider."""
    riders = defaultdict(lambda: {
        "career_outs": 0, "career_rides": 0, "career_scores": [],
        "career_bull_scores": [], "career_bull_ids": set(),
        "recent_5": [], "recent_10": [], "recent_20": [],
    })
    
    rider_features = []
    
    for idx, row in df.iterrows():
        name = row["rider_name"]
        if not name or pd.isna(name):
            rider_features.append({})
            continue
        
        r = riders[name]
        
        # Features BEFORE this ride (look-ahead bias free)
        feats = {}
        
        # Career stats
        if r["career_outs"] > 0:
            feats["rider_career_outs"] = r["career_outs"]
            feats["rider_career_qual_pct"] = r["career_rides"] / r["career_outs"]
            feats["rider_career_avg_score"] = np.mean(r["career_scores"])
            feats["rider_career_max_score"] = max(r["career_scores"]) if r["career_scores"] else 0
            feats["rider_career_avg_bull_faced"] = np.mean(r["career_bull_scores"]) if r["career_bull_scores"] else 0
            feats["rider_career_unique_bulls"] = len(r["career_bull_ids"])
        else:
            feats["rider_career_outs"] = 0
            feats["rider_career_qual_pct"] = 0
            feats["rider_career_avg_score"] = 0
            feats["rider_career_max_score"] = 0
            feats["rider_career_avg_bull_faced"] = 0
            feats["rider_career_unique_bulls"] = 0
        
        # Recent form (last 5, 10, 20)
        for window, key in [(5, "5"), (10, "10"), (20, "20")]:
            recent = r[f"recent_{key}"]
            if recent:
                feats[f"rider_last{key}_qual_pct"] = sum(1 for s in recent if s and s > 0) / len(recent)
                feats[f"rider_last{key}_avg_score"] = np.mean([s for s in recent if s and s > 0]) if any(s and s > 0 for s in recent) else 0
            else:
                feats[f"rider_last{key}_qual_pct"] = 0
                feats[f"rider_last{key}_avg_score"] = 0
        
        rider_features.append(feats)
        
        # UPDATE rider state with this ride's outcome
        r["career_outs"] += 1
        if row["qualified"]:
            r["career_rides"] += 1
            if not pd.isna(row["score"]):
                r["career_scores"].append(row["score"])
        if not pd.isna(row["bull_score"]):
            r["career_bull_scores"].append(row["bull_score"])
        if row["bull_id"]:
            r["career_bull_ids"].add(row["bull_id"])
        
        # Update recent windows
        outcome = row["score"] if row["qualified"] else 0
        for window in [5, 10, 20]:
            key = str(window)
            r[f"recent_{key}"].append(outcome)
            if len(r[f"recent_{key}"]) > window:
                r[f"recent_{key}"].pop(0)
    
    return pd.DataFrame(rider_features)


# ── BULL FEATURES ─────────────────────────────────
def build_bull_features(df):
    """Historical bull performance before each out."""
    bulls = defaultdict(lambda: {
        "career_outs": 0, "career_scores": [],
        "career_ridden": 0, "recent_scores": [],
    })
    
    bull_features = []
    
    for idx, row in df.iterrows():
        bull_id = row["bull_id"]
        if not bull_id or pd.isna(bull_id):
            bull_features.append({})
            continue
        
        b = bulls[bull_id]
        feats = {}
        
        if b["career_outs"] > 0:
            feats["bull_career_outs"] = b["career_outs"]
            feats["bull_career_avg_score"] = np.mean(b["career_scores"])
            feats["bull_career_max_score"] = max(b["career_scores"]) if b["career_scores"] else 0
            feats["bull_career_buckoff_pct"] = (b["career_outs"] - b["career_ridden"]) / b["career_outs"]
            feats["bull_career_times_ridden"] = b["career_ridden"]
            feats["bull_recent_avg_score"] = np.mean(b["recent_scores"]) if b["recent_scores"] else 0
        else:
            feats["bull_career_outs"] = 0
            feats["bull_career_avg_score"] = 0
            feats["bull_career_max_score"] = 0
            feats["bull_career_buckoff_pct"] = 0
            feats["bull_career_times_ridden"] = 0
            feats["bull_recent_avg_score"] = 0
        
        bull_features.append(feats)
        
        # Update bull state
        b["career_outs"] += 1
        if not pd.isna(row["bull_score"]):
            b["career_scores"].append(row["bull_score"])
            b["recent_scores"].append(row["bull_score"])
            if len(b["recent_scores"]) > 5:
                b["recent_scores"].pop(0)
        if row["qualified"]:
            b["career_ridden"] += 1
    
    return pd.DataFrame(bull_features)


# ── EVENT / CONTEXT FEATURES ──────────────────────
def build_event_features(df):
    """Event-level and round-level features."""
    feats = pd.DataFrame(index=df.index)
    
    # Organization
    feats["is_pbr"] = (df["evt_org"] == "PBR").astype(int)
    feats["is_prca"] = (df["evt_org"] == "PRCA").astype(int)
    feats["is_cpra"] = (df["evt_org"] == "CPRA").astype(int)
    
    # Tour class
    feats["is_utb"] = (df["evt_tour_class"].str.contains("UTB|Premier|BFTS", na=False)).astype(int)
    feats["is_team_series"] = (df["evt_tour_class"] == "Team Series").astype(int)
    feats["is_prca_rodeo"] = (df["evt_tour_class"] == "PRCA").astype(int)
    
    # Round features
    feats["round"] = df["go"].fillna(1).astype(int)
    feats["is_championship_round"] = (feats["round"] >= 3).astype(int)
    feats["is_short_go"] = (feats["round"] >= 2).astype(int)
    feats["is_long_round"] = (feats["round"] == 1).astype(int)
    
    # Event stats (from Probullstats)
    if "stat_avg_bull_score" in df.columns:
        feats["event_avg_bull_score"] = df["stat_avg_bull_score"].fillna(0)
    if "stat_qualified_pct" in df.columns:
        feats["event_qual_pct"] = df["stat_qualified_pct"].fillna(0)
    if "stat_records" in df.columns:
        feats["event_size"] = df["stat_records"].fillna(0)
    
    # Event note signals pressure
    feats["is_xtreme_bulls"] = df["evt_note"].str.contains("Xtreme|PRCAX", na=False).astype(int)
    feats["is_finals"] = df["evt_note"].str.contains("Finals|NFR", na=False).astype(int)
    feats["is_15_15"] = df["evt_note"].str.contains("15/15", na=False).astype(int)
    
    # Month (seasonality)
    try:
        dates = pd.to_datetime(df["evt_date"], format="mixed", errors="coerce")
        feats["month"] = dates.dt.month.fillna(6).astype(int)
        feats["is_winter"] = feats["month"].isin([12, 1, 2]).astype(int)
        feats["is_summer"] = feats["month"].isin([6, 7, 8]).astype(int)
    except:
        feats["month"] = 6
        feats["is_winter"] = 0
        feats["is_summer"] = 0
    
    return feats


# ── MATCH FEATURES ────────────────────────────────
def build_match_features(df):
    """Features about the specific rider-bull matchup."""
    feats = pd.DataFrame(index=df.index)
    
    # Judge score agreement (higher agreement = more consistent scoring)
    if "judge1" in df.columns and "judge2" in df.columns:
        j1 = df["judge1"].fillna(0)
        j2 = df["judge2"].fillna(0)
        feats["judge_agreement"] = 1 - (abs(j1 - j2) / (j1 + j2 + 0.01))
        feats["judge_avg"] = (j1 + j2) / 2
        feats["judge_range"] = abs(j1 - j2)
    
    # Did they get the 8 seconds?
    if "ride_time" in df.columns:
        feats["got_8_seconds"] = (df["ride_time"] >= 7.9).fillna(0).astype(int)
    
    # Performance order (earlier in the round = less pressure)
    feats["perf_order"] = df.groupby(["rid", "go"]).cumcount()
    
    return feats


# ── MAIN ───────────────────────────────────────────
def main():
    print(f"DataSpur Feature Pipeline — {datetime.now()}")
    print("=" * 50)
    
    # Load
    df = load_rides()
    
    # Build features
    print("Building rider features...")
    rider_feats = build_rider_features(df)
    
    print("Building bull features...")
    bull_feats = build_bull_features(df)
    
    print("Building event features...")
    event_feats = build_event_features(df)
    
    print("Building match features...")
    match_feats = build_match_features(df)
    
    # Combine
    feature_df = pd.concat([rider_feats, bull_feats, event_feats, match_feats], axis=1)
    
    # Targets
    feature_df["target_qualified"] = df["qualified"].astype(int)
    feature_df["target_score"] = df["score"].where(df["qualified"], np.nan)
    feature_df["rider_name"] = df["rider_name"]
    feature_df["bull_name"] = df["bull_name"]
    feature_df["rid"] = df["rid"]
    feature_df["evt_date"] = df["evt_date"]
    feature_df["evt_org"] = df["evt_org"]
    
    # Drop rows with no targets (non-bull riding)
    feature_df = feature_df[feature_df["target_qualified"].notna()].copy()
    
    # Fill remaining NaN with 0
    feature_df = feature_df.fillna(0)
    
    print(f"\nFeature matrix: {feature_df.shape[0]:,} rows × {feature_df.shape[1]} columns")
    print(f"Qualified rides: {feature_df['target_qualified'].sum():,} ({feature_df['target_qualified'].mean()*100:.1f}%)")
    print(f"Mean score (qualified): {feature_df['target_score'].dropna().mean():.1f}")
    
    # Temporal split (80% train, 20% test by time)
    split_idx = int(len(feature_df) * 0.8)
    train = feature_df.iloc[:split_idx]
    test = feature_df.iloc[split_idx:]
    print(f"\nTemporal split: train={len(train):,} test={len(test):,}")
    print(f"Train qualified pct: {train['target_qualified'].mean()*100:.1f}%")
    print(f"Test qualified pct: {test['target_qualified'].mean()*100:.1f}%")
    
    # Save
    out = DATA_DIR / "features_br.parquet"
    feature_df.to_parquet(out, index=False)
    print(f"\nSaved features to {out}")
    
    # Feature importance preview (simple correlation with targets)
    # Drop data-leak features: got_8_seconds IS the target, ride_time encodes it
    feature_cols = [c for c in feature_df.columns 
                    if c not in ["target_qualified", "target_score", "rider_name", 
                                 "bull_name", "rid", "evt_date", "evt_org",
                                 "got_8_seconds", "judge_avg", "judge_range"]]
    
    corr_qual = feature_df[feature_cols + ["target_qualified"]].corr()["target_qualified"].drop("target_qualified")
    print("\nTop features correlated with qualified ride:")
    for feat, corr in corr_qual.abs().sort_values(ascending=False).head(10).items():
        direction = "+" if corr_qual[feat] > 0 else "-"
        print(f"  {direction} {feat}: {corr:.4f}")
    
    corr_score = feature_df[feature_cols + ["target_score"]].corr()["target_score"].drop("target_score")
    print("\nTop features correlated with ride score:")
    for feat, corr in corr_score.abs().sort_values(ascending=False).head(10).items():
        direction = "+" if corr_score[feat] > 0 else "-"
        print(f"  {direction} {feat}: {corr:.4f}")
    
    return feature_df, train, test


if __name__ == "__main__":
    df, train, test = main()