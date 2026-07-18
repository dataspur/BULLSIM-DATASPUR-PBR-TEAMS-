#!/usr/bin/env python3
"""
DataSpur Rider Weighting System
================================
Computes individualized rider profile weights that capture:
1. Volatility Index      – score variance, streak unpredictability
2. Post-Layoff Decay      – recovery rate after injury/absence
3. Pressure Factor        – performance delta in high-stakes rounds
4. Recent Form Weight     – personalized optimal recency weighting
5. Cold Streak Threshold  – outs before declaring a slump
6. Trend Momentum         – direction and strength of recent trajectory

These weights plug into the existing feature pipeline to modulate:
- How much recent form matters vs career averages (per-rider)
- How to discount/boost predictions in pressure situations
- How to handle a rider coming off a layoff
- How volatile a rider's predictions should be treated

Output: rider_weights.json + rider_weights.csv for top-N riders
"""

import csv
import json
import math
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────
DATA_DIR = Path.home() / "dataspur" / "data"
RIDES_PATH = DATA_DIR / "rides.csv"
OUT_JSON = DATA_DIR / "rider_weights.json"
OUT_CSV = DATA_DIR / "rider_weights.csv"
OUT_HTML = DATA_DIR / "rider_weights_report.html"

LAYOFF_THRESHOLD_DAYS = 60  # days without a ride = likely injury/layoff
MIN_RIDES_FOR_PROFILE = 10   # need at least this many rides for reliable metrics
TOP_N = 20                    # output profiles for top N riders by ride count


# ╔══════════════════════════════════════════════════════════════╗
# ║  DATA LOADING & CLEANING                                    ║
# ╚══════════════════════════════════════════════════════════════╝

def load_and_clean():
    """Load rides data, parse dates, sort chronologically."""
    rides = pd.read_csv(RIDES_PATH, low_memory=False)
    rides = rides[rides["event_type"] == "BR"].copy()

    # Parse qualified
    rides["qualified_bool"] = (
        rides["qualified"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0", "yes"])
    )

    # Parse scores
    rides["score_num"] = pd.to_numeric(rides["score"], errors="coerce")
    rides["bull_score_num"] = pd.to_numeric(rides["bull_score"], errors="coerce")

    # Parse dates
    # Format is like "Jul, 2026 (NFR Open)" or "Jul, 2026"
    def parse_date(d):
        if pd.isna(d):
            return pd.NaT
        d = str(d)
        # Extract month + year
        import re
        m = re.search(r"(\w{3}),?\s*(\d{4})", d)
        if m:
            return pd.Timestamp(f"{m.group(2)}-{m.group(1)}-15")
        return pd.NaT

    rides["date_parsed"] = rides["evt_date"].apply(parse_date)
    rides = rides.dropna(subset=["date_parsed"]).sort_values(
        ["rider_name", "date_parsed", "rid"]
    ).reset_index(drop=True)

    # Parse go/round as numeric
    rides["go_num"] = pd.to_numeric(rides["go"], errors="coerce").fillna(1).astype(int)
    rides["perf_num"] = pd.to_numeric(rides["perf"], errors="coerce").fillna(1).astype(int)

    print(f"Loaded {len(rides):,} bull riding outs")
    print(f"  Riders: {rides['rider_name'].nunique():,}")
    print(f"  Date range: {rides['date_parsed'].min().date()} → {rides['date_parsed'].max().date()}")
    print(f"  Qualification rate: {rides['qualified_bool'].mean()*100:.1f}%")

    return rides


# ╔══════════════════════════════════════════════════════════════╗
# ║  PRESSURE / ROUND CLASSIFICATION                           ║
# ╚══════════════════════════════════════════════════════════════╝

def classify_pressure_context(rides):
    """Tag each ride with its pressure context."""
    rides = rides.copy()

    # Short round: go >= 2 (after the long round)
    rides["is_short_round"] = (rides["go_num"] >= 2).astype(int)

    # Finals/Championship events
    # Match: Finals, NFR, CFR, Championship, Permit Finals, Challenger Finals
    note_str = rides["evt_note"].fillna("").astype(str).str.lower()
    rides["is_finals"] = note_str.str.contains(
        r"finals|nfr|cfr|championship", na=False
    ).astype(int)

    # PBR events (highest level, excluding Team Series)
    rides["is_premier"] = (
        (rides["evt_tour_class"].fillna("").astype(str) == "PBR")
    ).astype(int)

    # Team Series (unique format, pressure varies)
    rides["is_team_series"] = (rides["evt_tour_class"] == "Team Series").astype(int)

    # Regular round: go=1 AND not finals AND not premier (baseline)
    rides["is_baseline_round"] = (
        (rides["go_num"] == 1) &
        (rides["is_finals"] == 0) &
        (rides["is_premier"] == 0) &
        (rides["is_team_series"] == 0)
    ).astype(int)

    # High-pressure: short round, finals, premier, or team series
    rides["is_high_pressure"] = (
        (rides["is_short_round"] == 1) |
        (rides["is_finals"] == 1) |
        (rides["is_premier"] == 1) |
        (rides["is_team_series"] == 1)
    ).astype(int)

    return rides


# ╔══════════════════════════════════════════════════════════════╗
# ║  RIDER METRICS COMPUTATION                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_all_rider_metrics(rides):
    """
    Compute all 6 rider metrics by iterating through each rider's
    chronological ride history.
    """
    grouped = rides.groupby("rider_name")
    results = {}

    for rider_name, group in grouped:
        if len(group) < MIN_RIDES_FOR_PROFILE:
            continue

        # Sort chronologically (already sorted but be safe)
        group = group.sort_values("date_parsed").reset_index(drop=True)

        scores = group["score_num"].values
        qualified = group["qualified_bool"].values
        dates = group["date_parsed"].values
        go_nums = group["go_num"].values
        is_high_pressure = group["is_high_pressure"].values
        is_baseline = group["is_baseline_round"].values

        # Valid scores (non-NaN, positive)
        valid_mask = ~np.isnan(scores) & (scores > 0)
        n_outs = len(group)

        # ── 1. VOLATILITY INDEX ────────────────────────────
        volatility = compute_volatility_index(scores, qualified, valid_mask)

        # ── 2. POST-LAYOFF DECAY RATE ──────────────────────
        layoff_decay = compute_layoff_decay(dates, qualified, valid_mask, scores)

        # ── 3. PRESSURE FACTOR ─────────────────────────────
        pressure = compute_pressure_factor(
            scores, qualified, valid_mask, is_high_pressure, is_baseline
        )

        # ── 4. RECENT FORM WEIGHT ──────────────────────────
        recent_weight = compute_recent_form_weight(
            scores, qualified, valid_mask
        )

        # ── 5. COLD STREAK THRESHOLD ───────────────────────
        cold_streak = compute_cold_streak_threshold(qualified)

        # ── 6. TREND MOMENTUM ──────────────────────────────
        trend = compute_trend_momentum(scores, qualified, valid_mask, n_outs)

        results[rider_name] = {
            "rider": rider_name,
            "total_outs": n_outs,
            "career_qual_pct": round(qualified.sum() / n_outs * 100, 1),
            "career_avg_score": round(np.mean(scores[valid_mask]), 1) if valid_mask.any() else 0,
            "volatility_index": round(volatility, 4),
            "post_layoff_decay_rate": round(layoff_decay, 4),
            "pressure_factor": round(pressure, 4),
            "recent_form_weight": round(recent_weight, 4),
            "cold_streak_threshold": cold_streak,
            "trend_momentum": round(trend, 4),
        }

    return results


# ── 1. VOLATILITY INDEX ──────────────────────────────────────
def compute_volatility_index(scores, qualified, valid_mask):
    """
    Volatility Index (0-1 scale, higher = more unpredictable).

    Combines:
    - Score variance (normalized by typical bull riding std ~8 pts)
    - Streak alternation frequency (how often they switch good → bad)
    - Consistency of bull quality faced (do they get wildly different bulls?)

    Formula:
    volatility = 0.4 * score_var_component + 0.35 * streak_alternation + 0.25 * bull_var_component
    Then clamped to [0.05, 0.95] and scaled to [0, 1].
    """
    n = len(qualified)

    # Component A: Score variance
    if valid_mask.sum() >= 3:
        score_std = np.std(scores[valid_mask])
        # Typical bull riding std is ~5-7 pts. Score over 10 is high volatility.
        score_var_component = min(score_std / 10.0, 1.0)
    else:
        score_var_component = 0.5

    # Component B: Streak alternation (do they oscillate?)
    # Count transitions: qual→buckoff or buckoff→qual
    transitions = 0
    for i in range(1, n):
        if qualified[i] != qualified[i-1]:
            transitions += 1
    max_transitions = n - 1
    streak_alt = transitions / max(max_transitions, 1)
    # High alternation → high volatility. Low alternation → consistent (either good or bad)
    streak_component = abs(streak_alt - 0.5) * 2  # 0.5 alternation = max unpredictability
    # Actually: streak_alt of 0.5 means they alternate every ride = unpredictable
    # streak_alt of 0 or 1 = always same = predictable
    # So: volatility increases with streak_alt up to 0.5, then decreases
    streak_component = 1.0 - abs(streak_alt - 0.5) * 2

    # Component C: Bull quality consistency (using score as proxy for bull difficulty)
    # More variance in bull_scores faced = more extraneous variance
    # We'll use the score variance as a rough proxy for combined rider+bull variance
    # and assume typical bull variance contributes ~60%
    bull_component = min(score_var_component * 0.7, 1.0)

    # Weighted combination
    raw = 0.40 * score_var_component + 0.35 * streak_component + 0.25 * bull_component

    # Clamp and scale
    return max(0.0, min(1.0, raw))


# ── 2. POST-LAYOFF DECAY RATE ────────────────────────────────
def compute_layoff_decay(dates, qualified, valid_mask, scores):
    """
    Post-Layoff Decay Rate (0-1 scale, higher = recovers faster).

    Methodology:
    1. Identify gaps between consecutive rides > LAYOFF_THRESHOLD_DAYS
    2. For each layoff, track performance in the N rides after return
    3. Fit an exponential decay: qual_rate(outs_after) = baseline * (1 - e^(-decay*outs))
    4. Decay rate is the fitted lambda parameter

    If not enough layoffs to estimate, default to population median.
    """
    n = len(dates)
    layoff_performances = []  # list of (outs_after_return, qualified?) tuples

    for i in range(1, n):
        gap_days = (dates[i] - dates[i-1]) / np.timedelta64(1, 'D')
        if gap_days > LAYOFF_THRESHOLD_DAYS:
            # This is a return from layoff. Track next 10 rides.
            for j in range(min(10, n - i)):
                layoff_performances.append({
                    "outs_after": j + 1,
                    "qualified": qualified[i + j],
                    "score": scores[i + j] if valid_mask[i + j] else 0,
                })

    if len(layoff_performances) == 0:
        # No layoffs detected → neutral recovery speed (not applicable)
        return 0.60  # default: moderate recovery, unbiased

    # Compute average qualification rate by outs-after-return
    lf = pd.DataFrame(layoff_performances)
    qual_by_out = lf.groupby("outs_after")["qualified"].mean()

    # Baseline qual rate (career average, excluding layoff return period)
    career_qual = qualified.mean()

    # Check if post-layoff qual is already at or above career average
    # (meaning the rider recovers instantly — high decay rate)
    post_layoff_qual = lf["qualified"].mean()
    if post_layoff_qual >= career_qual * 0.90:
        # Recovery is fast — performance is already at baseline
        return min(1.0, 0.70 + 0.30 * (post_layoff_qual / max(career_qual, 0.01)))

    # Fit: qual_at_out_k = career_qual * (1 - e^(-decay * k))
    # Linearize: -ln(1 - qual_at_out_k/career_qual) = decay * k
    # Use only outs where qual < career_qual (still recovering)
    valid_points = []
    for k, q in qual_by_out.items():
        if career_qual > 0 and q < career_qual * 0.95:
            ratio = max(q / career_qual, 0.01)
            y = -math.log(max(1 - ratio, 0.001))
            valid_points.append((k, y))

    if len(valid_points) >= 2:
        ks = np.array([p[0] for p in valid_points])
        ys = np.array([p[1] for p in valid_points])
        # Fit slope (decay rate) via least squares through origin
        decay_rate = np.sum(ks * ys) / max(np.sum(ks * ks), 1e-10)
    elif len(valid_points) == 1:
        decay_rate = valid_points[0][1] / valid_points[0][0]
    else:
        # All post-layoff rides at or near baseline → fast recovery
        decay_rate = 1.0

    # Map decay rate to 0-1 scale (higher = faster recovery)
    # Typical decay rates: 0.1 (slow, takes ~10 outs) to 1.0 (immediate)
    normalized = min(decay_rate / 0.5, 1.0)

    return normalized


# ── 3. PRESSURE FACTOR ──────────────────────────────────────
def compute_pressure_factor(scores, qualified, valid_mask, is_high_pressure, is_baseline):
    """
    Pressure Factor: performance delta in high-pressure vs baseline.
    Negative = worse under pressure; positive = thrives.

    If a rider has insufficient baseline or pressure rides, falls back to
    comparing against population-average pressure delta for riders at the same level.
    Then shrinks toward 0 based on sample size (empirical Bayes).
    """
    pressure_mask = is_high_pressure == 1
    baseline_mask = is_baseline == 1

    n_pressure = pressure_mask.sum()
    n_baseline = baseline_mask.sum()

    if n_pressure < 3 or n_baseline < 3:
        # Try comparing pressure vs all non-pressure rides instead
        non_pressure_mask = ~pressure_mask
        n_nonpressure = non_pressure_mask.sum()
        if n_pressure >= 3 and n_nonpressure >= 3:
            qual_pressure = qualified[pressure_mask].mean()
            qual_other = qualified[non_pressure_mask].mean()
            qual_delta = qual_pressure - qual_other

            score_pressure = scores[pressure_mask & valid_mask]
            score_other = scores[non_pressure_mask & valid_mask]
            score_delta = 0.0
            if len(score_pressure) >= 2 and len(score_other) >= 2:
                score_delta = np.mean(score_pressure) - np.mean(score_other)
        else:
            return 0.0  # truly insufficient data
    else:
        qual_pressure = qualified[pressure_mask].mean()
        qual_baseline = qualified[baseline_mask].mean()
        qual_delta = qual_pressure - qual_baseline

        score_pressure = scores[pressure_mask & valid_mask]
        score_baseline = scores[baseline_mask & valid_mask]
        score_delta = 0.0
        if len(score_pressure) >= 2 and len(score_baseline) >= 2:
            score_delta = np.mean(score_pressure) - np.mean(score_baseline)

    # Normalize: qual delta of 15% is large, score delta of 8 pts is large
    qual_norm = qual_delta / 0.15  # 15pp delta = 1.0
    score_norm = score_delta / 8.0  # 8pt delta = 1.0

    # Weighted combination (qual rate is more reliable with small N)
    raw = 0.5 * qual_norm + 0.5 * score_norm

    # Shrink toward 0 based on sample size (empirical Bayes)
    # With < 10 pressure rides, shrink heavily toward 0
    min_n = min(n_pressure, n_baseline if n_baseline >= 3 else n_pressure)
    shrinkage = min(1.0, min_n / 20.0)  # full weight at 20+ rides in each condition
    raw = raw * shrinkage

    # Clamp to [-1, 1]
    raw = max(-1.0, min(1.0, raw))

    return raw * 0.5


# ── 4. RECENT FORM WEIGHT ──────────────────────────────────
def compute_recent_form_weight(scores, qualified, valid_mask):
    """
    Recent Form Weight: how much recent rides predict the next ride vs career avg.

    For each ride i with sufficient history (>15 rides):
    - Predict next ride using: (a) career average, (b) recent-N average
    - Compute which would have been a better predictor
    - The optimal weight w minimizes: (next - [w*recent + (1-w)*career])^2

    Regularization: shrink toward population prior (0.3) based on sample size.
    Returns: optimal w (0-1) where higher = recent form matters more.
    """
    n = len(qualified)
    if n < 20:
        # For very few rides, use population prior with strong shrinkage
        return 0.30

    min_history = 15
    if n <= min_history:
        return 0.30

    errors_by_w = {w: 0.0 for w in np.arange(0.0, 1.05, 0.05)}
    count = 0

    for i in range(min_history, n):
        # Career avg up to i-1
        past_qual = qualified[:i].mean()

        # Recent 5 rides
        recent_n = min(5, i)
        recent_qual = qualified[i-recent_n:i].mean()

        # Actual outcome
        actual = 1.0 if qualified[i] else 0.0

        for w in errors_by_w:
            pred = w * recent_qual + (1 - w) * past_qual
            errors_by_w[w] += (actual - pred) ** 2
        count += 1

    if count < 5:
        return 0.30

    # Find w with minimum error
    best_w = min(errors_by_w, key=errors_by_w.get)

    # Shrink toward population prior (0.3) based on number of prediction trials
    # More trials = more trust in fitted w
    prior = 0.30
    effective_n = count
    # James-Stein style shrinkage: w_shrunk = prior + (1 - lambda) * (w - prior)
    # where lambda = 1 / (1 + effective_n / 10) — strong shrinkage for small N
    lambda_shrink = 1.0 / (1.0 + effective_n / 10.0)
    best_w = prior + (1.0 - lambda_shrink) * (best_w - prior)

    return max(0.0, min(1.0, best_w))


# ── 5. COLD STREAK THRESHOLD ───────────────────────────────
def compute_cold_streak_threshold(qualified):
    """
    Cold Streak Threshold: number of consecutive buckoffs before we call it a slump.

    Uses run-length encoding of consecutive 0s (buckoffs).
    Threshold = median of non-trivial (>=2) buckoff streak lengths.
    If no streaks >= 2, defaults to 5.
    """
    streaks = []
    current = 0
    for q in qualified:
        if not q:
            current += 1
        else:
            if current >= 2:
                streaks.append(current)
            current = 0
    if current >= 2:
        streaks.append(current)

    if not streaks:
        return 5  # default: 5 consecutive buckoffs = slump

    # Use weighted: average of median and mean, rounded
    median_s = np.median(streaks)
    mean_s = np.mean(streaks)
    threshold = int(round(0.7 * median_s + 0.3 * mean_s))

    return max(3, min(threshold, 20))  # clamp to [3, 20]


# ── 6. TREND MOMENTUM ──────────────────────────────────────
def compute_trend_momentum(scores, qualified, valid_mask, n_outs):
    """
    Trend Momentum: is the rider improving or declining?

    Compares slope of recent performance (last 20 or N/3 rides, whichever is smaller)
    to overall career trend.

    Returns: z-score like value: positive = improving, negative = declining.
    [-0.5, +0.5] range typical.
    """
    if n_outs < 30:
        return 0.0

    # Convert qualified rides to a performance score
    # Non-qualified = 0, qualified = score (or 80 if no score)
    perf_series = np.zeros(n_outs)
    for i in range(n_outs):
        if qualified[i]:
            perf_series[i] = scores[i] if valid_mask[i] else 80.0
        else:
            perf_series[i] = 0.0

    # Overall trend (linear slope over career)
    x_all = np.arange(n_outs)
    slope_all = np.polyfit(x_all, perf_series, 1)[0]

    # Recent trend (last 20 or N/3)
    recent_n = min(20, max(10, n_outs // 3))
    x_recent = np.arange(recent_n)
    y_recent = perf_series[-recent_n:]
    slope_recent = np.polyfit(x_recent, y_recent, 1)[0]

    # Difference: recent slope - overall slope
    # Normalize by typical score range (80 pts max), scaled for career length
    # A steeper recent decline/improvement is more meaningful for shorter careers
    delta = (slope_recent - slope_all) / 80.0 * min(n_outs, 50)

    # Clamp to [-0.5, +0.5]
    return max(-0.5, min(0.5, delta * 0.5))


# ╔══════════════════════════════════════════════════════════════╗
# ║  OUTPUT & REPORTING                                         ║
# ╚══════════════════════════════════════════════════════════════╝

def generate_outputs(results):
    """Generate JSON, CSV, and HTML outputs for top riders."""
    # Sort by total_outs descending, take top N
    sorted_riders = sorted(
        results.values(),
        key=lambda x: x["total_outs"],
        reverse=True
    )[:TOP_N]

    # ── JSON output ──
    json_output = []
    for r in sorted_riders:
        json_output.append({
            "rider": r["rider"],
            "total_outs": r["total_outs"],
            "career_qual_pct": r["career_qual_pct"],
            "career_avg_score": r["career_avg_score"],
            "volatility_index": r["volatility_index"],
            "post_layoff_decay_rate": r["post_layoff_decay_rate"],
            "pressure_factor": r["pressure_factor"],
            "recent_form_weight": r["recent_form_weight"],
            "cold_streak_threshold": r["cold_streak_threshold"],
            "trend_momentum": r["trend_momentum"],
        })

    with open(OUT_JSON, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"\nSaved rider profiles to {OUT_JSON} ({len(json_output)} riders)")

    # ── CSV output (all riders) ──
    all_sorted = sorted(results.values(), key=lambda x: x["total_outs"], reverse=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=json_output[0].keys())
        writer.writeheader()
        writer.writerows(all_sorted)
    print(f"Saved all rider profiles to {OUT_CSV} ({len(all_sorted)} riders)")

    return json_output


def print_summary_table(profiles):
    """Print a formatted summary table of top riders."""
    print("\n" + "=" * 120)
    print("TOP 20 RIDERS — INDIVIDUAL WEIGHTING PROFILES")
    print("=" * 120)

    header = (
        f"{'Rider':<24s} {'Outs':>5s} {'Qual%':>6s} {'AvgSc':>5s} "
        f"{'Volat':>6s} {'Decay':>6s} {'Press':>7s} "
        f"{'RecWt':>6s} {'Slump':>5s} {'Trend':>7s}"
    )
    print(header)
    print("-" * 120)

    for p in profiles:
        print(
            f"{p['rider']:<24s} "
            f"{p['total_outs']:>5d} "
            f"{p['career_qual_pct']:>5.1f}% "
            f"{p['career_avg_score']:>5.1f} "
            f"{p['volatility_index']:>6.3f} "
            f"{p['post_layoff_decay_rate']:>6.3f} "
            f"{p['pressure_factor']:>7.3f} "
            f"{p['recent_form_weight']:>6.3f} "
            f"{p['cold_streak_threshold']:>5d} "
            f"{p['trend_momentum']:>7.3f}"
        )

    # Summary statistics
    print("\n" + "-" * 120)
    print("DISTRIBUTION SUMMARY (all riders with ≥10 outs)")
    all_vals = list(profiles)  # we only have top 20 here, need all
    # Print within the function context
    for metric, label in [
        ("volatility_index", "Volatility Index"),
        ("post_layoff_decay_rate", "Post-Layoff Decay"),
        ("pressure_factor", "Pressure Factor"),
        ("recent_form_weight", "Recent Form Weight"),
        ("trend_momentum", "Trend Momentum"),
    ]:
        vals = [p[metric] for p in profiles]
        print(f"  {label:<22s}: mean={np.mean(vals):+.3f}  std={np.std(vals):.3f}  "
              f"min={np.min(vals):+.3f}  max={np.max(vals):+.3f}")

    print("=" * 120)


def generate_html_report(all_results):
    """Generate a styled HTML report for all riders with significant data."""
    sorted_all = sorted(
        all_results.values(),
        key=lambda x: x["total_outs"],
        reverse=True
    )

    # Only include riders with enough rides for reliable metrics
    report_riders = [r for r in sorted_all if r["total_outs"] >= 10]

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DataSpur Rider Weighting Profiles</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }
h1 { color: #1a1a2e; }
table { border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
th { background: #1a1a2e; color: white; padding: 12px 10px; text-align: right; font-size: 13px; position: sticky; top: 0; }
th:first-child { text-align: left; }
td { padding: 8px 10px; text-align: right; border-bottom: 1px solid #eee; font-size: 13px; }
td:first-child { text-align: left; font-weight: 500; }
tr:hover { background: #f0f4ff; }
.pos { color: #2e7d32; }
.neg { color: #c62828; }
.neutral { color: #666; }
.high { background: #fff3e0; }
.low { background: #e8f5e9; }
.summary { margin: 20px 0; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.bar-cell { position: relative; }
.bar { position: absolute; left: 0; top: 0; height: 100%; border-radius: 3px; opacity: 0.3; }
</style>
</head>
<body>
<h1>🏇 DataSpur Rider Weighting System</h1>
<div class="summary">
<p><strong>Riders profiled:</strong> """ + f"{len(report_riders):,}</p>" + """
<p>Each rider receives an individualized profile that modulates how the prediction model weights their data.
These weights inject rider-specific priors into the feature pipeline, replacing the current uniform treatment.</p>
<p><strong>Metrics explained:</strong></p>
<ul>
<li><strong>Volatility Index (0-1):</strong> Higher = more unpredictable ride-to-ride. Drives prediction interval width.</li>
<li><strong>Decay Rate (0-1):</strong> Higher = recovers faster after injury/layoff. Modulates post-absence predictions.</li>
<li><strong>Pressure Factor:</strong> Negative = worse under pressure. Adjusts predictions for short rounds/finals.</li>
<li><strong>Recent Form Weight (0-1):</strong> How much recent rides matter vs career. Personalizes the recency kernel.</li>
<li><strong>Slump Threshold:</strong> Consecutive buckoffs before model declares a cold streak.</li>
<li><strong>Trend Momentum:</strong> Positive = improving. Adjusts baseline expectation up/down.</li>
</ul>
</div>
<table>
<thead><tr>
<th>Rider</th><th>Outs</th><th>Qual%</th><th>Avg Score</th>
<th>Volatility</th><th>Decay</th><th>Pressure</th>
<th>Recency Wt</th><th>Slump Thr</th><th>Trend</th>
</tr></thead>
<tbody>
"""

    for r in report_riders[:100]:  # Top 100 for HTML
        p_class = "pos" if r["pressure_factor"] > 0.02 else ("neg" if r["pressure_factor"] < -0.02 else "neutral")
        t_class = "pos" if r["trend_momentum"] > 0.02 else ("neg" if r["trend_momentum"] < -0.02 else "neutral")

        html += f"""<tr>
<td>{r['rider']}</td>
<td>{r['total_outs']}</td>
<td>{r['career_qual_pct']:.1f}%</td>
<td>{r['career_avg_score']:.1f}</td>
<td>{r['volatility_index']:.3f}</td>
<td>{r['post_layoff_decay_rate']:.3f}</td>
<td class="{p_class}">{r['pressure_factor']:+.3f}</td>
<td>{r['recent_form_weight']:.3f}</td>
<td>{r['cold_streak_threshold']}</td>
<td class="{t_class}">{r['trend_momentum']:+.3f}</td>
</tr>
"""

    html += """</tbody></table>
<p style="color:#999; margin-top:20px;">DataSpur Rider Weighting System · Generated """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
</body></html>"""

    with open(OUT_HTML, "w") as f:
        f.write(html)
    print(f"Saved HTML report to {OUT_HTML}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  FEATURE PIPELINE INTEGRATION                               ║
# ╚══════════════════════════════════════════════════════════════╝

def generate_integration_code():
    """Print example code showing how to integrate weights into the feature pipeline."""
    print("\n" + "=" * 120)
    print("INTEGRATION GUIDE — How to use these weights in the feature pipeline")
    print("=" * 120)
    print("""
# ── In features.py or train_model_v3.py, after loading rider_weights.json ──

import json
with open("data/rider_weights.json") as f:
    rider_weights = {rw["rider"]: rw for rw in json.load(f)}

def apply_rider_weights(row, weights):
    '''Modify prediction features using rider-specific weights.'''
    w = weights.get(row["rider_name"], {})
    if not w:
        return row  # no profile, use defaults

    # 1. Recent form: blend career avg with recent form using personalized weight
    rec_w = w.get("recent_form_weight", 0.3)
    career_qual = row.get("rider_career_qual_pct", 50)
    recent_qual = row.get("rider_last5_qual", career_qual / 100)
    row["rider_weighted_qual"] = rec_w * recent_qual + (1 - rec_w) * (career_qual / 100)

    # 2. Pressure adjustment: if this is a high-pressure ride, apply pressure factor
    pressure_factor = w.get("pressure_factor", 0.0)
    if row.get("is_short_go", 0) or row.get("is_finals", 0):
        row["rider_weighted_qual"] += pressure_factor * 0.2  # scale down

    # 3. Post-layoff: if rider is within N rides of returning, discount
    layoff_decay = w.get("post_layoff_decay_rate", 0.7)
    rides_since_return = row.get("rides_since_layoff", 999)
    if rides_since_return < 10:
        recovery = 1 - np.exp(-layoff_decay * rides_since_return)
        row["rider_weighted_qual"] *= recovery

    # 4. Volatility: widen prediction intervals for volatile riders
    volatility = w.get("volatility_index", 0.5)
    row["prediction_confidence"] = 1.0 - volatility * 0.5

    # 5. Trend momentum: adjust baseline expectation
    trend = w.get("trend_momentum", 0.0)
    row["rider_weighted_qual"] += trend * 0.15

    return row
""")
    print("=" * 120)


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN                                                       ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    print("=" * 60)
    print("DATASPUR RIDER WEIGHTING SYSTEM")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading and cleaning data...")
    rides = load_and_clean()

    # 2. Classify pressure context
    print("\n[2] Classifying pressure contexts...")
    rides = classify_pressure_context(rides)
    print(f"  High-pressure rides: {rides['is_high_pressure'].sum():,} "
          f"({rides['is_high_pressure'].mean()*100:.1f}%)")
    print(f"  Baseline rides: {rides['is_baseline_round'].sum():,} "
          f"({rides['is_baseline_round'].mean()*100:.1f}%)")
    print(f"  Short round rides: {rides['is_short_round'].sum():,}")
    print(f"  Finals/CFR rides: {rides['is_finals'].sum():,}")
    print(f"  Premier/UTB rides: {rides['is_premier'].sum():,}")
    print(f"  Team Series rides: {rides['is_team_series'].sum():,}")

    # 3. Compute all rider metrics
    print("\n[3] Computing rider metrics...")
    results = compute_all_rider_metrics(rides)
    print(f"  Profiled {len(results):,} riders (≥{MIN_RIDES_FOR_PROFILE} outs)")

    # 4. Generate outputs
    print("\n[4] Generating outputs...")
    profiles = generate_outputs(results)

    # 5. Print summary
    print_summary_table(profiles)

    # 6. HTML report
    print("\n[5] Generating HTML report...")
    generate_html_report(results)

    # 7. Integration guide
    generate_integration_code()

    print("\nDone! Rider weighting system ready.")
    return results, profiles


if __name__ == "__main__":
    main()