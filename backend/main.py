#!/usr/bin/env python3
"""
DataSpur API — PBR Teams Matchup Simulator Backend
Serves roster data, runs Hungarian optimization with game theory,
spin×handedness matchup analysis, and opponent-aware strategy recommendations.
Enterprise-grade: validation, error handling, typed interfaces.
"""
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any
import csv, re, json, math, os, logging
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dataspur")

DATA = Path(__file__).parent / "data"
app = FastAPI(title="DataSpur PBR Teams Simulator", version="2.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ====== LOAD ALL DATA AT STARTUP ======
rides_data = []
bull_profiles = {}       # bull_id -> profile dict
rider_profiles = {}      # rider_slug -> profile dict

with open(DATA / "rides.csv") as f:
    rides_data = list(csv.DictReader(f))

with open(DATA / "bull_profiles.csv") as f:
    for row in csv.DictReader(f):
        bull_profiles[row["bull_id"]] = row

with open(DATA / "rider_profiles.csv") as f:
    for row in csv.DictReader(f):
        rider_profiles[row["rider_slug"]] = row

logger.info(f"Loaded {len(rides_data)} rides, {len(bull_profiles)} bulls, {len(rider_profiles)} riders")

# ====== PYDANTIC MODELS ======
class SimulateRequest(BaseModel):
    team_code: str
    bulls: List[Any] = []  # Accept {id, name} dicts OR bare ID strings
    opponent_team: Optional[str] = None
    opponent_bulls: Optional[List[Any]] = None
    manual_matchups: Optional[Dict[str, str]] = None

# ====== TEAM DEFINITIONS ======
TEAMS = {
    'AUS': 'Austin Gamblers', 'AZ': 'Arizona Ridge Riders', 'CAR': 'Carolina Cowboys',
    'FL': 'Florida Freedom', 'KANS': 'Kansas City Outlaws', 'MIS': 'Missouri Thunder',
    'NASH': 'Nashville Stampede', 'NY': 'New York Mavericks', 'OKL': 'Oklahoma Wildcatters',
    'TEX': 'Texas Rattlers'
}

# ====== RIDER-TEAM ASSIGNMENTS (2026 Team Series data only) ======
ts_rides = [r for r in rides_data 
            if r.get('evt_tour_class') == 'Team Series' 
            and r['event_type'] == 'BR'
            and '2026' in str(r.get('evt_date',''))]

rider_team_counts = defaultdict(lambda: defaultdict(int))
for r in ts_rides:
    note = r.get('evt_note', '').strip()
    teams = re.findall(r'[A-Z]{2,4}', note)
    for t in teams:
        rider_team_counts[r['rider_name']][t] += 1

RIDER_TO_TEAM = {}
for rider, counts in rider_team_counts.items():
    if not counts: continue
    best = max(counts, key=counts.get)
    if counts[best] / sum(counts.values()) >= 0.45:
        RIDER_TO_TEAM[rider] = best

def get_team_roster(team_code):
    return sorted([r for r, t in RIDER_TO_TEAM.items() if t == team_code])

# ====== COMPUTE STATS ======
rider_stats = defaultdict(lambda: {'outs': 0, 'covers': 0, 'total_score': 0, 'scores': []})
bull_stats = defaultdict(lambda: {'outs': 0, 'buckoffs': 0})

for r in rides_data:
    if r['event_type'] != 'BR': continue
    rider = r['rider_name']
    bull_id = r.get('bull_id', '').strip()
    qualified = str(r.get('qualified', '')).lower() in ('true', '1', '1.0', 'yes')
    rider_stats[rider]['outs'] += 1
    if qualified:
        rider_stats[rider]['covers'] += 1
        try:
            s = float(r.get('score', 0) or 0)
            rider_stats[rider]['total_score'] += s
            rider_stats[rider]['scores'].append(s)
        except: pass
    if bull_id:
        bull_stats[bull_id]['outs'] += 1
        if not qualified:
            bull_stats[bull_id]['buckoffs'] += 1

# ====== HELPERS ======
def _normalize_bulls(bulls_input: List[Any]) -> List[Dict[str, str]]:
    """Normalize bull input: accept {id, name} dicts OR bare ID strings."""
    result = []
    for b in bulls_input:
        if isinstance(b, dict):
            result.append({"id": b.get("id", ""), "name": b.get("name", "")})
        elif isinstance(b, str):
            result.append({"id": b, "name": b})
        else:
            continue
    return result

def predict_cover(rider_name: str, bull_id: str) -> Dict[str, Any]:
    """
    Predict cover probability and expected score for a rider-bull matchup.
    Uses career stats weighted average, enhanced with spin×handedness when available.
    """
    rs = rider_stats.get(rider_name, {'outs': 0, 'covers': 0, 'total_score': 0, 'scores': []})
    bs = bull_stats.get(bull_id, {'outs': 0, 'buckoffs': 0})
    
    rider_cov_pct = rs['covers'] / max(rs['outs'], 1)
    bull_buckoff_pct = bs['buckoffs'] / max(bs['outs'], 1)
    
    # Base probability from rider and bull stats
    if rs['outs'] >= 10 and bs['outs'] >= 5:
        prob = (rider_cov_pct * 0.6 + (1 - bull_buckoff_pct) * 0.4)
    elif rs['outs'] >= 10:
        prob = rider_cov_pct * 0.8 + 0.25 * 0.2
    elif bs['outs'] >= 5:
        prob = (1 - bull_buckoff_pct) * 0.7 + 0.25 * 0.3
    else:
        prob = 0.25
    
    # Spin×handedness adjustment (matchup biomechanics)
    spin_hand_bonus = 0
    bp = bull_profiles.get(bull_id, {})
    # Find rider slug in rider_profiles (best effort match)
    rider_slug = rider_name.replace(" ", "+")
    rp = rider_profiles.get(rider_slug, {})
    
    bull_spin = bp.get('hand_advantage_dir', '').strip()
    rider_hand = rp.get('riding_hand', '').strip()
    
    if bull_spin and rider_hand:
        if bull_spin == "LEFT" and rider_hand == "Right":
            spin_hand_bonus = 0.08  # Favorable: bull spins into rope hand
        elif bull_spin == "RIGHT" and rider_hand == "Left":
            spin_hand_bonus = 0.08  # Favorable
        elif bull_spin == "LEFT" and rider_hand == "Left":
            spin_hand_bonus = -0.06  # Unfavorable: bull spins away from rope hand
        elif bull_spin == "RIGHT" and rider_hand == "Right":
            spin_hand_bonus = -0.06
    
    prob += spin_hand_bonus
    prob = max(0.02, min(0.95, prob))
    
    # Expected score
    if rs['covers'] > 0:
        exp_score = rs['total_score'] / rs['covers']
    else:
        exp_score = 85.0
    
    variance = float(np.std(rs['scores'])) if len(rs['scores']) > 3 else 5.0
    
    return {
        'cover_prob': round(prob, 4),
        'exp_score': round(exp_score, 1),
        'score_std': round(variance, 1),
        'rider_outs': rs['outs'],
        'rider_covers': rs['covers'],
        'bull_outs': bs['outs'],
        'bull_buckoffs': bs['buckoffs'],
        'spin_hand_match': {
            'bull_spin': bull_spin or 'unknown',
            'rider_hand': rider_hand or 'unknown',
            'bonus': round(spin_hand_bonus, 3),
        } if bull_spin or rider_hand else None,
    }

# ====== API ENDPOINTS ======

@app.get("/health")
@app.get("/api/health")
def health():
    """Health check for Railway and frontend."""
    return {
        "status": "ok",
        "loaded_rides": len(rides_data),
        "bulls_with_profiles": len(bull_profiles),
        "riders_with_profiles": len(rider_profiles),
        "teams_tracked": len(TEAMS),
    }

@app.get("/api/teams")
def list_teams():
    teams = []
    for code, name in TEAMS.items():
        roster = get_team_roster(code)
        teams.append({
            'code': code, 'name': name,
            'roster': roster,
            'roster_count': len(roster)
        })
    return {'teams': teams}

@app.get("/api/team/{team_code}")
def team_detail(team_code: str):
    """Team detail with stats."""
    if team_code not in TEAMS:
        raise HTTPException(404, f"Team not found: {team_code}")
    roster = get_team_roster(team_code)
    riders_info = []
    for rider in roster:
        rs = rider_stats.get(rider, {})
        cov_pct = rs['covers'] / max(rs['outs'], 1) * 100 if rs['outs'] > 0 else 0
        avg_score = rs['total_score'] / rs['covers'] if rs['covers'] > 0 else 0
        riders_info.append({
            'name': rider, 'outs': rs['outs'], 'covers': rs['covers'],
            'cover_pct': round(cov_pct, 1), 'avg_score': round(avg_score, 1),
        })
    avg_cover = sum(r['cover_pct'] for r in riders_info) / max(len(riders_info), 1)
    return {
        'team': TEAMS[team_code], 'code': team_code,
        'riders': riders_info,
        'avg_cover_pct': round(avg_cover, 1),
    }

@app.get("/api/team/{team_code}/roster")
def team_roster(team_code: str):
    if team_code not in TEAMS:
        raise HTTPException(404, f"Team not found: {team_code}")
    roster = get_team_roster(team_code)
    riders_info = []
    for rider in roster:
        rs = rider_stats.get(rider, {})
        cov_pct = rs['covers'] / max(rs['outs'], 1) * 100 if rs['outs'] > 0 else 0
        avg_score = rs['total_score'] / rs['covers'] if rs['covers'] > 0 else 0
        riders_info.append({
            'name': rider, 'outs': rs['outs'], 'covers': rs['covers'],
            'cover_pct': round(cov_pct, 1), 'avg_score': round(avg_score, 1),
        })
    return {'team': TEAMS[team_code], 'code': team_code, 'riders': riders_info}

@app.post("/api/simulate")
def simulate(request: SimulateRequest):
    """
    Simulate optimal matchups for a team against a bull pen.
    Handles both {id, name} objects and bare ID strings for bulls.
    """
    if request.team_code not in TEAMS:
        raise HTTPException(400, f"Invalid team code: {request.team_code}")
    
    roster = get_team_roster(request.team_code)
    if not roster:
        raise HTTPException(400, f"No riders found for team: {request.team_code}")
    
    bulls = _normalize_bulls(request.bulls)
    if len(bulls) == 0:
        raise HTTPException(400, "No valid bulls provided")
    
    opponent_bulls = _normalize_bulls(request.opponent_bulls or [])
    
    # Rank riders by cover %
    riders_ranked = sorted(roster, key=lambda r: 
        rider_stats[r]['covers'] / max(rider_stats[r]['outs'], 1), reverse=True)
    
    n = min(len(roster), len(bulls))
    selected_riders = riders_ranked[:n]
    
    # ====== OPTIMAL MATCHUP (Hungarian Algorithm) ======
    cost = np.zeros((n, n))
    prob_matrix = np.zeros((n, n))
    spin_hand_info = {}
    
    for i, rider in enumerate(selected_riders):
        for j, bull in enumerate(bulls[:n]):
            pred = predict_cover(rider, bull.get('id', ''))
            cost[i, j] = -pred['exp_score'] * pred['cover_prob']
            prob_matrix[i, j] = pred['cover_prob']
            spin_hand_info[(i, j)] = pred.get('spin_hand_match')
    
    row_ind, col_ind = linear_sum_assignment(cost)
    optimal_score = float(-cost[row_ind, col_ind].sum())
    
    optimal_matchups = []
    for i, j in zip(row_ind, col_ind):
        pred = predict_cover(selected_riders[i], bulls[j].get('id', ''))
        matchup = {
            'rider': selected_riders[i],
            'bull_name': bulls[j].get('name', '?'),
            'bull_id': bulls[j].get('id', ''),
            'cover_prob': pred['cover_prob'],
            'exp_score': pred['exp_score'],
            'rider_career': f"{pred['rider_covers']}/{pred['rider_outs']}",
            'bull_record': f"{pred['bull_buckoffs']} buckoffs / {pred['bull_outs']} outs",
        }
        if pred.get('spin_hand_match'):
            matchup['spin_hand_match'] = pred['spin_hand_match']
        optimal_matchups.append(matchup)
    
    # ====== OPPONENT SIMULATION ======
    opponent_analysis = None
    if request.opponent_team and opponent_bulls:
        if request.opponent_team not in TEAMS:
            raise HTTPException(400, f"Invalid opponent team code: {request.opponent_team}")
        opp_roster = get_team_roster(request.opponent_team)
        if opp_roster and len(opponent_bulls) >= 3:
            opp_ranked = sorted(opp_roster, key=lambda r: 
                rider_stats[r]['covers'] / max(rider_stats[r]['outs'], 1), reverse=True)
            m = min(len(opp_ranked), len(opponent_bulls))
            opp_selected = opp_ranked[:m]
            
            opp_cost = np.zeros((m, m))
            opp_prob = np.zeros((m, m))
            for i, rider in enumerate(opp_selected):
                for j, bull in enumerate(opponent_bulls[:m]):
                    pred = predict_cover(rider, bull.get('id', ''))
                    opp_cost[i, j] = -pred['exp_score'] * pred['cover_prob']
                    opp_prob[i, j] = pred['cover_prob']
            
            opp_ri, opp_ci = linear_sum_assignment(opp_cost)
            opp_optimal_score = float(-opp_cost[opp_ri, opp_ci].sum())
            opp_expected_covers = float(sum(opp_prob[opp_ri[i], opp_ci[i]] for i in range(m)))
            
            if opp_expected_covers <= 1.5:
                rec = 'CONSERVATIVE — opponent weak. Need 2+ covers to win. Prioritize high-probability rides.'
            elif opp_expected_covers <= 2.5:
                rec = 'BALANCED — opponent average. Need 3+ covers or high scores.'
            else:
                rec = 'AGGRESSIVE — opponent strong. Need maximum points. Accept lower cover probability for higher scores.'
            
            opponent_analysis = {
                'team': TEAMS.get(request.opponent_team, request.opponent_team),
                'code': request.opponent_team,
                'optimal_score': round(opp_optimal_score, 1),
                'expected_covers': round(opp_expected_covers, 1),
                'strategy_recommendation': rec,
                'score_diff': round(optimal_score - opp_optimal_score, 1),
                'cover_diff': round(float(sum(prob_matrix[row_ind[i], col_ind[i]] for i in range(n))) - opp_expected_covers, 1),
            }
    
    # ====== STRATEGY ======
    exp_covers = float(sum(prob_matrix[row_ind[i], col_ind[i]] for i in range(n)))
    
    if opponent_analysis:
        target_covers = max(2, math.ceil(opponent_analysis['expected_covers'] + 0.5))
        strategy = 'CONSERVATIVE' if exp_covers >= target_covers + 0.5 else \
                  'AGGRESSIVE' if exp_covers < target_covers else 'BALANCED'
    else:
        strategy = 'MAXIMIZE SCORE'
        target_covers = 3
    
    # ====== MANUAL MATCHUP CHECK ======
    manual_analysis = None
    if request.manual_matchups:
        manual_results = []
        manual_score = 0
        manual_covers = 0
        for rider, bull_name in request.manual_matchups.items():
            bull = next((b for b in bulls if b.get('name', '') == bull_name), None)
            if bull and rider in selected_riders:
                pred = predict_cover(rider, bull.get('id', ''))
                manual_results.append({
                    'rider': rider, 'bull_name': bull_name,
                    'cover_prob': pred['cover_prob'],
                    'exp_score': pred['exp_score'],
                    'is_optimal': any(m['rider'] == rider and m['bull_name'] == bull_name for m in optimal_matchups),
                })
                manual_score += pred['exp_score'] * pred['cover_prob']
                manual_covers += pred['cover_prob']
        
        manual_analysis = {
            'matchups': manual_results,
            'expected_score': round(manual_score, 1),
            'expected_covers': round(manual_covers, 1),
            'score_diff_vs_optimal': round(manual_score - optimal_score, 1),
            'covers_diff_vs_optimal': round(manual_covers - exp_covers, 1),
            'correct_picks': sum(1 for m in manual_results if m['is_optimal']),
            'total_picks': len(manual_results),
        }
    
    return {
        'team': TEAMS.get(request.team_code, request.team_code),
        'team_code': request.team_code,
        'riders_used': selected_riders,
        'bulls_used': [b.get('name', '?') for b in bulls[:n]],
        'optimal_matchups': optimal_matchups,
        'optimal_score': round(optimal_score, 1),
        'expected_covers': round(exp_covers, 1),
        'strategy': strategy,
        'target_covers': target_covers,
        'opponent_analysis': opponent_analysis,
        'manual_analysis': manual_analysis,
    }

@app.get("/api/bulls/search")
def search_bulls(q: str = "", limit: int = 20):
    """Search for bulls by name or ID. Returns top matches with stats."""
    results = []
    q_lower = q.lower()
    seen = set()
    for r in rides_data:
        name = r.get('bull_name', '')
        bid = r.get('bull_id', '').strip()
        if q_lower in name.lower() or q == bid:
            key = (bid, name)
            if key not in seen:
                seen.add(key)
                bs = bull_stats.get(bid, {})
                buckoff_pct = bs['buckoffs'] / max(bs['outs'], 1) * 100
                results.append({
                    'id': bid, 'name': name,
                    'outs': bs['outs'],
                    'buckoff_pct': round(buckoff_pct, 1)
                })
                if len(results) >= limit:
                    break
    return {'bulls': results}

@app.get("/api/reports")
def reports():
    """Season report data — league-wide stats."""
    all_teams = []
    for code, name in TEAMS.items():
        roster = get_team_roster(code)
        riders_info = []
        for rider in roster:
            rs = rider_stats.get(rider, {})
            cov_pct = rs['covers'] / max(rs['outs'], 1) * 100 if rs['outs'] > 0 else 0
            avg_score = rs['total_score'] / rs['covers'] if rs['covers'] > 0 else 0
            riders_info.append({
                'name': rider, 'outs': rs['outs'], 'covers': rs['covers'],
                'cover_pct': round(cov_pct, 1), 'avg_score': round(avg_score, 1),
            })
        all_teams.append({
            'code': code, 'name': name,
            'roster_count': len(roster),
            'avg_cover_pct': round(sum(r['cover_pct'] for r in riders_info) / max(len(riders_info), 1), 1),
            'riders': riders_info,
        })
    
    # League stats
    all_br = [r for r in rides_data if r['event_type'] == 'BR']
    total_rides = len(all_br)
    total_covers = sum(1 for r in all_br if str(r.get('qualified', '')).lower() in ('true', '1', '1.0', 'yes'))
    league_cover_pct = round(total_covers / max(total_rides, 1) * 100, 1)
    
    # Spin×handedness matchup stats
    spin_match_rides = 0
    spin_match_covers = 0
    for r in all_br:
        bull_id = r.get('bull_id', '').strip()
        bp = bull_profiles.get(bull_id, {})
        spin = bp.get('hand_advantage_dir', '').strip()
        if not spin:
            continue
        rider = r['rider_name']
        slug = rider.replace(" ", "+")
        rp = rider_profiles.get(slug, {})
        hand = rp.get('riding_hand', '').strip()
        if not hand:
            continue
        spin_match_rides += 1
        is_favorable = (spin == "LEFT" and hand == "Right") or (spin == "RIGHT" and hand == "Left")
        qualified = str(r.get('qualified', '')).lower() in ('true', '1', '1.0', 'yes')
        if qualified:
            spin_match_covers += 1
    
    return {
        'teams': all_teams,
        'league': {
            'total_rides': total_rides,
            'total_covers': total_covers,
            'league_cover_pct': league_cover_pct,
            'spin_match_rides': spin_match_rides,
            'spin_match_coverage': round(spin_match_rides / max(total_rides, 1) * 100, 1),
            'bulls_tracked': len(bull_stats),
            'riders_tracked': len(rider_stats),
        }
    }

@app.get("/api/bulls/{bull_id}")
def bull_detail(bull_id: str):
    """Get detailed bull profile including spin direction, power rating."""
    bp = bull_profiles.get(bull_id)
    if not bp:
        raise HTTPException(404, f"Bull not found: {bull_id}")
    bs = bull_stats.get(bull_id, {'outs': 0, 'buckoffs': 0})
    buckoff_pct = bs['buckoffs'] / max(bs['outs'], 1) * 100
    return {
        'id': bull_id,
        'name': bp.get('name', ''),
        'power_rating': float(bp.get('power_rating', 0) or 0),
        'spin_direction': bp.get('hand_advantage_dir', '').strip(),
        'hand_pct': float(bp.get('hand_advantage_pct', 0) or 0),
        'lh_pct': float(bp.get('lh_pct', 0) or 0),
        'rh_pct': float(bp.get('rh_pct', 0) or 0),
        'career_outs': bs['outs'],
        'career_buckoffs': bs['buckoffs'],
        'buckoff_pct': round(buckoff_pct, 1),
        'contractor': bp.get('contractor', ''),
        'active': bp.get('active', '') == 'True',
    }

# Frontend served by lovable.dev/Vercel — not bundled in API
FRONTEND_PATH = Path(__file__).parent.parent / "frontend"
if FRONTEND_PATH.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_PATH), html=True), name="frontend")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8420))
    logger.info(f"Starting DataSpur API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)