#!/usr/bin/env python3
"""
DataSpur API — PBR Teams Matchup Simulator Backend
Serves roster data, runs Hungarian optimization with game theory,
and provides opponent-aware strategy recommendations.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import csv, re, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import uvicorn

DATA = Path.home() / "dataspur" / "data"
app = FastAPI(title="DataSpur PBR Teams Simulator", version="1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ====== LOAD ALL DATA AT STARTUP ======
rides_data = []
with open(DATA / "rides.csv") as f:
    rides_data = list(csv.DictReader(f))

# Rider-team assignments (from 2026 Team Series data ONLY)
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

# Compute rider and bull stats
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

def predict_cover(rider_name, bull_id, include_variance=True):
    """Predict cover probability and expected score for a rider-bull matchup."""
    rs = rider_stats.get(rider_name, {'outs': 0, 'covers': 0, 'total_score': 0, 'scores': []})
    bs = bull_stats.get(bull_id, {'outs': 0, 'buckoffs': 0})
    
    rider_cov_pct = rs['covers'] / max(rs['outs'], 1)
    bull_buckoff_pct = bs['buckoffs'] / max(bs['outs'], 1)
    
    if rs['outs'] >= 10 and bs['outs'] >= 5:
        prob = (rider_cov_pct * 0.6 + (1 - bull_buckoff_pct) * 0.4)
    elif rs['outs'] >= 10:
        prob = rider_cov_pct * 0.8 + 0.25 * 0.2
    elif bs['outs'] >= 5:
        prob = (1 - bull_buckoff_pct) * 0.7 + 0.25 * 0.3
    else:
        prob = 0.25
    
    prob = max(0.02, min(0.95, prob))
    
    if rs['covers'] > 0:
        exp_score = rs['total_score'] / rs['covers']
    else:
        exp_score = 85.0
    
    variance = np.std(rs['scores']) if len(rs['scores']) > 3 else 5.0
    
    return {
        'cover_prob': round(prob, 4),
        'exp_score': round(exp_score, 1),
        'score_std': round(variance, 1),
        'rider_outs': rs['outs'],
        'rider_covers': rs['covers'],
        'bull_outs': bs['outs'],
        'bull_buckoffs': bs['buckoffs'],
    }

# Team definitions
TEAMS = {
    'AUS': 'Austin Gamblers', 'AZ': 'Arizona Ridge Riders', 'CAR': 'Carolina Cowboys',
    'FL': 'Florida Freedom', 'KANS': 'Kansas City Outlaws', 'MIS': 'Missouri Thunder',
    'NASH': 'Nashville Stampede', 'NY': 'New York Mavericks', 'OKL': 'Oklahoma Wildcatters',
    'TEX': 'Texas Rattlers'
}

# Build team rosters
def get_team_roster(team_code):
    return sorted([r for r, t in RIDER_TO_TEAM.items() if t == team_code])

# ====== API ENDPOINTS ======

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

@app.get("/api/team/{team_code}/roster")
def team_roster(team_code: str):
    if team_code not in TEAMS:
        raise HTTPException(404, "Team not found")
    roster = get_team_roster(team_code)
    riders_info = []
    for rider in roster:
        rs = rider_stats.get(rider, {})
        cov_pct = rs['covers'] / max(rs['outs'], 1) * 100 if rs['outs'] > 0 else 0
        avg_score = rs['total_score'] / rs['covers'] if rs['covers'] > 0 else 0
        riders_info.append({
            'name': rider,
            'outs': rs['outs'],
            'covers': rs['covers'],
            'cover_pct': round(cov_pct, 1),
            'avg_score': round(avg_score, 1),
        })
    return {'team': TEAMS[team_code], 'code': team_code, 'riders': riders_info}

@app.post("/api/simulate")
def simulate(request: dict):
    """
    Simulate optimal matchups for a team against a bull pen.
    Body: {
        "team_code": "CAR",
        "bulls": [{"id": "123", "name": "Rock'n Roll"}, ...],
        "opponent_team": "AUS",  // optional
        "opponent_bulls": [{"id": "456", "name": "Whoa"}, ...],  // optional
        "manual_matchups": {"Rider Name": "Bull Name"}  // optional — check manual picks
    }
    """
    team_code = request.get('team_code')
    if team_code not in TEAMS:
        raise HTTPException(400, "Invalid team code")
    
    roster = get_team_roster(team_code)
    bulls = request.get('bulls', [])
    opponent_team = request.get('opponent_team')
    opponent_bulls = request.get('opponent_bulls', [])
    manual = request.get('manual_matchups', {})
    
    if len(bulls) == 0:
        raise HTTPException(400, "No bulls provided")
    
    # If fewer riders than bulls, pick top riders by cover%
    riders_ranked = sorted(roster, key=lambda r: 
        rider_stats[r]['covers'] / max(rider_stats[r]['outs'], 1), reverse=True)
    
    n = min(len(roster), len(bulls))
    selected_riders = riders_ranked[:n]
    
    # ====== OPTIMAL MATCHUP (Hungarian Algorithm) ======
    cost = np.zeros((n, n))
    prob_matrix = np.zeros((n, n))
    for i, rider in enumerate(selected_riders):
        for j, bull in enumerate(bulls[:n]):
            pred = predict_cover(rider, bull.get('id', ''))
            cost[i, j] = -pred['exp_score'] * pred['cover_prob']
            prob_matrix[i, j] = pred['cover_prob']
    
    row_ind, col_ind = linear_sum_assignment(cost)
    optimal_score = float(-cost[row_ind, col_ind].sum())
    
    optimal_matchups = []
    for i, j in zip(row_ind, col_ind):
        pred = predict_cover(selected_riders[i], bulls[j].get('id', ''))
        optimal_matchups.append({
            'rider': selected_riders[i],
            'bull_name': bulls[j].get('name', '?'),
            'bull_id': bulls[j].get('id', ''),
            'cover_prob': pred['cover_prob'],
            'exp_score': pred['exp_score'],
            'rider_career': f"{pred['rider_covers']}/{pred['rider_outs']}",
            'bull_record': f"{pred['bull_buckoffs']} buckoffs / {pred['bull_outs']} outs",
        })
    
    # ====== OPPONENT SIMULATION ======
    opponent_analysis = None
    if opponent_team and opponent_bulls:
        opp_roster = get_team_roster(opponent_team)
        if opp_roster and len(opponent_bulls) >= 3:
            opp_ranked = sorted(opp_roster, key=lambda r: 
                rider_stats[r]['covers'] / max(rider_stats[r]['outs'], 1), reverse=True)
            m = min(len(opp_ranked), len(opponent_bulls))
            opp_selected = opp_ranked[:m]
            
            opp_cost = np.zeros((m, m))
            for i, rider in enumerate(opp_selected):
                for j, bull in enumerate(opponent_bulls[:m]):
                    pred = predict_cover(rider, bull.get('id', ''))
                    opp_cost[i, j] = -pred['exp_score'] * pred['cover_prob']
            
            opp_ri, opp_ci = linear_sum_assignment(opp_cost)
            opp_optimal_score = float(-opp_cost[opp_ri, opp_ci].sum())
            
            prob_opp = np.zeros((m, m))
            for i, rider in enumerate(opp_selected):
                for j, bull in enumerate(opponent_bulls[:m]):
                    pred = predict_cover(rider, bull.get('id', ''))
                    prob_opp[i, j] = pred['cover_prob']
            
            opp_expected_covers = sum(prob_opp[opp_ri[i], opp_ci[i]] for i in range(m))
            
            # Game theory: recommended strategy based on opponent strength
            opponent_analysis = {
                'team': TEAMS.get(opponent_team, opponent_team),
                'optimal_score': round(opp_optimal_score, 1),
                'expected_covers': round(opp_expected_covers, 1),
                'strategy_recommendation': '',
            }
            
            if opp_expected_covers <= 1.5:
                opponent_analysis['strategy_recommendation'] = 'CONSERVATIVE — opponent weak. Need 2+ covers to win. Prioritize high-probability rides.'
            elif opp_expected_covers <= 2.5:
                opponent_analysis['strategy_recommendation'] = 'BALANCED — opponent average. Need 3+ covers or high scores.'
            else:
                opponent_analysis['strategy_recommendation'] = 'AGGRESSIVE — opponent strong. Need maximum points. Accept lower cover probability for higher scores.'
    
    # ====== GAME THEORY STRATEGY ======
    # Expected covers from optimal matchup
    exp_covers = sum(prob_matrix[row_ind[i], col_ind[i]] for i in range(n))
    
    if opponent_analysis:
        target_covers = max(2, math.ceil(opponent_analysis['expected_covers'] + 0.5))
        strategy = 'CONSERVATIVE' if exp_covers >= target_covers + 0.5 else \
                  'AGGRESSIVE' if exp_covers < target_covers else 'BALANCED'
    else:
        strategy = 'MAXIMIZE SCORE'
        target_covers = 3
    
    # ====== MANUAL MATCHUP CHECK ======
    manual_analysis = None
    if manual:
        manual_results = []
        manual_score = 0
        manual_covers = 0
        for rider, bull_name in manual.items():
            # Find bull
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
        'team': TEAMS.get(team_code, team_code),
        'team_code': team_code,
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
    """Search for bulls by name or ID."""
    results = []
    q_lower = q.lower()
    for r in rides_data:
        name = r.get('bull_name', '')
        bid = r.get('bull_id', '').strip()
        if q_lower in name.lower() or q == bid:
            key = (bid, name)
            if key not in [(x['id'], x['name']) for x in results]:
                bs = bull_stats.get(bid, {})
                buckoff_pct = bs['buckoffs'] / max(bs['outs'], 1) * 100
                results.append({
                    'id': bid,
                    'name': name,
                    'outs': bs['outs'],
                    'buckoff_pct': round(buckoff_pct, 1)
                })
                if len(results) >= limit:
                    break
    return {'bulls': results}

@app.get("/api/dashboard/{team_code}")
def dashboard(team_code: str):
    """Team dashboard with key stats."""
    if team_code not in TEAMS:
        raise HTTPException(404)
    roster = get_team_roster(team_code)
    
    riders_data = []
    for rider in roster:
        rs = rider_stats.get(rider, {})
        cov = rs['covers'] / max(rs['outs'], 1) * 100 if rs['outs'] > 0 else 0
        avg = rs['total_score'] / rs['covers'] if rs['covers'] > 0 else 0
        riders_data.append({
            'name': rider, 'outs': rs['outs'], 'covers': rs['covers'],
            'cover_pct': round(cov, 1), 'avg_score': round(avg, 1)
        })
    
    return {
        'team': TEAMS[team_code],
        'code': team_code,
        'riders': riders_data,
        'avg_cover_pct': round(sum(r['cover_pct'] for r in riders_data) / max(len(riders_data), 1), 1),
    }

# Serve static frontend
FRONTEND_PATH = Path.home() / "dataspur" / "frontend"
FRONTEND_PATH.mkdir(exist_ok=True)
app.mount("/app", StaticFiles(directory=str(FRONTEND_PATH), html=True), name="frontend")

if __name__ == "__main__":
    print("Starting DataSpur API on http://localhost:8420")
    print("Frontend: http://localhost:8420/app")
    uvicorn.run(app, host="0.0.0.0", port=8420)