#!/usr/bin/env python3
"""
Deep-dive analysis: show specific bad matchup decisions that cost games.
For each team, identify the worst coaching decisions and what 
the optimal lineup would have been.
"""
import csv, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

DATA = Path.home() / "dataspur" / "data"

# Load and infer teams (same as analyzer)
with open(DATA / "rides.csv") as f:
    all_rides = list(csv.DictReader(f))
ts_rides = [r for r in all_rides if r.get('evt_tour_class') == 'Team Series' and r['event_type'] == 'BR']

rider_team_counts = defaultdict(lambda: defaultdict(int))
for r in ts_rides:
    note = r.get('evt_note', '').strip()
    teams = re.findall(r'[A-Z]{2,4}', note)
    for t in teams:
        rider_team_counts[r['rider_name']][t] += 1

rider_to_team = {}
for rider, counts in rider_team_counts.items():
    if not counts: continue
    best = max(counts, key=counts.get)
    if counts[best] / sum(counts.values()) >= 0.45:
        rider_to_team[rider] = best

# Compute stats from ALL rides
rider_stats = defaultdict(lambda: {'outs': 0, 'covers': 0, 'total_score': 0})
bull_stats = defaultdict(lambda: {'outs': 0, 'buckoffs': 0})

for r in all_rides:
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
        except: pass
    
    if bull_id:
        bull_stats[bull_id]['outs'] += 1
        if not qualified:
            bull_stats[bull_id]['buckoffs'] += 1

def predict(rider, bull_id):
    rs = rider_stats.get(rider, {'outs': 0, 'covers': 0, 'total_score': 0})
    bs = bull_stats.get(bull_id, {'outs': 0, 'buckoffs': 0})
    
    rider_cov = rs['covers'] / max(rs['outs'], 1)
    bull_buck = bs['buckoffs'] / max(bs['outs'], 1)
    
    if rs['outs'] >= 5 and bs['outs'] >= 5:
        prob = (rider_cov + (1 - bull_buck)) / 2
    elif rs['outs'] >= 5:
        prob = rider_cov
    elif bs['outs'] >= 5:
        prob = 1 - bull_buck
    else:
        prob = 0.25
    
    exp_score = rs['total_score'] / max(rs['covers'], 1) if rs['covers'] > 0 else 85.0
    return prob, exp_score

# Build games
by_rid = defaultdict(list)
for r in ts_rides:
    by_rid[r['rid']].append(r)

team_codes = {'AZ': 'Arizona Ridge Riders', 'TEX': 'Texas Rattlers', 'MIS': 'Missouri Thunder',
              'FL': 'Florida Freedom', 'CAR': 'Carolina Cowboys', 'AUS': 'Austin Gamblers',
              'KANS': 'Kansas City Outlaws', 'NASH': 'Nashville Stampede', 'NY': 'New York Mavericks',
              'OKL': 'Oklahoma Wildcatters'}

# Deep dive: for each team, find the top 5 worst coaching decisions
team_games = defaultdict(list)

for rid, rides in by_rid.items():
    note = rides[0].get('evt_note', '').strip()
    date = rides[0].get('evt_date', '')
    teams_in_note = re.findall(r'[A-Z]{2,4}', note)
    if len(teams_in_note) < 2: continue
    
    team_a, team_b = teams_in_note[:2]
    
    team_a_rides = [r for r in rides if rider_to_team.get(r['rider_name']) == team_a]
    team_b_rides = [r for r in rides if rider_to_team.get(r['rider_name']) == team_b]
    
    if len(team_a_rides) < 3 or len(team_b_rides) < 3: continue
    
    def compute_matchups(team_rides):
        riders = [r['rider_name'] for r in team_rides]
        bulls = [(r.get('bull_id', ''), r.get('bull_name', '')) for r in team_rides]
        n = len(riders)
        
        cost = np.zeros((n, n))
        for i, rider in enumerate(riders):
            for j, (bid, bname) in enumerate(bulls):
                prob, exp_score = predict(rider, bid)
                cost[i, j] = -exp_score * prob
        
        row_ind, col_ind = linear_sum_assignment(cost)
        
        actual_score = sum(float(r.get('score', 0) or 0) for r in team_rides)
        actual_qual = sum(1 for r in team_rides if str(r.get('qualified', '')).lower() in ('true', '1', '1.0', 'yes'))
        optimal_score = -cost[row_ind, col_ind].sum()
        
        matchups = []
        for idx, (ri, bj) in enumerate(zip(row_ind, col_ind)):
            rider = riders[ri]
            bull_name = bulls[bj][1]
            prob, score = predict(rider, bulls[bj][0])
            actual_rider_bull = team_rides[ri]
            actual_bull = actual_rider_bull.get('bull_name', '?')
            is_optimal = (bull_name == actual_bull)
            matchups.append({
                'rider': rider, 'actual_bull': actual_bull,
                'opt_bull': bull_name, 'is_optimal': is_optimal,
                'cover_prob': prob, 'exp_score': score,
                'actual_score': float(actual_rider_bull.get('score', 0) or 0),
                'actual_qual': str(actual_rider_bull.get('qualified', '')).lower() in ('true', '1', '1.0', 'yes'),
            })
        
        return actual_score, actual_qual, optimal_score, matchups
    
    actual_a, qual_a, opt_a, matchups_a = compute_matchups(team_a_rides)
    actual_b, qual_b, opt_b, matchups_b = compute_matchups(team_b_rides)
    
    for match in matchups_a:
        if not match['is_optimal']:
            team_games[team_a].append({
                'rid': rid, 'date': date, 'opponent': team_b, 'note': note,
                'rider': match['rider'], 'actual_bull': match['actual_bull'],
                'opt_bull': match['opt_bull'], 'cover_prob': match['cover_prob'],
                'actual_score': match['actual_score'], 'actual_qual': match['actual_qual'],
                'team_score': actual_a, 'opp_score': actual_b,
                'delta': opt_a - actual_a,
            })
    
    for match in matchups_b:
        if not match['is_optimal']:
            team_games[team_b].append({
                'rid': rid, 'date': date, 'opponent': team_a, 'note': note,
                'rider': match['rider'], 'actual_bull': match['actual_bull'],
                'opt_bull': match['opt_bull'], 'cover_prob': match['cover_prob'],
                'actual_score': match['actual_score'], 'actual_qual': match['actual_qual'],
                'team_score': actual_b, 'opp_score': actual_a,
                'delta': opt_b - actual_b,
            })

# Print report
print("=" * 70)
print("PBR TEAMS — WORST COACHING DECISIONS BY TEAM")
print("=" * 70)

for team in sorted(team_games, key=lambda t: len(team_games[t]), reverse=True):
    games = team_games[team]
    if len(games) < 3:
        continue
    
    total_bad = len(games)
    total_delta = sum(g['delta'] for g in games)
    avg_delta = total_delta / max(total_bad, 1)
    
    print(f"\n{'='*70}")
    print(f"TEAM: {team_codes.get(team, team)} ({team})")
    print(f"  Total bad assignments: {total_bad}")
    print(f"  Total recoverable points: {total_delta:.0f}")
    print(f"  Average pts per bad decision: {avg_delta:.1f}")
    print(f"\n  TOP 5 WORST SINGLE DECISIONS (rider assigned to wrong bull):")
    
    worst = sorted(games, key=lambda g: g['delta'], reverse=True)[:5]
    for i, g in enumerate(worst):
        print(f"\n  [{i+1}] {g['rid']} | {g['date']} | {g['note']}")
        print(f"      Rider: {g['rider']}")
        print(f"      ACTUALLY rode: {g['actual_bull']} → {'QUALIFIED' if g['actual_qual'] else 'BUCKED OFF'} ({g['actual_score']:.1f} pts)")
        print(f"      SHOULD have rode: {g['opt_bull']} → {g['cover_prob']*100:.0f}% cover probability")
        print(f"      Team score: {g['team_score']:.1f} vs {g['opp_score']:.1f}")
        print(f"      Improvement: {g['delta']:.0f} pts | Game winnable: {'YES' if g['delta'] > abs(g['team_score']-g['opp_score']) else 'NO'}")

# Summary table
print(f"\n{'='*70}")
print("SUMMARY: BAD DECISIONS PER TEAM")
print(f"{'='*70}")
print(f"  {'TEAM':<6s} {'BAD PICKS':>10s} {'TOTAL PTS LOST':>15s} {'AVG/BAD':>8s}")
print(f"  {'-'*45}")
for team in sorted(team_games, key=lambda t: sum(g['delta'] for g in team_games[t]), reverse=True):
    games = team_games[team]
    if len(games) < 3: continue
    td = sum(g['delta'] for g in games)
    avg = td / max(len(games), 1)
    print(f"  {team:<6s} {len(games):>10d} {td:>15.0f} {avg:>8.1f}")