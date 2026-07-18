#!/usr/bin/env python3
"""
PBR Teams Matchup Analyzer
1. Scrape team rosters and game results from Probullstats
2. Infer rider-team assignments from event data
3. Run Hungarian assignment optimizer to find optimal matchups
4. Flag games where bad matchups cost wins
"""
import csv, re, time, json
from collections import defaultdict, Counter
from pathlib import Path
from itertools import permutations
import numpy as np
from scipy.optimize import linear_sum_assignment

DATA = Path.home() / "dataspur" / "data"

# ============ 1. INFER RIDER-TEAM ASSIGNMENTS ============
print("=" * 60)
print("PBR TEAMS MATCHUP ANALYZER")
print("=" * 60)

# Load Team Series rides
with open(DATA / "rides.csv") as f:
    all_rides = list(csv.DictReader(f))

ts_rides = [r for r in all_rides 
            if r.get('evt_tour_class') == 'Team Series' 
            and r['event_type'] == 'BR']

print(f"\n[1] Team Series rides: {len(ts_rides):,}")

# Infer rider teams from event notes
rider_team_counts = defaultdict(lambda: defaultdict(int))
for r in ts_rides:
    note = r.get('evt_note', '').strip()
    rider = r['rider_name']
    teams = re.findall(r'[A-Z]{2,4}', note)
    for t in teams:
        rider_team_counts[rider][t] += 1

# Assign each rider to their most frequent team code
rider_to_team = {}
for rider, counts in rider_team_counts.items():
    if not counts:
        continue
    best_team = max(counts, key=counts.get)
    total = sum(counts.values())
    confidence = counts[best_team] / total
    if confidence >= 0.45:  # Lower threshold since opponents inflate other counts
        rider_to_team[rider] = best_team

print(f"  Riders assigned: {len(rider_to_team):,}")
team_sizes = Counter(rider_to_team.values())
for team, count in team_sizes.most_common():
    print(f"    {team}: {count} riders")

# ============ 2. BUILD GAME-LEVEL DATA ============
print(f"\n[2] Building game-level data...")

# Group rides by event (each event = one game)
by_rid = defaultdict(list)
for r in ts_rides:
    by_rid[r['rid']].append(r)

games = []
for rid, rides in by_rid.items():
    note = rides[0].get('evt_note', '').strip()
    date = rides[0].get('evt_date', '')
    teams_in_note = re.findall(r'[A-Z]{2,4}', note)
    
    if len(teams_in_note) < 2:
        continue
    
    team_a, team_b = teams_in_note[:2]
    
    # Assign rides to teams
    team_a_rides = []
    team_b_rides = []
    unassigned = []
    
    for r in rides:
        rider = r['rider_name']
        if rider in rider_to_team:
            if rider_to_team[rider] == team_a:
                team_a_rides.append(r)
            elif rider_to_team[rider] == team_b:
                team_b_rides.append(r)
            else:
                unassigned.append(r)
        else:
            unassigned.append(r)
    
    # Only include games where we can assign both teams
    if len(team_a_rides) >= 3 and len(team_b_rides) >= 3:
        # Calculate team scores
        def team_score(rides_list):
            total = 0
            qualified = 0
            for r in rides_list:
                s = r.get('score', '')
                try:
                    s = float(s) if s and s.replace('.','').replace('-','').isdigit() else 0
                except:
                    s = 0
                total += s
                if s > 0:
                    qualified += 1
            return total, qualified
        
        score_a, qual_a = team_score(team_a_rides)
        score_b, qual_b = team_score(team_b_rides)
        
        winner = team_a if score_a > score_b else team_b if score_b > score_a else 'TIE'
        margin = abs(score_a - score_b)
        
        games.append({
            'rid': rid, 'date': date, 'note': note,
            'team_a': team_a, 'team_b': team_b,
            'score_a': score_a, 'qual_a': qual_a,
            'score_b': score_b, 'qual_b': qual_b,
            'winner': winner, 'margin': margin,
            'riders_a': [r['rider_name'] for r in team_a_rides],
            'riders_b': [r['rider_name'] for r in team_b_rides],
            'bulls_a': [{'name': r.get('bull_name',''), 'id': r.get('bull_id','')} for r in team_a_rides],
            'bulls_b': [{'name': r.get('bull_name',''), 'id': r.get('bull_id','')} for r in team_b_rides],
            'rides': team_a_rides + team_b_rides + unassigned,
        })

print(f"  Games reconstructed: {len(games):,}")

# ============ 3. COMPUTE RIDER-BULL COVER PROBABILITIES ============
print(f"\n[3] Computing rider-bull cover probabilities...")

# For each rider and bull in our data, compute historical cover rate
# This is our model's output — probability rider covers this bull type

# Use bull_id as unique identifier, with rider career stats
rider_stats = defaultdict(lambda: {'outs': 0, 'covers': 0, 'total_score': 0})
bull_stats = defaultdict(lambda: {'outs': 0, 'buckoffs': 0})

# Compute stats from ALL rides (not just Teams) for richer data
for r in all_rides:
    if r['event_type'] != 'BR':
        continue
    rider = r['rider_name']
    bull_id = r.get('bull_id', '').strip()
    qualified = str(r.get('qualified', '')).lower() in ('true', '1', '1.0', 'yes')
    
    rider_stats[rider]['outs'] += 1
    if qualified:
        rider_stats[rider]['covers'] += 1
        try:
            s = float(r.get('score', 0) or 0)
            rider_stats[rider]['total_score'] += s
        except:
            pass
    
    if bull_id:
        bull_stats[bull_id]['outs'] += 1
        if not qualified:
            bull_stats[bull_id]['buckoffs'] += 1

print(f"  Riders with stats: {len(rider_stats):,}")
print(f"  Bulls with stats: {len(bull_stats):,}")

def predict_cover(rider_name, bull_id):
    """Simple model: combine rider cover% with bull buckoff%."""
    rs = rider_stats.get(rider_name, {'outs': 0, 'covers': 0})
    bs = bull_stats.get(bull_id, {'outs': 0, 'buckoffs': 0})
    
    rider_cover_pct = rs['covers'] / max(rs['outs'], 1)
    bull_buckoff_pct = bs['buckoffs'] / max(bs['outs'], 1)
    
    # Blend: if we have data for both, weight equally
    if rs['outs'] >= 5 and bs['outs'] >= 5:
        prob = (rider_cover_pct + (1 - bull_buckoff_pct)) / 2
    elif rs['outs'] >= 5:
        prob = rider_cover_pct
    elif bs['outs'] >= 5:
        prob = 1 - bull_buckoff_pct
    else:
        prob = 0.25  # league average
    
    # Expected score if covered
    if rs['outs'] > 0 and rs['covers'] > 0:
        exp_score = rs['total_score'] / rs['covers']
    else:
        exp_score = 85.0
    
    return prob, exp_score

# ============ 4. OPTIMIZE MATCHUPS ============
print(f"\n[4] Running matchup optimization...")

results = []

for game in games:
    riders_a = game['riders_a']
    bulls_a = game['bulls_a']  # Bulls faced by team A
    riders_b = game['riders_b']
    bulls_b = game['bulls_b']  # Bulls faced by team B
    
    # Team A optimization: given their riders and the bulls they faced,
    # what was the optimal assignment?
    n_a = len(riders_a)
    
    # Build cost matrix: negative expected score (we minimize, so maximize exp score)
    cost_a = np.zeros((n_a, n_a))
    for i, rider in enumerate(riders_a):
        for j, bull in enumerate(bulls_a):
            prob, exp_score = predict_cover(rider, bull.get('id', ''))
            cost_a[i, j] = -exp_score * prob  # Negative for minimization
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_a)
    optimal_score_a = -cost_a[row_ind, col_ind].sum()
    
    # What was the actual score?
    actual_score_a = game['score_a']
    optimal_qual_a = sum(1 for i in range(n_a) if -cost_a[row_ind[i], col_ind[i]] > 0)
    
    # Team B optimization
    n_b = len(riders_b)
    cost_b = np.zeros((n_b, n_b))
    for i, rider in enumerate(riders_b):
        for j, bull in enumerate(bulls_b):
            prob, exp_score = predict_cover(rider, bull.get('id', ''))
            cost_b[i, j] = -exp_score * prob
    
    row_ind_b, col_ind_b = linear_sum_assignment(cost_b)
    optimal_score_b = -cost_b[row_ind_b, col_ind_b].sum()
    actual_score_b = game['score_b']
    optimal_qual_b = sum(1 for i in range(n_b) if -cost_b[row_ind_b[i], col_ind_b[i]] > 0)
    
    # Calculate what the game outcome WOULD have been with optimal matchups
    opt_winner = 'A' if optimal_score_a > optimal_score_b else 'B' if optimal_score_b > optimal_score_a else 'TIE'
    actual_winner = 'A' if actual_score_a > actual_score_b else 'B' if actual_score_b > actual_score_a else 'TIE'
    
    # Did a team lose that could have won?
    flipped = (actual_winner != opt_winner and actual_winner != 'TIE' and opt_winner != 'TIE')
    
    results.append({
        'rid': game['rid'], 'date': game['date'], 'note': game['note'],
        'actual_a': actual_score_a, 'actual_b': actual_score_b,
        'actual_winner': game['winner'],
        'opt_a': optimal_score_a, 'opt_b': optimal_score_b,
        'opt_winner': f"{game['team_a']} wins" if opt_winner == 'A' else f"{game['team_b']} wins" if opt_winner == 'B' else 'TIE',
        'actual_qual_a': game['qual_a'], 'actual_qual_b': game['qual_b'],
        'opt_qual_a': optimal_qual_a, 'opt_qual_b': optimal_qual_b,
        'flipped': flipped, 'margin': game['margin'],
        'delta_a': optimal_score_a - actual_score_a,
        'delta_b': optimal_score_b - actual_score_b,
        'team_a': game['team_a'], 'team_b': game['team_b'],
    })

# ============ 5. ANALYSIS ============
print(f"\n[5] Analysis ({len(results)} games)...")

# Games where winner would flip
flipped = [r for r in results if r['flipped']]
print(f"\n  FLIPPED OUTCOMES: {len(flipped)} games where better matchups change the winner")
for f in sorted(flipped, key=lambda x: x['margin'], reverse=True)[:15]:
    print(f"  {f['rid']} | {f['note']} | actual: {f['actual_winner']} won {f['actual_a']}-{f['actual_b']} | "
          f"optimal: {f['opt_winner']} ({f['opt_a']:.0f}-{f['opt_b']:.0f}) | "
          f"margin: {f['margin']:.1f} | delta: {f['delta_a']:.0f}/{f['delta_b']:.0f}")

# By team: total "points left on the table"
print(f"\n  POINTS LEFT ON TABLE (by team):")
team_deltas = defaultdict(lambda: {'total_delta': 0, 'games': 0, 'losses': 0, 'flipped_losses': 0})
for r in results:
    team_deltas[r['team_a']]['total_delta'] += r['delta_a']
    team_deltas[r['team_a']]['games'] += 1
    if r['actual_winner'] != r['team_a'] and r['actual_winner'] != 'TIE':
        team_deltas[r['team_a']]['losses'] += 1
        if r['flipped']:
            team_deltas[r['team_a']]['flipped_losses'] += 1
    
    team_deltas[r['team_b']]['total_delta'] += r['delta_b']
    team_deltas[r['team_b']]['games'] += 1
    if r['actual_winner'] != r['team_b'] and r['actual_winner'] != 'TIE':
        team_deltas[r['team_b']]['losses'] += 1
        if r['flipped']:
            team_deltas[r['team_b']]['flipped_losses'] += 1

for team in sorted(team_deltas, key=lambda t: team_deltas[t]['total_delta'], reverse=True):
    td = team_deltas[team]
    avg = td['total_delta'] / max(td['games'], 1)
    print(f"  {team:<6s}: {td['total_delta']:6.0f} total pts lost | avg {avg:5.1f}/game | "
          f"{td['flipped_losses']}/{td['losses']} losses could flip | {td['games']} games")

# Wins that could have been bigger
print(f"\n  WINS WITH UNOPTIMIZED LINEUPS (team still won but left points):")
big_misses = [r for r in results if not r['flipped'] and (r['delta_a'] > 5 or r['delta_b'] > 5)]
for r in sorted(big_misses, key=lambda x: max(x['delta_a'], x['delta_b']), reverse=True)[:10]:
    print(f"  {r['rid']} | {r['note']} | {r['actual_winner']} won | "
          f"left {max(r['delta_a'], r['delta_b']):.0f} pts on table")

print("\nDone.")