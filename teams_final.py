#!/usr/bin/env python3
"""
PBR Teams Complete Analysis — Every Team, Every Bad Decision
Answers:
1. Games flipped L→W per team (not just points)
2. Avg points lost per match
3. Championship-costing bad lineups
4. Detailed per-game breakdown
"""
import csv, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

DATA = Path.home() / "dataspur" / "data"

print("=" * 70)
print("PBR TEAMS — COMPLETE MATCHUP OPTIMIZATION AUDIT")
print("=" * 70)

# ====== LOAD & INFER TEAMS ======
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
            rider_stats[rider]['total_score'] += float(r.get('score', 0) or 0)
        except: pass
    
    if bull_id:
        bull_stats[bull_id]['outs'] += 1
        if not qualified:
            bull_stats[bull_id]['buckoffs'] += 1

def predict(rider, bull_id):
    rs, bs = rider_stats.get(rider, {}), bull_stats.get(bull_id, {})
    rider_cov = rs.get('covers', 0) / max(rs.get('outs', 1), 1)
    bull_buck = bs.get('buckoffs', 0) / max(bs.get('outs', 1), 1)
    
    if rs.get('outs', 0) >= 5 and bs.get('outs', 0) >= 5:
        prob = (rider_cov + (1 - bull_buck)) / 2
    elif rs.get('outs', 0) >= 5:
        prob = rider_cov
    elif bs.get('outs', 0) >= 5:
        prob = 1 - bull_buck
    else:
        prob = 0.25
    
    exp_score = rs.get('total_score', 0) / max(rs.get('covers', 1), 1) if rs.get('covers', 0) > 0 else 85.0
    return prob, exp_score

# ====== BUILD GAMES ======
by_rid = defaultdict(list)
for r in ts_rides:
    by_rid[r['rid']].append(r)

team_codes = {'AZ': 'Arizona Ridge Riders', 'TEX': 'Texas Rattlers', 'MIS': 'Missouri Thunder',
              'FL': 'Florida Freedom', 'CAR': 'Carolina Cowboys', 'AUS': 'Austin Gamblers',
              'KANS': 'Kansas City Outlaws', 'NASH': 'Nashville Stampede', 'NY': 'New York Mavericks',
              'OKL': 'Oklahoma Wildcatters'}

# ====== PER-GAME ANALYSIS ======
all_results = []

for rid, rides in by_rid.items():
    note = rides[0].get('evt_note', '').strip()
    date = rides[0].get('evt_date', '')
    teams_in_note = re.findall(r'[A-Z]{2,4}', note)
    if len(teams_in_note) < 2: continue
    
    ta, tb = teams_in_note[:2]
    
    rides_a = [r for r in rides if rider_to_team.get(r['rider_name']) == ta]
    rides_b = [r for r in rides if rider_to_team.get(r['rider_name']) == tb]
    
    if len(rides_a) < 3 or len(rides_b) < 3: continue
    
    def optimize(team_rides):
        riders = [r['rider_name'] for r in team_rides]
        bulls = [(r.get('bull_id', ''), r.get('bull_name', '')) for r in team_rides]
        n = len(riders)
        cost = np.zeros((n, n))
        for i, rider in enumerate(riders):
            for j, (bid, _) in enumerate(bulls):
                prob, exp_score = predict(rider, bid)
                cost[i, j] = -exp_score * prob
        row_ind, col_ind = linear_sum_assignment(cost)
        
        actual_score = sum(float(r.get('score', 0) or 0) for r in team_rides)
        actual_qual = sum(1 for r in team_rides if str(r.get('qualified', '')).lower() in ('true', '1', '1.0', 'yes'))
        optimal_score = float(-cost[row_ind, col_ind].sum())
        optimal_qual = sum(1 for i in range(n) if -cost[row_ind[i], col_ind[i]] > 0)
        
        return actual_score, actual_qual, optimal_score, optimal_qual, row_ind, col_ind, riders, bulls
    
    act_a, qual_a, opt_a, oqual_a, ri_a, ci_a, riders_a_list, bulls_a_list = optimize(rides_a)
    act_b, qual_b, opt_b, oqual_b, ri_b, ci_b, riders_b_list, bulls_b_list = optimize(rides_b)
    
    actual_winner = ta if act_a > act_b else tb if act_b > act_a else 'TIE'
    opt_winner = ta if opt_a > opt_b else tb if opt_b > opt_a else 'TIE'
    flipped = (actual_winner != opt_winner and actual_winner != 'TIE' and opt_winner != 'TIE')
    
    # Build bad assignment details for each team
    bad_a = []
    for idx, (ri, ci) in enumerate(zip(ri_a, ci_a)):
        actual_bull = rides_a[idx].get('bull_name', '?')
        opt_bull = bulls_a_list[ci][1]
        if actual_bull != opt_bull:
            rider = riders_a_list[ri]
            prob, exp_s = predict(rider, bulls_a_list[ci][0])
            act_s = float(rides_a[idx].get('score', 0) or 0)
            bad_a.append({'rider': rider, 'actual_bull': actual_bull, 'opt_bull': opt_bull,
                         'cover_prob': prob, 'actual_score': act_s, 'delta': exp_s * prob - act_s})
    
    bad_b = []
    for idx, (ri, ci) in enumerate(zip(ri_b, ci_b)):
        actual_bull = rides_b[idx].get('bull_name', '?')
        opt_bull = bulls_b_list[ci][1]
        if actual_bull != opt_bull:
            rider = riders_b_list[ri]
            prob, exp_s = predict(rider, bulls_b_list[ci][0])
            act_s = float(rides_b[idx].get('score', 0) or 0)
            bad_b.append({'rider': rider, 'actual_bull': actual_bull, 'opt_bull': opt_bull,
                         'cover_prob': prob, 'actual_score': act_s, 'delta': exp_s * prob - act_s})
    
    all_results.append({
        'rid': rid, 'date': date, 'note': note,
        'team_a': ta, 'team_b': tb,
        'act_a': act_a, 'act_b': act_b,
        'opt_a': opt_a, 'opt_b': opt_b,
        'qual_a': qual_a, 'qual_b': qual_b,
        'oqual_a': oqual_a, 'oqual_b': oqual_b,
        'actual_winner': actual_winner, 'opt_winner': opt_winner,
        'flipped': flipped,
        'margin': abs(act_a - act_b),
        'delta_a': opt_a - act_a,
        'delta_b': opt_b - act_b,
        'bad_a': bad_a, 'bad_b': bad_b,
    })

# ====== TEAM-BY-TEAM REPORT ======
print(f"\nAnalyzed {len(all_results)} games\n")

# Per-team stats
team_stats = defaultdict(lambda: {
    'games': 0, 'wins': 0, 'losses': 0, 'ties': 0,
    'flipped_losses': 0,  # losses that could be wins
    'flipped_wins': 0,    # wins that could be losses (bad coaching despite winning)
    'total_delta': 0.0,
    'bad_assignments': 0,
    'pt_diff': 0.0,        # total actual point differential
    'opt_pt_diff': 0.0,    # total optimal point differential
    'flipped_games': [],   # list of game details
    'worst_decisions': [], # top bad assignments
})

for r in all_results:
    for team, act_score, opt_score, delta, bad_list, opp_score in [
        (r['team_a'], r['act_a'], r['opt_a'], r['delta_a'], r['bad_a'], r['opt_b']),
        (r['team_b'], r['act_b'], r['opt_b'], r['delta_b'], r['bad_b'], r['opt_a']),
    ]:
        ts = team_stats[team]
        ts['games'] += 1
        ts['total_delta'] += delta
        ts['bad_assignments'] += len(bad_list)
        ts['pt_diff'] += act_score - opp_score
        
        if team == r['team_a']:
            ts['opt_pt_diff'] += opt_score - r['opt_b']
        else:
            ts['opt_pt_diff'] += opt_score - r['opt_a']
        
        won = (r['actual_winner'] == team)
        lost = (r['actual_winner'] != team and r['actual_winner'] != 'TIE')
        tied = (r['actual_winner'] == 'TIE')
        
        if won: ts['wins'] += 1
        elif tied: ts['ties'] += 1
        else: ts['losses'] += 1
        
        # Did better matchups flip this game?
        if lost and r['flipped']:
            ts['flipped_losses'] += 1
        elif won and r['flipped']:
            ts['flipped_wins'] += 1  # won but opponent could have won with better matchups
        
        # Store flipped game details for this team
        if (lost and r['flipped']) or (won and r['flipped']):
            ts['flipped_games'].append({
                'rid': r['rid'], 'date': r['date'], 'note': r['note'],
                'opponent': r['team_b'] if team == r['team_a'] else r['team_a'],
                'team_score': act_score, 'opp_score': opp_score,
                'actual_result': 'W' if won else 'L',
                'opt_team_score': opt_score,
                'opt_opp_score': r['opt_b'] if team == r['team_a'] else r['opt_a'],
                'delta': delta,
                'margin': abs(act_score - opp_score),
            })
        
        ts['worst_decisions'].extend(bad_list)

# Sort worst decisions per team
for team in team_stats:
    team_stats[team]['worst_decisions'].sort(key=lambda x: x['delta'], reverse=True)

# ====== PRINT TEAM REPORTS ======
# Sort teams by: most losses flipped (the key metric the user wants)
teams_by_flipped = sorted(team_stats.items(), key=lambda x: x[1]['flipped_losses'], reverse=True)

print("=" * 70)
print("RANKED BY GAMES THAT COULD HAVE BEEN WINS (L→W)")
print("=" * 70)
print(f"  {'TEAM':<30s} {'GAMES':>5s} {'W':>5s} {'L':>5s} {'FLIPPED L→W':>12s} {'BAD ASGN':>10s} {'Δ PTS/GAME':>11s} {'PT DIFF':>8s} {'OPT DIFF':>9s}")
print(f"  {'-'*100}")

for team, ts in teams_by_flipped:
    name = team_codes.get(team, team)
    avg_delta = ts['total_delta'] / max(ts['games'], 1)
    print(f"  {name:<30s} {ts['games']:>5d} {ts['wins']:>5d} {ts['losses']:>5d} "
          f"{ts['flipped_losses']:>12d} {ts['bad_assignments']:>10d} "
          f"{avg_delta:>+11.1f} {ts['pt_diff']:>+8.1f} {ts['opt_pt_diff']:>+9.1f}")

# ====== MOST LOSSES FLIPPED ======
print(f"\n{'='*70}")
print("TOP 5: TEAMS WITH MOST PREVENTABLE LOSSES")
print(f"{'='*70}")

for team, ts in teams_by_flipped[:5]:
    name = team_codes.get(team, team)
    print(f"\n  {name} ({team}): {ts['flipped_losses']}/{ts['losses']} losses were winnable "
          f"({ts['flipped_losses']/max(ts['losses'],1)*100:.0f}% of losses)")
    ties_str = f"-{ts['ties']}" if ts['ties'] else ""
    print(f"  Record: {ts['wins']}-{ts['losses']}{ties_str} "
          f"(would be {ts['wins']+ts['flipped_losses']}-{ts['losses']-ts['flipped_losses']} "
          f"with optimal matchups)")
    avg_delta = ts['total_delta'] / max(ts['games'], 1)
    print(f"  Avg pts lost per game: {avg_delta:+.1f}")
    print(f"  Worst games where coaching cost the win:")
    
    flipped = sorted(ts['flipped_games'], key=lambda g: g['delta'], reverse=True)[:5]
    for i, g in enumerate(flipped):
        if g['actual_result'] == 'L':  # Only show actual losses that could flip
            print(f"    [{i+1}] {g['rid']} | {g['date']} | vs {g['opponent']}")
            print(f"        Actual: LOST {g['team_score']:.1f} - {g['opp_score']:.1f} "
                  f"(margin: {g['margin']:.1f})")
            print(f"        Optimal: WIN {g['opt_team_score']:.1f} - {g['opt_opp_score']:.1f}")
            print(f"        Recoverable: {g['delta']:.0f} pts")
    
    # Show worst individual decisions
    print(f"\n  Worst single bad assignments:")
    for i, b in enumerate(ts['worst_decisions'][:3]):
        print(f"    [{i+1}] {b['rider']}: rode {b['actual_bull']} ({b['actual_score']:.1f} pts) "
              f"→ should have ridden {b['opt_bull']} ({b['cover_prob']*100:.0f}% cover) "
              f"[+{b['delta']:.0f} pts]")

# ====== CHAMPIONSHIP ANALYSIS ======
print(f"\n{'='*70}")
print("CHAMPIONSHIP IMPLICATIONS — DID BAD LINEUPS COST A TITLE?")
print(f"{'='*70}")

# Find championship/title-clinching scenarios
# PBR Teams Championship in Las Vegas (Nov)
# Look at late-season games where flipped losses would change standings

# Check which teams have the most flipped losses and calculate standing impact
# If a wildcard spot was decided by 1-2 games, those flipped games matter

# Find the standings effect
print("\nStandings impact (if all flipped games went optimal):")
for team, ts in sorted(team_stats.items(), 
                       key=lambda x: x[1]['flipped_losses'] * 2 + x[1]['flipped_wins'], 
                       reverse=True):
    name = team_codes.get(team, team)
    orig_win_pct = ts['wins'] / max(ts['games'], 1)
    opt_wins = ts['wins'] + ts['flipped_losses']
    opt_losses = ts['losses'] - ts['flipped_losses']
    opt_win_pct = opt_wins / max(ts['games'], 1)
    improvement = (opt_win_pct - orig_win_pct) * 100
    
    if ts['flipped_losses'] > 0:
        print(f"  {name}: {ts['wins']}-{ts['losses']} ({orig_win_pct*100:.0f}%) "
              f"→ {opt_wins}-{opt_losses} ({opt_win_pct*100:.0f}%) "
              f"[+{improvement:.0f}% pts | +{ts['flipped_losses']} wins, -{ts['flipped_wins']} at risk]")

# Look for specific Vegas/championship events
print("\nChampionship/Vegas events with bad lineups:")
vegas_events = [r for r in all_results if 'Vegas' in str(r.get('date', '')) or 'Las Vegas' in str(r.get('date', ''))
                or r['rid'].startswith('AT7')]  # AT7xx are likely late-season/championship
for r in vegas_events:
    if r['flipped']:
        losing_team = r['team_a'] if r['actual_winner'] == r['team_b'] else r['team_b']
        winning_team = r['actual_winner']
        print(f"  {r['rid']} | {r['date']} | {r['note']}: {losing_team} lost {r['act_a']:.1f}-{r['act_b']:.1f} "
              f"→ optimal {winning_team if r['opt_winner']==winning_team else losing_team} wins "
              f"{r['opt_a']:.1f}-{r['opt_b']:.1f}")

print("\nDone.")