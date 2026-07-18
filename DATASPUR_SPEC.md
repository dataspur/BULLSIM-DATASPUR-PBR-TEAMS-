# DataSpur — PBR Teams Matchup Intelligence Platform

## Overview
DataSpur is a sports analytics platform that optimizes rider-bull matchups for PBR (Professional Bull Riders) Team Series. It ingests historical ride data from Probullstats, computes per-rider and per-bull statistics, and uses combinatorial optimization (Hungarian algorithm) to recommend optimal lineup assignments. The platform includes opponent-aware game theory — adjusting strategy between conservative and aggressive based on the opposing team's predicted performance.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    FRONTEND (React)                   │
│  Team Selector │ Bull Pen Loader │ Simulator │ Reports│
└──────────────────────┬───────────────────────────────┘
                       │ REST API (JSON)
┌──────────────────────┴───────────────────────────────┐
│               BACKEND (FastAPI / Python)              │
│  /api/teams  /api/team/:code/roster  /api/simulate    │
│  /api/bulls/search  /api/dashboard/:code              │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────────┐
│                  DATA LAYER                           │
│  rides.csv (61,525 outs)  │  bull_profiles.csv (7,098)│
│  rider_profiles.csv (2,898)│ rider_weights.csv (1,401)│
└──────────────────────────────────────────────────────┘
```

## Tech Stack

### Frontend (React — built in lovable.dev)
- **React 18** with TypeScript
- **Tailwind CSS** for styling (dark theme)
- **shadcn/ui** components (select, card, button, badge, tabs, dialog)
- **Recharts** for data visualization (rider stats, team comparisons)
- **React Query** for API data fetching
- **Zustand** for state management

### Backend (Python API — deploy to Railway/Render)
- **FastAPI** — REST API server
- **NumPy + SciPy** — Hungarian algorithm for matchup optimization
- **Pandas** — data loading and preprocessing
- **Uvicorn** — ASGI server

### Data
- Static CSV files (shipped with deployment)
- No database required — data is read-once at startup into memory
- Update cadence: re-deploy when new Probullstats data is scraped

## Frontend Pages & Components

### 1. Dashboard (Home)
```
┌─────────────────────────────────────────────────────┐
│  DATASPUR                    PBR TEAMS SIMULATOR     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │ 10 Teams    │ │ 62 Riders   │ │ 7,098 Bulls │   │
│  │   Active    │ │  Profiled   │ │  Indexed    │   │
│  └─────────────┘ └─────────────┘ └─────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │ Quick Start: Select Your Team                │    │
│  │ [Team Dropdown] → Go to Simulator            │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ┌──────────────────┐ ┌──────────────────┐          │
│  │ Team Standings   │ │ Recent Activity  │          │
│  │ (if available)   │ │ (last simulations)│          │
│  └──────────────────┘ └──────────────────┘          │
└─────────────────────────────────────────────────────┘
```

### 2. Simulator (Main Tool)
```
┌─────────────────────────────────────────────────────┐
│  ← Back    SIMULATOR    Carolina Cowboys             │
├──────────────────┬──────────────────────────────────┤
│                  │                                  │
│  SETUP PANEL     │  RESULTS PANEL                   │
│                  │                                  │
│  Team: [CAR ▼]  │  STRATEGY: CONSERVATIVE          │
│                  │  Target: 2 covers to win          │
│  ROSTER          │                                  │
│  □ T. Salgado   │  ┌──────────────────────────┐    │
│    60% · 86.1   │  │ OPTIMAL LINEUP           │    │
│  □ J. Lockwood  │  │                          │    │
│    50% · 87.8   │  │ 1. T. Salgado → 035 Woody│    │
│  □ G. Valleiras │  │    72% cover · 86.1 exp  │    │
│    40% · 86.1   │  │                          │    │
│  □ K. Whitehorse│  │ 2. J. Lockwood → Doc Win │    │
│    38% · 84.0   │  │    65% cover · 87.8 exp  │    │
│  □ T. Kimzey    │  │    3. ...                 │    │
│    25% · 83.0   │  │                          │    │
│                  │  │ Expected: 345.2 pts      │    │
│  BULL PEN [5]   │  │ Expected: 2.8 covers     │    │
│  ┌────────────┐ │  └──────────────────────────┘    │
│  │035 Woody ×│ │                                    │
│  │320 Rock'n ×│ │  ┌──────────────────────────┐    │
│  │122 DocWin ×│ │  │ MANUAL TEST              │    │
│  └────────────┘ │  │ Pick rider→bull matches   │    │
│                  │  │ to compare vs optimal     │    │
│  + Add Bull     │  │ [Check vs Optimal]        │    │
│                  │  └──────────────────────────┘    │
│  OPPONENT        │                                  │
│  Team: [AUS ▼]  │  ┌──────────────────────────┐    │
│                  │  │ OPPONENT PREDICTION      │    │
│  Their Bulls [5] │  │ AUS Gamblers             │    │
│  ┌────────────┐ │  │ Expected: 1.3 covers     │    │
│  │030 Whoa × │ │  │ → CONSERVATIVE strategy   │    │
│  │816 Skippy ×│ │  └──────────────────────────┘    │
│  └────────────┘ │                                    │
│                  │                                  │
│  [RUN SIMULATION]│                                  │
└──────────────────┴──────────────────────────────────┘
```

### 3. Team Dashboard
- Per-team view with rider cards showing cover %, avg score, recent form
- Rider weights displayed: volatility, pressure factor, momentum trend
- IR list with expected return dates
- Season schedule

### 4. Bull Search
- Searchable table of 7,098 bulls with buckoff %, outs, power rating
- Spin direction filter (LEFT/RIGHT)
- Quick-add to bull pen

### 5. Reports / Analysis
- Team-by-team efficiency scores
- "Games you could have won" analysis
- Worst coaching decisions highlighted
- Export to PDF

## API Endpoints

### GET /api/teams
Returns all 10 teams with roster counts.
```json
{
  "teams": [
    {"code": "CAR", "name": "Carolina Cowboys", "roster_count": 5},
    ...
  ]
}
```

### GET /api/team/{code}/roster
Returns detailed roster for a team.
```json
{
  "team": "Carolina Cowboys",
  "code": "CAR",
  "riders": [
    {"name": "Jess Lockwood", "outs": 24, "covers": 12, "cover_pct": 50.0, "avg_score": 87.8},
    ...
  ]
}
```

### POST /api/simulate
Runs the matchup optimization.
```json
// Request:
{
  "team_code": "CAR",
  "bulls": [{"id": "53146", "name": "035 Woody"}, ...],
  "opponent_team": "AUS",
  "opponent_bulls": [{"id": "45633", "name": "030 Whoa"}, ...],
  "manual_matchups": {"Jess Lockwood": "035 Woody"}  // optional
}

// Response:
{
  "team": "Carolina Cowboys",
  "optimal_matchups": [
    {"rider": "Thiago Salgado", "bull_name": "035 Woody", "cover_prob": 0.72, "exp_score": 86.1},
    ...
  ],
  "optimal_score": 345.2,
  "expected_covers": 2.8,
  "strategy": "CONSERVATIVE",
  "target_covers": 2,
  "opponent_analysis": {
    "team": "Austin Gamblers",
    "optimal_score": 265.0,
    "expected_covers": 1.3,
    "strategy_recommendation": "CONSERVATIVE — opponent weak. Need 2+ covers to win."
  },
  "manual_analysis": {
    "matchups": [...],
    "expected_score": 310.0,
    "score_diff_vs_optimal": -35.2,
    "correct_picks": 2,
    "total_picks": 5
  }
}
```

### GET /api/bulls/search?q=woody
Searches bulls by name or ID.

### GET /api/dashboard/{code}
Returns team dashboard stats including rider weights.

## Data Files

### rides.csv (61,525 rows)
Columns: rid, event_type, perf, go, rider_slug, rider_name, bull_id, bull_name, score, qualified, bull_score, stock_score, ride_plus_minus, judge_scores, ride_time, comments, evt_org, evt_tour_class, evt_city, evt_state, evt_date, evt_note, evt_has_records, evt_rid

### bull_profiles.csv (7,098 rows)
Columns: bull_id, name, power_rating, avg_bull_score, avg_adjusted, attempts, rides, avg_ride_score, buckoff_pct, pre_ride_prob, hand_advantage_pct, hand_advantage_dir, lh_wins, lh_losses, lh_pct, lh_total, rh_wins, rh_losses, rh_pct, rh_total, top_riders_wins, top_riders_losses, top_riders_pct, contractor, active, round_wins, rider_round_wins

### rider_profiles.csv (2,898 rows)
Columns: rider_slug, riding_hand, career_qual_pct

### rider_weights.csv (1,401 rows)
Columns: rider, volatility_index, post_layoff_decay_rate, pressure_factor, recent_form_weight, cold_streak_threshold, trend_momentum

## Color System (Dark Theme)
- Background: #0a0e14 (deepest), #11161d (panels)
- Borders: #1f2937, #30363d
- Primary: #58a6ff (blue)
- Success: #3fb950 (green) — high cover probability
- Warning: #d29922 (amber) — medium  
- Danger: #f85149 (red) — low cover probability, bad matchups
- Text: #c9d1d9 (primary), #8b949e (secondary)

## Integration with lovable.dev

1. Create a new project in lovable.dev
2. Use this spec as the prompt
3. The backend runs separately on Railway/Render
4. Frontend connects to backend via API_URL environment variable
5. lovable.dev can generate the entire React frontend from this spec
6. All components use shadcn/ui for consistency
7. Dark theme implemented via Tailwind classes

## Deployment

### Backend (Railway)
```bash
cd backend
pip install -r requirements.txt
python api_server.py  # Runs on port 8420
```

### Frontend (Vercel via lovable.dev)
```bash
# Set env var: VITE_API_URL=https://your-railway-app.railway.app
npm run build
```

## What the simulator does that matters

1. **Hungarian Algorithm**: Given 5 riders and 5 bulls, finds the optimal assignment that maximizes expected team score. This is mathematically guaranteed to be the best possible matchup.

2. **Opponent-Aware Strategy**: Computes opponent's expected covers. If they're predicted 1/5, the strategy shifts to CONSERVATIVE — just need 2 covers. If they're predicted 4/5, shifts to AGGRESSIVE — maximize points at all costs.

3. **Manual Tester**: Coach picks their own matchups, system shows how far from optimal they are and which picks were correct.

4. **Rider Weights**: Individual profiles adjust predictions per rider — Lockwood's recent form matters 2.4x more than career; Leme takes many outs to recover from layoffs; Winckler crumbles under pressure.