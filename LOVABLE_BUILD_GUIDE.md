# DataSpur — lovable.dev Build Guide

## Step 1: Deploy Backend First

Before building the frontend, get the API live:

### Option A: Railway (easiest)
```
1. Go to railway.app → New Project → Deploy from GitHub
2. Point to your repo, or upload ~/dataspur-deploy/backend/
3. Railway auto-detects Python + requirements.txt
4. Set start command: python main.py
5. Deploy → copy the URL (e.g., https://dataspur-api.up.railway.app)
```

### Option B: Render
```
1. Go to render.com → New Web Service
2. Upload ~/dataspur-deploy/backend/
3. Build command: pip install -r requirements.txt
4. Start command: python main.py
5. Copy the URL
```

### Option C: Run locally
```
cd ~/dataspur-deploy/backend
pip install -r requirements.txt
python main.py
# → http://localhost:8420
```

---

## Step 2: lovable.dev Prompt

Paste this EXACT prompt into lovable.dev:

```
Build a dark-themed sports analytics dashboard for PBR (Professional Bull Riders) Team Series called "DataSpur". This is a professional tool for PBR coaches and GMs to optimize rider-bull matchups during team competitions.

BACKEND API: Connect to {YOUR_API_URL} (FastAPI server already deployed).

=== PAGES ===

1. DASHBOARD (home page / route)
- Dark background (#0a0e14), panel cards (#11161d), borders (#1f2937)
- Header: "DATASPUR" logo text in blue (#58a6ff) + "PBR TEAMS SIMULATOR" badge
- Three stat cards showing counts: 10 Teams, 62 Riders, 7,098 Bulls (fetch from GET /api/teams)
- Quick Start section: team dropdown → "Go to Simulator" button that navigates to /simulate
- Bottom: Team standings grid showing all 10 teams with win/loss if available

2. SIMULATOR (main page /simulate)
Two-column layout:

LEFT PANEL (Setup):
- Team selector dropdown (GET /api/teams)
- When team selected, show roster list from GET /api/team/{code}/roster
  - Each rider shows: checkbox, name, cover %, avg score
  - Highest cover riders auto-selected
- Bull Pen section:
  - Search input that calls GET /api/bulls/search?q={query}&limit=20
  - Search results show bull name, ID, buckoff %
  - Click to add bull to pen (max 5)
  - Selected bulls shown as removable tags
  - Quick-add input for bull IDs
- Opponent section:
  - Opponent team dropdown
  - Their bull pen (same add/remove UI)
- "RUN SIMULATION" button (green, prominent)

RIGHT PANEL (Results):
- Strategy badge: CONSERVATIVE (green) / BALANCED (amber) / AGGRESSIVE (red)
  based on POST /api/simulate response
- Two stat cards: "Optimal Score" and "Expected Covers"
- "OPTIMAL LINEUP" section:
  - Each rider-bull pair shown as a card
  - Rider name, bull name, cover probability (color-coded: green >60%, amber 35-60%, red <35%)
  - Expected score, career stats
- "OPPONENT PREDICTION" section (if opponent configured):
  - Shows their expected score and covers
  - Strategy recommendation text
- "MANUAL TEST" section:
  - Dropdown for each rider to pick a bull
  - "Check vs Optimal" button
  - Shows which picks were correct/wrong
  - Shows score difference vs optimal

3. TEAM DASHBOARD (/team/:code)
- Team header with name and stats
- Rider grid: cards showing cover %, avg score, volatility, pressure factor, momentum trend
- IR list if available

=== API ENDPOINTS ===

GET /api/teams → { teams: [{ code, name, roster_count }] }
GET /api/team/{code}/roster → { team, code, riders: [{ name, outs, covers, cover_pct, avg_score }] }
POST /api/simulate → {
  team_code: "CAR",
  bulls: [{ id, name }],
  opponent_team?: "AUS",
  opponent_bulls?: [{ id, name }],
  manual_matchups?: { "Rider Name": "Bull Name" }
}
Returns: { optimal_matchups, optimal_score, expected_covers, strategy, target_covers, opponent_analysis, manual_analysis }
GET /api/bulls/search?q=woody&limit=20 → { bulls: [{ id, name, outs, buckoff_pct }] }
GET /api/dashboard/{code} → { team, riders: [{ name, cover_pct, avg_score }] }

=== TECH REQUIREMENTS ===

Use:
- React 18 with TypeScript
- Tailwind CSS (dark theme — colors above)
- shadcn/ui components (Card, Button, Badge, Select, Input, Tabs, Dialog)
- React Query (@tanstack/react-query) for all API calls
- Zustand for state management (current team, bulls, opponent, simulation result)
- Recharts for any charts
- lucide-react for icons
- React Router for navigation (/, /simulate, /team/:code)

Store the API URL in an env variable: VITE_API_URL

=== DESIGN ===
Make this look like a major sports property analytics platform — think ESPN, Sportradar, or an NFL analytics dashboard. Dark, professional, data-dense.
```

---

## Step 3: lovable.dev Project Setup

After the prompt generates the app:

1. Go to Project Settings → Environment Variables
2. Add: `VITE_API_URL` = your Railway/Render URL (e.g., `https://dataspur-api.up.railway.app`)
3. If running locally: `VITE_API_URL` = `http://localhost:8420`

---

## Step 4: Plugins & Integrations in lovable.dev

No special plugins needed. lovable.dev handles:
- React + Vite + TypeScript (built-in)
- Tailwind CSS (auto-detected)
- shadcn/ui (auto-installed via prompt)

If shadcn/ui components don't auto-install, ask lovable.dev:
"Install shadcn/ui and add these components: card, button, badge, select, input, tabs, dialog, separator"

---

## Step 5: Test & Deploy

1. Click "Preview" in lovable.dev — test the simulator
2. Select Carolina Cowboys → add bulls → run simulation → verify results match
3. Click "Deploy" → deploys to Vercel automatically
4. Your professional PBR analytics platform is live

---

## NOTES

- The backend MUST be deployed first and accessible before lovable.dev can fetch data
- CORS is already enabled on the backend (allow_origins=["*"])
- The API reads CSV data at startup — first request may take 2-3 seconds
- Data updates: redeploy backend with new CSV files when scraping new Probullstats data