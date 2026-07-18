# DataSpur Frontend — React Component Spec for lovable.dev

## Setup
```
npx create-vite@latest dataspur-frontend --template react-ts
cd dataspur-frontend
npm install @tanstack/react-query zustand recharts lucide-react
npm install -D tailwindcss autoprefixer postcss
npx tailwindcss init -p
```

## Tailwind Config (tailwind.config.js)
```js
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ds: {
          bg: '#0a0e14',
          panel: '#11161d',
          border: '#1f2937',
          'border-hover': '#30363d',
          primary: '#58a6ff',
          success: '#3fb950',
          warning: '#d29922',
          danger: '#f85149',
          text: '#c9d1d9',
          muted: '#8b949e',
        }
      }
    }
  }
}
```

## Component Tree
```
App
├── Header (logo, nav)
├── Dashboard (home page)
│   ├── StatCards (teams, riders, bulls counts)
│   ├── QuickStart (team selector → navigate to simulator)
│   └── RecentActivity
├── Simulator (main tool)
│   ├── SetupPanel
│   │   ├── TeamSelector (dropdown)
│   │   ├── RosterList (rider cards with checkboxes)
│   │   ├── BullPen (search + tag list)
│   │   ├── OpponentPanel (team selector + bull pen)
│   │   └── RunButton
│   └── ResultsPanel
│       ├── StrategyBadge (conservative/balanced/aggressive)
│       ├── StatsGrid (optimal score, expected covers)
│       ├── OptimalLineup (rider-bull cards)
│       ├── OpponentPrediction
│       └── ManualTester
│           ├── MatchupSelectors (rider → bull dropdowns)
│           ├── CheckButton
│           └── FeedbackList (correct/wrong picks)
├── TeamDashboard (per-team view)
│   ├── TeamHeader (name, record, coach)
│   ├── RiderGrid (rider cards with stats + weights)
│   └── Schedule
├── BullSearch (searchable table)
│   ├── SearchInput
│   ├── FilterBar (spin direction, buckoff range)
│   └── BullTable
└── Reports (analysis view)
    ├── TeamEfficiencyChart
    └── GameOutcomeList
```

## Key Components

### TeamSelector
```tsx
interface TeamSelectorProps {
  value: string;
  onChange: (code: string) => void;
}
// Dropdown showing all 10 teams with roster counts
// Calls GET /api/teams
```

### RosterList
```tsx
interface RosterListProps {
  teamCode: string;
  selectedRiders: string[];
  onToggleRider: (name: string) => void;
}
// Shows rider cards with cover %, avg score
// Highlighted if selected
// Calls GET /api/team/{code}/roster
```

### BullPen
```tsx
interface BullPenProps {
  bulls: Bull[];
  onAdd: (bull: Bull) => void;
  onRemove: (id: string) => void;
}
// Search bar + tag display
// Max 5 bulls
// Calls GET /api/bulls/search?q=
```

### ResultsPanel
```tsx
interface ResultsPanelProps {
  simulation: SimulationResult;
}
// Shows strategy badge, stats, optimal lineup
// Color-coded cover probabilities (green >60%, amber 35-60%, red <35%)
```

### ManualTester
```tsx
interface ManualTesterProps {
  riders: Rider[];
  bulls: Bull[];
  simulation: SimulationResult;
  onCheck: (matchups: Record<string, string>) => void;
}
// Dropdowns for each rider → bull
// Shows comparison vs optimal after check
```

## API Store (Zustand)
```ts
interface AppStore {
  apiUrl: string;
  currentTeam: string | null;
  bulls: Bull[];
  opponentTeam: string | null;
  opponentBulls: Bull[];
  simulation: SimulationResult | null;
  
  setTeam: (code: string) => void;
  addBull: (bull: Bull) => void;
  removeBull: (id: string) => void;
  runSimulation: () => Promise<void>;
}
```

## Data Fetching (React Query)
```ts
const { data: teams } = useQuery({
  queryKey: ['teams'],
  queryFn: () => fetch(`${apiUrl}/api/teams`).then(r => r.json())
});

const { data: roster } = useQuery({
  queryKey: ['roster', teamCode],
  queryFn: () => fetch(`${apiUrl}/api/team/${teamCode}/roster`).then(r => r.json()),
  enabled: !!teamCode
});

const simulationMutation = useMutation({
  mutationFn: (body: SimRequest) => 
    fetch(`${apiUrl}/api/simulate`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    }).then(r => r.json())
});
```

## How to build in lovable.dev

1. Paste this entire spec into lovable.dev's prompt
2. It will generate the React project with Tailwind + shadcn/ui
3. Point the API_URL to your deployed backend
4. Test with the live API at your Railway/Render URL
5. Deploy frontend to Vercel (one click from lovable.dev)

## Backend Deployment (separate)

### Option A: Railway
```bash
cd backend
railway up
```

### Option B: Render
```bash
# Create Web Service, point to backend/main.py
# Build: pip install -r requirements.txt
# Start: python main.py
```

### Option C: Run locally
```bash
cd backend
pip install -r requirements.txt
python main.py
# → http://localhost:8420
```

The frontend connects to the backend via the VITE_API_URL env variable.