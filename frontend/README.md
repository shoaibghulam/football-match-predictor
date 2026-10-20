# MatchPredict frontend

Vite + React 19 + TypeScript (strict) + TanStack Query + React Router 7 + Tailwind CSS v4.
It shows calibrated football match probabilities from the MatchPredict API (`docs/04-api.md`).

## Run

```bash
cd frontend
npm install

# Against the real backend (FastAPI on http://localhost:8000; /api is proxied)
npm run dev                 # http://localhost:5173

# Stand-alone demo with realistic sample data (no backend needed, uses msw in the browser)
npm run dev:mock            # same as VITE_USE_MOCKS=1 npm run dev
```

| Script | What it does |
|---|---|
| `npm run dev` | Vite dev server; `/api` → `VITE_PROXY_TARGET` (default `http://localhost:8000`) |
| `npm run dev:mock` | Dev server with the in-browser mock API (`.env.mock`) |
| `npm run build` | `tsc -b && vite build` → `dist/` (static, any CDN) |
| `npm run preview` | Serve the production build locally |
| `npm test` | Vitest (jsdom + Testing Library + msw) |
| `npm run test:coverage` | Tests with v8 coverage (fails under 80% lines) |
| `npm run lint` | ESLint flat config (typescript-eslint, react-hooks) |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_BASE` | `/api/v1` | API base URL (relative or absolute) baked into the build |
| `VITE_PROXY_TARGET` | `http://localhost:8000` | Dev-server proxy target for `/api` |
| `VITE_USE_MOCKS` | unset | `1` serves sample data from `src/mocks` via a service worker |

See `.env.example`.

## Routes

| Path | Page |
|---|---|
| `/` | Matches: date picker, prev/next day, Today/Tomorrow/Next 7 days, league filter, team search. State lives in the URL (`?date=&to=&league=&q=`). |
| `/team`, `/team/:name` | Team search (combobox), next 14 days of fixtures with the team's win chance, recent form and ratings |
| `/predict` | Custom match-up: any home team vs any away team, with a swap button |
| `/about` | How accurate is this? Calibration, realistic accuracy, the model, the data source and a responsible-gambling notice |

## Structure

```
src/
  api/          client.ts (fetch wrapper + error envelope), endpoints.ts, hooks.ts (TanStack Query), types.ts
  components/   MatchCard, ProbabilityBar, PredictionStats, TopScores, ModelBadge, MatchDetailDialog, Modal,
                DateControls, LeagueSelect, FixtureList, TeamCombobox, FormChips, StatusStates, layout/, team/
  pages/        MatchesPage, TeamPage, PredictPage, AboutPage, NotFoundPage
  utils/        date (local-time yyyy-mm-dd), probability (formatting, sum-to-100 rounding), leagues, form
  mocks/        sample data + msw handlers (shared by dev mock mode and tests)
  test/         setup, msw server, render helpers, factories, jsdom env
```

## Notes

- Dates are the user's local calendar dates, sent to the API as `yyyy-mm-dd`. Kickoffs without a time zone are shown as-is.
- `/teams/{name}/form` is parsed leniently: results may be under `results`, `matches` or `last_results`, each with
  `home/away/home_goals/away_goals` or `opponent/venue/goals_for/goals_against/result`. Ratings are any numeric map.
- Tests use a small custom jsdom environment (`src/test/jsdom-env.ts`) that keeps Node's `AbortController`, because
  Node's `fetch` rejects jsdom's `AbortSignal`.
