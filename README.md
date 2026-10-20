# football-match-predictor

A football match prediction app. It shows today's matches (or any date range) in **your local time zone**, along with live scores and featured games. Pick any team to see its **win, draw and loss chances** for upcoming fixtures. The predictions come from **our own models**, trained on more than 250,000 club matches: 38 domestic leagues plus the UEFA Champions League, Europa League and Conference League.

**Stack:** Python 3.12 · FastAPI · SQLAlchemy (SQLite) · CatBoost · pandas · React 19 · Vite · TypeScript · Tailwind · TanStack Query

---

## Features

- **Matches page:** today by default, with a date picker, previous/next day, "Next 7 days", a league filter and team search. All dates and kick-off times are shown in the viewer's time zone.
- **Live now and Featured:** live UEFA matches with minute and score, refreshed every minute, followed by the day's biggest games.
- **Team page:** a team's upcoming fixtures with "win chance %", its recent form (W/D/L) and its current ratings.
- **Custom match-up:** pick any two teams for an instant prediction.
- **For every match:** calibrated home/draw/away probabilities, expected goals, Over 2.5, both teams to score, and the 5 most likely scorelines.
- **Auto-updating:** a daily results and fixtures ingest, live-score polling during UEFA matches, and pre-match predictions frozen at kick-off.

### Model accuracy

Walk-forward backtest over 4 seasons (2022/23–2025/26, ~41,000 domestic matches). Lower RPS is better.

| Model | RPS ↓ | Accuracy |
|---|---|---|
| Always predict the base rates | 0.2282 | 43.7% |
| Elo only | 0.2093 | 48.9% |
| Bookmaker (Bet365, margin removed) | 0.2036 | 50.4% |
| **Ours, ratings only** | 0.2078 | 49.5% |
| **Ours, with odds** | **0.2034** | **50.5%** |

The European model (UEFA competitions) scores RPS 0.2052, against 0.2090 for an Elo baseline, over 2,388 matches.

> Football is uncertain. Even the best models and bookmakers pick the right result only about half the time. This project is for information and entertainment, not betting advice.

---

## How to run it

### 1. Prerequisites

| Tool | Version | Install |
|---|---|---|
| Git | any | https://git-scm.com |
| Python | 3.12 | https://www.python.org/downloads/ |
| uv (Python package manager) | 0.5+ | `pip install uv` or https://docs.astral.sh/uv/ |
| Node.js (includes npm) | 20+ | https://nodejs.org |
| Docker (optional) | any | https://www.docker.com |

Check them:

```bash
git --version && python --version && uv --version && node --version && npm --version
```

### 2. Get the code

```bash
git clone https://github.com/shoaibghulam/football-match-predictor.git
cd football-match-predictor
```

### 3. Get the training data

The historical data is not stored in the repo because of its size.

1. On [Kaggle](https://www.kaggle.com/datasets), find the dataset **"Club Football Match Data (2000 - 2025)"** and download it. You need a free Kaggle account.
2. Unzip it and copy **`Matches.csv`** and **`EloRatings.csv`** into `data/raw/`:

```
football-match-predictor/
└── data/
    └── raw/
        ├── Matches.csv       (~45 MB, ~239k matches)
        └── EloRatings.csv    (~11 MB)
```

### 4. Get a free football-data.org API key

This key enables the Champions League schedule, its live scores and results, and a longer fixture horizon.

1. Register at https://www.football-data.org/client/register. It's free, and the key arrives by email.
2. Create your settings file from the template:

```bash
cd backend
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
```

3. Open `backend/.env` and paste your key:

```
FOOTBALL_DATA_ORG_KEY=your_key_here
```

`backend/.env` is git-ignored, so never commit it. Every other setting in the file is optional and already has a sensible default.

### 5. Set up the backend (first time only)

Run these from the `backend/` folder:

```bash
# Install Python dependencies into backend/.venv
uv sync

# Load the 238k historical matches into the local database (~30 s)
uv run python -m matchpredict.db.seed

# Download 15 seasons of UEFA results (openfootball) into the database (~1 min)
uv run python -m matchpredict.ingest.europe seed

# Train the domestic model: ratings + CatBoost + goals model, with a full backtest (~20 min)
uv run python -m matchpredict.predictor.train
#   faster trial run on a data subset:  uv run python -m matchpredict.predictor.train --quick

# Train the European (UEFA) model (a few minutes)
uv run python -m matchpredict.predictor.europe

# Pull the latest results and fixtures, and bring the ratings up to date (~1-2 min)
uv run python -m matchpredict.ingest.run
uv run python -m matchpredict.ingest.europe refresh
```

The trained models are saved to `backend/artifacts/`, with backtest metrics next to them in `metrics.json` and `europe_metrics.json`.

### 6. Start the backend API

```bash
# in backend/
uv run matchpredict
```

- API: http://127.0.0.1:8000/api/v1/health
- Interactive API docs: http://127.0.0.1:8000/api/v1/docs

While the server runs, it keeps itself up to date automatically: a daily ingest at 07:00 UTC, live UEFA scores every 2 minutes during matches, and a weekly retrain. Set `MP_SCHEDULER_ENABLED=false` in `.env` to switch these off.

### 7. Start the frontend

Open a **second terminal**:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. Use `localhost`, not `127.0.0.1`, because the dev server listens on `localhost`.

Want a quick look without the backend or any data? `npm run dev:mock` runs the UI with realistic sample data.

### 8. Every day after that

```bash
# terminal 1
cd backend && uv run matchpredict
# terminal 2
cd frontend && npm run dev
```

That's all you need. The backend fetches new results and fixtures on its own.

---

## Run with Docker (optional)

First complete steps 3–5 so that `backend/artifacts/` contains the trained models. Then, from the repo root:

```bash
docker compose up --build
```

Open **http://localhost:8080**. The frontend is served by nginx, which forwards `/api` to the backend container. The database lives in a Docker volume.

To pass your API key, either set it in your shell before running compose (`FOOTBALL_DATA_ORG_KEY=... docker compose up`) or put it in a root `.env` file next to `docker-compose.yml`.

---

## Configuration

All settings are environment variables, optionally set in `backend/.env`. See [`backend/.env.example`](backend/.env.example) for the full list with defaults. The most useful ones:

| Variable | Default | Purpose |
|---|---|---|
| `FOOTBALL_DATA_ORG_KEY` | (none) | Champions League fixtures, results and live scores |
| `MP_SCHEDULER_ENABLED` | `true` | Daily ingest, live refresh and weekly retrain |
| `MP_INGEST_HOUR_UTC` | `7` | Time of the daily results and fixtures ingest |
| `MP_PORT` | `8000` | API port |
| `MP_CORS_ORIGINS` | `http://localhost:5173` | Allowed frontend origins (comma-separated) |
| `MP_RATE_LIMIT` | `60/minute` | Requests per client IP |

Frontend: set `VITE_API_BASE` in `frontend/.env` if the API is not served at `/api/v1` on the same host.

---

## Tests

```bash
# backend
cd backend
uv run pytest
uv run ruff check src tests

# frontend
cd frontend
npm test
npm run lint
npm run build
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Predictions show "No prediction available" | The models are not trained or loaded yet. Run step 5, then restart `uv run matchpredict`. Check `/api/v1/health`, which should show `"model_loaded": true`. |
| No Champions League matches | `FOOTBALL_DATA_ORG_KEY` is missing in `backend/.env` (`/health` shows `football_data_org_key_configured`). Run `uv run python -m matchpredict.ingest.europe refresh`. |
| Few upcoming domestic fixtures | football-data.co.uk publishes only the next round, on Tuesdays and Fridays. The daily ingest picks it up. |
| Frontend loads but shows no data | Make sure the backend is running on port 8000, and open `http://localhost:5173`. |
| `Port 8000 already in use` | Stop the other process, or set `MP_PORT=8001` in `.env` and point the Vite proxy (`frontend/vite.config.ts`) at it. |
| SSL or handshake errors while downloading | The ingest retries and falls back to `curl` automatically. Make sure `curl` is installed. |
| Training is slow | Use `--quick` for a trial run. Full training uses all CPU cores and takes about 20 minutes. |

---

## Project layout

```
backend/
  src/matchpredict/
    predictor/   ratings engine, features, models, training, prediction service (v1, europe, v2)
    ingest/      football-data.co.uk, football-data.org, TheSportsDB, openfootball, scheduler
    db/          SQLAlchemy models, repository, seed
    api/         FastAPI app and routes (/api/v1)
  tests/
  artifacts/     trained models (git-ignored)
frontend/
  src/           React app: pages, components, API client, utils
data/raw/        input CSVs (git-ignored)
docker-compose.yml
```

## Data sources and credits

| Source | Used for | Terms |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk) | Historical results, stats and odds; daily results and fixtures | Free for personal use |
| [football-data.org](https://www.football-data.org) | Champions League schedule, results and live scores | Free tier, needs an API key |
| [openfootball](https://github.com/openfootball/champions-league) | UEFA competition results 2011–2026 | CC0 |
| [TheSportsDB](https://www.thesportsdb.com) | Europa / Conference League fixtures (fallback) | Free tier |
| [ClubElo](http://clubelo.com) | Cross-country club ratings (`EloRatings.csv`) | Credit ClubElo |

The data belongs to its respective providers. Check each provider's terms before any commercial use.

## License

[MIT](LICENSE) © 2026 Shoaib Ghulam
