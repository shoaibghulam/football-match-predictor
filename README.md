# football-match-predictor

A football match prediction app. It shows today's matches (or any date range), live scores and featured games. Pick a team to see its win, draw and loss chances for upcoming fixtures. The predictions come from **our own models**, trained on more than 250,000 club matches: 38 domestic leagues plus the UEFA Champions League, Europa League and Conference League.

**Stack:** FastAPI · SQLAlchemy (SQLite / Postgres) · CatBoost · pandas · React 19 + Vite + TypeScript · Tailwind · TanStack Query

## What it predicts

- Calibrated **home / draw / away** probabilities: of matches we give a 60% chance, about 6 in 10 are won.
- **Expected goals**, **Over 2.5**, **both teams to score**, and the **5 most likely scorelines**.
- **Domestic model:** our own Elo, pi-ratings, attack/defence ratings and form/shot trends → CatBoost, with an optional bookmaker-odds correction. Backtested over 4 seasons (~41k matches):

  | Model | RPS ↓ | Accuracy |
  |---|---|---|
  | Bookmaker (Bet365 de-vigged) | 0.2036 | 50.4% |
  | Ours, ratings only | 0.2078 | 49.5% |
  | Ours, with odds | **0.2034** | **50.5%** |

- **European model:** cross-country club strength for UEFA matches, trained on 15 seasons of European results. RPS is 0.2052, against 0.2090 for an Elo baseline.

Full details are in [docs/06-model-report.md](docs/06-model-report.md) and [docs/07-european-competitions.md](docs/07-european-competitions.md).

> Football is uncertain. Even the best models and bookmakers pick the right result only about half the time. For information and entertainment, not betting advice.

## Quick start (local)

Prerequisites: Python 3.12 with [uv](https://docs.astral.sh/uv/), and Node 20+.

```bash
# 1. Data: put Matches.csv and EloRatings.csv in data/raw/ (Kaggle "Club Football Match Data 2000-2025").

# 2. Backend
cd backend
uv sync
cp .env.example .env                          # then set FOOTBALL_DATA_ORG_KEY (free: football-data.org/client/register)
uv run python -m matchpredict.db.seed         # load history into the DB (~30 s)
uv run python -m matchpredict.predictor.train # train the domestic model (~20 min)
uv run python -m matchpredict.predictor.europe_train  # train the European model
uv run python -m matchpredict.ingest.run      # pull the latest results and fixtures
uv run matchpredict                           # API on http://127.0.0.1:8000 (docs: /api/v1/docs)

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev                                   # http://localhost:5173
```

Try the UI without a backend: `npm run dev:mock`.

### Docker

```bash
docker compose up --build    # app on http://localhost:8080
```

The model artifacts in `backend/artifacts/` must already exist; train them with the commands above.

## Data sources

| Source | Used for |
|---|---|
| [football-data.co.uk](https://www.football-data.co.uk) | Historical results, stats and odds; daily results and upcoming fixtures |
| [football-data.org](https://www.football-data.org) (free key) | Champions League schedule, results and live scores |
| [openfootball](https://github.com/openfootball/champions-league) | 15 seasons of UEFA competition results (CC0) |
| [TheSportsDB](https://www.thesportsdb.com) | Europa / Conference League fixtures (fallback) |
| [ClubElo](http://clubelo.com) | Cross-country club ratings (`EloRatings.csv`) |

## Project layout

```
backend/   FastAPI app — predictor/ (ML), ingest/ (data sources), db/, api/
frontend/  React + Vite app
docs/      research, architecture, model reports, API contract, roadmap
data/raw/  input CSVs (git-ignored)
```

## Tests

```bash
cd backend && uv run pytest && uv run ruff check src tests
cd frontend && npm test && npm run lint && npm run build
```

## Docs

[Research](docs/01-research.md) · [Architecture](docs/02-architecture.md) · [Model design](docs/03-model.md) · [API](docs/04-api.md) · [Roadmap](docs/05-roadmap.md)
