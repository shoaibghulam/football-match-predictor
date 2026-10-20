# 02 — Architecture

## Stack decision: FastAPI + Vite (React + TypeScript)

| Option | Verdict |
|---|---|
| **FastAPI** | ✅ **Chosen.** The ML lives in Python, so the model is loaded once in-process and served from async endpoints. Pydantic schemas give us an OpenAPI contract, and the frontend types are generated from it. Small surface area. |
| Django + DRF | ❌ Its strengths (admin, auth, a large ORM app) are not what this product needs. It is heavier for a read-mostly prediction API. |
| Frontend: **Vite + React + TS** | ✅ Fast dev server and a static build that any CDN can serve. Uses TanStack Query for data fetching. |
| DB: **SQLite (dev) → PostgreSQL (prod)** | Via SQLAlchemy 2.0, with the same code for both. Postgres runs through docker-compose. |
| Python tooling | `uv` for dependencies and venv, `ruff` for lint, `pytest` for tests. |

## Components

```
                ┌──────────────────── backend (FastAPI, Python 3.12) ─────────────────────┐
 football-data  │  ingest/   ──► db (matches, fixtures, teams, ratings, predictions)      │
 .co.uk CSVs ──►│   daily job      ▲                     │                                │
 football-data  │                  │                     ▼                                │
 .org (opt.) ──►│  predictor/  ratings engine ─► features ─► models (Dixon-Coles + CatBoost)│
                │                  └─ train.py (offline)   └─ artifacts/*.joblib          │
                │  api/  GET /fixtures  GET /teams  GET /predict  GET /leagues  /health   │
                └───────────────────────────────▲──────────────────────────────────────────┘
                                                │ JSON (OpenAPI)
                              frontend (Vite + React + TS)
                    Today / date picker · league filter · team search · match card
```

### Repository layout

```
shameer/
├── data/raw/                 # Matches.csv, EloRatings.csv (git-ignored)
├── data/processed/           # parquet features (git-ignored)
├── docs/                     # these docs
├── backend/
│   ├── pyproject.toml        # uv project "matchpredict"
│   ├── src/matchpredict/
│   │   ├── config.py         # pydantic-settings (env vars)
│   │   ├── predictor/        # pure ML: ratings.py, features.py, goals.py, model.py, metrics.py, train.py
│   │   ├── ingest/           # football_data_uk.py, football_data_org.py, scheduler.py
│   │   ├── db/               # SQLAlchemy models + session
│   │   └── api/              # FastAPI routers + Pydantic schemas
│   └── tests/
└── frontend/                 # Vite React TS app
```

## Key flows

1. **Train (offline, weekly):**
   - `python -m matchpredict.predictor.train` loads the history and runs the ratings engine once in chronological order.
   - It builds the feature table, runs the walk-forward evaluation, fits the final models on all data, and saves `artifacts/model-<date>.joblib` plus a metrics report.
2. **Daily ingest (07:00 UTC, APScheduler):**
   - Pull the current-season result CSVs and upsert finished matches.
   - Pull `fixtures.csv` and `new_league_fixtures.csv` and upsert the upcoming fixtures with their odds.
   - Optionally pull the football-data.org schedule for the 7–14 days ahead.
3. **Predict:**
   - For each upcoming fixture, compute features from the ratings *as of now* (the ratings state is updated incrementally with new results).
   - Run the model: the market model if odds exist, otherwise the base model.
   - Store the predictions, which `GET /fixtures` then serves.
4. **Team pick:** the user picks a team; the frontend calls `GET /predict?home=&away=&league=` for any pairing, plus that team's next fixtures.

## Non-functional

- **Security:**
  - No user accounts in v1.
  - All query params are validated by Pydantic: team names come from a whitelist in the DB, dates are ISO.
  - Rate limiting via `slowapi`.
  - CORS is restricted to the frontend origin, and secrets (the football-data.org key) live in env vars only.
- **Performance:** the model is loaded once at startup. Predictions are precomputed for listed fixtures, and the on-demand `/predict` responds in under 50 ms.
- **Observability:** structured logging; `/health` reports the model version, last ingest time and row counts.
