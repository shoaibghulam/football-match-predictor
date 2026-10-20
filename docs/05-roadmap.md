# 05 — Roadmap

## Phase 1: Model (current)
- [x] Research: data sources, models and baselines (`01-research.md`)
- [x] Scaffold the backend (`uv`, deps)
- [x] Data loader and cleaning (`predictor/data.py`)
- [x] Ratings engine: Elo, pi-ratings, process EWMAs, form, rest, league rates (`predictor/ratings.py`) with 8 leakage and behaviour tests
- [x] Metrics: RPS, log loss, Brier, accuracy, de-vig (`predictor/metrics.py`) with 4 tests
- [x] Feature table and walk-forward evaluation (`predictor/train.py`, `metrics.py`)
- [x] CatBoost base and market models with temperature calibration
- [x] Dixon-Coles goals model
- [x] Training report (RPS vs benchmarks) and saved artifact (`06-model-report.md`, `backend/artifacts/model.joblib`)

## Phase 2: Data ingest
- [x] football-data.co.uk client (fixtures, current-season results; httpx with a curl fallback) (`ingest/football_data_uk.py`, `ingest/http.py`)
- [x] SQLAlchemy models and DB init; seed the history from CSV (`db/`, `python -m matchpredict.db.seed`, ~25 s for 238,854 rows)
- [x] Daily scheduler (APScheduler, 07:00 UTC, plus a catch-up run at startup when the last run is over 12 h old); incremental ratings updates via `service.update_results` (`ingest/run.py`, `ingest/scheduler.py`)
- [x] Optional football-data.org key for a 14-day horizon; team alias table plus rapidfuzz matching (`ingest/football_data_org.py`)

## Phase 3: API
- [x] FastAPI app, routers per `04-api.md`, validation, CORS, rate limiting, structured JSON logs (`api/`)
- [x] Integration tests (httpx `TestClient`); ingest, DB and API coverage at 99%

## Phase 4: Frontend
- [x] Vite + React + TS + TanStack Query (`frontend/`, Tailwind v4, React Router 7; typed API client; msw mock mode `npm run dev:mock`)
- [x] Today view, date picker and league filter (plus prev/next day, Today/Tomorrow/Next 7 days, team search; state in the URL)
- [x] Team search → that team's fixtures and win chance; custom match-up predictor
- [x] Match card: probability bar, xG, top scorelines, "how accurate is this?" explainer (`/about`)
- [x] Tests: Vitest + Testing Library + msw (64 tests, ~99% line coverage); lint and build green

## Phase 4b: European competitions and live (see `07-european-competitions.md`)
- [x] UEFA leagues `UCL` / `UEL` / `UECL` listed first; canonical team-name alias table with country-aware fuzzy matching (`ingest/teamnames.py`)
- [x] openfootball history 2011-12 … 2025-26 (4,240 matches) parser, cache and DB seed (`ingest/openfootball.py`, `python -m matchpredict.ingest.europe seed`)
- [x] football-data.org CL fixtures, live scores and results (key); TheSportsDB fallback for UEL / UECL and for UCL without a key (`ingest/thesportsdb.py`)
- [x] Live status (`live`, minute, current score), additive DB migration, and a live refresh job (2 min, only within ±3 h of kick-off)
- [x] Cross-league European model (ClubElo-scale ratings + LR + Poisson goals), walk-forward RPS 0.2052 vs 0.2090 for the Elo baseline (`predictor/europe.py`)
- [x] API: `competition_type` / `featured` / `live` on fixtures, `GET /featured`, `/health` Europe fields, European fallback in `/predict`
- [x] Frontend: Live now strip (60 s refetch while live), Featured section, UEFA-first grouping with accent, key notice

## Phase 5: Ship
- [ ] docker-compose (api + postgres + static frontend)
- [ ] Weekly retrain job with promotion gate
- [ ] E2E tests (Playwright) on the critical flows
