# 04 — API Contract (v1)

Base: `/api/v1`. JSON only. Every error uses the envelope `{"error": {"code": str, "message": str}}`.

| Method | Path | Query | Returns |
|---|---|---|---|
| GET | `/health` | — | `{status, model_version, last_ingest, matches, fixtures}` |
| GET | `/leagues` | — | `[{code, name, country}]` |
| GET | `/teams` | `league?`, `q?` (search, ≥2 chars) | `[{name, league}]` |
| GET | `/fixtures` | `date?` (ISO, default today), `date_to?` (≤ 14 days after `date`), `league?`, `team?` | `[{id, league, kickoff, home, away, odds?, prediction}]` |
| GET | `/predict` | `home`, `away` (required, known teams), `league?` | `prediction` for an arbitrary pairing at the home ground of `home` |
| GET | `/teams/{name}/form` | — | the last 10 results plus current ratings |

`prediction` object:

```json
{
  "probs": {"home": 0.52, "draw": 0.26, "away": 0.22},
  "expected_goals": {"home": 1.61, "away": 0.98},
  "over_2_5": 0.49,
  "btts": 0.47,
  "top_scores": [{"score": "1-0", "p": 0.12}, {"score": "1-1", "p": 0.11}],
  "model": "market|base",
  "model_version": "2026-10-20"
}
```

Validation:
- Unknown team → `404 TEAM_NOT_FOUND`.
- `home == away` → `422`.
- A date range wider than 14 days → `422`.

## Implementation notes (v1, backward-compatible additions)

Served by `backend/src/matchpredict/api/` (`uv run matchpredict` → `http://127.0.0.1:8000`; OpenAPI at `/api/v1/openapi.json`, Swagger UI at `/api/v1/docs`). All fields listed in the table above are present; the extras below are additions.

### `GET /health`

```json
{"status": "ok|degraded", "model_loaded": true, "model_version": "2026-10-20",
 "last_result_date": "2026-10-19", "last_ingest": "2026-10-20T07:00:41.512",
 "matches": 248472, "fixtures": 47}
```

`status` is `degraded` (and `model_version` null) while no model artifact is loaded. `matches` = finished matches in the DB, `fixtures` = scheduled matches from today on. `last_ingest` is the UTC finish time of the last successful ingest.

### `GET /leagues`

All 38 leagues, in a fixed order (England first, extra countries last): `[{"code": "E0", "name": "Premier League", "country": "England"}, ...]`.

### `GET /teams`

`[{"name": "Arsenal", "league": "E0"}]`, sorted by name. `league` is the league of the team's most recent match. Teams are those that played within ~400 days of the latest stored match, plus any team the model knows. `q` is a case-insensitive substring match (min 2 chars).

### `GET /fixtures`

Every match in `[date, date_to]` (inclusive, both ISO `YYYY-MM-DD`; `date` defaults to today in UK time, `date_to` defaults to `date`). Upcoming and finished matches are both returned, ordered by date and kickoff:

```json
{
  "id": 248473,
  "league": "E1",
  "date": "2026-10-20",
  "time": "19:45",
  "kickoff": "2026-10-20T19:45:00+01:00",
  "home": "Wolves",
  "away": "Portsmouth",
  "status": "scheduled",
  "score": null,
  "odds": {"home": 1.53, "draw": 4.1, "away": 5.25},
  "prediction": { "...": "prediction object" }
}
```

- `time` is the UK local kickoff time published by football-data.co.uk (`null` when unknown). `kickoff` is the same instant as a timezone-aware ISO 8601 datetime (Europe/London offset), or `null` when the time is unknown. Use `kickoff` to show local times.
- `status` is `"scheduled"` or `"finished"`. For finished matches, `score` is `{"home": int, "away": int}`; otherwise it is `null`.
- `odds` holds pre-match decimal odds (Bet365, else the market average), or `null`.
- `prediction` is **nullable**. It is `null` when the model is not loaded yet, or when the model cannot price that match (for example, a newly promoted team it has never seen). With odds, `prediction.model` is `"market"`. Without odds, it is `"base"`.
- `team` filters on the exact team name (as returned by `/teams`).

### `GET /predict`

Returns the prediction object plus the resolved `home`, `away` and `league`:

```json
{"home": "Arsenal", "away": "Chelsea", "league": "E0", "probs": {...}, "expected_goals": {...},
 "over_2_5": 0.49, "btts": 0.47, "top_scores": [...], "model": "base", "model_version": "2026-10-20"}
```

Team names are matched case-insensitively against the model's known teams. `league` defaults to the home team's league. The prediction is for a match played today at the home team's ground, without odds.

### `GET /teams/{name}/form`

```json
{
  "team": "Arsenal",
  "league": "E0",
  "results": [
    {"date": "2026-10-18", "league": "E0", "home": "Nott'm Forest", "away": "Arsenal",
     "home_goals": 0, "away_goals": 1, "venue": "A", "opponent": "Nott'm Forest",
     "goals_for": 1, "goals_against": 0, "result": "W"}
  ],
  "ratings": {"elo": 1850.2, "pi_home": 1.1, "pi_away": 0.8, "form5": 12.0}
}
```

`results` holds the last 10 finished matches, newest first. `ratings` is `null` when the model is not loaded or does not know the team.

### UEFA competitions, live scores and featured matches (additive)

See `07-european-competitions.md` for the sources, the setup and the model.

**`GET /leagues`** now starts with `UCL` (UEFA Champions League), `UEL` (UEFA Europa League) and `UECL` (UEFA Conference League), all with country `"Europe"`. There are 41 leagues in total.

**Fixture items** (`/fixtures`, `/featured`) gain these fields:

```json
{
  "id": 252001, "league": "UCL", "date": "2026-10-20", "time": "20:00",
  "kickoff": "2026-10-20T20:00:00+01:00", "home": "Paris SG", "away": "Barcelona",
  "status": "live", "score": {"home": 3, "away": 2}, "odds": null,
  "prediction": {"probs": {"home": 0.37, "draw": 0.26, "away": 0.37}, "...": "...",
                 "model": "europe", "model_version": "europe-2026-10-21", "confidence": "high"},
  "competition_type": "europe",
  "stage": "league",
  "featured": true,
  "live": {"minute": 40, "minute_estimated": true, "score": {"home": 3, "away": 2}}
}
```

- `status` can now also be `"live"`. While a match is live, `score` holds the current score.
- `live` is `null` unless the match is live. When the feed gives no minute, `minute` is estimated from the kick-off time and `minute_estimated` is `true`.
- `competition_type` is `"europe"` for UCL, UEL and UECL, and `"domestic"` otherwise.
- `stage` is `qual`, `league`, `ko`, `final` or `null`.
- `featured` is true for UEFA matches, and for matches where both clubs are rated at least 1780 on the ClubElo scale.
- Which model prices a match:
  - UEFA fixtures always use the European model (`model: "europe"`). A club the model has never seen gets a competition prior and `confidence: "low"`.
  - Domestic fixtures fall back to the European model when the two teams' leagues are in different countries, or when the domestic model cannot price the match.
- `prediction.confidence` is `null` for domestic-model predictions.

**`GET /featured?date=`** returns at most 12 fixture items, `featured: true`, for `date` (default today). They come in this order:

1. live matches (within ±1 day);
2. the day's UEFA matches;
3. domestic matches from European leagues where both clubs are rated at least 1600, ranked by their combined strength.

**`GET /health`** gains these fields:

```json
{"europe_model_loaded": true, "europe_model_version": "europe-2026-10-21",
 "football_data_org_key_configured": true, "live_matches": 7}
```

The key itself is never returned.

**`GET /predict`:**

- `league=UCL|UEL|UECL` prices the pairing with the European model.
- Clubs that only the European model knows (e.g. `Slavia Praha`) are resolved through it.
- With no `league`, a pairing from leagues in different countries uses the European model and `league` is returned as `"UCL"`.
- Errors are the same as before. `503 MODEL_NOT_LOADED` is returned when the European model is needed but missing.

New configuration variables:

| Variable | Default |
|---|---|
| `MP_EUROPE_ARTIFACT_PATH` | `backend/artifacts/europe.joblib` |
| `MP_OPENFOOTBALL_DIR` | `backend/data/openfootball` |
| `MP_THESPORTSDB_KEY` | `123` (free) |
| `MP_EUROPE_INGEST_ENABLED` | `true` |
| `MP_LIVE_REFRESH_MINUTES` | `2` |
| `MP_LIVE_WINDOW_HOURS` | `3` |

### Errors

Every error, including framework validation errors, uses `{"error": {"code", "message"}}`.

| Status | `code` | When |
|---|---|---|
| 404 | `TEAM_NOT_FOUND` | Unknown team in `/predict` or `/teams/{name}/form` |
| 404 | `LEAGUE_NOT_FOUND` | Unknown `league` code |
| 404 | `NOT_FOUND` | Unknown route |
| 405 | `METHOD_NOT_ALLOWED` | Anything other than GET |
| 422 | `SAME_TEAM` | `home == away` (case-insensitive) |
| 422 | `DATE_RANGE_TOO_WIDE` | `date_to` more than 14 days after `date` |
| 422 | `INVALID_DATE_RANGE` | `date_to` before `date` |
| 422 | `VALIDATION_ERROR` | Malformed parameters: a non-ISO date, `q` shorter than 2 characters, a lowercase or invalid league code, or a team name containing illegal characters |
| 429 | `RATE_LIMITED` | More than `MP_RATE_LIMIT` (default `60/minute`) requests per client IP. `/health` is exempt |
| 503 | `MODEL_NOT_LOADED` | `/predict` called before a model artifact exists |
| 500 | `INTERNAL_ERROR` | Unexpected failure (details are logged, never returned) |

### Configuration (env vars, prefix `MP_`)

| Variable | Default |
|---|---|
| `MP_DATABASE_URL` | `sqlite:///backend/data/app.db` |
| `MP_ARTIFACT_PATH` | `backend/artifacts/model.joblib` |
| `MP_HISTORY_CSV` | `data/raw/Matches.csv` |
| `MP_CORS_ORIGINS` | `http://localhost:5173` (comma-separated) |
| `MP_SCHEDULER_ENABLED` | `true` |
| `MP_INGEST_HOUR_UTC` | `7` |
| `MP_RATE_LIMIT` | `60/minute` |
| `MP_HOST` | `127.0.0.1` |
| `MP_PORT` | `8000` |
| `MP_FOOTBALL_DATA_ORG_KEY` or `FOOTBALL_DATA_ORG_KEY` | unset (when set, extends the fixture horizon to 14 days) |
