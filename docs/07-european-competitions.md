# 07 — European competitions (UCL / UEL / UECL), live scores and featured matches

UEFA club competitions are now listed, priced and shown live:

- Champions League (`UCL`), Europa League (`UEL`) and Conference League (`UECL`).
- They are listed first in `/leagues` with country `"Europe"`.
- The Matches page starts with a **Live now** strip and a **Featured** section.

## Setup: the free football-data.org key

1. Register at <https://www.football-data.org/client/register>. The free tier includes the Champions League ("CL") at 10 requests/minute.
2. Put the key in `backend/.env`. See `backend/.env.example`; the file is git-ignored.

   ```
   FOOTBALL_DATA_ORG_KEY=your-key
   ```

   You can also export `FOOTBALL_DATA_ORG_KEY` or `MP_FOOTBALL_DATA_ORG_KEY` in the environment instead.
3. Restart the API. `GET /api/v1/health` then reports `"football_data_org_key_configured": true`.

Without a key, everything still works:

- UCL, UEL and UECL come from TheSportsDB's free key, which covers the last and the next round of each competition.
- The UI shows a dismissible notice: "Champions League schedule needs a free football-data.org key".

| Needs the key | Works without it |
|---|---|
| Full Champions League schedule (−3 … +7 days), CL live scores and CL results, every 2 minutes during matches | UEL / UECL fixtures and results (TheSportsDB), UCL from TheSportsDB (last + next round), predictions, history, featured |

The free tier of football-data.org has no Europa League or Conference League data, and it does not send a match minute. The API therefore estimates the minute from the kick-off time and sets `minute_estimated: true`.

## One-off commands (run from `backend/`)

```bash
uv run python -m matchpredict.predictor.europe          # download openfootball, train, evaluate -> artifacts/europe.joblib
uv run python -m matchpredict.ingest.europe seed        # openfootball history (2011-12 …) -> DB (4,240 matches)
uv run python -m matchpredict.ingest.europe refresh     # UEFA fixtures / live / results now
uv run python -m matchpredict.ingest.europe live        # one live poll (no-op outside the ±3 h window)
```

- The daily ingest (`python -m matchpredict.ingest.run`, or the scheduler at 07:00 UTC) also runs `refresh`.
- With `MP_SCHEDULER_ENABLED=true`, the API also starts the live job, which runs every `MP_LIVE_REFRESH_MINUTES` (default 2).
  - It only sends requests while a UEFA match is live or kicks off within ±`MP_LIVE_WINDOW_HOURS` (default 3).
  - Each poll is one football-data.org request, or a few TheSportsDB requests without the key. Both stay within the free tier limits.

## Data

| Source | What | Volume |
|---|---|---|
| [openfootball/champions-league](https://github.com/openfootball/champions-league) (CC0, Football.TXT) | UCL 2011-12 … 2025-26; UEL and UECL main draws 2020-21 / 2021-22 … 2024-25; CL/EL/ECL qualifiers 2024-25 and 2025-26 | 29 files, **4,240 played matches**: UCL 2,179, UEL 976, UECL 1,085. Cached in `backend/data/openfootball/` |
| `data/raw/EloRatings.csv` | ClubElo snapshots 2000-07 … 2026-09-01, cross-league comparable | 631 snapshots, 19 countries |
| `data/raw/Matches.csv` | Domestic matches with ClubElo `HomeElo`/`AwayElo` | 140,711 usable rows (optional training domain) |
| football-data.org v4 `competitions/CL/matches` | CL fixtures, IN_PLAY/PAUSED/FINISHED, current score | live |
| TheSportsDB `eventspastleague` / `eventsnextleague` / `eventsround` (ids 4480/4481/5071) | UCL/UEL/UECL rounds, statuses, scores (UTF-8 decoded explicitly) | live |

openfootball has no Europa League or Conference League main-draw files for 2025-26; only the qualifiers exist. Nothing before 2011-12 is available.

Parsing rules:

- The **90-minute** score is used:
  - `3-2 a.e.t. (3-0, 1-0)` → 3-0.
  - `4-3 pen. 1-1 a.e.t. (1-1, 0-1)` → 1-1.
- `[cancelled]` and `[awarded]` matches are skipped.
- A date line without a year takes the season year closest to the previous date. This handles the COVID final tournament in August 2020.
- Stages are `qual`, `league` (group stage / league phase), `ko` and `final`. The final is played at a neutral ground.

## Team names

The canonical name is the football-data.co.uk spelling used in our DB. `EloRatings.csv` uses the same spelling (e.g. `Paris SG`, `Bayern Munich`, `Ath Madrid`, `Sp Lisbon`). The mapping lives in `ingest/teamnames.py` and works in this order:

1. A manual alias table, about 200 entries, such as "FC Barcelona"→"Barcelona", "Paris Saint-Germain FC"→"Paris SG", "FC Internazionale Milano"→"Inter", "Sporting Clube de Portugal"→"Sp Lisbon", "PSV"→"PSV Eindhoven", "Racing Club de Lens"→"Lens" and "PAE AEK"→"AEK".
2. An exact or case-insensitive match.
3. A normalised match: accents stripped; FC/CF/AC/SC/SK/KV/… removed; city exonyms such as Prague→Praha.
4. A rapidfuzz `token_set_ratio` match:
   - ≥ 88 within the same country, when the source gives one (openfootball does).
   - ≥ 93 when it does not (TheSportsDB, football-data.org).

Clubs that cannot be mapped keep their raw name, for example "Qarabağ Ağdam FK". Rows that contain one are flagged `unmapped=true` in the DB, and misses are logged. The real CL names from football-data.org for 2026-10-20/21 were checked and all map, e.g. "Como 1907"→Como, "LASK Linz"→LASK and "FK Bodø/Glimt"→Bodo/Glimt.

**Coverage (openfootball):**

- 100% of appearances by clubs from ClubElo-covered countries are mapped.
- 73.7% of all appearances are mapped (193 of 382 unique names).
- The other 189 names are clubs from countries that `EloRatings.csv` does not cover: CZE, UKR, SRB, CRO, CYP, BUL, SVK, AZE, KAZ, … and the small associations. They keep their raw name and get a dynamic rating (see below).

## Model (`predictor/europe.py`, `predictor/europe_train.py`)

### Strength

One rating per club on the ClubElo scale.

- **Historical features:** the latest ClubElo snapshot strictly before the match day.
- **Clubs outside ClubElo:** a dynamic Elo using the ClubElo formula (K = 20, K = 40 for the first 10 games, a goal-margin multiplier, +60 home advantage).
  - A new club is seeded from the mean rating of clubs from its country.
  - If there are none, it is seeded by stage: qualifier 1300, league phase 1550, knock-out 1600.
- **Serving:** the 2026-09-01 snapshot. Ratings then move with DB results played after it:
  - domestic results with +65 home advantage;
  - European results with +60.
  - Each rating carries a timestamp, so a match only moves ratings that are older than the match. Nothing is counted twice.
  - The daily ingest and every UEFA refresh call `EuropeService.update_results`.
- **Never-seen club in a UEFA fixture:** it gets the median rating of recent entrants in that competition minus 50, and the prediction has `confidence: "low"`. So every UCL/UEL/UECL fixture is priced.

### Outcome model

A multinomial logistic regression (C = 0.05) on these features:

- `elo_diff` and `|elo_diff|`;
- neutral venue;
- second leg plus the first-leg goal difference (two-legged ties are detected automatically);
- "rating not from ClubElo" flags for each side, and their interaction with `elo_diff`.

The chosen variant, `europe_lr_dom`, also trains on the domestic `Matches.csv` rows at weight 0.2 with a domain flag. That variant also lets the model price domestic cross-league games: when the two teams come from leagues in different countries, or the domestic model cannot price a match, the API falls back to it.

### Goals

Two Poisson GLMs on the same features give an 11×11 score matrix. The matrix is rescaled so its home-win, draw and away-win sums equal the outcome model's probabilities. Over 2.5, BTTS and the top 5 scores then come from that matrix.

### Output

`contracts.Prediction` with `model: "europe"`, plus an optional `confidence` field.

## Walk-forward evaluation (honest, out-of-sample)

For each test season, every model is trained on the earlier seasons only. Domestic rows are cut at the season start (July 1).

Metrics: RPS (ranked probability score) and log loss, lower is better.

- **Elo baseline:** the Elo expectation with +60 home advantage and a constant draw share taken from training.
- **base_rates:** the training outcome frequencies.

| Season | Model | n | RPS | Log loss | Brier | Accuracy |
|---|---|---|---|---|---|---|
| 2022-23 | elo_baseline | 407 | 0.1958 | 0.9601 | 0.5677 | 0.568 |
| 2022-23 | europe_lr | 407 | 0.1962 | 0.9611 | 0.5693 | 0.560 |
| 2022-23 | **europe_lr_dom** | 407 | 0.1953 | 0.9596 | 0.5678 | 0.565 |
| 2023-24 | elo_baseline | 407 | 0.1982 | 0.9610 | 0.5644 | 0.543 |
| 2023-24 | europe_lr | 407 | 0.1950 | 0.9447 | 0.5572 | 0.550 |
| 2023-24 | **europe_lr_dom** | 407 | 0.1956 | 0.9473 | 0.5581 | 0.543 |
| 2024-25 | elo_baseline | 955 | 0.2152 | 0.9885 | 0.5855 | 0.533 |
| 2024-25 | europe_lr | 955 | 0.2112 | 0.9748 | 0.5794 | 0.533 |
| 2024-25 | **europe_lr_dom** | 955 | 0.2109 | 0.9749 | 0.5792 | 0.536 |
| 2025-26 | elo_baseline | 619 | 0.2151 | 1.0144 | 0.6016 | 0.511 |
| 2025-26 | europe_lr | 619 | 0.2088 | 0.9912 | 0.5893 | 0.535 |
| 2025-26 | **europe_lr_dom** | 619 | 0.2093 | 0.9935 | 0.5907 | 0.532 |
| **all** | base_rates | 2388 | 0.2314 | 1.0412 | 0.6280 | 0.485 |
| **all** | elo_baseline | 2388 | 0.2090 | 0.9857 | 0.5831 | 0.535 |
| **all** | europe_lr | 2388 | 0.2053 | 0.9716 | 0.5764 | 0.541 |
| **all** | **europe_lr_dom** (served) | 2388 | **0.2052** | 0.9724 | 0.5767 | 0.541 |

### Reading the results

- The model beats the Elo baseline by about 1.8% RPS overall and by 0.013 in log loss.
- Most of the gain comes from 2024-25 onwards. Those seasons have many qualifiers and Conference League clubs outside ClubElo, and that is where the coverage features help.
- In 2022-23 it is level with the baseline.
- On older validation seasons (2019-20 to 2021-22, Champions League only, with little training data) the plain Elo baseline was slightly better (0.1983 vs about 0.200). Fewer features did not change this much.
- The served variant was picked among three candidates on the folds above. The two finalists are within 0.0001 RPS, so treat the choice as a tie.
- No bookmaker odds are used; a market model would be stronger.
- These numbers are close to good published European-cup Elo models. They are not "world best", and nobody should expect more than about 54% accuracy on 1X2.

### Sample: Barcelona vs Paris SG (UCL, at Barcelona, 2026-10-21 ratings)

`GET /api/v1/predict?home=FC Barcelona&away=Paris SG&league=UCL`

| | Value |
|---|---|
| Win probabilities | home 52.3%, draw 23.9%, away 23.8% |
| Expected goals | 1.72 – 1.10 |
| Over 2.5 goals | 53.7% |
| Both teams score | 55.0% |
| Most likely scores | 1-1, 1-0, 2-1 |

The real match of 2026-10-20 was played in Paris. Its pre-match prediction was home 37%, draw 26%, away 37%. PSG won 4-2.

## API additions (all additive)

All additions are documented in `04-api.md`:

- **Fixture items:** `competition_type`, `stage`, `featured`, `live` (`{minute, minute_estimated, score}`), status `"live"`, and `score` set while live.
- **New endpoint:** `GET /featured`.
- **`/health`:** `europe_model_loaded`, `europe_model_version`, `football_data_org_key_configured`, `live_matches`.
- **`/predict`:** `league=UCL|UEL|UECL` uses the European model, and so do clubs the domestic model does not know.
- **Predictions:** an optional `confidence` field.

## Known limitations

- Free football-data.org has no minute (it is estimated) and no UEL/UECL. TheSportsDB free rounds can lag by a few minutes.
- Live scores only refresh while the scheduler is enabled. Otherwise run `python -m matchpredict.ingest.europe live` or `refresh`.
- Finished and live fixtures are re-priced with the current ratings. The model has already absorbed the result, so this is not a stored pre-match snapshot.
- Clubs outside ClubElo (e.g. Czech, Ukrainian and Croatian clubs) are rated only from European results and, where we have them, domestic results (e.g. Switzerland and Ireland). They have wider error.
- When a source spells the same small club in two different ways, the DB can hold a duplicate fixture, for example "Omonia Nikosia" and "Omonia Nicosia" from different years. Better aliases fix this.
