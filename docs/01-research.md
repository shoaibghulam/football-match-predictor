# 01 — Research Findings

_Last updated: 2026-10-20. Sources were checked live on that date where marked **LIVE**._

## 1. Our starting data

| File | Rows | Range | Contents |
|---|---|---|---|
| `data/raw/Matches.csv` | 238,859 | 2000-07-28 → 2026-09-03 | 38 divisions, results, HT score, shots/SoT/corners/fouls/cards (~51%), Elo (63%), Form3/5, Bet365 pre-match 1X2 (99%), O/U 2.5 and Asian handicap (~66%) |
| `data/raw/EloRatings.csv` | 273,973 | 2000-07-01 → 2026-09-01 | ClubElo snapshots: date, club, country, elo |

Findings from profiling:

- **Source = football-data.co.uk.** Division codes, column layout and team names match. Every team in today's `fixtures.csv` exists in our data (**LIVE**). The five unknown names in `new_league_fixtures.csv` are promoted clubs or spelling variants that need an alias.
- **`OddHome/Draw/Away` = Bet365 *pre-match* odds** (equal to `B365H`), not closing odds. football-data.co.uk collects them on Friday/Tuesday afternoons. The same odds appear in the free `fixtures.csv` before each round, so **odds can be used as a feature at prediction time** for fixtures that have them.
- **Elo is missing** for ARG, BRA, CHN, IRL, JAP, MEX, USA and SUI, and is sparse for lower divisions. **We compute our own ratings for every league.**
- Outcomes: Home 44.6%, Draw 26.5%, Away 28.9%.

### Baselines on our data (test = matches from 2024-07-01)

| Model | Accuracy | RPS ↓ |
|---|---|---|
| Base rates (always 44.6/26.5/28.9) | — | 0.2288 |
| Elo difference → logistic | 48.5% | 0.2103 |
| **Bet365 pre-match odds (de-vigged)** | **50.5%** | **0.2036** |

The odds row is the honest ceiling. A model within about 0.005 RPS of it is a strong result.

## 2. Prediction models — what the evidence says

- **2017 Soccer Prediction Challenge** (216k matches, 52 leagues, goals only):
  - The winner, Hubáček et al., used **gradient boosting on pi-ratings and PageRank ratings**: about RPS 0.206 and 52.4% accuracy. _Mach Learn_ 108:29–47 (2019).
  - The organisers (Berrar et al.) found that **domain-knowledge rating features matter more than the algorithm**.
  - Hierarchical Poisson models scored 0.205–0.209 (Tsokos et al., arXiv 1807.01623).
- **2023 Soccer Prediction Challenge:** the bookmaker consensus scored about 0.206. CatBoost + pi-ratings was a top validation model (0.2085). Deep learning did not beat it (Yeung et al., arXiv 2309.14807).
- **Review** (Bunker, Yeung & Fujii 2024, arXiv 2403.07669): **CatBoost on pi-ratings is the strongest goals-only approach.** Bookmaker odds remain hard to beat.
- **Dixon-Coles / Poisson with time decay** (Ley et al. 2019): simple models with a decay half-life of about 1 season predict best, and they give **scorelines, expected goals and O/U** for free.
- **Shots and corners** ("GAP ratings", Wheatcroft IJF 2020) carry signal beyond goals. That is useful because we have shots for about 51% of matches and xG for 2026/27.
- **Calibration:** temperature scaling improved GBDT calibration, while isotonic made it worse.

**Realistic expectations to show users:** about 49–54% 3-way accuracy and RPS 0.20–0.21. Draws are almost never the favourite, so the UI shows calibrated probabilities, not just "picks".

## 3. Data sources (free)

| Source | Use | Auth / limits | Notes |
|---|---|---|---|
| **football-data.co.uk** `fixtures.csv`, `new_league_fixtures.csv` | **Primary: upcoming fixtures + pre-match odds** | None | Next round only (published Tue/Fri). Exact name match (**LIVE**) |
| **football-data.co.uk** `mmz4281/{YYZZ}/{Div}.csv`, `new/{CTRY}.csv` | **Daily results + stats + xG** | None | 2026/27 files updated daily. `HxG`/`AxG` present for most main divisions (**LIVE**) |
| football-data.org v4 | Fallback: full schedule for the next 7–14 days | Free key, 10 req/min, 12 competitions | PL, ELC, BL1, SA, PD, FL1, DED, PPL, BSA, CL, EC, WC. Needs name aliases |
| OpenLigaDB | Bundesliga 1–3 live/fixtures | None | German names |
| TheSportsDB `eventsday.php` | Daily match list for other leagues | Key `123`, 30 req/min | Free tier is otherwise crippled |
| ClubElo | — | — | **Fixtures API deactivated; other endpoints returned 502** (**LIVE**). Not relied on |
| API-Football free | Not used | 100 req/day | Free plan restricts seasons, likely no current season (unverified) |
| FBref | Not used | — | Opta xG removed in Jan 2026 |
| FotMob | **Not used** | — | ToS forbids scraping |

> Python's `urllib`/`ssl` fails the TLS handshake with football-data.co.uk on this machine, but `curl` succeeds. The ingest client must use `httpx` with a fallback and retries. Plain HTTP redirects to HTTPS.

## 4. Libraries

- **penaltyblog** 1.13 (actively maintained): Dixon-Coles and other goal models, pi-ratings, Elo, RPS metric, odds de-vigging.
- **catboost** 1.2: the main classifier; handles missing stats natively.
- **scikit-learn**: logistic baselines and calibration utilities.
