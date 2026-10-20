# 06 — Model Report

_Trained 2026-10-20 · artifact `backend/artifacts/model.joblib` (2.1 MB, `model_version` = `2026-10-20`) · full run 18.5 min on CPU._
_Reproduce with `cd backend && uv run python -m matchpredict.predictor.train` (add `--quick` for a ~4 min subset run). Raw numbers are in `backend/artifacts/metrics.json` and `report_tables.md`._

## Summary against the targets

| Target (`03-model.md`) | Result (pooled over 4 test seasons, 40,871 matches) | Status |
|---|---|---|
| Base model (no odds) RPS ≤ 0.207 | **0.2078** | **Missed by 0.0008.** Clearly better than Elo-logit (0.2093) and the goals model (0.2083) |
| Market model RPS ≤ 0.2036 (at least as good as the odds) | **0.2034** vs bookmaker 0.2036 | **Met.** Better than or equal to the bookmaker in every fold |
| ECE ≤ 0.02 (per class) | base 0.0037 / 0.0040 / 0.0057 · market 0.0054 / 0.0095 / 0.0061 | **Met** |

The market model's edge over the odds is real but small: about 0.0002 RPS and 0.0008 log loss. It works as a calibrated, slightly bias-corrected version of the odds, not as a model that beats the market. The base model sits at the level the literature reports for goals-only models (0.206–0.209). It is about 0.004 RPS behind the bookmaker.

## Walk-forward protocol

- **Season** = 1 Jul to 30 Jun. For test season N:
  - **Train:** every match before the start of season N−1, minus a 365-day ratings burn-in at the start of the data (2000/01).
  - **Calibrate:** season N−1, used for CatBoost early stopping and the temperature fit.
  - **Test:** season N.
- **Ratings** are computed in one leakage-free chronological pass over all history. The engine reads its state before applying each day's results, so every match's features are strictly pre-match.
- **Evaluation rows:** all models are scored on the same test rows, namely those with valid Bet365 odds (about 99%). That makes the bookmaker row a like-for-like comparison.
- **Coverage gap:** the extra-country leagues (ARG, BRA, USA, JAP, MEX, …) end in December 2024 in `Matches.csv`. The 2025/26 fold and the per-league table below therefore cover the 22 European divisions only.

## Pooled results (2022/23 – 2025/26)

| Model | n | RPS | Log loss | Brier | Accuracy | ECE H / D / A |
|---|---|---|---|---|---|---|
| base rates | 40871 | 0.2282 | 1.0746 | 0.6502 | 43.7% | 0.0099 / 0.0025 / 0.0124 |
| Elo → logistic | 40871 | 0.2093 | 1.0194 | 0.6110 | 48.9% | 0.0135 / 0.0026 / 0.0105 |
| **Bet365 de-vigged** | 40871 | **0.2036** | 1.0012 | 0.5986 | 50.4% | 0.0137 / 0.0067 / 0.0136 |
| goals model 1X2 (Poisson GBM + Dixon-Coles) | 40871 | 0.2083 | 1.0161 | 0.6088 | 49.3% | 0.0075 / 0.0080 / 0.0126 |
| **base** (CatBoost, no odds) | 40871 | **0.2078** | 1.0144 | 0.6077 | 49.5% | 0.0037 / 0.0040 / 0.0057 |
| **market** (CatBoost residual on odds) | 40871 | **0.2034** | 1.0004 | 0.5983 | 50.5% | 0.0054 / 0.0095 / 0.0061 |

## Per test season (RPS / accuracy)

| Season | n | base rates | Elo-logit | Bookmaker | Goals | **Base** | **Market** |
|---|---|---|---|---|---|---|---|
| 2022/23 | 12334 | 0.2279 | 0.2085 | 0.2029 / 50.9% | 0.2076 | **0.2071** / 49.6% | **0.2026** / 50.8% |
| 2023/24 | 12137 | 0.2285 | 0.2098 | 0.2045 / 49.8% | 0.2087 | **0.2082** / 49.2% | **0.2043** / 49.8% |
| 2024/25 | 8766 | 0.2275 | 0.2092 | 0.2032 / 50.5% | 0.2085 | **0.2079** / 49.8% | **0.2030** / 50.7% |
| 2025/26 | 7634 | 0.2290 | 0.2097 | 0.2038 / 50.6% | 0.2087 | **0.2081** / 49.3% | **0.2036** / 50.8% |

The ranking market < bookmaker < base < goals < Elo-logit < base rates holds in every fold.

## Per league, last fold (2025/26, RPS)

| League | n | Elo-logit | Bookmaker | Goals | Base | Market |
|---|---|---|---|---|---|---|
| B1 | 311 | 0.2176 | 0.2112 | 0.2146 | 0.2145 | 0.2107 |
| D1 | 306 | 0.1950 | 0.1904 | 0.1945 | 0.1923 | 0.1890 |
| D2 | 306 | 0.2256 | 0.2180 | 0.2214 | 0.2221 | 0.2173 |
| E0 | 380 | 0.2115 | 0.2057 | 0.2114 | 0.2113 | 0.2066 |
| E1 | 552 | 0.2224 | 0.2165 | 0.2211 | 0.2205 | 0.2173 |
| E2 | 552 | 0.2175 | 0.2114 | 0.2181 | 0.2166 | 0.2113 |
| E3 | 552 | 0.2206 | 0.2131 | 0.2173 | 0.2167 | 0.2124 |
| EC | 540 | 0.2141 | 0.2061 | 0.2170 | 0.2178 | 0.2064 |
| F1 | 306 | 0.2055 | 0.1994 | 0.2038 | 0.2031 | 0.1987 |
| F2 | 305 | 0.2225 | 0.2139 | 0.2214 | 0.2204 | 0.2141 |
| G1 | 236 | 0.1889 | 0.1848 | 0.1860 | 0.1846 | 0.1845 |
| I1 | 380 | 0.2039 | 0.1966 | 0.1996 | 0.2007 | 0.1965 |
| I2 | 380 | 0.2010 | 0.1985 | 0.2013 | 0.1994 | 0.1976 |
| N1 | 306 | 0.1966 | 0.1946 | 0.2008 | 0.1977 | 0.1963 |
| P1 | 306 | 0.1756 | 0.1740 | 0.1725 | 0.1733 | 0.1732 |
| SC0 | 228 | 0.1945 | 0.1923 | 0.1970 | 0.1946 | 0.1920 |
| SC1 | 180 | 0.2141 | 0.2095 | 0.2148 | 0.2116 | 0.2109 |
| SC2 | 180 | 0.2160 | 0.2085 | 0.2143 | 0.2138 | 0.2079 |
| SC3 | 180 | 0.2278 | 0.2166 | 0.2238 | 0.2249 | 0.2174 |
| SP1 | 380 | 0.1997 | 0.1957 | 0.2001 | 0.1985 | 0.1950 |
| SP2 | 462 | 0.2227 | 0.2158 | 0.2200 | 0.2216 | 0.2155 |
| T1 | 306 | 0.1936 | 0.1850 | 0.1929 | 0.1927 | 0.1842 |

Per-league samples hold only 180–550 matches, so differences under about 0.003 are noise. The base model's largest gaps to the bookmaker are in EC (National League) and the lower English and Scottish tiers. These are leagues where team news, budgets and squad turnover carry information that results-based ratings do not see.

## What the models are

- **Features** (`predictor/features.py`, 48 numeric + league; the 4 xG EWMAs are empty until 2026/27 data arrives):
  - Own Elo (K = 20, home advantage 60) and a fast Elo (K = 40)
  - Pi-ratings
  - Online Poisson attack/defence ratings, giving per-league λ_home and λ_away plus their independent-Poisson 1X2
  - Form5 and goal difference over the last 5
  - Rest days and matches played
  - EWMAs (α = 0.15) of goals, shots, shots on target and corners, for and against
  - Rolling league home-win and draw rates
  - League as a CatBoost categorical
- **Base model:** CatBoost `MultiClass`, depth 6, learning rate 0.05, l2 = 5, early-stopped on the calibration season (261–557 trees per fold). Temperature scaling is fitted on the same season (T = 0.99–1.04, so CatBoost was already well calibrated).
- **Market model:** CatBoost boosts *from* the de-vigged Bet365 log-probabilities, which are used as the baseline, so the trees learn only a residual correction. Learning rate 0.02. It is used only when the fixture has valid odds.
- **Goals model:**
  - Two `HistGradientBoostingRegressor(loss="poisson")` models, one for home goals and one for away goals, on the rating features.
  - Dixon-Coles ρ fitted by maximum likelihood on the training rows (final ρ = −0.046).
  - 0–10 goal score matrix, giving P(over 2.5), BTTS and the top-5 scorelines.
  - At serving time the matrix is rescaled so its H/D/A regions equal the classifier's 1X2 (coherence). Expected goals, O2.5, BTTS and the top scores are read from the rescaled matrix.
- **Final artifact:**
  - Hyperparameters and temperature come from a fit that trains on everything up to 365 days before the last match and early-stops and calibrates on the last 365 days.
  - The models are then refit on all 233,262 usable matches with 1.1× the tuned tree count, keeping the fitted T.
  - It also contains the ratings engine state after all 238,854 matches.

## How choices were made (no test-fold peeking)

Choices were made on a separate pre-test validation split: train before 2020-07-01, early-stop on 2020/21, score on 2021/22. The test folds were run once with the chosen settings.

| Experiment (validation 2021/22, odds rows) | Log loss | RPS |
|---|---|---|
| Base, original engine features, learning rate 0.1 | 1.02000 | 0.20942 |
| + fast Elo + Poisson attack/defence ratings | 1.02017 | 0.20940 |
| Learning rate 0.05, depth 4 / **6** / 8 | 1.01946 / **1.01908** / 1.01969 | 0.20928 / **0.20913** / 0.20935 |
| Compact 18-feature set | 1.02207 | 0.21012 |
| Bookmaker de-vigged | 1.00230 | 0.20390 |
| Market: CatBoost with odds as plain features | 1.00282 | 0.20402 (worse than the odds) |
| Market: logistic stack of log-odds and base logits | 1.00171 | 0.20375 |
| **Market: CatBoost residual on log-odds (chosen)** | **1.00152** | **0.20371** |

The extra rating features were neutral on validation (results-based information is saturated). They were kept because the goals model uses them and they cost nothing. The decisive change for the market model was switching from "odds as features" to "odds as baseline". Plain CatBoost could not even match the odds it was given.

## Honest caveats and known limitations

1. **The base target is missed by 0.0008.** Closing the gap probably needs information the results history does not carry: lineups and injuries, transfer spend, xG (only available from 2026/27), or a cross-league strength model for promotion and relegation. More tuning of the same features gave nothing on validation.
2. **The early-stopping and calibration season is the same season.** Temperature is therefore fitted on data that also chose the tree count. The effect is tiny (T ≈ 1), and the test-fold ECE confirms calibration holds out of sample.
3. **The 2020/21 validation season was played in empty stadiums** with reduced home advantage. It was used only for relative model selection.
4. **O/U 2.5, BTTS and scorelines are not benchmarked** against the bookmaker's O/U odds in this report. Only the 1X2 head of the goals model is evaluated.
5. **Extra-country leagues** stop at December 2024 in the training data. Their ratings are only as fresh as whatever `ModelService.update_results` has applied since; it de-duplicates per league, so backfills are accepted.
6. **Newly promoted or never-seen teams:** in a known league they are seeded at the league mean minus 100 Elo, below-average attack and defence, and neutral form. In a completely unknown league, `predict` raises `KeyError`.
7. **Ratings are updated incrementally between weekly retrains,** but the classifiers are not. A retrain with the promotion gate from `03-model.md` is still a Phase 5 item.
