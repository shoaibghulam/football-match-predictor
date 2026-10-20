# 03 — Data Pipeline & Prediction Model

## Outputs per match
- P(Home win), P(Draw), P(Away win). These are calibrated and sum to 1.
- Expected goals for each side, the most likely scorelines, P(Over 2.5) and P(both teams to score). These come from the goals model.
- A "win chance for team X" view, derived from the above for the selected team.

## Ratings engine (one chronological pass, strictly pre-kickoff)

Matches are sorted by `(MatchDate, MatchTime)`. Each match **reads** the state first and **then updates** it, so a match never sees its own result.

| Rating | Description |
|---|---|
| **Own Elo** | ClubElo-style. Margin-of-victory multiplier, per-league home advantage, K≈20. Covers all 38 leagues; new teams are seeded at their league's mean of the bottom four. |
| **Pi-ratings** | Constantinou & Fenton 2013: separate home and away ratings, goal-difference error with diminishing returns, λ=0.06, γ=0.5 (to be tuned). |
| **Process (GAP-style) EWMAs** | Exponentially weighted shots, shots on target, corners and goals, for and against, plus xG where present. NaN when no stats exist, and CatBoost handles the gaps. |
| **Form & context** | Points over the last 3/5/10 matches, rest days, matches played this season, promoted flag, per-league rolling home-win and draw rates. |

## Models

- **(A) Goals model:** Dixon-Coles with time decay (`penaltyblog`), fitted per league on a rolling window of about 3 seasons with ξ tuned so the half-life is about 1 season. Outputs λ_home, λ_away, a score matrix, 1X2 and O/U 2.5. Its 1X2 is also fed into (B) as features.
- **(B) Base classifier:** CatBoost multiclass (`MultiClass` loss) on the rating differences and contexts above, plus the league as a categorical feature. **It uses no odds.** Used for any fixture without odds.
- **(B') Market classifier:** the same features plus de-vigged Bet365 pre-match probabilities. Used when the fixture has odds (`fixtures.csv` always does). It is legitimate because the same pre-match odds exist at serving time; closing odds (`B365C*`) are **never** used.
- **Calibration:** temperature scaling, fitted on the validation season.
- **Coherence:** the goals-model score matrix is rescaled so its H/D/A regions match the final 1X2.

## Validation (walk-forward, never shuffled)

- Train on seasons ≤ N−1, tune and calibrate on N−1, test on N. Repeat for N = 2022/23, 2023/24, 2024/25 and 2025/26.
- **Metrics:** RPS (primary, for comparison with the literature), log loss (selection), multiclass Brier, accuracy, and reliability per class.
- **Benchmarks:**
  - base rates: 0.2288
  - Elo logit: 0.2103
  - **Bet365 de-vigged: 0.2036**
- **Acceptance targets:**
  - Base model: RPS ≤ 0.207 pooled.
  - Market model: RPS ≤ 0.2036, i.e. at least as good as the odds alone.
  - ECE ≤ 0.02.

## Leakage checklist
- [ ] Features come only from matches with kickoff before this one; same-day matches without a time are handled as "unknown order", so they don't see each other.
- [ ] The match's own shots, cards and goals are never features.
- [ ] ClubElo values are not used (the internal Elo replaces them, and its snapshot timing is unknown).
- [ ] Encoders, calibrators and decay parameters are fit on training folds only.
- [ ] Closing odds are never used.

## Update loop
Each day ingest appends finished matches → the ratings state is updated incrementally (O(new matches)) → upcoming fixtures are re-predicted. The models are fully retrained weekly, and a new artifact is promoted only if its walk-forward RPS is no worse than the current one.
