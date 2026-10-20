import type { Fixture, Prediction } from '../api/types'

export function makePrediction(overrides: Partial<Prediction> = {}): Prediction {
  return {
    probs: { home: 0.52, draw: 0.26, away: 0.22 },
    expected_goals: { home: 1.61, away: 0.98 },
    over_2_5: 0.49,
    btts: 0.47,
    top_scores: [
      { score: '1-0', p: 0.12 },
      { score: '1-1', p: 0.11 },
      { score: '2-0', p: 0.1 },
      { score: '2-1', p: 0.09 },
      { score: '0-0', p: 0.08 },
    ],
    model: 'market',
    model_version: '2026-10-20',
    ...overrides,
  }
}

export function makeFixture(overrides: Partial<Fixture> = {}): Fixture {
  return {
    id: 1,
    league: 'E0',
    kickoff: '2026-10-20T15:00:00',
    home: 'Arsenal',
    away: 'Chelsea',
    odds: null,
    prediction: makePrediction(),
    status: 'scheduled',
    score: null,
    ...overrides,
  }
}
