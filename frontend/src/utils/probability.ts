import type { Fixture, Outcome, Probs } from '../api/types'

export const OUTCOMES: readonly Outcome[] = ['home', 'draw', 'away']

function clamp01(p: number): number {
  if (!Number.isFinite(p)) return 0
  return Math.min(1, Math.max(0, p))
}

/** "52%" (rounded); "<1%" for tiny non-zero values. */
export function formatPercent(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return '–'
  const v = clamp01(p)
  if (v > 0 && v < 0.005) return '<1%'
  return `${Math.round(v * 100)}%`
}

/**
 * Integer percentages for the three outcomes that always sum to 100
 * (largest-remainder method), so the bar labels never read 101%.
 */
export function toPercents(probs: Probs): Record<Outcome, number> {
  const raw = OUTCOMES.map((o) => clamp01(probs[o]))
  const total = raw.reduce((a, b) => a + b, 0)
  if (total <= 0) return { home: 0, draw: 0, away: 0 }
  const scaled = raw.map((p) => (p / total) * 100)
  const floors = scaled.map(Math.floor)
  let remaining = 100 - floors.reduce((a, b) => a + b, 0)
  const order = scaled
    .map((v, i) => ({ i, frac: v - Math.floor(v) }))
    .sort((a, b) => b.frac - a.frac)
  const result = [...floors]
  for (const { i } of order) {
    if (remaining <= 0) break
    result[i] = (result[i] ?? 0) + 1
    remaining -= 1
  }
  return { home: result[0] ?? 0, draw: result[1] ?? 0, away: result[2] ?? 0 }
}

/** The most likely outcome (ties resolve home > away > draw). */
export function favouriteOutcome(probs: Probs): Outcome {
  if (probs.home >= probs.away && probs.home >= probs.draw) return 'home'
  if (probs.away >= probs.draw) return 'away'
  return 'draw'
}

/** Expected goals with one decimal, e.g. "1.6". */
export function formatGoals(xg: number | null | undefined): string {
  if (xg === null || xg === undefined || !Number.isFinite(xg)) return '–'
  return Math.max(0, xg).toFixed(1)
}

/** Win probability for `team` in a fixture (from its home or away perspective), or null. */
export function teamWinChance(fixture: Fixture, team: string): number | null {
  const probs = fixture.prediction?.probs
  if (!probs) return null
  const t = team.trim().toLowerCase()
  if (fixture.home.toLowerCase() === t) return probs.home
  if (fixture.away.toLowerCase() === t) return probs.away
  return null
}

/** Plain-language strength of a probability; never implies certainty. */
export function likelihoodLabel(p: number): string {
  if (p >= 0.7) return 'Strong favourite'
  if (p >= 0.5) return 'Favourite'
  if (p >= 0.4) return 'Slight favourite'
  return 'Most likely, but open'
}
