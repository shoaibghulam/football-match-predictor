import type { OutcomeCode, ResultItem } from '../../api/results'

export const OUTCOME_NAMES: Record<OutcomeCode, string> = { H: 'Home win', D: 'Draw', A: 'Away win' }

/** "Chelsea", "Draw" or "Coventry" for an outcome of this match. */
export function outcomeLabel(item: Pick<ResultItem, 'home' | 'away'>, code: OutcomeCode): string {
  if (code === 'H') return item.home
  if (code === 'A') return item.away
  return 'Draw'
}

/** "52.3%" with one decimal (accuracy figures), "–" when unknown. */
export function pct1(p: number | null | undefined): string {
  return p === null || p === undefined || !Number.isFinite(p) ? '–' : `${(p * 100).toFixed(1)}%`
}

export function formatRps(v: number | null | undefined): string {
  return v === null || v === undefined || !Number.isFinite(v) ? '–' : v.toFixed(4)
}

export function formatCount(n: number): string {
  return n.toLocaleString('en-GB')
}
