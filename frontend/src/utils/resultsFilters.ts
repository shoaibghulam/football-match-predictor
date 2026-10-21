/** URL state <-> API filters for the Results page. */
import type { ConfidenceFilter, OutcomeFilter, ResultFilters, ResultStatus, SourceFilter } from '../api/results'
import { addDays, daysBetween, isValidISODate } from './date'

export type RangePreset = 'all' | 'yesterday' | '7d' | '30d' | 'season' | 'custom'

export const RANGE_PRESETS: readonly { value: RangePreset; label: string }[] = [
  { value: 'all', label: 'All time' },
  { value: 'yesterday', label: 'Yesterday' },
  { value: '7d', label: 'Last 7 days' },
  { value: '30d', label: 'Last 30 days' },
  { value: 'season', label: 'This season' },
  { value: 'custom', label: 'Custom' },
]

/** The API limits live/all ranges to 366 days. */
export const MAX_CUSTOM_DAYS = 366
const SEASON_START_MONTH = 7

export interface ResultsUrlState {
  /** null when the URL does not choose a source (the page picks live or backtest) */
  source: SourceFilter | null
  range: RangePreset
  from?: string
  to?: string
  league: string
  team: string
  outcome: OutcomeFilter
  confidence: ConfidenceFilter
  status: ResultStatus
}

function pick<T extends string>(value: string | null, allowed: readonly T[], fallback: T): T {
  return allowed.includes(value as T) ? (value as T) : fallback
}

export function readResultsUrl(params: URLSearchParams): ResultsUrlState {
  const rawSource = params.get('source')
  const source = rawSource === 'live' || rawSource === 'backtest' || rawSource === 'all' ? rawSource : null
  const from = params.get('from')
  const to = params.get('to')
  return {
    source,
    range: pick(params.get('range'), RANGE_PRESETS.map((p) => p.value), 'all'),
    from: isValidISODate(from) ? from : undefined,
    to: isValidISODate(to) ? to : undefined,
    league: params.get('league') ?? '',
    team: params.get('q') ?? '',
    outcome: pick(params.get('outcome'), ['all', 'correct', 'wrong'] as const, 'all'),
    confidence: pick(params.get('confidence'), ['all', 'high', 'low'] as const, 'all'),
    status: source !== 'backtest' && params.get('status') === 'pending' ? 'pending' : 'finished',
  }
}

/** First day (1 July) of the football season containing `today`. */
export function seasonStart(today: string): string {
  const [y, m] = today.split('-').map(Number)
  const year = (m ?? 1) >= SEASON_START_MONTH ? (y ?? 0) : (y ?? 0) - 1
  return `${year}-07-01`
}

export function presetRange(state: ResultsUrlState, today: string): { date?: string; dateTo?: string } {
  switch (state.range) {
    case 'yesterday': {
      const y = addDays(today, -1)
      return { date: y, dateTo: y }
    }
    case '7d':
      return { date: addDays(today, -6), dateTo: today }
    case '30d':
      return { date: addDays(today, -29), dateTo: today }
    case 'season':
      return { date: seasonStart(today), dateTo: today }
    case 'custom':
      return customRange(state.from, state.to)
    default:
      return {}
  }
}

function customRange(from?: string, to?: string): { date?: string; dateTo?: string } {
  if (from && to && daysBetween(from, to) < 0) return { date: to, dateTo: from }
  if (from && to && daysBetween(from, to) > MAX_CUSTOM_DAYS) return { date: from, dateTo: addDays(from, MAX_CUSTOM_DAYS) }
  return { date: from, dateTo: to }
}

export function toApiFilters(state: ResultsUrlState, source: SourceFilter, today: string): ResultFilters {
  const team = state.team.trim()
  return {
    source,
    status: source === 'backtest' ? 'finished' : state.status,
    ...presetRange(state, today),
    league: state.league || undefined,
    team: team.length > 0 ? team : undefined,
    outcome: state.outcome,
    confidence: state.confidence,
  }
}
