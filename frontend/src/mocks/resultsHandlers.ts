/** Mock /results and /results/summary (dev mock mode and tests). */
import { http, HttpResponse } from 'msw'
import { buildUrl } from '../api/client'
import type { OutcomeCode, ResultItem, ResultsSummary } from '../api/results'

const CODES: readonly OutcomeCode[] = ['H', 'D', 'A']

export function makeResult(overrides: Partial<ResultItem> = {}): ResultItem {
  return {
    id: 'live-1',
    match_id: 1,
    source: 'live',
    status: 'finished',
    date: '2026-10-19',
    kickoff: '2026-10-19T20:00:00+01:00',
    league: 'E0',
    home: 'Arsenal',
    away: 'Chelsea',
    score: { home: 2, away: 1 },
    actual: 'H',
    predicted: 'H',
    probs: { home: 0.55, draw: 0.25, away: 0.2 },
    prob_predicted: 0.55,
    prob_of_actual: 0.55,
    correct: true,
    rps: 0.1012,
    model: 'market',
    model_version: 'v2-2026-10-21',
    frozen_at: '2026-10-19T18:30:00Z',
    ...overrides,
  }
}

export const MOCK_RESULTS: ResultItem[] = [
  makeResult(),
  makeResult({ id: 'live-2', match_id: 2, home: 'Getafe', away: 'Barcelona', league: 'SP1', score: { home: 1, away: 1 },
    actual: 'D', predicted: 'A', probs: { home: 0.2, draw: 0.27, away: 0.53 }, prob_predicted: 0.53, prob_of_actual: 0.27,
    correct: false, rps: 0.1653 }),
  makeResult({ id: 'live-3', match_id: 3, date: '2026-10-24', kickoff: '2026-10-24T15:00:00+01:00', home: 'Liverpool',
    away: 'Everton', status: 'pending', score: null, actual: null, correct: null, prob_of_actual: null, rps: null }),
  makeResult({ id: 'backtest-4', match_id: 4, source: 'backtest', date: '2025-05-11', kickoff: null, home: 'Lazio',
    away: 'Juventus', league: 'I1', score: { home: 0, away: 2 }, actual: 'A', predicted: 'A',
    probs: { home: 0.3, draw: 0.28, away: 0.42 }, prob_predicted: 0.42, prob_of_actual: 0.42, model: 'base',
    model_version: 'v2-backtest-2024/25', frozen_at: null, rps: 0.2 }),
]

function stat(rows: readonly ResultItem[]) {
  const correct = rows.filter((r) => r.correct).length
  return { n: rows.length, correct, accuracy: rows.length ? correct / rows.length : null }
}

export function summarise(rows: readonly ResultItem[], all: readonly ResultItem[], pending: number): ResultsSummary {
  const done = rows.filter((r) => r.status === 'finished')
  const byCode = (key: 'actual' | 'predicted') =>
    Object.fromEntries(CODES.map((c) => [c, stat(done.filter((r) => r[key] === c))])) as ResultsSummary['by_outcome']
  const rps = done.length ? done.reduce((a, r) => a + (r.rps ?? 0), 0) / done.length : null
  const leagues = [...new Set(done.map((r) => r.league))]
  return {
    source: 'live', ...stat(done), rps, log_loss: null, brier: null,
    by_outcome: byCode('actual'), by_predicted: byCode('predicted'),
    by_league: leagues.map((league) => ({ league, ...stat(done.filter((r) => r.league === league)), rps: rps ?? 0 })),
    by_confidence_bucket: [{ bucket: '50-60%', lo: 0.5, hi: 0.6, ...stat(done), avg_prob: 0.54 }],
    bookmaker: done.length ? { n: done.length, accuracy: 0.5, rps: 0.2036, model_accuracy: stat(done).accuracy ?? 0, model_rps: rps ?? 0 } : null,
    daily: [...new Set(done.map((r) => r.date))].sort().map((date) => ({ date, ...stat(done.filter((r) => r.date === date)) })),
    monthly: [],
    first_date: done[0]?.date ?? null, last_date: done[done.length - 1]?.date ?? null,
    pending,
    available: { live: all.filter((r) => r.source === 'live' && r.status === 'finished').length,
      backtest: all.filter((r) => r.source === 'backtest').length },
  }  // prettier-ignore
}

function filterRows(url: URL, rows: readonly ResultItem[], forceFinished = false): ResultItem[] {
  const p = url.searchParams
  const source = p.get('source') ?? 'live'
  const status = forceFinished ? 'finished' : (p.get('status') ?? 'finished')
  const outcome = p.get('outcome')
  const league = p.get('league')
  const team = (p.get('team') ?? '').toLowerCase()
  return rows.filter(
    (r) =>
      (source === 'all' || r.source === source) &&
      r.status === status &&
      (!outcome || (outcome === 'correct' ? r.correct === true : r.correct === false)) &&
      (!league || r.league === league) &&
      (!team || r.home.toLowerCase().includes(team) || r.away.toLowerCase().includes(team)),
  )
}

/** Handlers over a given data set (tests pass their own rows). */
export function resultsHandlers(rows: readonly ResultItem[] = MOCK_RESULTS) {
  return [
    http.get(buildUrl('/results'), ({ request }) => {
      const url = new URL(request.url)
      const page = Number(url.searchParams.get('page') ?? '1')
      const size = Number(url.searchParams.get('page_size') ?? '25')
      const matched = filterRows(url, rows)
      return HttpResponse.json({ items: matched.slice((page - 1) * size, page * size), total: matched.length, page, page_size: size })
    }),
    http.get(buildUrl('/results/summary'), ({ request }) => {
      const url = new URL(request.url)
      const pendingUrl = new URL(url)
      pendingUrl.searchParams.set('status', 'pending')
      return HttpResponse.json(summarise(filterRows(url, rows, true), rows, filterRows(pendingUrl, rows).length))
    }),
  ]
}
