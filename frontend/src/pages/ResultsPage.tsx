import { useMemo } from 'react'
import { useSearchParams } from 'react-router'
import { useLeagues } from '../api/hooks'
import { useResults, useResultsSummary, type SourceFilter } from '../api/results'
import { AccuracyChart } from '../components/results/AccuracyChart'
import { CalibrationTable } from '../components/results/CalibrationTable'
import { formatCount } from '../components/results/labels'
import { ResultsBreakdown } from '../components/results/ResultsBreakdown'
import { ResultsFilters, type FilterPatch } from '../components/results/ResultsFilters'
import { ResultsList } from '../components/results/ResultsList'
import { ResultsStats } from '../components/results/ResultsStats'
import { EmptyState, ErrorState, LoadingList } from '../components/StatusStates'
import { useDebouncedValue } from '../hooks/useDebouncedValue'
import { localTimeZoneLabel, todayISO } from '../utils/date'
import { leagueLookup } from '../utils/leagues'
import { readResultsUrl, toApiFilters } from '../utils/resultsFilters'

const TEAM_DEBOUNCE_MS = 300
/** Characters the API accepts in a team name. */
const TEAM_UNSAFE = /[^\p{L}\p{N}_ .'&()/-]/gu

const SOURCE_NOTES: Record<SourceFilter, string> = {
  live: 'Live record: predictions this app froze before kick-off, scored after the final whistle.',
  backtest:
    'Backtest: every match of a season was predicted by a model trained only on matches before that season (walk-forward, out-of-sample). It is a simulation of past seasons, not live picks.',
  all: 'Live and backtest records together — each row is labelled with where it comes from.',
}

export function ResultsPage() {
  const [params, setParams] = useSearchParams()
  const today = todayISO()
  const state = readResultsUrl(params)
  const team = useDebouncedValue(state.team.replace(TEAM_UNSAFE, ''), TEAM_DEBOUNCE_MS)
  const query = { ...state, team }

  // No source in the URL: use the live record when it has a finished match, else the backtest.
  const probe = useResultsSummary(state.source === null ? toApiFilters(query, 'live', today) : null)
  const autoSource: SourceFilter | null = probe.data ? (probe.data.available.live > 0 ? 'live' : 'backtest') : null
  const source = state.source ?? autoSource
  const fellBack = state.source === null && autoSource === 'backtest'
  const filters = source ? toApiFilters(query, source, today) : null

  const summary = useResultsSummary(filters)
  const list = useResults(filters)
  const leagues = useLeagues()
  const lookup = useMemo(() => leagueLookup(leagues.data), [leagues.data])
  const items = useMemo(() => list.data?.pages.flatMap((p) => p.items) ?? [], [list.data])
  const total = list.data?.pages[0]?.total ?? 0

  function update(patch: FilterPatch) {
    const next = new URLSearchParams(params)
    for (const [k, v] of Object.entries(patch)) {
      if (v) next.set(k, v)
      else next.delete(k)
    }
    setParams(next, { replace: true })
  }

  function renderList() {
    if (list.isPending) return <LoadingList label="Loading results" />
    if (list.isError) {
      return <ErrorState error={list.error} onRetry={() => void list.refetch()} title="Could not load results" />
    }
    if (items.length === 0) {
      const pending = filters?.status === 'pending'
      return (
        <EmptyState title={pending ? 'No upcoming predicted matches' : 'No predicted matches fit these filters'}>
          <button type="button" className="btn mt-2" onClick={() => update({ range: undefined, from: undefined, to: undefined, league: undefined, q: undefined, outcome: undefined, confidence: undefined })}>
            Clear filters
          </button>
        </EmptyState>
      )
    }
    return (
      <>
        <p className="muted text-sm" aria-live="polite">
          Showing {formatCount(items.length)} of {formatCount(total)} {filters?.status === 'pending' ? 'upcoming' : 'finished'} matches
        </p>
        <ResultsList items={items} leagues={lookup} showSource={source === 'all'} />
        {list.hasNextPage && (
          <button type="button" className="btn self-center" disabled={list.isFetchingNextPage} onClick={() => void list.fetchNextPage()}>
            {list.isFetchingNextPage ? 'Loading…' : 'Load more'}
          </button>
        )}
      </>
    )
  }

  const blockingError = probe.isError && state.source === null ? probe.error : summary.isError ? summary.error : null

  return (
    <div className="flex flex-col gap-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Results</h1>
        <p className="muted text-sm">Every match we predicted — and whether we got it right.</p>
        <p className="muted mt-1 text-xs">Dates and kick-off times are shown in your time zone: {localTimeZoneLabel()}</p>
      </div>

      {fellBack && (
        <div role="note" className="rounded-xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100">
          No live prediction has a final result yet — live tracking only started recently. Showing the backtest
          record meanwhile; switch to <strong>Live</strong> to see upcoming predicted matches.
        </div>
      )}
      {source && <p className="text-sm">{SOURCE_NOTES[source]}</p>}

      {source && <ResultsFilters state={state} source={source} leagues={leagues.data ?? []} onChange={update} />}

      {blockingError ? (
        <ErrorState error={blockingError} onRetry={() => void (summary.isError ? summary.refetch() : probe.refetch())} title="Could not load results" />
      ) : summary.data && summary.data.n > 0 ? (
        <ResultsStats summary={summary.data} />
      ) : summary.data ? (
        <p className="muted text-sm">No finished matches to score for these filters{summary.data.pending ? ` (${formatCount(summary.data.pending)} awaiting a result)` : ''}.</p>
      ) : (
        <LoadingList count={1} label="Loading accuracy" />
      )}

      <section aria-labelledby="results-list" className="flex flex-col gap-3">
        <h2 id="results-list" className="text-lg font-semibold">
          {filters?.status === 'pending' ? 'Awaiting result' : 'Matches'}
        </h2>
        {filters ? renderList() : <LoadingList label="Loading results" />}
      </section>

      {summary.data && summary.data.n > 0 && (
        <>
          <ResultsBreakdown summary={summary.data} leagues={lookup} />
          <div className="grid gap-4 md:grid-cols-2">
            <CalibrationTable buckets={summary.data.by_confidence_bucket} />
            <AccuracyChart summary={summary.data} />
          </div>
        </>
      )}
    </div>
  )
}
