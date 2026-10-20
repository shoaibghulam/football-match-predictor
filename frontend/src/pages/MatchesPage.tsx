import { useMemo } from 'react'
import { useSearchParams } from 'react-router'
import { useFeatured, useFixtures, useHealth, useLeagues } from '../api/hooks'
import { DateControls, type DateRange } from '../components/DateControls'
import { FeaturedSection } from '../components/FeaturedSection'
import { FixtureList } from '../components/FixtureList'
import { KeyNotice } from '../components/KeyNotice'
import { LeagueSelect } from '../components/LeagueSelect'
import { EmptyState, ErrorState, LoadingList } from '../components/StatusStates'
import { addDays, daysBetween, inLocalRange, isValidISODate, localTimeZoneLabel, todayISO } from '../utils/date'
import { filterFixtures } from '../utils/leagues'

/** API dates are UK dates; fetch one extra day either side so every local day is complete. */
const TIMEZONE_PADDING_DAYS = 1
/** Keeps the padded request within the API's 14-day limit. */
export const MAX_RANGE_DAYS = 14 - 2 * TIMEZONE_PADDING_DAYS

function readRange(params: URLSearchParams, today: string): DateRange {
  const rawDate = params.get('date')
  const date = isValidISODate(rawDate) ? rawDate : today
  const rawTo = params.get('to')
  if (!isValidISODate(rawTo)) return { date }
  const span = daysBetween(date, rawTo)
  return span > 0 && span <= MAX_RANGE_DAYS ? { date, dateTo: rawTo } : { date }
}

export function MatchesPage() {
  const [params, setParams] = useSearchParams()
  const today = todayISO()
  const range = readRange(params, today)
  const league = params.get('league') ?? ''
  const search = params.get('q') ?? ''

  const rangeEnd = range.dateTo ?? range.date
  const fixtures = useFixtures({
    date: addDays(range.date, -TIMEZONE_PADDING_DAYS),
    dateTo: addDays(rangeEnd, TIMEZONE_PADDING_DAYS),
  })
  const leagues = useLeagues()
  const featured = useFeatured(range.date)
  const health = useHealth()
  const missingKey = health.data?.football_data_org_key_configured === false

  const localDay = useMemo(
    () => (fixtures.data ?? []).filter((f) => inLocalRange(f, range.date, rangeEnd)),
    [fixtures.data, range.date, rangeEnd],
  )
  const visible = useMemo(
    () => filterFixtures(localDay, { leagues: league ? [league] : [], search }),
    [localDay, league, search],
  )
  const featuredLocal = (featured.data ?? []).filter((f) => inLocalRange(f, range.date, rangeEnd))

  function update(patch: Record<string, string | undefined>) {
    const next = new URLSearchParams(params)
    for (const [k, v] of Object.entries(patch)) {
      if (v) next.set(k, v)
      else next.delete(k)
    }
    setParams(next, { replace: true })
  }

  const setRange = (r: DateRange) => update({ date: r.date === today && !r.dateTo ? undefined : r.date, to: r.dateTo })
  const hasFilters = Boolean(league || search.trim())

  function renderContent() {
    if (fixtures.isPending) return <LoadingList />
    if (fixtures.isError) {
      return <ErrorState error={fixtures.error} onRetry={() => void fixtures.refetch()} title="Could not load matches" />
    }
    if (localDay.length === 0) {
      return (
        <EmptyState title="No matches for this date — try another day">
          <button type="button" className="btn mt-2" onClick={() => setRange({ date: addDays(range.dateTo ?? range.date, 1) })}>
            Next day ›
          </button>
        </EmptyState>
      )
    }
    if (visible.length === 0) {
      return (
        <EmptyState title="No matches fit your filters">
          <button type="button" className="btn mt-2" onClick={() => update({ league: undefined, q: undefined })}>
            Clear filters
          </button>
        </EmptyState>
      )
    }
    return <FixtureList fixtures={visible} leagues={leagues.data} showDate={Boolean(range.dateTo)} />
  }

  return (
    <div className="flex flex-col gap-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Matches</h1>
        <p className="muted text-sm">Calibrated win, draw and loss probabilities for every fixture.</p>
        <p className="muted mt-1 text-xs" data-testid="timezone-note">
          Dates and kick-off times are shown in your time zone: {localTimeZoneLabel()}
        </p>
      </div>

      {missingKey && <KeyNotice />}
      {featuredLocal.length > 0 && <FeaturedSection fixtures={featuredLocal} leagues={leagues.data} />}

      <div className="card grid gap-4 p-4 md:grid-cols-[1.4fr_1fr]">
        <DateControls value={range} onChange={setRange} today={today} />
        <div className="flex flex-col gap-3">
          <LeagueSelect
            leagues={leagues.data ?? []}
            value={league}
            onChange={(code) => update({ league: code || undefined })}
            disabled={!leagues.data}
          />
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium">Find a team</span>
            <input
              type="search"
              className="input"
              placeholder="e.g. Arsenal"
              value={search}
              onChange={(e) => update({ q: e.target.value || undefined })}
            />
          </label>
        </div>
      </div>

      {fixtures.isSuccess && localDay.length > 0 && (
        <p className="muted text-sm" aria-live="polite">
          {visible.length} of {localDay.length} matches{hasFilters ? ' (filtered)' : ''}
        </p>
      )}
      {renderContent()}
    </div>
  )
}
