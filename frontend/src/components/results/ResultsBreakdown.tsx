import type { OutcomeCode, ResultsSummary } from '../../api/results'
import type { League } from '../../api/types'
import { formatCount, formatRps, OUTCOME_NAMES, pct1 } from './labels'

const CODES: readonly OutcomeCode[] = ['H', 'D', 'A']
const TOP_LEAGUES = 8

function Meter({ value }: { value: number | null }) {
  const width = value === null ? 0 : Math.round(value * 100)
  return (
    <span className="block h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800" aria-hidden="true">
      <span className="block h-full rounded-full bg-blue-600 dark:bg-blue-400" style={{ width: `${width}%` }} />
    </span>
  )
}

function OutcomePanel({ summary }: { summary: ResultsSummary }) {
  return (
    <section aria-labelledby="by-outcome" className="card flex flex-col gap-3 p-4">
      <h2 id="by-outcome" className="font-semibold">
        By outcome
      </h2>
      <table className="w-full text-sm">
        <thead>
          <tr className="muted text-left text-xs">
            <th scope="col" className="py-1 font-medium">Result</th>
            <th scope="col" className="py-1 font-medium">Matches</th>
            <th scope="col" className="py-1 font-medium">We called it</th>
            <th scope="col" className="py-1 font-medium">When we picked it</th>
          </tr>
        </thead>
        <tbody>
          {CODES.map((code) => {
            const actual = summary.by_outcome[code]
            const picked = summary.by_predicted[code]
            return (
              <tr key={code} className="border-t border-slate-100 dark:border-slate-800">
                <th scope="row" className="py-1.5 text-left font-medium">{OUTCOME_NAMES[code]}</th>
                <td className="tabular-nums">{formatCount(actual.n)}</td>
                <td className="tabular-nums">{pct1(actual.accuracy)}</td>
                <td className="tabular-nums">
                  {pct1(picked.accuracy)} <span className="muted text-xs">of {formatCount(picked.n)}</span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="muted text-xs">
        &ldquo;We called it&rdquo;: share of matches with that result where it was our most likely outcome. Draws are
        rarely the single most likely outcome, so this is usually low for draws.
      </p>
    </section>
  )
}

function LeaguePanel({ summary, leagues }: { summary: ResultsSummary; leagues: ReadonlyMap<string, League> }) {
  const rows = summary.by_league.slice(0, TOP_LEAGUES)
  return (
    <section aria-labelledby="by-league" className="card flex flex-col gap-3 p-4">
      <h2 id="by-league" className="font-semibold">
        By league <span className="muted text-xs font-normal">(most matches)</span>
      </h2>
      <ul className="flex flex-col gap-2 text-sm">
        {rows.map((r) => (
          <li key={r.league} className="flex flex-col gap-1">
            <span className="flex justify-between gap-2">
              <span className="truncate">{leagues.get(r.league)?.name ?? r.league}</span>
              <span className="tabular-nums">
                {pct1(r.accuracy)} <span className="muted text-xs">of {formatCount(r.n)} · RPS {formatRps(r.rps)}</span>
              </span>
            </span>
            <Meter value={r.accuracy} />
          </li>
        ))}
      </ul>
    </section>
  )
}

export function ResultsBreakdown({ summary, leagues }: { summary: ResultsSummary; leagues: ReadonlyMap<string, League> }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <OutcomePanel summary={summary} />
      {summary.by_league.length > 0 && <LeaguePanel summary={summary} leagues={leagues} />}
    </div>
  )
}
