import type { ResultItem } from '../../api/results'
import type { League } from '../../api/types'
import { formatDayLabel, localMatchDate, parseKickoff } from '../../utils/date'
import { formatPercent } from '../../utils/probability'
import { ProbabilityBar } from '../ProbabilityBar'
import { outcomeLabel } from './labels'
import { ResultBadge } from './ResultBadge'

interface ResultsListProps {
  items: readonly ResultItem[]
  leagues: ReadonlyMap<string, League>
  showSource: boolean
}

/** Stacked cards on small screens, a table from `md` up (one DOM, responsive CSS). */
const TD = 'block px-0 py-1 md:table-cell md:px-3 md:py-2 md:align-middle'
const LABEL = 'muted mr-1 text-xs md:hidden'

function MatchCell({ item }: { item: ResultItem }) {
  const score = item.score ? `${item.score.home}–${item.score.away}` : 'v'
  return (
    <span className="flex flex-wrap items-baseline gap-x-2 font-medium">
      <span className={item.actual === 'H' ? 'font-bold' : ''}>{item.home}</span>
      <span className="tabular-nums" aria-label={item.score ? `score ${score}` : 'versus'}>
        {score}
      </span>
      <span className={item.actual === 'A' ? 'font-bold' : ''}>{item.away}</span>
    </span>
  )
}

function ResultRow({ item, leagues, showSource }: { item: ResultItem; leagues: ReadonlyMap<string, League>; showSource: boolean }) {
  const day = localMatchDate(item) ?? item.date
  const time = parseKickoff(item.kickoff).time
  const leagueName = leagues.get(item.league)?.name ?? item.league
  return (
    <tr className="card mb-3 block p-3 md:mb-0 md:table-row md:rounded-none md:border-0 md:border-b md:p-0 md:shadow-none">
      <td className={TD}>
        <span className="text-sm">
          {formatDayLabel(day)}
          {time ? ` ${time}` : ''}
        </span>
        <span className="muted block text-xs">
          {leagueName}
          {showSource && (
            <span className="ml-1 rounded bg-slate-100 px-1.5 py-px text-[11px] font-semibold uppercase dark:bg-slate-800">
              {item.source}
            </span>
          )}
        </span>
      </td>
      <td className={TD}>
        <MatchCell item={item} />
      </td>
      <td className={TD}>
        <span className={LABEL}>Our pick:</span>
        <span className="text-sm font-semibold">
          {outcomeLabel(item, item.predicted)} · {formatPercent(item.prob_predicted)}
        </span>
        <span className="mt-1 block w-full max-w-48">
          <ProbabilityBar probs={item.probs} homeLabel={item.home} awayLabel={item.away} compact />
        </span>
      </td>
      <td className={TD}>
        <span className={LABEL}>Result:</span>
        <span className="text-sm">{item.actual ? outcomeLabel(item, item.actual) : '—'}</span>
        {item.prob_of_actual !== null && (
          <span className="muted block text-xs">we gave it {formatPercent(item.prob_of_actual)}</span>
        )}
      </td>
      <td className={TD}>
        <ResultBadge item={item} />
      </td>
    </tr>
  )
}

export function ResultsList({ items, leagues, showSource }: ResultsListProps) {
  return (
    <table className="block w-full md:table md:overflow-hidden md:rounded-xl md:border md:border-slate-200 md:bg-white md:shadow-sm md:dark:border-slate-800 md:dark:bg-slate-900">
      <caption className="sr-only">Our predictions and the actual results</caption>
      <thead className="sr-only md:not-sr-only md:table-header-group">
        <tr className="muted text-left text-xs uppercase tracking-wide">
          <th scope="col" className="px-3 py-2 font-medium">Date</th>
          <th scope="col" className="px-3 py-2 font-medium">Match</th>
          <th scope="col" className="px-3 py-2 font-medium">Our prediction</th>
          <th scope="col" className="px-3 py-2 font-medium">Actual result</th>
          <th scope="col" className="px-3 py-2 font-medium">Verdict</th>
        </tr>
      </thead>
      <tbody className="block md:table-row-group">
        {items.map((item) => (
          <ResultRow key={item.id} item={item} leagues={leagues} showSource={showSource} />
        ))}
      </tbody>
    </table>
  )
}
