import type { Outcome, Probs } from '../api/types'
import { favouriteOutcome, OUTCOMES, toPercents } from '../utils/probability'

interface ProbabilityBarProps {
  probs: Probs
  homeLabel: string
  awayLabel: string
  /** Hide the legend (labels + percentages) below the bar. */
  compact?: boolean
}

const SEGMENT_CLASS: Record<Outcome, string> = {
  home: 'bg-home',
  draw: 'bg-draw',
  away: 'bg-away',
}

export function ProbabilityBar({ probs, homeLabel, awayLabel, compact = false }: ProbabilityBarProps) {
  const pct = toPercents(probs)
  const fav = favouriteOutcome(probs)
  const names: Record<Outcome, string> = { home: homeLabel, draw: 'Draw', away: awayLabel }
  const summary = `Win probabilities: ${homeLabel} ${pct.home}%, draw ${pct.draw}%, ${awayLabel} ${pct.away}%`

  return (
    <div>
      <div
        role="img"
        aria-label={summary}
        className="flex h-2.5 w-full gap-0.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800"
      >
        {OUTCOMES.map((o) =>
          pct[o] > 0 ? (
            <span
              key={o}
              data-testid={`bar-${o}`}
              className={`${SEGMENT_CLASS[o]} h-full ${o === fav ? '' : 'opacity-70'}`}
              style={{ width: `${pct[o]}%` }}
            />
          ) : null,
        )}
      </div>
      {!compact && (
        <ul className="mt-2 grid grid-cols-3 gap-2 text-xs" aria-hidden="true">
          {OUTCOMES.map((o) => (
            <li
              key={o}
              className={`flex flex-col ${o === 'draw' ? 'items-center text-center' : ''} ${o === 'away' ? 'items-end text-right' : ''}`}
            >
              <span className="flex items-center gap-1">
                <span className={`inline-block size-2 rounded-full ${SEGMENT_CLASS[o]}`} />
                <span className="muted max-w-full truncate">{o === 'draw' ? 'Draw' : names[o]}</span>
              </span>
              <span className={`text-base tabular-nums ${o === fav ? 'font-bold' : 'font-medium'}`}>
                {pct[o]}%
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
