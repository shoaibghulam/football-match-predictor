import type { Fixture, League } from '../api/types'
import { liveMinuteLabel, shownScore } from '../utils/fixture'
import { isUefa, leagueLookup } from '../utils/leagues'
import { formatPercent } from '../utils/probability'

export function LiveBadge({ label = 'LIVE' }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-red-600 px-2 py-0.5 text-xs font-bold tracking-wide text-white">
      <span aria-hidden="true" className="relative flex size-2">
        <span className="absolute inline-flex size-full animate-ping rounded-full bg-white opacity-75" />
        <span className="relative inline-flex size-2 rounded-full bg-white" />
      </span>
      {label}
    </span>
  )
}

interface LiveStripProps {
  fixtures: readonly Fixture[]
  leagues: readonly League[] | undefined
  onOpen?: (fixture: Fixture) => void
}

/** Horizontal strip of matches in progress: minute, score and the pre-match favourite. */
export function LiveStrip({ fixtures, leagues, onOpen }: LiveStripProps) {
  const lookup = leagueLookup(leagues)
  if (fixtures.length === 0) return null
  return (
    <section aria-labelledby="live-now" className="flex flex-col gap-2">
      <h2 id="live-now" className="flex items-center gap-2 text-sm font-semibold">
        <LiveBadge /> <span>Live now</span>
        <span className="muted font-normal">{fixtures.length} {fixtures.length === 1 ? 'match' : 'matches'}</span>
      </h2>
      <ul className="-mx-1 flex snap-x gap-3 overflow-x-auto px-1 pb-1" aria-label="Live matches">
        {fixtures.map((f) => {
          const score = shownScore(f)
          const probs = f.prediction?.probs
          const minute = liveMinuteLabel(f)
          return (
            <li key={f.id} className="snap-start">
              <button
                type="button"
                onClick={() => onOpen?.(f)}
                aria-label={`${f.home} ${score?.home ?? 0}, ${f.away} ${score?.away ?? 0}, ${minute === 'LIVE' ? 'in play' : `minute ${minute}`}`}
                className={`card flex w-64 flex-col gap-1.5 p-3 text-left hover:shadow-md ${
                  isUefa(f.league) ? 'border-l-4 border-l-indigo-600 dark:border-l-indigo-400' : ''
                }`}
              >
                <span className="flex items-center justify-between text-xs">
                  <span className="muted truncate">{lookup.get(f.league)?.name ?? f.league}</span>
                  <span className="font-semibold tabular-nums text-red-700 dark:text-red-400">{minute}</span>
                </span>
                <span className="grid grid-cols-[1fr_auto] gap-x-2 text-sm font-semibold">
                  <span className="truncate">{f.home}</span>
                  <span className="tabular-nums">{score?.home ?? 0}</span>
                  <span className="truncate">{f.away}</span>
                  <span className="tabular-nums">{score?.away ?? 0}</span>
                </span>
                {probs && (
                  <span className="muted text-xs tabular-nums">
                    Pre-match: {formatPercent(probs.home)} · {formatPercent(probs.draw)} · {formatPercent(probs.away)}
                  </span>
                )}
              </button>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
