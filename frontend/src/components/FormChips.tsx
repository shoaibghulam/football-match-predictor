import type { FormEntry, ResultLetter } from '../utils/form'

const CHIP_CLASS: Record<ResultLetter, string> = {
  W: 'bg-emerald-700 text-white dark:bg-emerald-400 dark:text-slate-950',
  D: 'bg-slate-500 text-white dark:bg-slate-400 dark:text-slate-950',
  L: 'bg-red-700 text-white dark:bg-red-400 dark:text-slate-950',
}

const WORD: Record<ResultLetter, string> = { W: 'Win', D: 'Draw', L: 'Loss' }

interface FormChipsProps {
  entries: readonly FormEntry[]
}

/** Recent results, most recent first, as W/D/L chips with an accessible description. */
export function FormChips({ entries }: FormChipsProps) {
  if (entries.length === 0) return <p className="muted text-sm">No recent results available.</p>
  return (
    <ol className="flex flex-col gap-1.5" aria-label="Recent results, most recent first">
      {entries.map((e) => {
        const venue = e.venue === 'H' ? 'home' : e.venue === 'A' ? 'away' : ''
        return (
          <li key={e.key} className="flex items-center gap-3 text-sm">
            <span
              className={`inline-flex size-7 shrink-0 items-center justify-center rounded-md text-xs font-bold ${
                e.outcome ? CHIP_CLASS[e.outcome] : 'bg-slate-200 dark:bg-slate-800'
              }`}
              aria-label={e.outcome ? WORD[e.outcome] : 'Unknown result'}
            >
              {e.outcome ?? '?'}
            </span>
            <span className="tabular-nums font-medium">{e.scoreText ?? '–'}</span>
            <span className="min-w-0 flex-1 truncate">
              {venue === 'away' ? 'at ' : 'vs '}
              {e.opponent}
              {venue && <span className="sr-only"> ({venue})</span>}
            </span>
            {e.date && <span className="muted text-xs tabular-nums">{e.date.slice(0, 10)}</span>}
          </li>
        )
      })}
    </ol>
  )
}
