import type { ScoreProb } from '../api/types'
import { formatPercent } from '../utils/probability'

interface TopScoresProps {
  scores: readonly ScoreProb[]
  limit?: number
}

export function TopScores({ scores, limit = 3 }: TopScoresProps) {
  const shown = [...scores].sort((a, b) => b.p - a.p).slice(0, limit)
  if (shown.length === 0) return null
  return (
    <div>
      <h4 className="muted mb-1 text-xs font-medium">Most likely scores</h4>
      <ol className="flex flex-wrap gap-1.5" aria-label="Most likely scores">
        {shown.map((s) => (
          <li
            key={s.score}
            className="rounded-md border border-slate-200 px-2 py-0.5 text-xs tabular-nums dark:border-slate-700"
          >
            <span className="font-semibold">{s.score}</span>{' '}
            <span className="muted">{formatPercent(s.p)}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}
