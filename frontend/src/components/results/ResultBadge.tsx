import type { ResultItem } from '../../api/results'

const BASE = 'inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold whitespace-nowrap'

/** Correct / Wrong / Awaiting result — icon + text, never colour alone. */
export function ResultBadge({ item }: { item: Pick<ResultItem, 'status' | 'correct'> }) {
  if (item.status === 'pending' || item.correct === null) {
    return (
      <span className={`${BASE} bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200`}>
        <span aria-hidden="true">⏳</span> Awaiting result
      </span>
    )
  }
  if (item.correct) {
    return (
      <span className={`${BASE} bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-200`}>
        <span aria-hidden="true">✓</span> Correct
      </span>
    )
  }
  return (
    <span className={`${BASE} bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-200`}>
      <span aria-hidden="true">✗</span> Wrong
    </span>
  )
}
