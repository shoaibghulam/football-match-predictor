import type { ResultsSummary } from '../../api/results'
import { formatCount, pct1 } from './labels'

interface Point {
  key: string
  label: string
  n: number
  accuracy: number | null
}

const MONTHLY_FROM_POINTS = 3

function points(summary: ResultsSummary): { title: string; data: Point[] } {
  if (summary.monthly.length >= MONTHLY_FROM_POINTS) {
    return {
      title: 'Accuracy by month',
      data: summary.monthly.map((m) => ({ key: m.month, label: m.month, n: m.n, accuracy: m.accuracy })),
    }
  }
  return {
    title: 'Accuracy by day',
    data: summary.daily.map((d) => ({ key: d.date, label: d.date.slice(5), n: d.n, accuracy: d.accuracy })),
  }
}

/** Plain CSS bar chart (no chart library); a dashed line marks 50%. */
export function AccuracyChart({ summary }: { summary: ResultsSummary }) {
  const { title, data } = points(summary)
  if (data.length === 0) return null
  return (
    <section aria-labelledby="accuracy-chart" className="card flex flex-col gap-3 p-4">
      <h2 id="accuracy-chart" className="font-semibold">
        {title}
      </h2>
      <div className="relative h-40">
        <span
          className="pointer-events-none absolute inset-x-0 top-1/2 border-t border-dashed border-slate-400 dark:border-slate-600"
          aria-hidden="true"
        />
        <span className="muted absolute right-0 top-1/2 -translate-y-full text-[10px]" aria-hidden="true">
          50%
        </span>
        <ul className="flex h-full items-end gap-px" aria-label={title}>
          {data.map((p) => {
            const height = p.accuracy === null ? 0 : Math.max(2, Math.round(p.accuracy * 100))
            const text = `${p.label}: ${pct1(p.accuracy)} of ${formatCount(p.n)}`
            return (
              <li key={p.key} className="flex h-full min-w-0 flex-1 items-end" title={text}>
                <span className="sr-only">{text}</span>
                <span
                  aria-hidden="true"
                  className="block w-full rounded-t bg-blue-600 opacity-80 hover:opacity-100 dark:bg-blue-400"
                  style={{ height: `${height}%` }}
                />
              </li>
            )
          })}
        </ul>
      </div>
      <p className="muted flex justify-between text-xs" aria-hidden="true">
        <span>{data[0]?.label}</span>
        <span>{data[data.length - 1]?.label}</span>
      </p>
    </section>
  )
}
