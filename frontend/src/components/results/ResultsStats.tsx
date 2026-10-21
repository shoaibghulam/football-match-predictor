import type { ResultsSummary } from '../../api/results'
import { formatCount, formatRps, pct1 } from './labels'

interface StatProps {
  label: string
  value: string
  hint?: string
}

function Stat({ label, value, hint }: StatProps) {
  return (
    <div className="flex flex-col gap-0.5 rounded-lg bg-slate-50 p-3 dark:bg-slate-800/60">
      <dt className="muted text-xs font-medium uppercase tracking-wide">{label}</dt>
      <dd className="text-2xl font-bold tabular-nums sm:text-3xl">{value}</dd>
      {hint && <dd className="muted text-xs">{hint}</dd>}
    </div>
  )
}

export function ResultsStats({ summary }: { summary: ResultsSummary }) {
  const book = summary.bookmaker
  const vsBook = book
    ? `${pct1(book.model_accuracy)} vs ${pct1(book.accuracy)}`
    : null
  return (
    <section aria-labelledby="results-stats" className="card flex flex-col gap-3 p-4">
      <h2 id="results-stats" className="sr-only">
        Accuracy summary
      </h2>
      <dl className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="Matches predicted" value={formatCount(summary.n)} />
        <Stat label="Correct" value={formatCount(summary.correct)} hint={`${formatCount(summary.n - summary.correct)} wrong`} />
        <Stat label="Accuracy" value={pct1(summary.accuracy)} hint="our most likely outcome happened" />
        <Stat label="RPS" value={formatRps(summary.rps)} hint="lower is better (0 = perfect)" />
        {vsBook && book ? (
          <Stat
            label="Us vs bookmaker"
            value={vsBook}
            hint={`RPS ${formatRps(book.model_rps)} vs ${formatRps(book.rps)} on ${formatCount(book.n)} matches with odds`}
          />
        ) : (
          <Stat label="Us vs bookmaker" value="–" hint="no pre-match odds for these matches" />
        )}
      </dl>
      <p className="muted text-sm">
        Around 50% is strong for three-way football predictions: about a quarter of matches end in a draw, and draws
        are very hard to call. The bookmaker&apos;s favourite wins about 50.5% of the time in our data.
      </p>
    </section>
  )
}
