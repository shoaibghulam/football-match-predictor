import { ApiError } from '../../api/client'
import { useTeamForm } from '../../api/hooks'
import { normalizeForm, normalizeRatings, summarizeForm } from '../../utils/form'
import { FormChips } from '../FormChips'
import { ErrorState } from '../StatusStates'

interface TeamFormPanelProps {
  team: string
}

export function TeamFormPanel({ team }: TeamFormPanelProps) {
  const query = useTeamForm(team)

  if (query.isPending) {
    return (
      <p role="status" className="muted text-sm">
        Loading recent form…
      </p>
    )
  }
  if (query.isError) {
    const notFound = query.error instanceof ApiError && query.error.status === 404
    return (
      <ErrorState
        error={query.error}
        title={notFound ? `We don't know a team called “${team}”` : 'Could not load recent form'}
        onRetry={notFound ? undefined : () => void query.refetch()}
      />
    )
  }

  const entries = normalizeForm(query.data, team)
  const ratings = normalizeRatings(query.data)
  const summary = summarizeForm(entries)

  return (
    <div className="grid gap-4 md:grid-cols-[1.5fr_1fr]">
      <section className="card p-4" aria-labelledby="form-heading">
        <h2 id="form-heading" className="mb-1 font-semibold">
          Recent form
        </h2>
        {entries.length > 0 && (
          <p className="muted mb-3 text-sm">
            Last {entries.length}: {summary.W} won, {summary.D} drawn, {summary.L} lost
          </p>
        )}
        <FormChips entries={entries} />
      </section>
      <section className="card p-4" aria-labelledby="ratings-heading">
        <h2 id="ratings-heading" className="mb-3 font-semibold">
          Current ratings
        </h2>
        {ratings.length === 0 ? (
          <p className="muted text-sm">No ratings available.</p>
        ) : (
          <dl className="grid grid-cols-2 gap-2">
            {ratings.map((r) => (
              <div key={r.key} className="rounded-lg bg-slate-100 p-2 dark:bg-slate-800">
                <dt className="muted text-xs">{r.label}</dt>
                <dd className="text-lg font-semibold tabular-nums">{r.value}</dd>
              </div>
            ))}
          </dl>
        )}
        <p className="muted mt-3 text-xs">
          Elo and pi-ratings are updated after every result. Higher means stronger.
        </p>
      </section>
    </div>
  )
}
