import { useMemo } from 'react'
import type { SourceFilter } from '../../api/results'
import type { League } from '../../api/types'
import { groupLeaguesByCountry } from '../../utils/leagues'
import { RANGE_PRESETS, type ResultsUrlState } from '../../utils/resultsFilters'

export type FilterPatch = Record<string, string | undefined>

interface ResultsFiltersProps {
  state: ResultsUrlState
  source: SourceFilter
  leagues: readonly League[]
  onChange: (patch: FilterPatch) => void
}

const SOURCES: readonly { value: SourceFilter; label: string; hint: string }[] = [
  { value: 'live', label: 'Live', hint: 'Predictions frozen before kick-off by this app' },
  { value: 'backtest', label: 'Backtest', hint: 'Walk-forward: each season predicted by a model trained only on earlier seasons' },
  { value: 'all', label: 'All', hint: 'Live and backtest together (each row is labelled)' },
]

interface SelectProps {
  label: string
  value: string
  onChange: (v: string) => void
  options: readonly { value: string; label: string }[]
}

function Select({ label, value, onChange, options }: SelectProps) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="font-medium">{label}</span>
      <select className="input" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  )
}

export function ResultsFilters({ state, source, leagues, onChange }: ResultsFiltersProps) {
  const groups = useMemo(() => groupLeaguesByCountry(leagues), [leagues])
  const allValue = (v: string) => (v === 'all' ? undefined : v)
  return (
    <div className="card flex flex-col gap-4 p-4">
      <div role="group" aria-label="Prediction record" className="flex flex-wrap gap-2">
        {SOURCES.map((s) => (
          <button
            key={s.value}
            type="button"
            className="chip"
            aria-pressed={source === s.value}
            title={s.hint}
            onClick={() => onChange({ source: s.value })}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div role="group" aria-label="Date range" className="flex flex-wrap gap-2">
        {RANGE_PRESETS.map((p) => (
          <button
            key={p.value}
            type="button"
            className="chip"
            aria-pressed={state.range === p.value}
            onClick={() => onChange({ range: p.value === 'all' ? undefined : p.value, from: undefined, to: undefined })}
          >
            {p.label}
          </button>
        ))}
      </div>
      {state.range === 'custom' && (
        <div className="grid grid-cols-2 gap-3 sm:max-w-md">
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium">From</span>
            <input type="date" className="input" value={state.from ?? ''} onChange={(e) => onChange({ from: e.target.value || undefined })} />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium">To</span>
            <input type="date" className="input" value={state.to ?? ''} onChange={(e) => onChange({ to: e.target.value || undefined })} />
          </label>
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <label className="flex flex-col gap-1 text-sm">
          <span className="font-medium">League</span>
          <select className="input" value={state.league} onChange={(e) => onChange({ league: e.target.value || undefined })}>
            <option value="">All leagues</option>
            <option value="UEFA">All UEFA competitions</option>
            {groups.map((g) => (
              <optgroup key={g.country} label={g.country}>
                {g.leagues.map((l) => (
                  <option key={l.code} value={l.code}>
                    {l.name}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm">
          <span className="font-medium">Team</span>
          <input
            type="search"
            className="input"
            placeholder="e.g. Arsenal"
            maxLength={64}
            value={state.team}
            onChange={(e) => onChange({ q: e.target.value || undefined })}
          />
        </label>
        <Select
          label="Outcome"
          value={state.outcome}
          onChange={(v) => onChange({ outcome: allValue(v) })}
          options={[
            { value: 'all', label: 'All' },
            { value: 'correct', label: 'Correct only' },
            { value: 'wrong', label: 'Wrong only' },
          ]}
        />
        <Select
          label="Confidence"
          value={state.confidence}
          onChange={(v) => onChange({ confidence: allValue(v) })}
          options={[
            { value: 'all', label: 'All' },
            { value: 'high', label: 'High (favourite ≥ 60%)' },
            { value: 'low', label: 'Low (favourite < 60%)' },
          ]}
        />
        {source !== 'backtest' && (
          <Select
            label="Status"
            value={state.status}
            onChange={(v) => onChange({ status: v === 'pending' ? 'pending' : undefined })}
            options={[
              { value: 'finished', label: 'Finished' },
              { value: 'pending', label: 'Pending (not played yet)' },
            ]}
          />
        )}
      </div>
    </div>
  )
}
