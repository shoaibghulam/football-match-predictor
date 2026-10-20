import { useMemo } from 'react'
import type { League } from '../api/types'
import { groupLeaguesByCountry } from '../utils/leagues'

interface LeagueSelectProps {
  leagues: readonly League[]
  value: string
  onChange: (code: string) => void
  disabled?: boolean
}

/** Single-select league filter with leagues grouped by country ("" = all leagues). */
export function LeagueSelect({ leagues, value, onChange, disabled = false }: LeagueSelectProps) {
  const groups = useMemo(() => groupLeaguesByCountry(leagues), [leagues])
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="font-medium">League</span>
      <select className="input" value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
        <option value="">All leagues</option>
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
  )
}
