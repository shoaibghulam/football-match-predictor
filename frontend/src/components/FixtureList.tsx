import { useMemo, useState } from 'react'
import type { Fixture, League } from '../api/types'
import { groupFixturesByLeague, isUefa, leagueLookup } from '../utils/leagues'
import { MatchCard } from './MatchCard'
import { MatchDetailDialog } from './MatchDetailDialog'

interface FixtureListProps {
  fixtures: readonly Fixture[]
  leagues: readonly League[] | undefined
  showDate?: boolean
}

/** Fixtures grouped under league headings, with a detail dialog for the selected match. */
export function FixtureList({ fixtures, leagues, showDate = false }: FixtureListProps) {
  const [selected, setSelected] = useState<Fixture | null>(null)
  const groups = useMemo(() => groupFixturesByLeague(fixtures, leagues), [fixtures, leagues])
  const lookup = useMemo(() => leagueLookup(leagues), [leagues])

  return (
    <div className="flex flex-col gap-6">
      {groups.map((g) => (
        <section key={g.code} aria-labelledby={`league-${g.code}`}>
          <h2
            id={`league-${g.code}`}
            className={`mb-2 flex items-baseline gap-2 text-sm font-semibold ${
              isUefa(g.code) ? 'text-indigo-800 dark:text-indigo-300' : ''
            }`}
          >
            <span>{g.name}</span>
            {g.country && <span className="muted font-normal">{g.country}</span>}
            <span className="muted ml-auto font-normal">
              {g.fixtures.length} {g.fixtures.length === 1 ? 'match' : 'matches'}
            </span>
          </h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {g.fixtures.map((f) => (
              <MatchCard key={f.id} fixture={f} showDate={showDate} onOpen={setSelected} />
            ))}
          </div>
        </section>
      ))}
      {selected && (
        <MatchDetailDialog
          fixture={selected}
          leagueName={lookup.get(selected.league)?.name ?? selected.league}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  )
}
