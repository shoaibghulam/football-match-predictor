import { useState } from 'react'
import { useFixtures } from '../../api/hooks'
import type { Fixture } from '../../api/types'
import { addDays, todayISO } from '../../utils/date'
import { formatPercent, teamWinChance } from '../../utils/probability'
import { MatchCard } from '../MatchCard'
import { MatchDetailDialog } from '../MatchDetailDialog'
import { EmptyState, ErrorState, LoadingList } from '../StatusStates'

export const TEAM_HORIZON_DAYS = 14

interface UpcomingFixturesProps {
  team: string
}

export function UpcomingFixtures({ team }: UpcomingFixturesProps) {
  const today = todayISO()
  const query = useFixtures({ date: today, dateTo: addDays(today, TEAM_HORIZON_DAYS), team })
  const [selected, setSelected] = useState<Fixture | null>(null)

  if (query.isPending) return <LoadingList count={2} label="Loading fixtures" />
  if (query.isError) {
    return <ErrorState error={query.error} onRetry={() => void query.refetch()} title="Could not load fixtures" />
  }
  if (query.data.length === 0) {
    return (
      <EmptyState title={`No fixtures for ${team} in the next ${TEAM_HORIZON_DAYS} days`}>
        Fixtures are usually published a few days before each round.
      </EmptyState>
    )
  }

  return (
    <ul className="grid gap-3 sm:grid-cols-2">
      {query.data.map((f) => {
        const chance = teamWinChance(f, team)
        return (
          <li key={f.id} className="flex flex-col gap-1.5">
            {chance !== null && (
              <p className="text-sm">
                {team} win chance: <span className="font-bold tabular-nums">{formatPercent(chance)}</span>
              </p>
            )}
            <MatchCard fixture={f} showDate onOpen={setSelected} />
          </li>
        )
      })}
      {selected && <MatchDetailDialog fixture={selected} onClose={() => setSelected(null)} />}
    </ul>
  )
}
