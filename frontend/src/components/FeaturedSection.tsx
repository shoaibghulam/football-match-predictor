import { useState } from 'react'
import type { Fixture, League } from '../api/types'
import { leagueLookup } from '../utils/leagues'
import { LiveStrip } from './LiveStrip'
import { MatchCard } from './MatchCard'
import { MatchDetailDialog } from './MatchDetailDialog'

interface FeaturedSectionProps {
  fixtures: readonly Fixture[]
  leagues: readonly League[] | undefined
}

/** "Live now" strip plus the featured (UEFA and big domestic) matches of the day. */
export function FeaturedSection({ fixtures, leagues }: FeaturedSectionProps) {
  const [selected, setSelected] = useState<Fixture | null>(null)
  const lookup = leagueLookup(leagues)
  const live = fixtures.filter((f) => f.status === 'live')
  const rest = fixtures.filter((f) => f.status !== 'live')
  if (fixtures.length === 0) return null

  return (
    <div className="flex flex-col gap-5">
      <LiveStrip fixtures={live} leagues={leagues} onOpen={setSelected} />
      {rest.length > 0 && (
        <section aria-labelledby="featured-matches">
          <h2 id="featured-matches" className="mb-2 flex items-baseline gap-2 text-sm font-semibold">
            <span>Featured</span>
            <span className="muted font-normal">Champions League, Europa League and the biggest games</span>
          </h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {rest.map((f) => (
              <MatchCard
                key={f.id}
                fixture={f}
                leagueName={lookup.get(f.league)?.name}
                onOpen={setSelected}
              />
            ))}
          </div>
        </section>
      )}
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
