import { useState } from 'react'
import { usePrediction } from '../api/hooks'
import type { Team } from '../api/types'
import { PredictionPanel } from '../components/PredictionPanel'
import { ErrorState } from '../components/StatusStates'
import { TeamCombobox } from '../components/TeamCombobox'

export function PredictPage() {
  const [home, setHome] = useState<Team | null>(null)
  const [away, setAway] = useState<Team | null>(null)

  const sameTeam = Boolean(home && away && home.name === away.name)
  const params = home && away && !sameTeam ? { home: home.name, away: away.name, league: home.league } : null
  const prediction = usePrediction(params)

  function swap() {
    setHome(away)
    setAway(home)
  }

  return (
    <div className="flex flex-col gap-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Custom match-up</h1>
        <p className="muted text-sm">
          Pick any two teams. The prediction assumes the match is played at the first team&apos;s home ground.
        </p>
      </div>

      <div className="card grid items-end gap-3 p-4 sm:grid-cols-[1fr_auto_1fr]">
        <TeamCombobox label="Home team" value={home?.name ?? null} onSelect={setHome} />
        <button
          type="button"
          className="btn"
          onClick={swap}
          disabled={!home && !away}
          aria-label="Swap home and away teams"
        >
          ⇄ Swap
        </button>
        <TeamCombobox label="Away team" value={away?.name ?? null} onSelect={setAway} />
      </div>

      {sameTeam && (
        <p role="alert" className="text-sm font-medium text-red-700 dark:text-red-300">
          Choose two different teams.
        </p>
      )}

      {!params && !sameTeam && (
        <p className="muted text-sm">Choose a home team and an away team to see the prediction.</p>
      )}

      {params && (
        <section className="card p-4" aria-labelledby="matchup-heading" aria-busy={prediction.isPending}>
          <h2 id="matchup-heading" className="mb-4 text-lg font-semibold">
            {params.home} vs {params.away}
          </h2>
          {prediction.isPending && (
            <p role="status" className="muted text-sm">
              Calculating…
            </p>
          )}
          {prediction.isError && (
            <ErrorState
              error={prediction.error}
              title="Could not get a prediction"
              onRetry={() => void prediction.refetch()}
            />
          )}
          {prediction.isSuccess && (
            <PredictionPanel prediction={prediction.data} home={params.home} away={params.away} />
          )}
        </section>
      )}
    </div>
  )
}
