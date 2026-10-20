import type { Fixture } from '../api/types'
import { formatDayLabel, parseKickoff } from '../utils/date'
import { isFinished, isLive, liveMinuteLabel, shownScore } from '../utils/fixture'
import { isUefa } from '../utils/leagues'
import { LiveBadge } from './LiveStrip'
import { favouriteOutcome, formatPercent, likelihoodLabel } from '../utils/probability'
import { ModelBadge } from './ModelBadge'
import { PredictionStats } from './PredictionStats'
import { ProbabilityBar } from './ProbabilityBar'
import { TopScores } from './TopScores'

interface MatchCardProps {
  fixture: Fixture
  /** Show the kickoff date as well as the time (multi-day views). */
  showDate?: boolean
  /** Competition name shown above the teams (featured lists mix competitions). */
  leagueName?: string
  onOpen?: (fixture: Fixture) => void
}


function KickoffLabel({ fixture, showDate }: { fixture: Fixture; showDate: boolean }) {
  const { date, time } = parseKickoff(fixture.kickoff)
  const day = showDate && date ? `${formatDayLabel(date)} · ` : ''
  if (isLive(fixture)) {
    return (
      <span className="inline-flex items-center gap-2">
        <LiveBadge />
        <span className="font-semibold tabular-nums text-red-700 dark:text-red-400">{liveMinuteLabel(fixture)}</span>
      </span>
    )
  }
  if (isFinished(fixture)) return <span className="font-semibold">{day}Full time</span>
  return (
    <span className="tabular-nums">
      {day}
      {time ?? 'Time TBC'}
    </span>
  )
}

function FavouriteNote({ fixture }: { fixture: Fixture }) {
  const probs = fixture.prediction?.probs
  if (!probs) return null
  const fav = favouriteOutcome(probs)
  const who = fav === 'home' ? fixture.home : fav === 'away' ? fixture.away : 'A draw'
  return (
    <p className="muted text-xs">
      {likelihoodLabel(probs[fav])}: <span className="font-medium text-slate-800 dark:text-slate-200">{who}</span>{' '}
      ({formatPercent(probs[fav])}) — not a certainty.
    </p>
  )
}

export function MatchCard({ fixture, showDate = false, leagueName, onOpen }: MatchCardProps) {
  const { prediction } = fixture
  const finished = isFinished(fixture)
  const live = isLive(fixture)
  const score = shownScore(fixture)
  const title = `${fixture.home} vs ${fixture.away}`
  const uefa = fixture.competition_type === 'europe' || isUefa(fixture.league)
  const accent = uefa ? 'border-l-4 border-l-indigo-600 dark:border-l-indigo-400' : ''

  return (
    <article
      className={`card relative flex flex-col gap-3 p-4 transition-shadow hover:shadow-md ${accent}`}
      aria-label={title}
    >
      {leagueName && (
        <p className={`-mb-2 text-xs font-semibold ${uefa ? 'text-indigo-700 dark:text-indigo-300' : 'muted'}`}>
          {leagueName}
        </p>
      )}
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="muted">
          <KickoffLabel fixture={fixture} showDate={showDate} />
        </span>
        {prediction && <ModelBadge model={prediction.model} />}
      </div>

      <h3 className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 text-base font-semibold">
        <span className="truncate">{fixture.home}</span>
        <span className="tabular-nums">{score ? score.home : ''}</span>
        <span className="truncate">{fixture.away}</span>
        <span className="tabular-nums">{score ? score.away : ''}</span>
      </h3>
      {score && (
        <p className="sr-only">
          {live ? 'Current score' : 'Final score'} {fixture.home} {score.home}, {fixture.away} {score.away}
        </p>
      )}

      {prediction ? (
        <>
          {(finished || live) && <PredictionLabel frozenAt={fixture.prediction_frozen_at} />}
          <ProbabilityBar probs={prediction.probs} homeLabel={fixture.home} awayLabel={fixture.away} />
          <FavouriteNote fixture={fixture} />
          <PredictionStats prediction={prediction} />
          <TopScores scores={prediction.top_scores} limit={3} />
          {prediction.confidence === 'low' && (
            <p className="text-xs text-amber-800 dark:text-amber-300">
              Lower confidence: at least one club has few rated games.
            </p>
          )}
        </>
      ) : (
        <p className="muted text-sm">No prediction available for this match yet.</p>
      )}

      {onOpen && (
        <button
          type="button"
          onClick={() => onOpen(fixture)}
          aria-label={`Details: ${title}`}
          className="self-start text-sm font-medium text-blue-700 underline-offset-2 after:absolute after:inset-0 after:rounded-xl hover:underline dark:text-blue-400"
        >
          Details
        </button>
      )}
    </article>
  )
}

function PredictionLabel({ frozenAt }: { frozenAt?: string | null }) {
  if (!frozenAt) {
    return (
      <p className="muted -mb-1 text-xs" title="No snapshot was stored before kickoff; priced with current ratings.">
        Model view (current ratings)
      </p>
    )
  }
  const when = new Date(frozenAt)
  const title = Number.isNaN(when.getTime()) ? undefined : `Frozen before kickoff (last updated ${when.toLocaleString()})`
  return (
    <p className="muted -mb-1 text-xs" title={title}>
      Pre-match prediction
    </p>
  )
}
