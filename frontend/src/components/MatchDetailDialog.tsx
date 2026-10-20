import { Link } from 'react-router'
import type { Fixture } from '../api/types'
import { formatLongDate, parseKickoff } from '../utils/date'
import { isFinished } from '../utils/fixture'
import { Modal } from './Modal'
import { PredictionPanel } from './PredictionPanel'

interface MatchDetailDialogProps {
  fixture: Fixture
  leagueName?: string
  onClose: () => void
}

export function MatchDetailDialog({ fixture, leagueName, onClose }: MatchDetailDialogProps) {
  const { date, time } = parseKickoff(fixture.kickoff)
  const finished = isFinished(fixture)
  const when = [date ? formatLongDate(date) : null, time].filter(Boolean).join(' · ')

  return (
    <Modal title={`${fixture.home} vs ${fixture.away}`} onClose={onClose}>
      <p className="muted mb-3 text-sm">
        {[leagueName, when].filter(Boolean).join(' — ')}
      </p>
      {finished && (
        <p className="mb-3 text-base font-semibold">
          Full time: {fixture.home} {fixture.score?.home} – {fixture.score?.away} {fixture.away}
        </p>
      )}
      {fixture.prediction ? (
        <PredictionPanel prediction={fixture.prediction} home={fixture.home} away={fixture.away} />
      ) : (
        <p className="muted text-sm">No prediction available for this match yet.</p>
      )}
      <div className="mt-4 flex flex-wrap gap-2 border-t border-slate-200 pt-4 text-sm dark:border-slate-800">
        <Link className="btn" to={`/team/${encodeURIComponent(fixture.home)}`}>
          {fixture.home}
        </Link>
        <Link className="btn" to={`/team/${encodeURIComponent(fixture.away)}`}>
          {fixture.away}
        </Link>
      </div>
    </Modal>
  )
}
