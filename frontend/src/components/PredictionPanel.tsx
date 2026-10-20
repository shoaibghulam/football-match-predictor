import { Link } from 'react-router'
import type { Prediction } from '../api/types'
import { favouriteOutcome, formatPercent, likelihoodLabel } from '../utils/probability'
import { ModelBadge } from './ModelBadge'
import { PredictionStats } from './PredictionStats'
import { ProbabilityBar } from './ProbabilityBar'
import { TopScores } from './TopScores'

interface PredictionPanelProps {
  prediction: Prediction
  home: string
  away: string
}

/** Full prediction view: bar, stats, all top scorelines and the model notice. */
export function PredictionPanel({ prediction, home, away }: PredictionPanelProps) {
  const fav = favouriteOutcome(prediction.probs)
  const favName = fav === 'home' ? `${home} win` : fav === 'away' ? `${away} win` : 'Draw'

  return (
    <div className="flex flex-col gap-4">
      <ProbabilityBar probs={prediction.probs} homeLabel={home} awayLabel={away} />
      <p className="text-sm">
        <span className="font-semibold">{likelihoodLabel(prediction.probs[fav])}:</span> {favName} at{' '}
        {formatPercent(prediction.probs[fav])}. Football is unpredictable, so even the most likely outcome often
        does not happen.
      </p>
      <PredictionStats prediction={prediction} />
      <TopScores scores={prediction.top_scores} limit={5} />
      <div className="muted flex flex-wrap items-center gap-2 text-xs">
        <ModelBadge model={prediction.model} />
        <span>Model {prediction.model_version}</span>
        <Link to="/about" className="font-medium text-blue-700 underline dark:text-blue-400">
          How accurate is this?
        </Link>
      </div>
    </div>
  )
}
