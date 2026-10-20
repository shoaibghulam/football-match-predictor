import type { Prediction } from '../api/types'
import { formatGoals, formatPercent } from '../utils/probability'

interface PredictionStatsProps {
  prediction: Prediction
}

export function PredictionStats({ prediction }: PredictionStatsProps) {
  const xg = prediction.expected_goals
  const items = [
    { label: 'Expected goals', value: `${formatGoals(xg?.home)} – ${formatGoals(xg?.away)}` },
    { label: 'Over 2.5 goals', value: formatPercent(prediction.over_2_5) },
    { label: 'Both teams score', value: formatPercent(prediction.btts) },
  ]
  return (
    <dl className="grid grid-cols-3 gap-2 text-center">
      {items.map((it) => (
        <div key={it.label} className="rounded-lg bg-slate-100 px-1 py-1.5 dark:bg-slate-800">
          <dt className="muted text-[11px] leading-tight">{it.label}</dt>
          <dd className="text-sm font-semibold tabular-nums">{it.value}</dd>
        </div>
      ))}
    </dl>
  )
}
