interface ModelBadgeProps {
  model: string
}

const EUROPE_TITLE =
  'Cross-league European model: club strength on the ClubElo scale, trained on UEFA matches since 2011.'

export function ModelBadge({ model }: ModelBadgeProps) {
  if (model === 'europe') {
    return (
      <span
        title={EUROPE_TITLE}
        className="inline-flex items-center rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-900 dark:bg-indigo-950 dark:text-indigo-200"
      >
        European model
      </span>
    )
  }
  const usesMarket = model === 'market'
  const label = usesMarket ? 'uses market odds' : 'ratings only'
  const title = usesMarket
    ? 'This prediction combines our ratings with pre-match bookmaker odds.'
    : 'No odds were available, so this prediction uses team ratings only.'
  return (
    <span
      title={title}
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
        usesMarket
          ? 'bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200'
          : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200'
      }`}
    >
      {label}
    </span>
  )
}
