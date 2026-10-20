import { addDays, formatLongDate, isValidISODate, todayISO } from '../utils/date'

export interface DateRange {
  date: string
  /** Inclusive end date for multi-day views; undefined for a single day. */
  dateTo?: string
}

interface DateControlsProps {
  value: DateRange
  onChange: (next: DateRange) => void
  today?: string
}

export const WEEK_SPAN_DAYS = 6

export function DateControls({ value, onChange, today = todayISO() }: DateControlsProps) {
  const tomorrow = addDays(today, 1)
  const weekEnd = addDays(today, WEEK_SPAN_DAYS)
  const isRange = Boolean(value.dateTo)

  const chips = [
    { label: 'Today', active: !isRange && value.date === today, range: { date: today } },
    { label: 'Tomorrow', active: !isRange && value.date === tomorrow, range: { date: tomorrow } },
    {
      label: 'Next 7 days',
      active: value.date === today && value.dateTo === weekEnd,
      range: { date: today, dateTo: weekEnd },
    },
  ]

  const heading = isRange
    ? `${formatLongDate(value.date)} – ${formatLongDate(value.dateTo as string)}`
    : formatLongDate(value.date)

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <button
          type="button"
          className="btn px-2.5"
          onClick={() => onChange({ date: addDays(value.date, -1) })}
          aria-label="Previous day"
        >
          ‹
        </button>
        <label className="flex-1">
          <span className="sr-only">Match date</span>
          <input
            type="date"
            className="input"
            value={value.date}
            onChange={(e) => {
              if (isValidISODate(e.target.value)) onChange({ date: e.target.value })
            }}
          />
        </label>
        <button
          type="button"
          className="btn px-2.5"
          onClick={() => onChange({ date: addDays(value.dateTo ?? value.date, 1) })}
          aria-label="Next day"
        >
          ›
        </button>
      </div>
      <div className="flex flex-wrap gap-2" role="group" aria-label="Quick dates">
        {chips.map((c) => (
          <button
            key={c.label}
            type="button"
            className="chip"
            aria-pressed={c.active}
            onClick={() => onChange(c.range)}
          >
            {c.label}
          </button>
        ))}
      </div>
      <p className="muted text-sm" aria-live="polite">
        Showing {heading}
      </p>
    </div>
  )
}
