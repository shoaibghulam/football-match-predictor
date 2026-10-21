import type { Health } from '../api/types'
import { SETUP_MINUTES, setupPercent, setupStageLabel } from '../utils/setup'

interface SetupNoticeProps {
  health: Health | undefined
}

/** First-start banner: the backend is still downloading history / fixtures. */
export function SetupNotice({ health }: SetupNoticeProps) {
  const percent = setupPercent(health)
  const detail = health?.setup?.detail
  return (
    <div
      role="status"
      aria-live="polite"
      data-testid="setup-notice"
      className="flex flex-col gap-2 rounded-lg border border-sky-300 bg-sky-50 px-3 py-3 text-sm text-sky-950 dark:border-sky-800 dark:bg-sky-950 dark:text-sky-100"
    >
      <p className="font-semibold">
        Setting up — downloading match history and today's fixtures (first start only, about {SETUP_MINUTES}{' '}
        minutes). This page refreshes automatically.
      </p>
      <p className="text-xs">
        {setupStageLabel(health)}
        {detail ? `: ${detail}` : ''}
      </p>
      {percent !== null && (
        <div
          role="progressbar"
          aria-label="Setup progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          className="h-1.5 w-full overflow-hidden rounded-full bg-sky-200 dark:bg-sky-900"
        >
          <div className="h-full rounded-full bg-sky-600 dark:bg-sky-400" style={{ width: `${percent}%` }} />
        </div>
      )}
    </div>
  )
}
