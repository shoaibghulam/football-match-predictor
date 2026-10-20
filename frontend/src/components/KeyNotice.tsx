import { useState } from 'react'

export const KEY_NOTICE_STORAGE = 'mp.keyNoticeDismissed'

function readDismissed(): boolean {
  try {
    return window.localStorage.getItem(KEY_NOTICE_STORAGE) === '1'
  } catch {
    return false
  }
}

/** Small dismissible hint shown when the backend has no football-data.org key. */
export function KeyNotice() {
  const [dismissed, setDismissed] = useState(readDismissed)
  if (dismissed) return null

  function dismiss() {
    setDismissed(true)
    try {
      window.localStorage.setItem(KEY_NOTICE_STORAGE, '1')
    } catch {
      // storage may be unavailable (private mode); dismissing still works for this visit
    }
  }

  return (
    <div
      role="note"
      className="flex items-start gap-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-950 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-100"
    >
      <p className="flex-1">
        Champions League schedule needs a free football-data.org key — see README
        (docs/07-european-competitions.md). Europa and Conference League use a limited fallback.
      </p>
      <button type="button" className="font-medium underline" onClick={dismiss} aria-label="Dismiss notice">
        Dismiss
      </button>
    </div>
  )
}
