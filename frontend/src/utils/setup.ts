import type { Health } from '../api/types'

/** Health is polled this often while the backend is still setting up. */
export const SETUP_POLL_MS = 10_000
/** Rough first-start duration shown to the user (history download + first ingest). */
export const SETUP_MINUTES = '5–10'

/**
 * True while the backend's first-start setup is running. Backends without the `setup`
 * field fall back to "no fixtures and never ingested".
 */
export function isSettingUp(health: Health | undefined): boolean {
  if (!health) return false
  if (health.setup) return health.setup.stage !== 'ready'
  return health.fixtures === 0 && !health.last_ingest
}

const STAGE_LABELS: Record<string, string> = {
  starting: 'Starting up',
  seeding_history: 'Downloading match history',
  seeding_uefa: 'Loading European competition history',
  ingesting: "Downloading today's fixtures",
}

export function setupStageLabel(health: Health | undefined): string {
  const stage = health?.setup?.stage
  return (stage && STAGE_LABELS[stage]) || 'Setting up'
}

/** 0..100 when the backend reports progress for the current stage, else null. */
export function setupPercent(health: Health | undefined): number | null {
  const p = health?.setup?.progress
  if (typeof p !== 'number' || !Number.isFinite(p)) return null
  return Math.round(Math.min(1, Math.max(0, p)) * 100)
}
