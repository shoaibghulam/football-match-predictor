import { describe, expect, it } from 'vitest'
import { setupRefetch } from '../api/hooks'
import type { Health } from '../api/types'
import { isSettingUp, SETUP_POLL_MS, setupPercent, setupStageLabel } from './setup'

const base: Health = { status: 'ok', fixtures: 10, last_ingest: '2026-10-20T07:00:00' }

describe('setup helpers', () => {
  it('detects the setup stage, with a fallback for old backends', () => {
    expect(isSettingUp(undefined)).toBe(false)
    expect(isSettingUp(base)).toBe(false)
    expect(isSettingUp({ ...base, setup: { stage: 'ingesting' } })).toBe(true)
    expect(isSettingUp({ ...base, setup: { stage: 'ready' } })).toBe(false)
    expect(isSettingUp({ status: 'degraded', fixtures: 0, last_ingest: null })).toBe(true)
    // a finished setup is trusted even when nothing was ingested (e.g. offline)
    expect(isSettingUp({ status: 'degraded', fixtures: 0, last_ingest: null, setup: { stage: 'ready' } })).toBe(false)
  })

  it('labels stages and clamps progress', () => {
    expect(setupStageLabel({ ...base, setup: { stage: 'seeding_history' } })).toBe('Downloading match history')
    expect(setupStageLabel({ ...base, setup: { stage: 'something-new' } })).toBe('Setting up')
    expect(setupStageLabel(undefined)).toBe('Setting up')
    expect(setupPercent({ ...base, setup: { stage: 'seeding_history', progress: 0.256 } })).toBe(26)
    expect(setupPercent({ ...base, setup: { stage: 'seeding_history', progress: 3 } })).toBe(100)
    expect(setupPercent({ ...base, setup: { stage: 'ingesting', progress: null } })).toBeNull()
    expect(setupPercent(undefined)).toBeNull()
  })

  it('polls health only while setting up', () => {
    expect(setupRefetch({ state: { data: { ...base, setup: { stage: 'seeding_uefa' } } } })).toBe(SETUP_POLL_MS)
    expect(setupRefetch({ state: { data: base } })).toBe(false)
    expect(setupRefetch({ state: {} })).toBe(false)
  })
})
