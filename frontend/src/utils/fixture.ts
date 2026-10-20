import type { Fixture } from '../api/types'

export function isFinished(fixture: Fixture): boolean {
  return fixture.status === 'finished' && Boolean(fixture.score)
}

export function isLive(fixture: Fixture): boolean {
  return fixture.status === 'live'
}

/** Current score while live, final score when finished. */
export function shownScore(fixture: Fixture) {
  if (isLive(fixture)) return fixture.live?.score ?? fixture.score ?? { home: 0, away: 0 }
  return isFinished(fixture) ? fixture.score : null
}

export function liveMinuteLabel(fixture: Fixture): string {
  const minute = fixture.live?.minute
  if (minute === null || minute === undefined) return 'LIVE'
  return fixture.live?.minute_estimated ? `~${minute}'` : `${minute}'`
}
