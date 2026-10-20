import { describe, expect, it } from 'vitest'
import { hasLive, LIVE_REFETCH_MS, liveRefetch } from '../api/hooks'
import type { League } from '../api/types'
import { featuredForDate, fixturesForRange, uefaFixturesForDate } from '../mocks/data'
import { makeFixture } from '../test/factories'
import { isLive, liveMinuteLabel, shownScore } from './fixture'
import { groupFixturesByLeague, groupLeaguesByCountry, isUefa } from './leagues'

const LEAGUES: League[] = [
  { code: 'E0', name: 'Premier League', country: 'England' },
  { code: 'UECL', name: 'UEFA Conference League', country: 'Europe' },
  { code: 'UCL', name: 'UEFA Champions League', country: 'Europe' },
  { code: 'SP1', name: 'La Liga', country: 'Spain' },
]

describe('UEFA helpers', () => {
  it('puts Europe and the UEFA competitions first', () => {
    const groups = groupLeaguesByCountry(LEAGUES)
    expect(groups.map((g) => g.country)).toEqual(['Europe', 'England', 'Spain'])
    expect(groups[0]?.leagues.map((l) => l.code)).toEqual(['UCL', 'UECL'])
    const fixtures = [
      makeFixture({ id: 1, league: 'SP1' }),
      makeFixture({ id: 2, league: 'UECL' }),
      makeFixture({ id: 3, league: 'E0' }),
      makeFixture({ id: 4, league: 'UCL' }),
    ]
    expect(groupFixturesByLeague(fixtures, LEAGUES).map((g) => g.code)).toEqual(['UCL', 'UECL', 'E0', 'SP1'])
    expect(isUefa('UEL')).toBe(true)
    expect(isUefa('E0')).toBe(false)
    expect(isUefa(undefined)).toBe(false)
  })

  it('formats live minute and score', () => {
    const live = makeFixture({ status: 'live', score: { home: 1, away: 0 }, live: { minute: 30, score: { home: 2, away: 0 } } })
    expect(isLive(live)).toBe(true)
    expect(shownScore(live)).toEqual({ home: 2, away: 0 })
    expect(liveMinuteLabel(live)).toBe("30'")
    expect(liveMinuteLabel({ ...live, live: { minute: 30, minute_estimated: true, score: { home: 2, away: 0 } } })).toBe("~30'")
    expect(liveMinuteLabel({ ...live, live: null })).toBe('LIVE')
    expect(shownScore({ ...live, live: null })).toEqual({ home: 1, away: 0 })
    expect(shownScore({ ...live, live: null, score: null })).toEqual({ home: 0, away: 0 })
    const finished = makeFixture({ status: 'finished', score: { home: 3, away: 1 } })
    expect(shownScore(finished)).toEqual({ home: 3, away: 1 })
    expect(shownScore(makeFixture())).toBeNull()
  })

  it('polls every minute only while something is live', () => {
    const live = makeFixture({ status: 'live' })
    expect(hasLive(undefined)).toBe(false)
    expect(hasLive([makeFixture(), live])).toBe(true)
    expect(liveRefetch({ state: { data: [live] } })).toBe(LIVE_REFETCH_MS)
    expect(liveRefetch({ state: { data: [makeFixture()] } })).toBe(false)
  })

  it('mock data includes a live Barcelona v Paris SG UCL match and a featured list', () => {
    const today = '2026-10-20'
    const uefa = uefaFixturesForDate(today, today)
    expect(uefa[0]).toMatchObject({ league: 'UCL', home: 'Barcelona', away: 'Paris SG', status: 'live' })
    expect(uefa[0]?.prediction?.model).toBe('europe')
    expect(uefaFixturesForDate('2026-10-21', today)).toEqual([])
    expect(fixturesForRange(today, today, today).filter((f) => isUefa(f.league))).toHaveLength(3)
    const featured = featuredForDate(today, today)
    expect(featured[0]?.status).toBe('live')
    expect(featured.every((f) => f.featured)).toBe(true)
    expect(featured.length).toBeLessThanOrEqual(12)
  })
})
