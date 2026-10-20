import { describe, expect, it } from 'vitest'
import type { League } from '../api/types'
import { makeFixture } from '../test/factories'
import { isFinished } from './fixture'
import { filterFixtures, groupFixturesByLeague, groupLeaguesByCountry, matchesTeamSearch } from './leagues'

const LEAGUES: League[] = [
  { code: 'SP1', name: 'La Liga', country: 'Spain' },
  { code: 'E1', name: 'Championship', country: 'England' },
  { code: 'E0', name: 'Premier League', country: 'England' },
  { code: 'X', name: 'Mystery', country: '' },
]

describe('league utils', () => {
  it('groups leagues by country alphabetically', () => {
    const groups = groupLeaguesByCountry(LEAGUES)
    expect(groups.map((g) => g.country)).toEqual(['England', 'Other', 'Spain'])
    expect(groups[0]?.leagues.map((l) => l.code)).toEqual(['E1', 'E0'])
  })

  it('groups fixtures by league and sorts by kickoff', () => {
    const fixtures = [
      makeFixture({ id: 1, league: 'SP1', kickoff: '2026-10-20T20:00:00' }),
      makeFixture({ id: 2, league: 'E0', kickoff: '2026-10-20T17:30:00' }),
      makeFixture({ id: 3, league: 'E0', kickoff: '2026-10-20T12:30:00' }),
      makeFixture({ id: 4, league: 'ZZ' }),
    ]
    const groups = groupFixturesByLeague(fixtures, LEAGUES)
    expect(groups.map((g) => g.code)).toEqual(['ZZ', 'E0', 'SP1'])
    expect(groups[1]?.fixtures.map((f) => f.id)).toEqual([3, 2])
    expect(groups[0]?.name).toBe('ZZ')
    expect(groupFixturesByLeague(fixtures, undefined)).toHaveLength(3)
  })

  it('filters by league and team search', () => {
    const fixtures = [
      makeFixture({ id: 1, league: 'E0', home: 'Arsenal', away: 'Chelsea' }),
      makeFixture({ id: 2, league: 'SP1', home: 'Getafe', away: 'Barcelona' }),
    ]
    expect(filterFixtures(fixtures, { leagues: [], search: '' })).toHaveLength(2)
    expect(filterFixtures(fixtures, { leagues: ['SP1'], search: '' }).map((f) => f.id)).toEqual([2])
    expect(filterFixtures(fixtures, { leagues: [], search: 'chel' }).map((f) => f.id)).toEqual([1])
    expect(matchesTeamSearch(fixtures[1]!, '  BARCA ')).toBe(false)
  })

  it('detects finished fixtures only when a score exists', () => {
    expect(isFinished(makeFixture({ status: 'finished', score: { home: 1, away: 0 } }))).toBe(true)
    expect(isFinished(makeFixture({ status: 'finished', score: null }))).toBe(false)
    expect(isFinished(makeFixture())).toBe(false)
  })
})
