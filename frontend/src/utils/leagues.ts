import type { Fixture, League } from '../api/types'
import { kickoffSortValue } from './date'

export interface CountryGroup {
  country: string
  leagues: League[]
}

export interface FixtureGroup {
  code: string
  name: string
  country: string
  fixtures: Fixture[]
}

const byText = (a: string, b: string) => a.localeCompare(b)

/** UEFA club competitions, listed first everywhere. */
export const UEFA_CODES: readonly string[] = ['UCL', 'UEL', 'UECL']
export const EUROPE_COUNTRY = 'Europe'

export function isUefa(code: string | null | undefined): boolean {
  return UEFA_CODES.includes(code ?? '')
}

/** Europe first, then alphabetical. */
function byCountry(a: string, b: string): number {
  if (a === EUROPE_COUNTRY || b === EUROPE_COUNTRY) return a === b ? 0 : a === EUROPE_COUNTRY ? -1 : 1
  return byText(a, b)
}

function uefaRank(code: string): number {
  const i = UEFA_CODES.indexOf(code)
  return i === -1 ? UEFA_CODES.length : i
}

/** Leagues grouped by country (Europe / UEFA first), both sorted alphabetically. */
export function groupLeaguesByCountry(leagues: readonly League[]): CountryGroup[] {
  const map = new Map<string, League[]>()
  for (const l of leagues) {
    const country = l.country || 'Other'
    map.set(country, [...(map.get(country) ?? []), l])
  }
  return [...map.entries()]
    .sort(([a], [b]) => byCountry(a, b))
    .map(([country, ls]) => ({
      country,
      leagues: [...ls].sort((a, b) => uefaRank(a.code) - uefaRank(b.code) || byText(a.name, b.name)),
    }))
}

export function leagueLookup(leagues: readonly League[] | undefined): Map<string, League> {
  return new Map((leagues ?? []).map((l) => [l.code, l]))
}

/** Fixtures grouped by league (UEFA first, then country and league name), each sorted by kickoff. */
export function groupFixturesByLeague(
  fixtures: readonly Fixture[],
  leagues: readonly League[] | undefined,
): FixtureGroup[] {
  const lookup = leagueLookup(leagues)
  const map = new Map<string, Fixture[]>()
  for (const f of fixtures) map.set(f.league, [...(map.get(f.league) ?? []), f])
  return [...map.entries()]
    .map(([code, fs]) => {
      const league = lookup.get(code)
      return {
        code,
        name: league?.name ?? code,
        country: league?.country ?? '',
        fixtures: [...fs].sort((a, b) => kickoffSortValue(a.kickoff) - kickoffSortValue(b.kickoff)),
      }
    })
    .sort(
      (a, b) => uefaRank(a.code) - uefaRank(b.code) || byCountry(a.country, b.country) || byText(a.name, b.name),
    )
}

/** Case-insensitive match of a search term against either team name. */
export function matchesTeamSearch(fixture: Fixture, term: string): boolean {
  const t = term.trim().toLowerCase()
  if (!t) return true
  return fixture.home.toLowerCase().includes(t) || fixture.away.toLowerCase().includes(t)
}

export function filterFixtures(
  fixtures: readonly Fixture[],
  opts: { leagues: readonly string[]; search: string },
): Fixture[] {
  const set = new Set(opts.leagues)
  return fixtures.filter((f) => (set.size === 0 || set.has(f.league)) && matchesTeamSearch(f, opts.search))
}
