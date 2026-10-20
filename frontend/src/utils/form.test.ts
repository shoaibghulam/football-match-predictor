import { describe, expect, it } from 'vitest'
import { normalizeForm, normalizeRatings, summarizeForm, toFormEntry } from './form'

describe('form utils', () => {
  it('derives results from home/away goals for the given team', () => {
    expect(toFormEntry({ home: 'Arsenal', away: 'Spurs', home_goals: 2, away_goals: 1 }, 'Arsenal', 0)).toMatchObject({
      outcome: 'W',
      opponent: 'Spurs',
      venue: 'H',
      scoreText: '2-1',
    })
    expect(toFormEntry({ home: 'Spurs', away: 'Arsenal', home_goals: 2, away_goals: 1 }, 'arsenal', 1)).toMatchObject({
      outcome: 'L',
      opponent: 'Spurs',
      venue: 'A',
      scoreText: '1-2',
    })
  })

  it('accepts alternative shapes (opponent/venue/goals_for/result)', () => {
    expect(
      toFormEntry({ opponent: 'Leeds', venue: 'away', goals_for: 0, goals_against: 0 }, 'Hull', 0),
    ).toMatchObject({ outcome: 'D', opponent: 'Leeds', venue: 'A', scoreText: '0-0' })
    expect(toFormEntry({ opponent: 'Leeds', result: 'win' }, 'Hull', 2)).toMatchObject({
      outcome: 'W',
      venue: null,
      scoreText: null,
    })
    expect(toFormEntry({}, 'Hull', 3)).toMatchObject({ outcome: null, opponent: 'Unknown' })
  })

  it('normalises the list from any known key, capped at 10', () => {
    const many = Array.from({ length: 12 }, () => ({ result: 'D' }))
    expect(normalizeForm({ matches: many }, 'X')).toHaveLength(10)
    expect(normalizeForm({ last_results: [{ result: 'L' }] }, 'X')).toHaveLength(1)
    expect(normalizeForm({}, 'X')).toEqual([])
    expect(normalizeForm(undefined, 'X')).toEqual([])
  })

  it('formats ratings and skips non-numeric values', () => {
    expect(normalizeRatings({ ratings: { elo: 1890.6, pi_home: 0.4567, custom_metric: 2, bad: null } })).toEqual([
      { key: 'elo', label: 'Elo rating', value: '1891' },
      { key: 'pi_home', label: 'Pi-rating (home)', value: '0.46' },
      { key: 'custom_metric', label: 'Custom metric', value: '2.00' },
    ])
    expect(normalizeRatings(undefined)).toEqual([])
  })

  it('summarises W/D/L counts', () => {
    const entries = normalizeForm({ results: [{ result: 'W' }, { result: 'W' }, { result: 'L' }, {}] }, 'X')
    expect(summarizeForm(entries)).toEqual({ W: 2, D: 0, L: 1 })
  })
})
