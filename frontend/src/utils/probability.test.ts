import { describe, expect, it } from 'vitest'
import { makeFixture } from '../test/factories'
import { favouriteOutcome, formatGoals, formatPercent, likelihoodLabel, teamWinChance, toPercents } from './probability'

describe('probability utils', () => {
  it('formats percentages', () => {
    expect(formatPercent(0.523)).toBe('52%')
    expect(formatPercent(0.003)).toBe('<1%')
    expect(formatPercent(0)).toBe('0%')
    expect(formatPercent(1.4)).toBe('100%')
    expect(formatPercent(undefined)).toBe('–')
    expect(formatPercent(Number.NaN)).toBe('–')
  })

  it('rounds three outcomes so they always sum to 100', () => {
    expect(toPercents({ home: 0.335, draw: 0.335, away: 0.33 })).toEqual({ home: 34, draw: 33, away: 33 })
    const p = toPercents({ home: 0.456, draw: 0.267, away: 0.277 })
    expect(p.home + p.draw + p.away).toBe(100)
    expect(toPercents({ home: 0, draw: 0, away: 0 })).toEqual({ home: 0, draw: 0, away: 0 })
    expect(toPercents({ home: 0.1, draw: 0.1, away: 0.2 })).toEqual({ home: 25, draw: 25, away: 50 })
  })

  it('finds the favourite outcome', () => {
    expect(favouriteOutcome({ home: 0.5, draw: 0.3, away: 0.2 })).toBe('home')
    expect(favouriteOutcome({ home: 0.2, draw: 0.3, away: 0.5 })).toBe('away')
    expect(favouriteOutcome({ home: 0.3, draw: 0.4, away: 0.3 })).toBe('draw')
  })

  it('formats expected goals', () => {
    expect(formatGoals(1.61)).toBe('1.6')
    expect(formatGoals(-1)).toBe('0.0')
    expect(formatGoals(null)).toBe('–')
  })

  it('computes win chance from the team perspective', () => {
    const f = makeFixture()
    expect(teamWinChance(f, 'Arsenal')).toBe(0.52)
    expect(teamWinChance(f, 'chelsea ')).toBe(0.22)
    expect(teamWinChance(f, 'Spurs')).toBeNull()
    expect(teamWinChance(makeFixture({ prediction: null }), 'Arsenal')).toBeNull()
  })

  it('never labels an outcome as certain', () => {
    expect(likelihoodLabel(0.75)).toBe('Strong favourite')
    expect(likelihoodLabel(0.55)).toBe('Favourite')
    expect(likelihoodLabel(0.45)).toBe('Slight favourite')
    expect(likelihoodLabel(0.36)).toBe('Most likely, but open')
  })
})
