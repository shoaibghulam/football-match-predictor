import { describe, expect, it } from 'vitest'
import {
  addDays,
  daysBetween,
  formatDayLabel,
  formatLongDate,
  inLocalRange,
  isValidISODate,
  kickoffSortValue,
  localMatchDate,
  localTimeZoneLabel,
  parseISODate,
  parseKickoff,
  toISODate,
  todayISO,
} from './date'

const NOW = new Date(2026, 9, 20, 10, 30)

describe('local timezone handling', () => {
  it('derives the local calendar date from a timezone-aware kickoff', () => {
    const kickoff = '2026-10-20T23:30:00+01:00'
    const expected = toISODate(new Date(kickoff))
    expect(localMatchDate({ kickoff })).toBe(expected)
  })

  it('falls back to the published date when the kickoff time is unknown', () => {
    expect(localMatchDate({ kickoff: null, date: '2026-10-21' })).toBe('2026-10-21')
    expect(localMatchDate({ kickoff: '2026-10-22' })).toBe('2026-10-22')
    expect(localMatchDate({ kickoff: 'garbage' })).toBeNull()
  })

  it('keeps only matches whose local date falls inside the selected range', () => {
    const inside = { kickoff: `2026-10-21T12:00:00` }
    const before = { kickoff: `2026-10-19T12:00:00` }
    const after = { kickoff: `2026-10-24T12:00:00` }
    const unknown = { kickoff: null, date: '2026-10-22' }
    const kept = [inside, before, after, unknown].filter((f) => inLocalRange(f, '2026-10-21', '2026-10-22'))
    expect(kept).toEqual([inside, unknown])
  })

  it('treats a single-day range as from == to', () => {
    expect(inLocalRange({ kickoff: '2026-10-21T08:00:00' }, '2026-10-21')).toBe(true)
    expect(inLocalRange({ kickoff: '2026-10-22T08:00:00' }, '2026-10-21')).toBe(false)
  })

  it('names the viewer timezone with its UTC offset', () => {
    const zone = Intl.DateTimeFormat().resolvedOptions().timeZone
    const label = localTimeZoneLabel(NOW)
    expect(label).toContain(zone)
    expect(label).toMatch(/GMT|UTC/)
  })
})

describe('date utils', () => {
  it('formats local dates as yyyy-mm-dd with zero padding', () => {
    expect(toISODate(new Date(2026, 0, 5))).toBe('2026-01-05')
    expect(todayISO(NOW)).toBe('2026-10-20')
  })

  it('parses valid ISO dates and rejects invalid ones', () => {
    expect(parseISODate('2026-10-20')?.getDate()).toBe(20)
    expect(parseISODate('2026-02-30')).toBeNull()
    expect(parseISODate('20-10-2026')).toBeNull()
    expect(parseISODate(null)).toBeNull()
    expect(isValidISODate('2026-12-31')).toBe(true)
    expect(isValidISODate('nope')).toBe(false)
  })

  it('adds days across month and year boundaries', () => {
    expect(addDays('2026-10-31', 1)).toBe('2026-11-01')
    expect(addDays('2026-12-31', 1)).toBe('2027-01-01')
    expect(addDays('2026-03-01', -1)).toBe('2026-02-28')
    expect(() => addDays('bad', 1)).toThrow('Invalid date')
  })

  it('counts days between dates (DST-safe)', () => {
    expect(daysBetween('2026-10-20', '2026-11-03')).toBe(14)
    expect(daysBetween('2026-03-28', '2026-03-30')).toBe(2)
    expect(daysBetween('2026-10-21', '2026-10-20')).toBe(-1)
    expect(() => daysBetween('x', '2026-10-20')).toThrow('Invalid date range')
  })

  it('labels relative days', () => {
    expect(formatDayLabel('2026-10-20', NOW)).toBe('Today')
    expect(formatDayLabel('2026-10-21', NOW)).toBe('Tomorrow')
    expect(formatDayLabel('2026-10-19', NOW)).toBe('Yesterday')
    expect(formatDayLabel('2026-10-25', NOW)).toMatch(/25/)
    expect(formatDayLabel('bad', NOW)).toBe('bad')
  })

  it('formats long dates', () => {
    expect(formatLongDate('2026-10-20')).toMatch(/2026/)
    expect(formatLongDate('bad')).toBe('bad')
  })

  it('parses kickoffs with and without a time', () => {
    expect(parseKickoff('2026-10-20T15:00:00')).toEqual({ date: '2026-10-20', time: '15:00' })
    expect(parseKickoff('2026-10-20')).toEqual({ date: '2026-10-20', time: null })
    expect(parseKickoff('garbage')).toEqual({ date: null, time: null })
    expect(parseKickoff(undefined)).toEqual({ date: null, time: null })
  })

  it('sorts unknown kickoffs last', () => {
    const early = kickoffSortValue('2026-10-20T12:30:00')
    const late = kickoffSortValue('2026-10-20T20:00:00')
    expect(early).toBeLessThan(late)
    expect(kickoffSortValue('2026-10-20')).toBeLessThan(early)
    expect(kickoffSortValue(null)).toBe(Number.MAX_SAFE_INTEGER)
    expect(kickoffSortValue('nope')).toBe(Number.MAX_SAFE_INTEGER)
  })
})
