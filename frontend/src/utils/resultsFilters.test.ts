import { describe, expect, it } from 'vitest'
import { presetRange, readResultsUrl, seasonStart, toApiFilters } from './resultsFilters'

const TODAY = '2026-10-21'

describe('resultsFilters', () => {
  it('reads defaults and ignores invalid values', () => {
    const s = readResultsUrl(new URLSearchParams('source=nope&range=bogus&outcome=x&from=2026-13-01'))
    expect(s).toMatchObject({ source: null, range: 'all', outcome: 'all', confidence: 'all', status: 'finished' })
    expect(s.from).toBeUndefined()
  })

  it('never asks the backtest for pending matches', () => {
    const s = readResultsUrl(new URLSearchParams('source=backtest&status=pending'))
    expect(s.status).toBe('finished')
    expect(toApiFilters({ ...s, status: 'pending' }, 'backtest', TODAY).status).toBe('finished')
  })

  it('turns presets into date ranges', () => {
    const base = readResultsUrl(new URLSearchParams())
    expect(presetRange({ ...base, range: 'yesterday' }, TODAY)).toEqual({ date: '2026-10-20', dateTo: '2026-10-20' })
    expect(presetRange({ ...base, range: '30d' }, TODAY)).toEqual({ date: '2026-09-22', dateTo: TODAY })
    expect(presetRange({ ...base, range: 'season' }, TODAY)).toEqual({ date: '2026-07-01', dateTo: TODAY })
    expect(presetRange(base, TODAY)).toEqual({})
    expect(presetRange({ ...base, range: 'custom', from: '2026-10-10', to: '2026-10-01' }, TODAY)).toEqual({
      date: '2026-10-01',
      dateTo: '2026-10-10',
    })
    expect(presetRange({ ...base, range: 'custom', from: '2020-01-01', to: '2026-01-01' }, TODAY).dateTo).toBe(
      '2021-01-01',
    )
  })

  it('knows the season start', () => {
    expect(seasonStart('2026-03-05')).toBe('2025-07-01')
    expect(seasonStart('2026-07-01')).toBe('2026-07-01')
  })

  it('builds API filters', () => {
    const s = readResultsUrl(new URLSearchParams('league=UEFA&q=%20Arsenal%20&confidence=high&status=pending'))
    expect(toApiFilters(s, 'all', TODAY)).toEqual({
      source: 'all',
      status: 'pending',
      league: 'UEFA',
      team: 'Arsenal',
      outcome: 'all',
      confidence: 'high',
    })
  })
})
