import { screen, waitFor, within } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { buildUrl } from '../api/client'
import type { Fixture } from '../api/types'
import { KEY_NOTICE_STORAGE } from '../components/KeyNotice'
import { makeFixture, makePrediction } from '../test/factories'
import { renderRoute } from '../test/render'
import { server } from '../test/server'

const LIVE = makeFixture({
  id: 10,
  league: 'UCL',
  home: 'Barcelona',
  away: 'Paris SG',
  kickoff: '2026-10-20T20:00:00+01:00',
  status: 'live',
  score: { home: 2, away: 1 },
  competition_type: 'europe',
  featured: true,
  live: { minute: 67, minute_estimated: false, score: { home: 2, away: 1 } },
  prediction: makePrediction({ model: 'europe', confidence: 'high' }),
  prediction_frozen_at: '2026-10-20T18:30:00Z',
})
const UEL = makeFixture({
  id: 11,
  league: 'UEL',
  home: 'Celtic',
  away: 'Benfica',
  competition_type: 'europe',
  featured: true,
  prediction: makePrediction({ model: 'europe', confidence: 'low' }),
})
const DOMESTIC = makeFixture({ id: 12, league: 'E0', home: 'Arsenal', away: 'Chelsea' })

let featuredCalls = 0

function memoryStorage(): Storage {
  const data = new Map<string, string>()
  return {
    get length() {
      return data.size
    },
    clear: () => data.clear(),
    getItem: (k: string) => data.get(k) ?? null,
    key: (i: number) => [...data.keys()][i] ?? null,
    removeItem: (k: string) => void data.delete(k),
    setItem: (k: string, v: string) => void data.set(k, v),
  }
}

function setStorage(storage: Storage) {
  Object.defineProperty(window, 'localStorage', { value: storage, configurable: true, writable: true })
}

function serve(featured: Fixture[], fixtures: Fixture[], keyConfigured = true) {
  server.use(
    http.get(buildUrl('/featured'), () => {
      featuredCalls += 1
      return HttpResponse.json(featured)
    }),
    http.get(buildUrl('/fixtures'), () => HttpResponse.json(fixtures)),
    http.get(buildUrl('/health'), () =>
      HttpResponse.json({ status: 'ok', football_data_org_key_configured: keyConfigured, europe_model_loaded: true }),
    ),
  )
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(new Date(2026, 9, 20, 21, 7))
  featuredCalls = 0
  setStorage(memoryStorage())
})
afterEach(() => vi.useRealTimers())

describe('MatchesPage — UEFA, live and featured', () => {
  it('shows the live strip with minute and score, then featured matches', async () => {
    serve([LIVE, UEL], [LIVE, UEL, DOMESTIC])
    const { user } = renderRoute('/')
    const strip = await screen.findByRole('region', { name: /Live now/ })
    const live = within(strip).getByRole('button', { name: "Barcelona 2, Paris SG 1, minute 67'" })
    expect(live).toHaveTextContent('UEFA Champions League')
    expect(within(strip).getByText("67'")).toBeInTheDocument()

    const featured = screen.getByRole('region', { name: /Featured/ })
    const card = within(featured).getByRole('article', { name: 'Celtic vs Benfica' })
    expect(within(card).getByText('European model')).toBeInTheDocument()
    expect(within(card).getByText(/Lower confidence/)).toBeInTheDocument()
    expect(within(card).getByText('UEFA Europa League')).toBeInTheDocument()

    await user.click(live)
    expect(screen.getByRole('dialog', { name: 'Barcelona vs Paris SG' })).toBeInTheDocument()
    await user.keyboard('{Escape}')
    await user.click(within(featured).getByRole('button', { name: 'Details: Celtic vs Benfica' }))
    expect(screen.getByRole('dialog', { name: 'Celtic vs Benfica' })).toBeInTheDocument()
  })

  it('lists UEFA competitions first, with the live card showing LIVE and the current score', async () => {
    serve([], [DOMESTIC, UEL, LIVE])
    renderRoute('/')
    await screen.findAllByRole('article')
    const headings = screen.getAllByRole('heading', { level: 2 }).map((h) => h.textContent ?? '')
    expect(headings[0]).toMatch(/UEFA Champions League/)
    expect(headings[1]).toMatch(/UEFA Europa League/)
    expect(headings[2]).toMatch(/Premier League/)
    const liveCard = screen.getByRole('article', { name: 'Barcelona vs Paris SG' })
    expect(within(liveCard).getByText('LIVE')).toBeInTheDocument()
    expect(within(liveCard).getByText('Current score Barcelona 2, Paris SG 1')).toBeInTheDocument()
    expect(within(liveCard).getByText('Pre-match prediction')).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: /Live now/ })).not.toBeInTheDocument()
  })

  it('groups UEFA competitions first in the league filter', async () => {
    serve([], [DOMESTIC, UEL])
    const { user } = renderRoute('/')
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'League' })).toBeEnabled())
    const select = screen.getByRole('combobox', { name: 'League' })
    const groups = within(select).getAllByRole('group')
    expect(groups[0]).toHaveAttribute('label', 'Europe')
    expect(within(groups[0] as HTMLElement).getAllByRole('option').map((o) => o.textContent)).toEqual([
      'UEFA Champions League',
      'UEFA Europa League',
      'UEFA Conference League',
    ])
    await user.selectOptions(select, 'UEL')
    expect(screen.getAllByRole('article').map((a) => a.getAttribute('aria-label'))).toEqual(['Celtic vs Benfica'])
  })

  it('shows a dismissible notice when no football-data.org key is configured', async () => {
    serve([], [DOMESTIC], false)
    const { user, unmount } = renderRoute('/')
    const note = await screen.findByRole('note')
    expect(note).toHaveTextContent('Champions League schedule needs a free football-data.org key — see README')
    await user.click(within(note).getByRole('button', { name: 'Dismiss notice' }))
    expect(screen.queryByRole('note')).not.toBeInTheDocument()
    expect(window.localStorage.getItem(KEY_NOTICE_STORAGE)).toBe('1')
    unmount()
    renderRoute('/')
    await screen.findAllByRole('article')
    expect(screen.queryByRole('note')).not.toBeInTheDocument()
  })

  it('keeps working when localStorage throws', async () => {
    const blocked = () => {
      throw new Error('blocked')
    }
    setStorage({ ...memoryStorage(), getItem: blocked, setItem: blocked })
    serve([], [DOMESTIC], false)
    const { user } = renderRoute('/')
    const note = await screen.findByRole('note')
    await user.click(within(note).getByRole('button', { name: 'Dismiss notice' }))
    expect(screen.queryByRole('note')).not.toBeInTheDocument()
  })

  it('hides the notice when the key is configured and requests featured for the date', async () => {
    serve([{ ...UEL, kickoff: '2026-10-22T20:00:00+01:00' }], [{ ...DOMESTIC, kickoff: '2026-10-22T15:00:00' }])
    renderRoute('/?date=2026-10-22')
    await screen.findByRole('region', { name: /Featured/ })
    expect(screen.queryByRole('note')).not.toBeInTheDocument()
    expect(featuredCalls).toBeGreaterThan(0)
  })
})
