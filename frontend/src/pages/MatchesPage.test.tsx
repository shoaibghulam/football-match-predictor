import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { buildUrl } from '../api/client'
import type { Fixture } from '../api/types'
import { apiError } from '../mocks/handlers'
import { makeFixture, makePrediction } from '../test/factories'
import { renderRoute } from '../test/render'
import { server } from '../test/server'

const FIXTURES: Fixture[] = [
  makeFixture({ id: 1, league: 'E0', home: 'Arsenal', away: 'Chelsea', kickoff: '2026-10-20T17:30:00' }),
  makeFixture({ id: 2, league: 'E0', home: 'Liverpool', away: 'Everton', kickoff: '2026-10-20T12:30:00' }),
  makeFixture({
    id: 3,
    league: 'SP1',
    home: 'Getafe',
    away: 'Barcelona',
    kickoff: '2026-10-20T20:00:00',
    prediction: makePrediction({ probs: { home: 0.2, draw: 0.25, away: 0.55 }, model: 'base' }),
  }),
]

let requests: URL[] = []

function serveFixtures(body: Fixture[] = FIXTURES) {
  server.use(
    http.get(buildUrl('/fixtures'), ({ request }) => {
      requests.push(new URL(request.url))
      return HttpResponse.json(body)
    }),
  )
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(new Date(2026, 9, 20, 10, 0))
  requests = []
  server.use(http.get(buildUrl('/featured'), () => HttpResponse.json([])))
})
afterEach(() => vi.useRealTimers())

describe('MatchesPage', () => {
  it("requests today's fixtures and groups them by league in kickoff order", async () => {
    serveFixtures()
    renderRoute('/')
    expect(screen.getByRole('status')).toHaveTextContent('Loading matches')

    const pl = await screen.findByRole('region', { name: /Premier League/ })
    const cards = within(pl).getAllByRole('article')
    expect(cards.map((c) => c.getAttribute('aria-label'))).toEqual(['Liverpool vs Everton', 'Arsenal vs Chelsea'])
    expect(screen.getByRole('region', { name: /La Liga/ })).toBeInTheDocument()
    // UK API dates are padded one day either side so the viewer's local day is complete
    expect(requests[0]?.searchParams.get('date')).toBe('2026-10-19')
    expect(requests[0]?.searchParams.get('date_to')).toBe('2026-10-21')
    expect(screen.getByText('3 of 3 matches')).toBeInTheDocument()
  })

  it('filters by league and team search, and can clear filters', async () => {
    serveFixtures()
    const { user } = renderRoute('/')
    await screen.findAllByRole('article')
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'League' })).toBeEnabled())

    await user.selectOptions(screen.getByRole('combobox', { name: 'League' }), 'SP1')
    expect(screen.getAllByRole('article')).toHaveLength(1)
    expect(screen.getByText('1 of 3 matches (filtered)')).toBeInTheDocument()

    await user.selectOptions(screen.getByRole('combobox', { name: 'League' }), '')
    await user.type(screen.getByRole('searchbox', { name: 'Find a team' }), 'zzz')
    expect(screen.getByText('No matches fit your filters')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Clear filters' }))
    expect(screen.getAllByRole('article')).toHaveLength(3)

    await user.type(screen.getByRole('searchbox', { name: 'Find a team' }), 'ever')
    expect(screen.getAllByRole('article')).toHaveLength(1)
  })

  it("shows only matches on the viewer's local day and names their time zone", async () => {
    serveFixtures([
      makeFixture({ id: 21, league: 'E0', home: 'Arsenal', away: 'Chelsea', kickoff: '2026-10-20T17:30:00' }),
      makeFixture({ id: 22, league: 'E0', home: 'Leeds', away: 'Fulham', kickoff: '2026-10-19T20:00:00' }),
      makeFixture({ id: 23, league: 'E0', home: 'Spurs', away: 'Wolves', kickoff: '2026-10-21T19:45:00' }),
      makeFixture({ id: 24, league: 'E0', home: 'Brighton', away: 'Everton', kickoff: null, date: '2026-10-20' }),
    ])
    renderRoute('/')
    const pl = await screen.findByRole('region', { name: /Premier League/ })
    const labels = within(pl).getAllByRole('article').map((c) => c.getAttribute('aria-label'))
    expect(labels).toEqual(['Arsenal vs Chelsea', 'Brighton vs Everton'])
    const zone = Intl.DateTimeFormat().resolvedOptions().timeZone
    expect(screen.getByTestId('timezone-note')).toHaveTextContent(zone)
  })

  it('shows the empty state and moves to the next day', async () => {
    serveFixtures([])
    const { user } = renderRoute('/')
    expect(await screen.findByText('No matches for this date — try another day')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Next day ›' }))
    await waitFor(() => expect(requests.at(-1)?.searchParams.get('date_to')).toBe('2026-10-22'))
  })

  it('shows an error with a working retry', async () => {
    server.use(http.get(buildUrl('/fixtures'), () => apiError(500, 'INTERNAL', 'Database unavailable')))
    const { user } = renderRoute('/')
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('Could not load matches')
    expect(alert).toHaveTextContent('Database unavailable')

    serveFixtures()
    await user.click(within(alert).getByRole('button', { name: 'Try again' }))
    expect(await screen.findAllByRole('article')).toHaveLength(3)
  })

  it('navigates dates with chips and arrows', async () => {
    serveFixtures()
    const { user, router } = renderRoute('/')
    await screen.findAllByRole('article')
    const search = () => new URLSearchParams(router.state.location.search)

    await user.click(screen.getByRole('button', { name: 'Tomorrow' }))
    await waitFor(() => expect(requests.at(-1)?.searchParams.get('date_to')).toBe('2026-10-22'))
    expect(screen.getByRole('button', { name: 'Tomorrow' })).toHaveAttribute('aria-pressed', 'true')

    await user.click(screen.getByRole('button', { name: 'Next 7 days' }))
    await waitFor(() => expect(requests.at(-1)?.searchParams.get('date_to')).toBe('2026-10-27'))
    expect(requests.at(-1)?.searchParams.get('date')).toBe('2026-10-19')

    await user.click(screen.getByRole('button', { name: 'Previous day' }))
    await waitFor(() => expect(requests.at(-1)?.searchParams.get('date')).toBe('2026-10-18'))
    expect(requests.at(-1)?.searchParams.get('date_to')).toBe('2026-10-20')

    await user.click(screen.getByRole('button', { name: 'Today' }))
    expect(search().has('date')).toBe(false)
    expect(screen.getByRole('button', { name: 'Today' })).toHaveAttribute('aria-pressed', 'true')
    await user.click(screen.getByRole('button', { name: 'Next day' }))
    expect(search().get('date')).toBe('2026-10-21')

    fireEvent.change(screen.getByLabelText('Match date'), { target: { value: '2026-10-30' } })
    expect(search().get('date')).toBe('2026-10-30')
    fireEvent.change(screen.getByLabelText('Match date'), { target: { value: '' } })
    expect(search().get('date')).toBe('2026-10-30')
  })

  it('reads the date range from the URL and ignores invalid values', async () => {
    serveFixtures()
    renderRoute('/?date=2026-10-22&to=2026-10-24')
    await waitFor(() => expect(requests[0]?.searchParams.get('date_to')).toBe('2026-10-25'))
    expect(requests[0]?.searchParams.get('date')).toBe('2026-10-21')

    requests = []
    serveFixtures()
    renderRoute('/?date=not-a-date&to=2027-01-01')
    await waitFor(() => expect(requests[0]?.searchParams.get('date')).toBe('2026-10-19'))
    expect(requests[0]?.searchParams.get('date_to')).toBe('2026-10-21')
  })

  it('opens a detail dialog with all five scorelines and closes it with Escape', async () => {
    serveFixtures()
    const { user } = renderRoute('/')
    await screen.findAllByRole('article')

    const trigger = screen.getByRole('button', { name: 'Details: Arsenal vs Chelsea' })
    await user.click(trigger)
    const dialog = screen.getByRole('dialog', { name: 'Arsenal vs Chelsea' })
    expect(within(dialog).getAllByRole('listitem').length).toBeGreaterThanOrEqual(5)
    expect(within(dialog).getByText(/Premier League/)).toBeInTheDocument()
    expect(within(dialog).getByRole('button', { name: 'Close' })).toHaveFocus()

    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
  })

  it('traps focus within the dialog and closes from the close button', async () => {
    serveFixtures()
    const { user } = renderRoute('/')
    await screen.findAllByRole('article')
    await user.click(screen.getByRole('button', { name: 'Details: Getafe vs Barcelona' }))
    const dialog = screen.getByRole('dialog')
    const close = within(dialog).getByRole('button', { name: 'Close' })

    await user.keyboard('{Shift>}{Tab}{/Shift}')
    expect(within(dialog).getByRole('link', { name: 'Barcelona' })).toHaveFocus()
    await user.tab()
    expect(close).toHaveFocus()

    await user.click(close)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })
})
