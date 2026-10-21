import { QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { createMemoryRouter, RouterProvider } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { buildUrl } from '../api/client'
import type { Fixture, Health } from '../api/types'
import { routes } from '../routes'
import { makeFixture } from '../test/factories'
import { testQueryClient } from '../test/render'
import { server } from '../test/server'

const FIXTURES: Fixture[] = [
  makeFixture({ id: 1, league: 'E0', home: 'Arsenal', away: 'Chelsea', kickoff: '2026-10-20T17:30:00' }),
]

function health(overrides: Partial<Health> = {}): Health {
  return {
    status: 'ok',
    last_ingest: null,
    matches: 120000,
    fixtures: 0,
    football_data_org_key_configured: true,
    setup: {
      stage: 'seeding_history',
      detail: 'Downloading match history from football-data.co.uk (300/610 files)',
      progress: 0.49,
      first_start: true,
    },
    ...overrides,
  }
}

function serve(h: Health, fixtures: Fixture[]) {
  server.use(
    http.get(buildUrl('/health'), () => HttpResponse.json(h)),
    http.get(buildUrl('/fixtures'), () => HttpResponse.json(fixtures)),
  )
}

function renderPage() {
  const client = testQueryClient()
  const router = createMemoryRouter(routes, { initialEntries: ['/'] })
  render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
  return client
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(new Date(2026, 9, 20, 10, 0))
  server.use(http.get(buildUrl('/featured'), () => HttpResponse.json([])))
})
afterEach(() => vi.useRealTimers())

describe('MatchesPage first-start setup', () => {
  it('shows the setup banner, progress and a waiting empty state', async () => {
    serve(health(), [])
    renderPage()

    const notice = await screen.findByTestId('setup-notice')
    expect(notice).toHaveTextContent('Setting up — downloading match history')
    expect(notice).toHaveTextContent('This page refreshes automatically')
    expect(notice).toHaveTextContent('300/610 files')
    expect(screen.getByRole('progressbar', { name: 'Setup progress' })).toHaveAttribute('aria-valuenow', '49')
    expect(await screen.findByText('Fixtures are on their way')).toBeInTheDocument()
    expect(screen.queryByText(/No matches for this date/)).not.toBeInTheDocument()
  })

  it('refetches fixtures when setup finishes and hides the banner', async () => {
    serve(health(), [])
    const client = renderPage()
    await screen.findByText('Fixtures are on their way')

    serve(health({ fixtures: 1, last_ingest: '2026-10-20T09:00:00', setup: { stage: 'ready' } }), FIXTURES)
    await client.refetchQueries({ queryKey: ['health'] })

    expect(await screen.findByRole('article', { name: 'Arsenal vs Chelsea' })).toBeInTheDocument()
    expect(screen.queryByTestId('setup-notice')).not.toBeInTheDocument()
  })

  it('treats an old backend with no fixtures and no ingest as setting up', async () => {
    serve(health({ setup: undefined }), [])
    renderPage()
    const notice = await screen.findByTestId('setup-notice')
    expect(notice).toHaveTextContent('Setting up')
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
  })

  it('shows no banner once the backend is ready', async () => {
    serve(health({ fixtures: 1, last_ingest: '2026-10-20T09:00:00', setup: { stage: 'ready' } }), FIXTURES)
    renderPage()
    await screen.findByRole('article', { name: 'Arsenal vs Chelsea' })
    expect(screen.queryByTestId('setup-notice')).not.toBeInTheDocument()
  })
})
