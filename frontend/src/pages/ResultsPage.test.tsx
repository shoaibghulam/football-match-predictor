import { screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ResultItem } from '../api/results'
import { makeResult, MOCK_RESULTS, resultsHandlers } from '../mocks/resultsHandlers'
import { renderRoute } from '../test/render'
import { server } from '../test/server'

let requests: URL[] = []
const onRequest = ({ request }: { request: Request }) => {
  const url = new URL(request.url)
  if (url.pathname.includes('/results')) requests.push(url)
}

function lastList(): URL | undefined {
  return [...requests].reverse().find((u) => u.pathname.endsWith('/results'))
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(new Date(2026, 9, 21, 10, 0))
  requests = []
  server.events.on('request:start', onRequest)
})
afterEach(() => {
  server.events.removeAllListeners()
  vi.useRealTimers()
})

function rows(): HTMLElement[] {
  return within(screen.getByRole('table', { name: /predictions and the actual results/i }))
    .getAllByRole('row')
    .slice(1)
}

describe('ResultsPage', () => {
  it('defaults to the live record and shows stats with correct / wrong badges', async () => {
    renderRoute('/results')
    expect(await screen.findByText('Live record', { exact: false })).toBeInTheDocument()
    const stats = await screen.findByRole('region', { name: 'Accuracy summary' })
    expect(within(stats).getByText('Matches predicted').nextSibling).toHaveTextContent('2')
    expect(within(stats).getByText('Accuracy').nextSibling).toHaveTextContent('50.0%')
    expect(within(stats).getByText('Us vs bookmaker')).toBeInTheDocument()

    await waitFor(() => expect(rows()).toHaveLength(2))
    const [first, second] = rows()
    expect(first).toHaveTextContent('Arsenal')
    expect(first).toHaveTextContent('2–1')
    expect(within(first as HTMLElement).getByText('Correct')).toBeInTheDocument()
    expect(within(second as HTMLElement).getByText('Wrong')).toBeInTheDocument()
    expect(second).toHaveTextContent('we gave it 27%')
    expect(lastList()?.searchParams.get('source')).toBe('live')
    expect(screen.getByRole('link', { name: 'Results' })).toHaveAttribute('aria-current', 'page')
  })

  it('falls back to the backtest with an explanation when no live match has finished', async () => {
    const backtestOnly = MOCK_RESULTS.filter((r) => r.source === 'backtest')
    server.use(...resultsHandlers(backtestOnly))
    renderRoute('/results')
    expect(await screen.findByRole('note')).toHaveTextContent(/No live prediction has a final result yet/)
    await waitFor(() => expect(rows()).toHaveLength(1))
    expect(rows()[0]).toHaveTextContent('Lazio')
    expect(lastList()?.searchParams.get('source')).toBe('backtest')
    expect(screen.getByRole('button', { name: 'Backtest' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('updates the URL and the request when filters change', async () => {
    const { user, router } = renderRoute('/results?source=live')
    await waitFor(() => expect(rows()).toHaveLength(2))

    await user.selectOptions(screen.getByLabelText('Outcome'), 'wrong')
    await waitFor(() => expect(lastList()?.searchParams.get('outcome')).toBe('wrong'))
    expect(router.state.location.search).toContain('outcome=wrong')
    await waitFor(() => expect(rows()).toHaveLength(1))
    expect(rows()[0]).toHaveTextContent('Getafe')

    await user.click(screen.getByRole('button', { name: 'Last 7 days' }))
    await waitFor(() => expect(lastList()?.searchParams.get('date')).toBe('2026-10-15'))
    expect(lastList()?.searchParams.get('date_to')).toBe('2026-10-21')
    expect(router.state.location.search).toContain('range=7d')

    await user.click(screen.getByRole('button', { name: 'Backtest' }))
    await waitFor(() => expect(lastList()?.searchParams.get('source')).toBe('backtest'))
    expect(screen.queryByLabelText('Status')).not.toBeInTheDocument()
  })

  it('lists pending predictions as awaiting a result', async () => {
    const { user } = renderRoute('/results?source=live')
    await waitFor(() => expect(rows()).toHaveLength(2))
    await user.selectOptions(screen.getByLabelText('Status'), 'pending')
    await waitFor(() => expect(lastList()?.searchParams.get('status')).toBe('pending'))
    await waitFor(() => expect(rows()).toHaveLength(1))
    expect(rows()[0]).toHaveTextContent('Liverpool')
    expect(within(rows()[0] as HTMLElement).getByText('Awaiting result')).toBeInTheDocument()
  })

  it('shows an empty state when nothing matches', async () => {
    renderRoute('/results?source=live&league=D1')
    expect(await screen.findByText('No predicted matches fit these filters')).toBeInTheDocument()
    expect(screen.getByText(/No finished matches to score/)).toBeInTheDocument()
  })

  it('loads more rows page by page', async () => {
    const many: ResultItem[] = Array.from({ length: 30 }, (_, i) =>
      makeResult({ id: `live-${i + 10}`, match_id: i + 10, home: `Home ${i}`, away: `Away ${i}` }),
    )
    server.use(...resultsHandlers(many))
    const { user } = renderRoute('/results?source=live')
    await waitFor(() => expect(rows()).toHaveLength(25))
    expect(screen.getByText('Showing 25 of 30 finished matches')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Load more' }))
    await waitFor(() => expect(rows()).toHaveLength(30))
    expect(screen.queryByRole('button', { name: 'Load more' })).not.toBeInTheDocument()
  })
})
