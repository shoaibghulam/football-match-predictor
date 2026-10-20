import { screen, within } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { describe, expect, it } from 'vitest'
import { buildUrl } from '../api/client'
import { makeFixture } from '../test/factories'
import { renderRoute } from '../test/render'
import { server } from '../test/server'

describe('TeamPage', () => {
  it('lets the user pick a team from the combobox', async () => {
    const { user, router } = renderRoute('/team')
    expect(screen.getByRole('heading', { level: 1, name: 'Teams' })).toBeInTheDocument()

    const input = screen.getByRole('combobox', { name: 'Choose a team' })
    await user.type(input, 'liv')
    const option = await screen.findByRole('option', { name: /Liverpool/ })
    expect(input).toHaveAttribute('aria-expanded', 'true')

    await user.keyboard('{ArrowDown}')
    expect(option).toHaveAttribute('aria-selected', 'true')
    await user.keyboard('{Enter}')
    expect(router.state.location.pathname).toBe('/team/Liverpool')
  })

  it('shows no results for an unknown search and closes on Escape', async () => {
    const { user } = renderRoute('/team')
    const input = screen.getByRole('combobox', { name: 'Choose a team' })
    await user.type(input, 'qqqq')
    expect(await screen.findByText('No teams found.', { selector: 'div' })).toBeInTheDocument()
    await user.keyboard('{Escape}')
    expect(input).toHaveAttribute('aria-expanded', 'false')
  })

  it("shows upcoming fixtures with the team's win chance, form and ratings", async () => {
    server.use(
      http.get(buildUrl('/fixtures'), ({ request }) => {
        const url = new URL(request.url)
        expect(url.searchParams.get('team')).toBe('Chelsea')
        expect(url.searchParams.get('date_to')).toBeTruthy()
        return HttpResponse.json([makeFixture()])
      }),
    )
    renderRoute('/team/Chelsea')
    expect(await screen.findByText(/Chelsea win chance:/)).toHaveTextContent('Chelsea win chance: 22%')

    const form = await screen.findByRole('list', { name: 'Recent results, most recent first' })
    expect(within(form).getAllByRole('listitem')).toHaveLength(10)
    expect(screen.getByText(/^Last 10:/)).toBeInTheDocument()
    expect(screen.getByText('Elo rating')).toBeInTheDocument()
  })

  it('shows an empty state when the team has no fixtures', async () => {
    server.use(http.get(buildUrl('/fixtures'), () => HttpResponse.json([])))
    renderRoute('/team/Arsenal')
    expect(await screen.findByText('No fixtures for Arsenal in the next 14 days')).toBeInTheDocument()
  })

  it('explains when the team is unknown', async () => {
    renderRoute('/team/Nobody%20FC')
    expect(await screen.findByText("We don't know a team called “Nobody FC”")).toBeInTheDocument()
  })

  it('opens the match detail from an upcoming fixture', async () => {
    server.use(http.get(buildUrl('/fixtures'), () => HttpResponse.json([makeFixture()])))
    const { user } = renderRoute('/team/Arsenal')
    await user.click(await screen.findByRole('button', { name: 'Details: Arsenal vs Chelsea' }))
    expect(screen.getByRole('dialog', { name: 'Arsenal vs Chelsea' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Close' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('retries after a fixtures error', async () => {
    server.use(http.get(buildUrl('/fixtures'), () => HttpResponse.json({ error: { code: 'X', message: 'Down' } }, { status: 503 })))
    const { user } = renderRoute('/team/Arsenal')
    const alert = await screen.findByRole('alert')
    server.use(http.get(buildUrl('/fixtures'), () => HttpResponse.json([makeFixture()])))
    await user.click(within(alert).getByRole('button', { name: 'Try again' }))
    expect(await screen.findByText(/Arsenal win chance:/)).toBeInTheDocument()
  })
})
