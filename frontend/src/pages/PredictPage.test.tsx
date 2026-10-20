import { screen, within } from '@testing-library/react'
import { http } from 'msw'
import { describe, expect, it } from 'vitest'
import { buildUrl } from '../api/client'
import { apiError } from '../mocks/handlers'
import { renderRoute } from '../test/render'
import { server } from '../test/server'

async function pick(user: ReturnType<typeof renderRoute>['user'], label: string, query: string, option: RegExp) {
  await user.type(screen.getByRole('combobox', { name: label }), query)
  await user.click(await screen.findByRole('option', { name: option }))
}

describe('PredictPage', () => {
  it('predicts a custom match-up and swaps teams', async () => {
    const { user } = renderRoute('/predict')
    expect(screen.getByText(/Choose a home team and an away team/)).toBeInTheDocument()

    await pick(user, 'Home team', 'arse', /Arsenal/)
    expect(screen.getByRole('combobox', { name: 'Home team' })).toHaveValue('Arsenal')
    await pick(user, 'Away team', 'getaf', /Getafe/)

    const result = await screen.findByRole('region', { name: 'Arsenal vs Getafe' })
    expect(await within(result).findByRole('img', { name: /Win probabilities: Arsenal/ })).toBeInTheDocument()
    expect(within(result).getByText('ratings only')).toBeInTheDocument()
    expect(within(result).getByRole('list', { name: 'Most likely scores' }).children).toHaveLength(5)

    await user.click(screen.getByRole('button', { name: 'Swap home and away teams' }))
    expect(await screen.findByRole('region', { name: 'Getafe vs Arsenal' })).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: 'Home team' })).toHaveValue('Getafe')
  })

  it('warns when the same team is chosen twice', async () => {
    const { user } = renderRoute('/predict')
    await pick(user, 'Home team', 'inter', /Inter/)
    await pick(user, 'Away team', 'inter', /Inter/)
    expect(screen.getByRole('alert')).toHaveTextContent('Choose two different teams.')
  })

  it('shows API errors from /predict', async () => {
    server.use(http.get(buildUrl('/predict'), () => apiError(404, 'TEAM_NOT_FOUND', 'Unknown team: Arsenal')))
    const { user } = renderRoute('/predict')
    await pick(user, 'Home team', 'arse', /Arsenal/)
    await pick(user, 'Away team', 'chel', /Chelsea/)
    expect(await screen.findByRole('alert')).toHaveTextContent('Unknown team: Arsenal')
  })
})
