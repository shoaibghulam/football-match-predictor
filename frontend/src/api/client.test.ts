import { http, HttpResponse } from 'msw'
import { describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { ApiError, apiGet, buildUrl, errorMessage, fallbackMessage } from './client'
import { getFixtures, getHealth, getPrediction, getTeamForm } from './endpoints'
import { shouldRetry } from './queryClient'

describe('api client', () => {
  it('builds absolute URLs and drops empty params', () => {
    const url = new URL(buildUrl('/fixtures', { date: '2026-10-20', league: '', team: undefined, n: 3 }))
    expect(url.pathname).toBe('/api/v1/fixtures')
    expect(url.searchParams.get('date')).toBe('2026-10-20')
    expect(url.searchParams.has('league')).toBe(false)
    expect(url.searchParams.get('n')).toBe('3')
  })

  it('returns parsed JSON from endpoints', async () => {
    await expect(getHealth()).resolves.toMatchObject({ status: 'ok' })
    const pred = await getPrediction({ home: 'Arsenal', away: 'Chelsea', league: 'E0' })
    expect(pred.probs.home + pred.probs.draw + pred.probs.away).toBeCloseTo(1, 2)
    await expect(getTeamForm('Arsenal')).resolves.toMatchObject({ team: 'Arsenal' })
    await expect(getFixtures({ date: '2026-10-20', dateTo: '2026-10-22' })).resolves.toBeInstanceOf(Array)
  })

  it('parses the error envelope into ApiError', async () => {
    const err = await getPrediction({ home: 'Nobody FC', away: 'Arsenal' }).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err).toMatchObject({ status: 404, code: 'TEAM_NOT_FOUND' })
    const range = await getFixtures({ date: '2026-10-01', dateTo: '2026-10-30' }).catch((e: unknown) => e)
    expect(range).toMatchObject({ status: 422 })
  })

  it('falls back to a friendly message when the body is not an envelope', async () => {
    server.use(http.get(buildUrl('/leagues'), () => new HttpResponse('oops', { status: 503 })))
    await expect(apiGet('/leagues')).rejects.toMatchObject({ status: 503, code: 'HTTP_503' })
    expect(fallbackMessage(404)).toBe('Not found.')
    expect(fallbackMessage(422)).toMatch(/not valid/)
    expect(fallbackMessage(429)).toMatch(/Too many/)
    expect(fallbackMessage(418)).toMatch(/418/)
  })

  it('rejects invalid JSON on success and handles empty bodies', async () => {
    server.use(http.get(buildUrl('/leagues'), () => new HttpResponse('{not json', { status: 200 })))
    await expect(apiGet('/leagues')).rejects.toMatchObject({ code: 'INVALID_RESPONSE' })
    server.use(http.get(buildUrl('/leagues'), () => new HttpResponse(null, { status: 200 })))
    await expect(apiGet('/leagues')).resolves.toBeNull()
  })

  it('maps network failures to NETWORK_ERROR and rethrows aborts', async () => {
    server.use(http.get(buildUrl('/leagues'), () => HttpResponse.error()))
    await expect(apiGet('/leagues')).rejects.toMatchObject({ status: 0, code: 'NETWORK_ERROR' })
    const controller = new AbortController()
    controller.abort()
    await expect(apiGet('/health', undefined, controller.signal)).rejects.toMatchObject({ name: 'AbortError' })
  })

  it('does not retry client errors', () => {
    expect(shouldRetry(0, new ApiError(404, 'X', 'x'))).toBe(false)
    expect(shouldRetry(0, new ApiError(500, 'X', 'x'))).toBe(true)
    expect(shouldRetry(2, new Error('net'))).toBe(false)
  })

  it('extracts error messages', () => {
    expect(errorMessage(new Error('boom'))).toBe('boom')
    expect(errorMessage('weird')).toBe('Something went wrong.')
  })
})
