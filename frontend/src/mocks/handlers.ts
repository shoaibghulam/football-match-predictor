import { http, HttpResponse } from 'msw'
import { buildUrl } from '../api/client'
import { daysBetween, isValidISODate, todayISO } from '../utils/date'
import { featuredForDate, findTeam, fixturesForRange, LEAGUES, MOCK_MODEL_VERSION, mockPrediction, mockTeamForm, TEAMS } from './data'

const MAX_RANGE_DAYS = 14

export function apiError(status: number, code: string, message: string) {
  return HttpResponse.json({ error: { code, message } }, { status })
}

export const handlers = [
  http.get(buildUrl('/health'), () =>
    HttpResponse.json({
      status: 'ok',
      model_version: MOCK_MODEL_VERSION,
      last_ingest: null,
      matches: 238000,
      fixtures: 40,
      europe_model_loaded: true,
      europe_model_version: 'europe-2026-10-21',
      football_data_org_key_configured: true,
      live_matches: 1,
      setup: { stage: 'ready', detail: 'Setup finished', progress: null, first_start: false },
    }),
  ),

  http.get(buildUrl('/featured'), ({ request }) => {
    const date = new URL(request.url).searchParams.get('date') ?? todayISO()
    if (!isValidISODate(date)) return apiError(422, 'VALIDATION_ERROR', 'Dates must be YYYY-MM-DD.')
    return HttpResponse.json(featuredForDate(date))
  }),

  http.get(buildUrl('/leagues'), () => HttpResponse.json(LEAGUES)),

  http.get(buildUrl('/teams'), ({ request }) => {
    const url = new URL(request.url)
    const q = (url.searchParams.get('q') ?? '').trim().toLowerCase()
    const league = url.searchParams.get('league')
    const teams = TEAMS.filter((t) => (!league || t.league === league) && (q.length < 2 || t.name.toLowerCase().includes(q)))
    return HttpResponse.json(teams)
  }),

  http.get(buildUrl('/teams/:name/form'), ({ params }) => {
    const team = findTeam(decodeURIComponent(String(params.name)))
    if (!team) return apiError(404, 'TEAM_NOT_FOUND', `Unknown team: ${String(params.name)}`)
    return HttpResponse.json(mockTeamForm(team))
  }),

  http.get(buildUrl('/fixtures'), ({ request }) => {
    const url = new URL(request.url)
    const date = url.searchParams.get('date') ?? todayISO()
    const to = url.searchParams.get('date_to') ?? date
    if (!isValidISODate(date) || !isValidISODate(to)) return apiError(422, 'INVALID_DATE', 'Dates must be YYYY-MM-DD.')
    const span = daysBetween(date, to)
    if (span < 0 || span > MAX_RANGE_DAYS) return apiError(422, 'INVALID_RANGE', 'Date range must be 0-14 days.')
    const league = url.searchParams.get('league')
    const team = url.searchParams.get('team')?.toLowerCase()
    const fixtures = fixturesForRange(date, to).filter(
      (f) =>
        (!league || f.league === league) &&
        (!team || f.home.toLowerCase() === team || f.away.toLowerCase() === team),
    )
    return HttpResponse.json(fixtures)
  }),

  http.get(buildUrl('/predict'), ({ request }) => {
    const url = new URL(request.url)
    const home = findTeam(url.searchParams.get('home') ?? '')
    const away = findTeam(url.searchParams.get('away') ?? '')
    if (!home || !away) return apiError(404, 'TEAM_NOT_FOUND', 'Unknown team.')
    if (home.name === away.name) return apiError(422, 'SAME_TEAM', 'Home and away must be different teams.')
    return HttpResponse.json(mockPrediction(home.name, away.name, false))
  }),
]
