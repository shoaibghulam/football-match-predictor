import { apiGet } from './client'
import type { Fixture, Health, League, Prediction, Team, TeamForm } from './types'

export interface FixtureParams {
  date?: string
  dateTo?: string
  league?: string
  team?: string
}

export interface PredictParams {
  home: string
  away: string
  league?: string
}

export const getHealth = (signal?: AbortSignal) => apiGet<Health>('/health', undefined, signal)

export const getLeagues = (signal?: AbortSignal) => apiGet<League[]>('/leagues', undefined, signal)

export const getTeams = (params: { q?: string; league?: string }, signal?: AbortSignal) =>
  apiGet<Team[]>('/teams', { q: params.q, league: params.league }, signal)

export const getFixtures = (params: FixtureParams, signal?: AbortSignal) =>
  apiGet<Fixture[]>(
    '/fixtures',
    { date: params.date, date_to: params.dateTo, league: params.league, team: params.team },
    signal,
  )

/** Live matches, then UEFA matches of the day, then the biggest domestic games (max 12). */
export const getFeatured = (params: { date?: string }, signal?: AbortSignal) =>
  apiGet<Fixture[]>('/featured', { date: params.date }, signal)

export const getPrediction = (params: PredictParams, signal?: AbortSignal) =>
  apiGet<Prediction>('/predict', { home: params.home, away: params.away, league: params.league }, signal)

export const getTeamForm = (name: string, signal?: AbortSignal) =>
  apiGet<TeamForm>(`/teams/${encodeURIComponent(name)}/form`, undefined, signal)
