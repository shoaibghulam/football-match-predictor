import { keepPreviousData, useQuery } from '@tanstack/react-query'
import {
  getFeatured,
  getFixtures,
  getHealth,
  getLeagues,
  getPrediction,
  getTeamForm,
  getTeams,
  type FixtureParams,
  type PredictParams,
} from './endpoints'

import type { Fixture } from './types'

export const STALE_TIME_MS = 5 * 60 * 1000
/** Refetch interval while any listed match is live. */
export const LIVE_REFETCH_MS = 60 * 1000
const LEAGUES_STALE_MS = 60 * 60 * 1000
export const MIN_TEAM_QUERY = 2

export const queryKeys = {
  leagues: ['leagues'] as const,
  teams: (q: string) => ['teams', q] as const,
  fixtures: (p: FixtureParams) => ['fixtures', p] as const,
  featured: (date: string | undefined) => ['featured', date ?? ''] as const,
  health: ['health'] as const,
  predict: (p: PredictParams | null) => ['predict', p] as const,
  teamForm: (name: string) => ['teamForm', name] as const,
}

export function useLeagues() {
  return useQuery({
    queryKey: queryKeys.leagues,
    queryFn: ({ signal }) => getLeagues(signal),
    staleTime: LEAGUES_STALE_MS,
  })
}

export function useTeams(q: string) {
  const term = q.trim()
  return useQuery({
    queryKey: queryKeys.teams(term),
    queryFn: ({ signal }) => getTeams({ q: term }, signal),
    enabled: term.length >= MIN_TEAM_QUERY,
    placeholderData: keepPreviousData,
  })
}

export function hasLive(fixtures: readonly Fixture[] | undefined): boolean {
  return Boolean(fixtures?.some((f) => f.status === 'live'))
}

/** Poll every minute while something is live, otherwise not at all. */
export function liveRefetch(query: { state: { data?: readonly Fixture[] } }): number | false {
  return hasLive(query.state.data) ? LIVE_REFETCH_MS : false
}

export function useFixtures(params: FixtureParams, enabled = true) {
  return useQuery({
    queryKey: queryKeys.fixtures(params),
    queryFn: ({ signal }) => getFixtures(params, signal),
    enabled,
    refetchInterval: liveRefetch,
  })
}

export function useFeatured(date: string | undefined) {
  return useQuery({
    queryKey: queryKeys.featured(date),
    queryFn: ({ signal }) => getFeatured({ date }, signal),
    refetchInterval: liveRefetch,
  })
}

export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: ({ signal }) => getHealth(signal),
    staleTime: STALE_TIME_MS,
  })
}

export function usePrediction(params: PredictParams | null) {
  return useQuery({
    queryKey: queryKeys.predict(params),
    queryFn: ({ signal }) => getPrediction(params as PredictParams, signal),
    enabled: params !== null,
  })
}

export function useTeamForm(name: string | undefined) {
  return useQuery({
    queryKey: queryKeys.teamForm(name ?? ''),
    queryFn: ({ signal }) => getTeamForm(name as string, signal),
    enabled: Boolean(name),
  })
}
