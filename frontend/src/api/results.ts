/** Results page API: our predictions vs actual results (GET /results, /results/summary). */
import { keepPreviousData, useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { apiGet } from './client'
import type { Probs } from './types'

export type ResultSource = 'live' | 'backtest'
export type SourceFilter = ResultSource | 'all'
export type ResultStatus = 'finished' | 'pending'
export type OutcomeCode = 'H' | 'D' | 'A'
export type OutcomeFilter = 'all' | 'correct' | 'wrong'
export type ConfidenceFilter = 'all' | 'high' | 'low'

export interface ResultItem {
  id: string
  match_id: number
  source: ResultSource
  status: ResultStatus
  date: string
  kickoff: string | null
  league: string
  home: string
  away: string
  score: { home: number; away: number } | null
  actual: OutcomeCode | null
  predicted: OutcomeCode
  probs: Probs
  prob_predicted: number
  prob_of_actual: number | null
  correct: boolean | null
  rps: number | null
  model: string | null
  model_version: string
  frozen_at: string | null
}

export interface ResultsPage {
  items: ResultItem[]
  total: number
  page: number
  page_size: number
}

export interface OutcomeStat {
  n: number
  correct: number
  accuracy: number | null
}

export interface LeagueStat extends OutcomeStat {
  league: string
  rps: number
}

export interface BucketStat {
  bucket: string
  lo: number
  hi: number
  n: number
  accuracy: number | null
  avg_prob: number | null
}

export interface BookmakerStat {
  n: number
  accuracy: number
  rps: number
  model_accuracy: number
  model_rps: number
}

export interface ResultsSummary {
  source: SourceFilter
  n: number
  correct: number
  accuracy: number | null
  rps: number | null
  log_loss: number | null
  brier: number | null
  by_outcome: Record<OutcomeCode, OutcomeStat>
  by_predicted: Record<OutcomeCode, OutcomeStat>
  by_league: LeagueStat[]
  by_confidence_bucket: BucketStat[]
  bookmaker: BookmakerStat | null
  daily: (OutcomeStat & { date: string })[]
  monthly: (OutcomeStat & { month: string })[]
  first_date: string | null
  last_date: string | null
  pending: number
  available: Record<ResultSource, number>
}

export interface ResultFilters {
  source: SourceFilter
  status: ResultStatus
  date?: string
  dateTo?: string
  league?: string
  team?: string
  outcome: OutcomeFilter
  confidence: ConfidenceFilter
}

export const RESULTS_PAGE_SIZE = 25
const RESULTS_STALE_MS = 60 * 1000

function toParams(f: ResultFilters) {
  return {
    source: f.source,
    status: f.status === 'finished' ? undefined : f.status,
    date: f.date,
    date_to: f.dateTo,
    league: f.league,
    team: f.team,
    outcome: f.outcome === 'all' ? undefined : f.outcome,
    confidence: f.confidence === 'all' ? undefined : f.confidence,
  }
}

export const getResults = (f: ResultFilters, page: number, signal?: AbortSignal) =>
  apiGet<ResultsPage>('/results', { ...toParams(f), page, page_size: RESULTS_PAGE_SIZE }, signal)

/** Summary ignores `status` (always finished matches) and reports pending separately. */
export const getResultsSummary = (f: ResultFilters, signal?: AbortSignal) =>
  apiGet<ResultsSummary>('/results/summary', { ...toParams(f), status: undefined }, signal)

export const resultsKeys = {
  list: (f: ResultFilters) => ['results', 'list', f] as const,
  summary: (f: ResultFilters) => ['results', 'summary', { ...f, status: 'finished' }] as const,
}

export function useResults(filters: ResultFilters | null) {
  return useInfiniteQuery({
    queryKey: resultsKeys.list(filters ?? ({} as ResultFilters)),
    queryFn: ({ pageParam, signal }) => getResults(filters as ResultFilters, pageParam, signal),
    initialPageParam: 1,
    getNextPageParam: (last) => (last.page * last.page_size < last.total ? last.page + 1 : undefined),
    enabled: filters !== null,
    staleTime: RESULTS_STALE_MS,
  })
}

export function useResultsSummary(filters: ResultFilters | null) {
  return useQuery({
    queryKey: resultsKeys.summary(filters ?? ({} as ResultFilters)),
    queryFn: ({ signal }) => getResultsSummary(filters as ResultFilters, signal),
    enabled: filters !== null,
    staleTime: RESULTS_STALE_MS,
    placeholderData: keepPreviousData,
  })
}
