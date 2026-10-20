import { QueryClient } from '@tanstack/react-query'
import { ApiError } from './client'
import { STALE_TIME_MS } from './hooks'

const MAX_RETRIES = 2

export function shouldRetry(failureCount: number, error: unknown): boolean {
  if (error instanceof ApiError && error.status >= 400 && error.status < 500) return false
  return failureCount < MAX_RETRIES
}

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { staleTime: STALE_TIME_MS, retry: shouldRetry, refetchOnWindowFocus: false },
    },
  })
}
