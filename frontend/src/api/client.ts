import type { ApiErrorBody } from './types'

const DEFAULT_BASE = '/api/v1'

export const API_BASE: string = (import.meta.env.VITE_API_BASE ?? '').trim() || DEFAULT_BASE

export type QueryParams = Record<string, string | number | undefined | null>

export class ApiError extends Error {
  readonly status: number
  readonly code: string

  constructor(status: number, code: string, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

/** Absolute URL for an API path; relative bases resolve against the page origin. */
export function buildUrl(path: string, params: QueryParams = {}): string {
  const base = API_BASE.replace(/\/+$/, '')
  const url = new URL(`${base}${path}`, window.location.origin)
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.set(key, String(value))
    }
  }
  return url.toString()
}

function isErrorBody(body: unknown): body is ApiErrorBody {
  if (typeof body !== 'object' || body === null || !('error' in body)) return false
  const err = (body as { error: unknown }).error
  return typeof err === 'object' && err !== null && 'message' in err
}

export function fallbackMessage(status: number): string {
  if (status === 404) return 'Not found.'
  if (status === 422) return 'The request was not valid.'
  if (status === 429) return 'Too many requests. Please wait a moment and try again.'
  if (status >= 500) return 'The prediction server had a problem. Please try again shortly.'
  return `Request failed (HTTP ${status}).`
}

async function readJson(res: Response): Promise<unknown> {
  const text = await res.text()
  if (!text) return null
  try {
    return JSON.parse(text) as unknown
  } catch {
    return undefined
  }
}

function isAbort(err: unknown): boolean {
  return typeof err === 'object' && err !== null && (err as { name?: unknown }).name === 'AbortError'
}

export async function apiGet<T>(path: string, params?: QueryParams, signal?: AbortSignal): Promise<T> {
  let res: Response
  try {
    res = await fetch(buildUrl(path, params), { signal, headers: { Accept: 'application/json' } })
  } catch (err) {
    if (isAbort(err)) throw err
    throw new ApiError(0, 'NETWORK_ERROR', 'Could not reach the prediction server. Check your connection and try again.')
  }

  const body = await readJson(res)
  if (!res.ok) {
    if (isErrorBody(body)) {
      throw new ApiError(res.status, String(body.error.code ?? `HTTP_${res.status}`), String(body.error.message))
    }
    throw new ApiError(res.status, `HTTP_${res.status}`, fallbackMessage(res.status))
  }
  if (body === undefined) {
    throw new ApiError(res.status, 'INVALID_RESPONSE', 'The server sent an unexpected response.')
  }
  return body as T
}

export function errorMessage(err: unknown): string {
  if (err instanceof Error && err.message) return err.message
  return 'Something went wrong.'
}
