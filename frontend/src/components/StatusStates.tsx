import type { ReactNode } from 'react'
import { errorMessage } from '../api/client'

export function CardSkeleton() {
  return (
    <div className="card animate-pulse p-4" aria-hidden="true">
      <div className="mb-3 h-3 w-16 rounded bg-slate-200 dark:bg-slate-800" />
      <div className="mb-2 h-4 w-3/4 rounded bg-slate-200 dark:bg-slate-800" />
      <div className="mb-4 h-4 w-2/3 rounded bg-slate-200 dark:bg-slate-800" />
      <div className="mb-3 h-2.5 w-full rounded-full bg-slate-200 dark:bg-slate-800" />
      <div className="grid grid-cols-3 gap-2">
        <div className="h-10 rounded bg-slate-200 dark:bg-slate-800" />
        <div className="h-10 rounded bg-slate-200 dark:bg-slate-800" />
        <div className="h-10 rounded bg-slate-200 dark:bg-slate-800" />
      </div>
    </div>
  )
}

export function LoadingList({ count = 4, label = 'Loading matches' }: { count?: number; label?: string }) {
  return (
    <div role="status" aria-live="polite">
      <span className="sr-only">{label}…</span>
      <div className="grid gap-3 sm:grid-cols-2">
        {Array.from({ length: count }, (_, i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
    </div>
  )
}

interface EmptyStateProps {
  title: string
  children?: ReactNode
}

export function EmptyState({ title, children }: EmptyStateProps) {
  return (
    <div className="card flex flex-col items-center gap-2 px-6 py-10 text-center" role="status">
      <span aria-hidden="true" className="text-3xl">
        ⚽
      </span>
      <p className="font-semibold">{title}</p>
      {children && <div className="muted text-sm">{children}</div>}
    </div>
  )
}

interface ErrorStateProps {
  error: unknown
  onRetry?: () => void
  title?: string
}

export function ErrorState({ error, onRetry, title = 'Could not load data' }: ErrorStateProps) {
  return (
    <div
      role="alert"
      className="flex flex-col items-start gap-3 rounded-xl border border-red-300 bg-red-50 p-4 text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-100"
    >
      <div>
        <p className="font-semibold">{title}</p>
        <p className="text-sm">{errorMessage(error)}</p>
      </div>
      {onRetry && (
        <button type="button" className="btn" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  )
}
