import { setupWorker } from 'msw/browser'
import { handlers } from './handlers'
import { resultsHandlers } from './resultsHandlers'

/** Starts the in-browser mock API (enabled with VITE_USE_MOCKS=1). */
export async function startMockWorker(): Promise<void> {
  const worker = setupWorker(...handlers, ...resultsHandlers())
  await worker.start({ onUnhandledRequest: 'bypass', quiet: true })
}
