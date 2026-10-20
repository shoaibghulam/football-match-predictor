import { setupWorker } from 'msw/browser'
import { handlers } from './handlers'

/** Starts the in-browser mock API (enabled with VITE_USE_MOCKS=1). */
export async function startMockWorker(): Promise<void> {
  const worker = setupWorker(...handlers)
  await worker.start({ onUnhandledRequest: 'bypass', quiet: true })
}
