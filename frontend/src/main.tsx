import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import './index.css'

async function enableMocks(): Promise<void> {
  if (import.meta.env.VITE_USE_MOCKS !== '1') return
  const { startMockWorker } = await import('./mocks/browser')
  await startMockWorker()
}

const rootEl = document.getElementById('root')
if (!rootEl) throw new Error('Root element #root not found')

void enableMocks().finally(() => {
  createRoot(rootEl).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
})
