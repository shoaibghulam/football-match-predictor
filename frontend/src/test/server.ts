import { setupServer } from 'msw/node'
import { handlers } from '../mocks/handlers'
import { resultsHandlers } from '../mocks/resultsHandlers'

export const server = setupServer(...handlers, ...resultsHandlers())
