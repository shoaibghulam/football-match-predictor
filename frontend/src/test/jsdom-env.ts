/**
 * jsdom test environment that keeps Node's AbortController/AbortSignal.
 * Node's fetch/Request (undici) reject jsdom's AbortSignal, which breaks
 * TanStack Query cancellation and React Router navigations under jsdom.
 */
import type { Environment } from 'vitest/environments'
import { builtinEnvironments } from 'vitest/environments'

const jsdom = builtinEnvironments.jsdom

const env: Environment = {
  ...jsdom,
  name: 'jsdom-node-abort',
  async setup(global, options) {
    const NodeAbortController = global.AbortController
    const NodeAbortSignal = global.AbortSignal
    const result = await jsdom.setup(global, options)
    global.AbortController = NodeAbortController
    global.AbortSignal = NodeAbortSignal
    return result
  },
}

export default env
