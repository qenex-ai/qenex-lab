export * from "./client.js"
export * from "./server.js"

import { createQenexClient } from "./client.js"
import { createQenexServer } from "./server.js"
import type { ServerOptions } from "./server.js"

export async function createQenex(options?: ServerOptions) {
  const server = await createQenexServer({
    ...options,
  })

  const client = createQenexClient({
    baseUrl: server.url,
  })

  return {
    client,
    server,
  }
}

// Legacy alias for backward compatibility
export const createOpencode = createQenex
