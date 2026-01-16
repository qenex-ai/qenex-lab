/**
 * QENEX AI Local Provider
 *
 * OpenAI-compatible provider for llama.cpp server. Provides seamless integration
 * with local LLMs running on llama.cpp, with automatic model discovery and
 * health checking.
 */

import type { LanguageModelV2 } from "@ai-sdk/provider"
import { OpenAICompatibleChatLanguageModel } from "@ai-sdk/openai-compatible"
import { withoutTrailingSlash, withUserAgentSuffix, type FetchFunction } from "@ai-sdk/provider-utils"
import { LlamaCppClient } from "./llama-cpp-client"
import { matchModel, createUnknownModel, type QenexLocalModel } from "./models"
import { Log } from "../../util/log"

const log = Log.create({ service: "qenex-ai-local" })

const VERSION = "1.0.0"

/**
 * Settings for creating a QENEX AI Local provider
 */
export interface QenexAILocalProviderSettings {
  /**
   * Base URL for the llama.cpp server
   * @default "http://localhost:8080"
   */
  baseURL?: string

  /**
   * API key for authentication (optional, llama.cpp doesn't require auth by default)
   */
  apiKey?: string

  /**
   * Custom headers to include in requests
   */
  headers?: Record<string, string>

  /**
   * Custom fetch implementation
   */
  fetch?: FetchFunction

  /**
   * Provider name for identification
   * @default "qenex-ai-local"
   */
  name?: string

  /**
   * Request timeout in milliseconds
   * @default 300000 (5 minutes)
   */
  timeout?: number

  /**
   * Whether to auto-discover models from the server
   * @default true
   */
  autoDiscover?: boolean
}

/**
 * QENEX AI Local provider interface
 */
export interface QenexAILocalProvider {
  (modelId: string): LanguageModelV2
  chat(modelId: string): LanguageModelV2
  languageModel(modelId: string): LanguageModelV2

  /**
   * Check if the llama.cpp server is available
   */
  isAvailable(): Promise<boolean>

  /**
   * Discover models from the server
   */
  discoverModels(): Promise<QenexLocalModel[]>

  /**
   * Get the llama.cpp client for direct access
   */
  getClient(): LlamaCppClient
}

/**
 * Create a QENEX AI Local provider instance for llama.cpp
 */
export function createQenexAILocal(options: QenexAILocalProviderSettings = {}): QenexAILocalProvider {
  const baseURL = withoutTrailingSlash(options.baseURL ?? "http://localhost:8080")
  const providerName = options.name ?? "qenex-ai-local"
  const timeout = options.timeout ?? 300000 // 5 minutes default for local inference

  // Create llama.cpp client for health checks and discovery
  const client = new LlamaCppClient({
    baseURL,
    timeout: 5000, // Short timeout for health checks
    apiKey: options.apiKey,
    headers: options.headers,
  })

  // Build headers
  const headers: Record<string, string> = {
    ...(options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {}),
    ...options.headers,
  }

  const getHeaders = () => withUserAgentSuffix(headers, `qenex-ai-local/${VERSION}`)

  // Use provided fetch or default to global fetch
  const fetchFn = options.fetch

  /**
   * Create a chat language model for the given model ID
   */
  const createChatModel = (modelId: string): LanguageModelV2 => {
    log.info("creating chat model", { modelId, baseURL })

    return new OpenAICompatibleChatLanguageModel(modelId, {
      provider: `${providerName}.chat`,
      headers: getHeaders,
      url: ({ path }) => `${baseURL}${path}`,
      fetch: fetchFn,
    })
  }

  /**
   * Check if the server is available
   */
  const isAvailable = async (): Promise<boolean> => {
    return client.isAvailable()
  }

  /**
   * Discover models from the server
   */
  const discoverModels = async (): Promise<QenexLocalModel[]> => {
    const discovery = await client.discover()

    if (!discovery.available) {
      log.warn("llama.cpp server not available", { baseURL })
      return []
    }

    const models: QenexLocalModel[] = []

    // Process discovered models
    for (const serverModel of discovery.models) {
      const matched = matchModel(serverModel.id)
      if (matched) {
        models.push({
          ...matched,
          id: serverModel.id, // Use the actual server model ID
          contextSize: discovery.contextSize,
        })
      } else {
        models.push(createUnknownModel(serverModel.id, discovery.contextSize))
      }
    }

    // If no models found but server is running, use current model from props
    if (models.length === 0 && discovery.currentModel) {
      const matched = matchModel(discovery.currentModel)
      if (matched) {
        models.push({
          ...matched,
          id: discovery.currentModel,
          contextSize: discovery.contextSize,
        })
      } else {
        models.push(createUnknownModel(discovery.currentModel, discovery.contextSize))
      }
    }

    log.info("discovered models", { count: models.length, models: models.map((m) => m.id) })
    return models
  }

  /**
   * Get the llama.cpp client
   */
  const getClient = (): LlamaCppClient => client

  // Create the provider function
  const provider = function (modelId: string): LanguageModelV2 {
    return createChatModel(modelId)
  }

  // Attach methods
  provider.chat = createChatModel
  provider.languageModel = createChatModel
  provider.isAvailable = isAvailable
  provider.discoverModels = discoverModels
  provider.getClient = getClient

  return provider as QenexAILocalProvider
}

/**
 * Default QENEX AI Local provider instance
 */
export const qenexAILocal = createQenexAILocal()
