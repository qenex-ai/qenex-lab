/**
 * llama.cpp Server Client
 *
 * Client for interacting with llama.cpp server API to check health,
 * discover loaded models, and get server capabilities.
 */

import { Log } from "../../util/log"

const log = Log.create({ service: "qenex-ai-local:llama-cpp" })

/**
 * Response from llama.cpp /health endpoint
 */
export interface LlamaCppHealthResponse {
  status: "ok" | "loading model" | "error" | "no slot available"
  slots_idle?: number
  slots_processing?: number
  error?: string
}

/**
 * Model info from llama.cpp /v1/models endpoint
 */
export interface LlamaCppModel {
  id: string
  object: "model"
  created: number
  owned_by: string
}

/**
 * Response from llama.cpp /v1/models endpoint
 */
export interface LlamaCppModelsResponse {
  object: "list"
  data: LlamaCppModel[]
}

/**
 * Response from llama.cpp /props endpoint (server properties)
 */
export interface LlamaCppPropsResponse {
  default_generation_settings?: {
    n_ctx?: number
    n_predict?: number
    model?: string
    seed?: number
    temperature?: number
    top_k?: number
    top_p?: number
    min_p?: number
    repeat_penalty?: number
    presence_penalty?: number
    frequency_penalty?: number
  }
  total_slots?: number
}

/**
 * Options for creating a LlamaCppClient
 */
export interface LlamaCppClientOptions {
  /** Base URL of the llama.cpp server (default: http://localhost:8080) */
  baseURL?: string
  /** Request timeout in milliseconds (default: 5000) */
  timeout?: number
  /** Custom headers to send with requests */
  headers?: Record<string, string>
  /** API key for authentication (optional, llama.cpp doesn't require auth by default) */
  apiKey?: string
}

/**
 * Client for llama.cpp server API
 */
export class LlamaCppClient {
  private baseURL: string
  private timeout: number
  private headers: Record<string, string>

  constructor(options: LlamaCppClientOptions = {}) {
    this.baseURL = (options.baseURL ?? "http://localhost:8080").replace(/\/$/, "")
    this.timeout = options.timeout ?? 5000
    this.headers = {
      "Content-Type": "application/json",
      ...(options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {}),
      ...options.headers,
    }
  }

  /**
   * Check if the llama.cpp server is running and healthy
   */
  async health(): Promise<LlamaCppHealthResponse> {
    try {
      const response = await fetch(`${this.baseURL}/health`, {
        method: "GET",
        headers: this.headers,
        signal: AbortSignal.timeout(this.timeout),
      })

      if (!response.ok) {
        return {
          status: "error",
          error: `HTTP ${response.status}: ${response.statusText}`,
        }
      }

      return (await response.json()) as LlamaCppHealthResponse
    } catch (error) {
      log.warn("llama.cpp health check failed", { error, baseURL: this.baseURL })
      return {
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error",
      }
    }
  }

  /**
   * Check if the server is available and ready
   */
  async isAvailable(): Promise<boolean> {
    const health = await this.health()
    return health.status === "ok"
  }

  /**
   * Get list of available models from the server
   */
  async listModels(): Promise<LlamaCppModel[]> {
    try {
      const response = await fetch(`${this.baseURL}/v1/models`, {
        method: "GET",
        headers: this.headers,
        signal: AbortSignal.timeout(this.timeout),
      })

      if (!response.ok) {
        log.warn("Failed to list models", {
          status: response.status,
          statusText: response.statusText,
        })
        return []
      }

      const data = (await response.json()) as LlamaCppModelsResponse
      return data.data ?? []
    } catch (error) {
      log.warn("Failed to list models from llama.cpp", { error })
      return []
    }
  }

  /**
   * Get server properties including default generation settings
   */
  async getProps(): Promise<LlamaCppPropsResponse | null> {
    try {
      const response = await fetch(`${this.baseURL}/props`, {
        method: "GET",
        headers: this.headers,
        signal: AbortSignal.timeout(this.timeout),
      })

      if (!response.ok) {
        return null
      }

      return (await response.json()) as LlamaCppPropsResponse
    } catch (error) {
      log.warn("Failed to get llama.cpp props", { error })
      return null
    }
  }

  /**
   * Get the currently loaded model name from props
   */
  async getCurrentModel(): Promise<string | null> {
    const props = await this.getProps()
    return props?.default_generation_settings?.model ?? null
  }

  /**
   * Get context size from server props
   */
  async getContextSize(): Promise<number> {
    const props = await this.getProps()
    return props?.default_generation_settings?.n_ctx ?? 4096
  }

  /**
   * Discover all available models and their capabilities
   */
  async discover(): Promise<{
    available: boolean
    models: LlamaCppModel[]
    currentModel: string | null
    contextSize: number
  }> {
    const available = await this.isAvailable()
    if (!available) {
      return {
        available: false,
        models: [],
        currentModel: null,
        contextSize: 4096,
      }
    }

    const [models, currentModel, contextSize] = await Promise.all([
      this.listModels(),
      this.getCurrentModel(),
      this.getContextSize(),
    ])

    return {
      available: true,
      models,
      currentModel,
      contextSize,
    }
  }
}

/**
 * Default llama.cpp client instance
 */
export const defaultClient = new LlamaCppClient()
