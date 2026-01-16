/**
 * QENEX AI Local Provider
 *
 * A local AI provider that integrates with llama.cpp server for running
 * local LLMs. Supports automatic model discovery, health checking, and
 * seamless integration with the OpenCode provider system.
 *
 * @module qenex-ai-local
 */

export { createQenexAILocal } from "./provider"
export type { QenexAILocalProviderSettings, QenexAILocalProvider } from "./provider"
export { LlamaCppClient } from "./llama-cpp-client"
export type { LlamaCppHealthResponse, LlamaCppModel } from "./llama-cpp-client"
export { QENEX_LOCAL_MODELS, getDefaultModels } from "./models"
export type { QenexLocalModel } from "./models"
