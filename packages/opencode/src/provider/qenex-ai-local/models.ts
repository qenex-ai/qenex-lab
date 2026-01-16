/**
 * QENEX AI Local Model Definitions
 *
 * Predefined model configurations for popular local LLMs that work with llama.cpp.
 * These models are auto-discovered or can be manually configured.
 */

/**
 * Model definition for QENEX AI Local provider
 */
export interface QenexLocalModel {
  /** Model identifier */
  id: string
  /** Human-readable name */
  name: string
  /** Model family (e.g., llama, mistral, qwen) */
  family: string
  /** Context window size in tokens */
  contextSize: number
  /** Maximum output tokens */
  maxOutput: number
  /** Whether the model supports tool/function calling */
  toolCall: boolean
  /** Whether the model supports vision/image input */
  vision: boolean
  /** Whether the model has reasoning capabilities */
  reasoning: boolean
  /** Whether temperature control is available */
  temperature: boolean
  /** Model description */
  description?: string
}

/**
 * Predefined models that are commonly used with llama.cpp
 * These serve as templates when auto-discovery finds a matching model
 */
export const QENEX_LOCAL_MODELS: Record<string, QenexLocalModel> = {
  // Llama 3.x models
  "llama-3.3-70b": {
    id: "llama-3.3-70b",
    name: "Llama 3.3 70B",
    family: "llama",
    contextSize: 131072,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Meta's Llama 3.3 70B - powerful open-source model",
  },
  "llama-3.2-90b-vision": {
    id: "llama-3.2-90b-vision",
    name: "Llama 3.2 90B Vision",
    family: "llama",
    contextSize: 131072,
    maxOutput: 8192,
    toolCall: true,
    vision: true,
    reasoning: false,
    temperature: true,
    description: "Meta's Llama 3.2 90B with vision capabilities",
  },
  "llama-3.2-11b-vision": {
    id: "llama-3.2-11b-vision",
    name: "Llama 3.2 11B Vision",
    family: "llama",
    contextSize: 131072,
    maxOutput: 8192,
    toolCall: true,
    vision: true,
    reasoning: false,
    temperature: true,
    description: "Meta's Llama 3.2 11B with vision capabilities",
  },
  "llama-3.1-405b": {
    id: "llama-3.1-405b",
    name: "Llama 3.1 405B",
    family: "llama",
    contextSize: 131072,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Meta's largest Llama 3.1 model",
  },
  "llama-3.1-70b": {
    id: "llama-3.1-70b",
    name: "Llama 3.1 70B",
    family: "llama",
    contextSize: 131072,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Meta's Llama 3.1 70B model",
  },
  "llama-3.1-8b": {
    id: "llama-3.1-8b",
    name: "Llama 3.1 8B",
    family: "llama",
    contextSize: 131072,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Meta's Llama 3.1 8B - efficient local model",
  },

  // Qwen models
  "qwen2.5-72b": {
    id: "qwen2.5-72b",
    name: "Qwen 2.5 72B",
    family: "qwen",
    contextSize: 131072,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Alibaba's Qwen 2.5 72B model",
  },
  "qwen2.5-coder-32b": {
    id: "qwen2.5-coder-32b",
    name: "Qwen 2.5 Coder 32B",
    family: "qwen",
    contextSize: 131072,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Alibaba's Qwen 2.5 Coder - optimized for code",
  },
  "qwen2.5-14b": {
    id: "qwen2.5-14b",
    name: "Qwen 2.5 14B",
    family: "qwen",
    contextSize: 131072,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Alibaba's Qwen 2.5 14B model",
  },
  "qwen2.5-7b": {
    id: "qwen2.5-7b",
    name: "Qwen 2.5 7B",
    family: "qwen",
    contextSize: 131072,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Alibaba's Qwen 2.5 7B - efficient model",
  },
  "qwq-32b": {
    id: "qwq-32b",
    name: "QwQ 32B",
    family: "qwen",
    contextSize: 131072,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: true,
    temperature: true,
    description: "Alibaba's QwQ 32B - reasoning-focused model",
  },

  // DeepSeek models
  "deepseek-r1": {
    id: "deepseek-r1",
    name: "DeepSeek R1",
    family: "deepseek",
    contextSize: 65536,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: true,
    temperature: true,
    description: "DeepSeek R1 - advanced reasoning model",
  },
  "deepseek-v3": {
    id: "deepseek-v3",
    name: "DeepSeek V3",
    family: "deepseek",
    contextSize: 65536,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "DeepSeek V3 - latest DeepSeek model",
  },
  "deepseek-coder-v2": {
    id: "deepseek-coder-v2",
    name: "DeepSeek Coder V2",
    family: "deepseek",
    contextSize: 65536,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "DeepSeek Coder V2 - optimized for code",
  },

  // Mistral models
  "mistral-large": {
    id: "mistral-large",
    name: "Mistral Large",
    family: "mistral",
    contextSize: 131072,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Mistral's largest model",
  },
  "mistral-nemo": {
    id: "mistral-nemo",
    name: "Mistral Nemo",
    family: "mistral",
    contextSize: 131072,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Mistral Nemo - efficient 12B model",
  },
  codestral: {
    id: "codestral",
    name: "Codestral",
    family: "mistral",
    contextSize: 32768,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Mistral's code-focused model",
  },

  // Phi models
  "phi-4": {
    id: "phi-4",
    name: "Phi-4",
    family: "phi",
    contextSize: 16384,
    maxOutput: 4096,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Microsoft's Phi-4 - efficient small model",
  },
  "phi-3.5-mini": {
    id: "phi-3.5-mini",
    name: "Phi-3.5 Mini",
    family: "phi",
    contextSize: 131072,
    maxOutput: 4096,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Microsoft's Phi-3.5 Mini - compact but capable",
  },

  // Gemma models
  "gemma-2-27b": {
    id: "gemma-2-27b",
    name: "Gemma 2 27B",
    family: "gemma",
    contextSize: 8192,
    maxOutput: 4096,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Google's Gemma 2 27B model",
  },
  "gemma-2-9b": {
    id: "gemma-2-9b",
    name: "Gemma 2 9B",
    family: "gemma",
    contextSize: 8192,
    maxOutput: 4096,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Google's Gemma 2 9B model",
  },

  // Command R models
  "command-r-plus": {
    id: "command-r-plus",
    name: "Command R+",
    family: "command",
    contextSize: 131072,
    maxOutput: 8192,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Cohere's Command R+ - enterprise RAG model",
  },

  // Fallback/default model
  default: {
    id: "default",
    name: "Local Model",
    family: "unknown",
    contextSize: 4096,
    maxOutput: 2048,
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: "Auto-detected local model",
  },
}

/**
 * Try to match a model name from llama.cpp to a known model template
 */
export function matchModel(modelName: string): QenexLocalModel | null {
  const normalized = modelName.toLowerCase()

  // Direct match
  if (QENEX_LOCAL_MODELS[normalized]) {
    return QENEX_LOCAL_MODELS[normalized]
  }

  // Pattern matching for common model naming conventions
  const patterns: Array<{ pattern: RegExp; modelId: string }> = [
    // Llama patterns
    { pattern: /llama[-_]?3\.3[-_]?70b/i, modelId: "llama-3.3-70b" },
    { pattern: /llama[-_]?3\.2[-_]?90b[-_]?vision/i, modelId: "llama-3.2-90b-vision" },
    { pattern: /llama[-_]?3\.2[-_]?11b[-_]?vision/i, modelId: "llama-3.2-11b-vision" },
    { pattern: /llama[-_]?3\.1[-_]?405b/i, modelId: "llama-3.1-405b" },
    { pattern: /llama[-_]?3\.1[-_]?70b/i, modelId: "llama-3.1-70b" },
    { pattern: /llama[-_]?3\.1[-_]?8b/i, modelId: "llama-3.1-8b" },

    // Qwen patterns
    { pattern: /qwen[-_]?2\.5[-_]?72b/i, modelId: "qwen2.5-72b" },
    { pattern: /qwen[-_]?2\.5[-_]?coder[-_]?32b/i, modelId: "qwen2.5-coder-32b" },
    { pattern: /qwen[-_]?2\.5[-_]?14b/i, modelId: "qwen2.5-14b" },
    { pattern: /qwen[-_]?2\.5[-_]?7b/i, modelId: "qwen2.5-7b" },
    { pattern: /qwq[-_]?32b/i, modelId: "qwq-32b" },

    // DeepSeek patterns
    { pattern: /deepseek[-_]?r1/i, modelId: "deepseek-r1" },
    { pattern: /deepseek[-_]?v3/i, modelId: "deepseek-v3" },
    { pattern: /deepseek[-_]?coder[-_]?v2/i, modelId: "deepseek-coder-v2" },

    // Mistral patterns
    { pattern: /mistral[-_]?large/i, modelId: "mistral-large" },
    { pattern: /mistral[-_]?nemo/i, modelId: "mistral-nemo" },
    { pattern: /codestral/i, modelId: "codestral" },

    // Phi patterns
    { pattern: /phi[-_]?4/i, modelId: "phi-4" },
    { pattern: /phi[-_]?3\.5[-_]?mini/i, modelId: "phi-3.5-mini" },

    // Gemma patterns
    { pattern: /gemma[-_]?2[-_]?27b/i, modelId: "gemma-2-27b" },
    { pattern: /gemma[-_]?2[-_]?9b/i, modelId: "gemma-2-9b" },

    // Command patterns
    { pattern: /command[-_]?r[-_]?\+|command[-_]?r[-_]?plus/i, modelId: "command-r-plus" },
  ]

  for (const { pattern, modelId } of patterns) {
    if (pattern.test(normalized)) {
      return QENEX_LOCAL_MODELS[modelId]
    }
  }

  return null
}

/**
 * Create a model definition for an unknown model
 */
export function createUnknownModel(modelName: string, contextSize: number = 4096): QenexLocalModel {
  return {
    id: modelName,
    name: modelName,
    family: "unknown",
    contextSize,
    maxOutput: Math.min(contextSize / 4, 8192),
    toolCall: true,
    vision: false,
    reasoning: false,
    temperature: true,
    description: `Local model: ${modelName}`,
  }
}

/**
 * Get default models for the QENEX AI Local provider
 * Returns a subset of commonly used models
 */
export function getDefaultModels(): QenexLocalModel[] {
  return [
    QENEX_LOCAL_MODELS["llama-3.1-8b"],
    QENEX_LOCAL_MODELS["qwen2.5-7b"],
    QENEX_LOCAL_MODELS["phi-4"],
    QENEX_LOCAL_MODELS["mistral-nemo"],
    QENEX_LOCAL_MODELS["gemma-2-9b"],
  ]
}

/**
 * Infer model family from model name
 */
export function inferFamily(modelName: string): string {
  const normalized = modelName.toLowerCase()

  if (normalized.includes("llama")) return "llama"
  if (normalized.includes("qwen") || normalized.includes("qwq")) return "qwen"
  if (normalized.includes("deepseek")) return "deepseek"
  if (normalized.includes("mistral") || normalized.includes("codestral")) return "mistral"
  if (normalized.includes("phi")) return "phi"
  if (normalized.includes("gemma")) return "gemma"
  if (normalized.includes("command")) return "command"
  if (normalized.includes("vicuna")) return "vicuna"
  if (normalized.includes("falcon")) return "falcon"
  if (normalized.includes("mpt")) return "mpt"
  if (normalized.includes("starcoder")) return "starcoder"

  return "unknown"
}
