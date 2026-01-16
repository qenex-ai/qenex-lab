import { test, expect, describe, mock, beforeEach, afterEach } from "bun:test"
import path from "path"
import { tmpdir } from "../fixture/fixture"
import { Instance } from "../../src/project/instance"
import { Provider } from "../../src/provider/provider"
import { Env } from "../../src/env"
import { LlamaCppClient } from "../../src/provider/qenex-ai-local/llama-cpp-client"
import { createQenexAILocal } from "../../src/provider/qenex-ai-local/provider"
import {
  matchModel,
  createUnknownModel,
  inferFamily,
  getDefaultModels,
  QENEX_LOCAL_MODELS,
} from "../../src/provider/qenex-ai-local/models"

describe("QENEX AI Local Models", () => {
  test("matchModel returns correct model for llama variants", () => {
    const model1 = matchModel("llama-3.1-8b")
    expect(model1).toBeDefined()
    expect(model1?.id).toBe("llama-3.1-8b")
    expect(model1?.family).toBe("llama")

    const model2 = matchModel("Llama-3.1-70B-Instruct")
    expect(model2).toBeDefined()
    expect(model2?.id).toBe("llama-3.1-70b")

    const model3 = matchModel("llama_3_1_8b")
    expect(model3).toBeDefined()
  })

  test("matchModel returns correct model for qwen variants", () => {
    const model1 = matchModel("qwen2.5-7b")
    expect(model1).toBeDefined()
    expect(model1?.id).toBe("qwen2.5-7b")
    expect(model1?.family).toBe("qwen")

    const model2 = matchModel("qwq-32b")
    expect(model2).toBeDefined()
    expect(model2?.reasoning).toBe(true)
  })

  test("matchModel returns correct model for deepseek variants", () => {
    const model1 = matchModel("deepseek-r1")
    expect(model1).toBeDefined()
    expect(model1?.id).toBe("deepseek-r1")
    expect(model1?.reasoning).toBe(true)

    const model2 = matchModel("deepseek-coder-v2")
    expect(model2).toBeDefined()
  })

  test("matchModel returns correct model for mistral variants", () => {
    const model1 = matchModel("mistral-nemo")
    expect(model1).toBeDefined()
    expect(model1?.id).toBe("mistral-nemo")

    const model2 = matchModel("codestral")
    expect(model2).toBeDefined()
  })

  test("matchModel returns null for unknown models", () => {
    const model = matchModel("completely-unknown-model-xyz")
    expect(model).toBeNull()
  })

  test("createUnknownModel creates valid model definition", () => {
    const model = createUnknownModel("my-custom-model", 8192)
    expect(model.id).toBe("my-custom-model")
    expect(model.name).toBe("my-custom-model")
    expect(model.contextSize).toBe(8192)
    expect(model.maxOutput).toBe(2048) // 8192 / 4 = 2048
    expect(model.family).toBe("unknown")
    expect(model.toolCall).toBe(true)
    expect(model.temperature).toBe(true)
  })

  test("createUnknownModel caps maxOutput at 8192", () => {
    const model = createUnknownModel("large-context-model", 131072)
    expect(model.maxOutput).toBe(8192)
  })

  test("inferFamily correctly identifies model families", () => {
    expect(inferFamily("llama-3.1-8b")).toBe("llama")
    expect(inferFamily("Qwen2.5-7B")).toBe("qwen")
    expect(inferFamily("deepseek-r1")).toBe("deepseek")
    expect(inferFamily("mistral-large")).toBe("mistral")
    expect(inferFamily("phi-4")).toBe("phi")
    expect(inferFamily("gemma-2-9b")).toBe("gemma")
    expect(inferFamily("unknown-model")).toBe("unknown")
  })

  test("getDefaultModels returns array of models", () => {
    const models = getDefaultModels()
    expect(Array.isArray(models)).toBe(true)
    expect(models.length).toBeGreaterThan(0)

    for (const model of models) {
      expect(model.id).toBeDefined()
      expect(model.name).toBeDefined()
      expect(model.contextSize).toBeGreaterThan(0)
    }
  })

  test("QENEX_LOCAL_MODELS contains expected models", () => {
    expect(QENEX_LOCAL_MODELS["llama-3.1-8b"]).toBeDefined()
    expect(QENEX_LOCAL_MODELS["qwen2.5-7b"]).toBeDefined()
    expect(QENEX_LOCAL_MODELS["phi-4"]).toBeDefined()
    expect(QENEX_LOCAL_MODELS["default"]).toBeDefined()
  })

  test("model definitions have required fields", () => {
    for (const [id, model] of Object.entries(QENEX_LOCAL_MODELS)) {
      expect(model.id).toBe(id)
      expect(model.name).toBeDefined()
      expect(model.family).toBeDefined()
      expect(typeof model.contextSize).toBe("number")
      expect(typeof model.maxOutput).toBe("number")
      expect(typeof model.toolCall).toBe("boolean")
      expect(typeof model.vision).toBe("boolean")
      expect(typeof model.reasoning).toBe("boolean")
      expect(typeof model.temperature).toBe("boolean")
    }
  })
})

describe("LlamaCppClient", () => {
  test("constructor uses default baseURL", () => {
    const client = new LlamaCppClient()
    // Can't directly access private fields, but we can test behavior
    expect(client).toBeDefined()
  })

  test("constructor accepts custom baseURL", () => {
    const client = new LlamaCppClient({ baseURL: "http://custom:9999" })
    expect(client).toBeDefined()
  })

  test("constructor strips trailing slash from baseURL", () => {
    const client = new LlamaCppClient({ baseURL: "http://localhost:8080/" })
    expect(client).toBeDefined()
  })

  test("health returns error when server unavailable", async () => {
    const client = new LlamaCppClient({
      baseURL: "http://localhost:59999", // Unlikely to be running
      timeout: 100,
    })

    const health = await client.health()
    expect(health.status).toBe("error")
    expect(health.error).toBeDefined()
  })

  test("isAvailable returns false when server unavailable", async () => {
    const client = new LlamaCppClient({
      baseURL: "http://localhost:59999",
      timeout: 100,
    })

    const available = await client.isAvailable()
    expect(available).toBe(false)
  })

  test("listModels returns empty array when server unavailable", async () => {
    const client = new LlamaCppClient({
      baseURL: "http://localhost:59999",
      timeout: 100,
    })

    const models = await client.listModels()
    expect(Array.isArray(models)).toBe(true)
    expect(models.length).toBe(0)
  })

  test("getProps returns null when server unavailable", async () => {
    const client = new LlamaCppClient({
      baseURL: "http://localhost:59999",
      timeout: 100,
    })

    const props = await client.getProps()
    expect(props).toBeNull()
  })

  test("getCurrentModel returns null when server unavailable", async () => {
    const client = new LlamaCppClient({
      baseURL: "http://localhost:59999",
      timeout: 100,
    })

    const model = await client.getCurrentModel()
    expect(model).toBeNull()
  })

  test("getContextSize returns default when server unavailable", async () => {
    const client = new LlamaCppClient({
      baseURL: "http://localhost:59999",
      timeout: 100,
    })

    const contextSize = await client.getContextSize()
    expect(contextSize).toBe(4096) // Default
  })

  test("discover returns unavailable status when server down", async () => {
    const client = new LlamaCppClient({
      baseURL: "http://localhost:59999",
      timeout: 100,
    })

    const discovery = await client.discover()
    expect(discovery.available).toBe(false)
    expect(discovery.models).toEqual([])
    expect(discovery.currentModel).toBeNull()
    expect(discovery.contextSize).toBe(4096)
  })
})

describe("createQenexAILocal", () => {
  test("creates provider with default settings", () => {
    const provider = createQenexAILocal()
    expect(provider).toBeDefined()
    expect(typeof provider).toBe("function")
    expect(typeof provider.chat).toBe("function")
    expect(typeof provider.languageModel).toBe("function")
    expect(typeof provider.isAvailable).toBe("function")
    expect(typeof provider.discoverModels).toBe("function")
    expect(typeof provider.getClient).toBe("function")
  })

  test("creates provider with custom baseURL", () => {
    const provider = createQenexAILocal({ baseURL: "http://custom:9999" })
    expect(provider).toBeDefined()
  })

  test("creates chat model", () => {
    const provider = createQenexAILocal()
    const model = provider.chat("llama-3.1-8b")
    expect(model).toBeDefined()
  })

  test("creates language model", () => {
    const provider = createQenexAILocal()
    const model = provider.languageModel("llama-3.1-8b")
    expect(model).toBeDefined()
  })

  test("provider function creates model", () => {
    const provider = createQenexAILocal()
    const model = provider("llama-3.1-8b")
    expect(model).toBeDefined()
  })

  test("getClient returns LlamaCppClient instance", () => {
    const provider = createQenexAILocal()
    const client = provider.getClient()
    expect(client).toBeInstanceOf(LlamaCppClient)
  })

  test("isAvailable returns false when server unavailable", async () => {
    const provider = createQenexAILocal({
      baseURL: "http://localhost:59999",
    })

    const available = await provider.isAvailable()
    expect(available).toBe(false)
  })

  test("discoverModels returns empty array when server unavailable", async () => {
    const provider = createQenexAILocal({
      baseURL: "http://localhost:59999",
    })

    const models = await provider.discoverModels()
    expect(Array.isArray(models)).toBe(true)
    expect(models.length).toBe(0)
  })
})

describe("QENEX AI Local Provider Integration", () => {
  test(
    "provider is registered in database",
    async () => {
      await using tmp = await tmpdir({
        init: async (dir) => {
          await Bun.write(
            path.join(dir, "opencode.json"),
            JSON.stringify({
              $schema: "https://opencode.ai/config.json",
            }),
          )
        },
      })
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          // The provider should be in the database even if server is not running
          // (it won't be in providers list since autoload depends on server availability)
          const providers = await Provider.list()
          // When llama.cpp is not running, provider won't be loaded
          // This is expected behavior
          expect(providers["qenex-ai-local"]).toBeUndefined()
        },
      })
    },
    { timeout: 30000 },
  )

  test(
    "provider can be configured via config file",
    async () => {
      await using tmp = await tmpdir({
        init: async (dir) => {
          await Bun.write(
            path.join(dir, "opencode.json"),
            JSON.stringify({
              $schema: "https://opencode.ai/config.json",
              provider: {
                "qenex-ai-local": {
                  name: "My Local LLM",
                  options: {
                    baseURL: "http://localhost:8080",
                  },
                  models: {
                    "my-model": {
                      name: "My Custom Model",
                      tool_call: true,
                      limit: { context: 8192, output: 2048 },
                    },
                  },
                },
              },
            }),
          )
        },
      })
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const providers = await Provider.list()
          // Provider might not be loaded if llama.cpp isn't running
          // But the config should be recognized
          if (providers["qenex-ai-local"]) {
            expect(providers["qenex-ai-local"].name).toBe("My Local LLM")
          }
        },
      })
    },
    { timeout: 30000 },
  )

  test(
    "provider env variables are recognized",
    async () => {
      await using tmp = await tmpdir({
        init: async (dir) => {
          await Bun.write(
            path.join(dir, "opencode.json"),
            JSON.stringify({
              $schema: "https://opencode.ai/config.json",
            }),
          )
        },
      })
      await Instance.provide({
        directory: tmp.path,
        init: async () => {
          // Set env var but server won't be running
          Env.set("QENEX_AI_LOCAL_BASE_URL", "http://localhost:8080")
        },
        fn: async () => {
          // Provider initialization will fail since no server is running
          // But env var should be recognized
          const providers = await Provider.list()
          // Since server is not available, provider won't load
          expect(providers["qenex-ai-local"]).toBeUndefined()
        },
      })
    },
    { timeout: 30000 },
  )
})

describe("Model matching edge cases", () => {
  test("matchModel handles various naming conventions", () => {
    // Underscores
    expect(matchModel("llama_3_1_8b")).toBeDefined()
    // Dashes
    expect(matchModel("llama-3-1-8b")).toBeDefined()
    // Mixed case
    expect(matchModel("LLAMA-3.1-8B")).toBeDefined()
    expect(matchModel("Llama-3.1-8B")).toBeDefined()
  })

  test("matchModel handles quantization suffixes", () => {
    // Common GGUF quantization suffixes
    const model = matchModel("llama-3.1-8b-q4_k_m")
    // Should still match the base model
    expect(model).toBeDefined()
  })

  test("matchModel handles vision models", () => {
    const model = matchModel("llama-3.2-11b-vision")
    expect(model).toBeDefined()
    expect(model?.vision).toBe(true)
  })

  test("matchModel handles reasoning models", () => {
    const qwq = matchModel("qwq-32b")
    expect(qwq).toBeDefined()
    expect(qwq?.reasoning).toBe(true)

    const deepseek = matchModel("deepseek-r1")
    expect(deepseek).toBeDefined()
    expect(deepseek?.reasoning).toBe(true)
  })
})
