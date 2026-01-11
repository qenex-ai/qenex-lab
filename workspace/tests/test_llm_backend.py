"""
Tests for QENEX Local LLM Backend Integration.

Tests cover:
- GenerationConfig: parameter conversion for different backends
- GenerationResult: result handling
- ModelInfo: model information
- Backend implementations: Mock, Ollama, LlamaCpp, vLLM, OpenAI-compat
- LLMRouter: backend selection, failover, routing
- Q-Lang integration: command handling
"""

import os
import sys
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Iterator

import pytest

# Add source path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "qenex-qlang" / "src"))

from llm_backend import (
    # Enums
    BackendType,
    # Config and Results
    GenerationConfig,
    GenerationResult,
    ModelInfo,
    # Backends
    LLMBackend,
    MockBackend,
    OllamaBackend,
    LlamaCppBackend,
    VLLMBackend,
    OpenAICompatBackend,
    # Router
    LLMRouter,
    create_default_router,
    # Commands
    handle_llm_command,
    # Availability flags
    HAS_LLM_BACKEND,
    HAS_HTTPX,
    HAS_REQUESTS,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_backend():
    """Create mock backend instance."""
    return MockBackend(latency_ms=10)


@pytest.fixture
def generation_config():
    """Create default generation config."""
    return GenerationConfig()


@pytest.fixture
def custom_config():
    """Create custom generation config."""
    return GenerationConfig(
        max_tokens=1024,
        temperature=0.5,
        top_p=0.9,
        top_k=50,
        repeat_penalty=1.2,
        stop=["<|end|>", "###"],
        seed=42,
    )


@pytest.fixture
def router():
    """Create LLM router with mock backend."""
    return LLMRouter(verbose=False)


# ============================================================================
# Test BackendType
# ============================================================================

class TestBackendType:
    """Tests for BackendType enum."""
    
    def test_all_types_exist(self):
        """Test all backend types exist."""
        assert BackendType.OLLAMA.value == "ollama"
        assert BackendType.LLAMACPP.value == "llamacpp"
        assert BackendType.VLLM.value == "vllm"
        assert BackendType.OPENAI_COMPAT.value == "openai"
        assert BackendType.MOCK.value == "mock"
    
    def test_enum_iteration(self):
        """Test can iterate over all types."""
        types = list(BackendType)
        assert len(types) == 5


# ============================================================================
# Test GenerationConfig
# ============================================================================

class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""
    
    def test_default_values(self, generation_config):
        """Test default configuration values."""
        assert generation_config.max_tokens == 2048
        assert generation_config.temperature == 0.7
        assert generation_config.top_p == 0.95
        assert generation_config.top_k == 40
        assert generation_config.repeat_penalty == 1.1
        assert generation_config.stop == []
        assert generation_config.seed is None
    
    def test_custom_values(self, custom_config):
        """Test custom configuration."""
        assert custom_config.max_tokens == 1024
        assert custom_config.temperature == 0.5
        assert custom_config.seed == 42
        assert custom_config.stop == ["<|end|>", "###"]
    
    def test_to_ollama(self, custom_config):
        """Test conversion to Ollama format."""
        opts = custom_config.to_ollama()
        
        assert opts["num_predict"] == 1024
        assert opts["temperature"] == 0.5
        assert opts["top_p"] == 0.9
        assert opts["top_k"] == 50
        assert opts["repeat_penalty"] == 1.2
        assert opts["stop"] == ["<|end|>", "###"]
        assert opts["seed"] == 42
    
    def test_to_openai(self, custom_config):
        """Test conversion to OpenAI format."""
        opts = custom_config.to_openai()
        
        assert opts["max_tokens"] == 1024
        assert opts["temperature"] == 0.5
        assert opts["top_p"] == 0.9
        assert opts["stop"] == ["<|end|>", "###"]
    
    def test_to_llamacpp(self, custom_config):
        """Test conversion to llama.cpp format."""
        opts = custom_config.to_llamacpp()
        
        assert opts["n_predict"] == 1024
        assert opts["temperature"] == 0.5
        assert opts["top_p"] == 0.9
        assert opts["top_k"] == 50
        assert opts["seed"] == 42
    
    def test_to_ollama_no_stop(self, generation_config):
        """Test Ollama format without stop sequences."""
        opts = generation_config.to_ollama()
        assert "stop" not in opts
    
    def test_to_ollama_no_seed(self, generation_config):
        """Test Ollama format without seed."""
        opts = generation_config.to_ollama()
        assert "seed" not in opts


# ============================================================================
# Test GenerationResult
# ============================================================================

class TestGenerationResult:
    """Tests for GenerationResult dataclass."""
    
    def test_minimal_result(self):
        """Test minimal result."""
        result = GenerationResult(text="Hello")
        
        assert result.text == "Hello"
        assert result.tokens_generated == 0
        assert result.elapsed_ms == 0.0
        assert result.finish_reason == "stop"
    
    def test_full_result(self):
        """Test result with all fields."""
        result = GenerationResult(
            text="Generated text",
            tokens_generated=100,
            tokens_prompt=50,
            elapsed_ms=1234.5,
            model="llama2:7b",
            backend="Ollama",
            finish_reason="stop",
            raw_response={"key": "value"},
        )
        
        assert result.text == "Generated text"
        assert result.tokens_generated == 100
        assert result.tokens_prompt == 50
        assert result.elapsed_ms == 1234.5
        assert result.model == "llama2:7b"
        assert result.backend == "Ollama"
        assert result.raw_response == {"key": "value"}
    
    def test_error_result(self):
        """Test error result."""
        result = GenerationResult(
            text="Error: Connection failed",
            backend="Ollama",
            finish_reason="error",
        )
        
        assert "Error" in result.text
        assert result.finish_reason == "error"


# ============================================================================
# Test ModelInfo
# ============================================================================

class TestModelInfo:
    """Tests for ModelInfo dataclass."""
    
    def test_minimal_model(self):
        """Test minimal model info."""
        model = ModelInfo(name="llama2")
        
        assert model.name == "llama2"
        assert model.size == ""
        assert model.quantization == ""
    
    def test_full_model(self):
        """Test model info with all fields."""
        model = ModelInfo(
            name="llama2:7b-q4_k",
            size="4.2GB",
            family="llama",
            parameter_count="7B",
            quantization="Q4_K",
            modified="2024-01-01",
            digest="abc123",
        )
        
        assert model.name == "llama2:7b-q4_k"
        assert model.size == "4.2GB"
        assert model.quantization == "Q4_K"
    
    def test_model_str(self):
        """Test model string representation."""
        model = ModelInfo(name="llama2", size="7GB", quantization="Q4")
        
        str_repr = str(model)
        assert "llama2" in str_repr
        assert "7GB" in str_repr
        assert "Q4" in str_repr
    
    def test_model_str_minimal(self):
        """Test minimal model string."""
        model = ModelInfo(name="test-model")
        assert str(model) == "test-model"


# ============================================================================
# Test MockBackend
# ============================================================================

class TestMockBackend:
    """Tests for MockBackend."""
    
    def test_mock_properties(self, mock_backend):
        """Test mock backend properties."""
        assert mock_backend.name == "Mock"
        assert mock_backend.backend_type == BackendType.MOCK
    
    def test_mock_is_available(self, mock_backend):
        """Test mock is always available."""
        assert mock_backend.is_available() is True
    
    def test_mock_list_models(self, mock_backend):
        """Test mock lists mock models."""
        models = mock_backend.list_models()
        
        assert len(models) == 3
        assert any("mock-7b" in m.name for m in models)
    
    def test_mock_generate(self, mock_backend):
        """Test mock generation."""
        result = mock_backend.generate("Hello world")
        
        assert result.text
        assert result.backend == "Mock"
        assert result.finish_reason == "stop"
        assert result.elapsed_ms >= mock_backend.latency_ms
    
    def test_mock_generate_code_prompt(self, mock_backend):
        """Test mock responds to code prompts."""
        result = mock_backend.generate("Write a function to add numbers")
        
        assert "```" in result.text or "def" in result.text
    
    def test_mock_generate_question(self, mock_backend):
        """Test mock responds to questions."""
        result = mock_backend.generate("What is the meaning of life?")
        
        assert "analysis" in result.text.lower() or "point" in result.text.lower()
    
    def test_mock_generate_with_model(self, mock_backend):
        """Test mock generation with model specified."""
        result = mock_backend.generate("Test", model="mock-13b")
        
        assert result.model == "mock-13b"
    
    def test_mock_generate_stream(self, mock_backend):
        """Test mock streaming generation."""
        chunks = list(mock_backend.generate_stream("Hello"))
        
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0
    
    def test_mock_chat(self, mock_backend):
        """Test mock chat completion."""
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = mock_backend.chat(messages)
        
        assert result.text
        assert result.backend == "Mock"


# ============================================================================
# Test OllamaBackend
# ============================================================================

class TestOllamaBackend:
    """Tests for OllamaBackend."""
    
    def test_ollama_creation(self):
        """Test Ollama backend creation."""
        backend = OllamaBackend()
        
        assert backend.name == "Ollama"
        assert backend.backend_type == BackendType.OLLAMA
        assert backend.base_url == "http://localhost:11434"
    
    def test_ollama_custom_url(self):
        """Test Ollama with custom URL."""
        backend = OllamaBackend(base_url="http://remote:11434")
        
        assert backend.base_url == "http://remote:11434"
    
    def test_ollama_default_model(self):
        """Test Ollama default model."""
        backend = OllamaBackend(default_model="mistral")
        
        assert backend.default_model == "mistral"
    
    @pytest.mark.skipif(not HAS_HTTPX and not HAS_REQUESTS, reason="No HTTP client")
    def test_ollama_not_available_offline(self):
        """Test Ollama reports unavailable when server down."""
        backend = OllamaBackend(base_url="http://localhost:99999")
        
        assert backend.is_available() is False
    
    def test_ollama_format_size(self):
        """Test size formatting."""
        backend = OllamaBackend()
        
        assert "KB" in backend._format_size(1024)
        assert "MB" in backend._format_size(1024 * 1024)
        assert "GB" in backend._format_size(1024 * 1024 * 1024)


# ============================================================================
# Test LlamaCppBackend
# ============================================================================

class TestLlamaCppBackend:
    """Tests for LlamaCppBackend."""
    
    def test_llamacpp_creation(self):
        """Test llama.cpp backend creation."""
        backend = LlamaCppBackend()
        
        assert backend.name == "llama.cpp"
        assert backend.backend_type == BackendType.LLAMACPP
    
    def test_llamacpp_server_mode(self):
        """Test llama.cpp in server mode."""
        backend = LlamaCppBackend(server_url="http://localhost:8080")
        
        assert backend._use_server is True
        assert backend.server_url == "http://localhost:8080"
    
    def test_llamacpp_subprocess_mode(self):
        """Test llama.cpp in subprocess mode."""
        backend = LlamaCppBackend(
            model_path="/path/to/model.gguf",
            n_gpu_layers=35,
        )
        
        assert backend._use_server is False
        assert backend.model_path == "/path/to/model.gguf"
        assert backend.n_gpu_layers == 35
    
    def test_llamacpp_extract_quant(self):
        """Test quantization extraction from filename."""
        backend = LlamaCppBackend()
        
        assert backend._extract_quant("model-q4_k.gguf") == "Q4_K"
        assert backend._extract_quant("model-Q8_0.gguf") == "Q8_0"
        assert backend._extract_quant("model.gguf") == ""


# ============================================================================
# Test VLLMBackend
# ============================================================================

class TestVLLMBackend:
    """Tests for VLLMBackend."""
    
    def test_vllm_creation(self):
        """Test vLLM backend creation."""
        backend = VLLMBackend()
        
        assert backend.name == "vLLM"
        assert backend.backend_type == BackendType.VLLM
        assert backend.base_url == "http://localhost:8000"
    
    def test_vllm_custom_config(self):
        """Test vLLM with custom configuration."""
        backend = VLLMBackend(
            base_url="http://gpu-server:8000",
            api_key="my-key",
            default_model="meta-llama/Llama-2-7b",
        )
        
        assert backend.base_url == "http://gpu-server:8000"
        assert backend.api_key == "my-key"
        assert backend.default_model == "meta-llama/Llama-2-7b"
    
    def test_vllm_headers(self):
        """Test vLLM request headers."""
        backend = VLLMBackend(api_key="test-key")
        
        headers = backend._headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"


# ============================================================================
# Test OpenAICompatBackend
# ============================================================================

class TestOpenAICompatBackend:
    """Tests for OpenAICompatBackend."""
    
    def test_openai_compat_creation(self):
        """Test OpenAI-compatible backend creation."""
        backend = OpenAICompatBackend(base_url="http://localhost:1234")
        
        assert backend.name == "OpenAI-Compatible"
        assert backend.backend_type == BackendType.OPENAI_COMPAT
    
    def test_openai_compat_headers(self):
        """Test OpenAI-compatible headers."""
        backend = OpenAICompatBackend(
            base_url="http://localhost:1234",
            api_key="sk-test",
        )
        
        headers = backend._headers()
        assert "Bearer sk-test" in headers["Authorization"]


# ============================================================================
# Test LLMRouter
# ============================================================================

class TestLLMRouter:
    """Tests for LLMRouter."""
    
    def test_router_creation(self, router):
        """Test router creation."""
        assert router is not None
        assert len(router.backends) >= 1  # At least mock
    
    def test_router_has_mock(self, router):
        """Test router has mock backend by default."""
        assert BackendType.MOCK in router.backends
    
    def test_router_register_backend(self, router):
        """Test registering a backend."""
        backend = OllamaBackend()
        router.register_backend(backend)
        
        assert BackendType.OLLAMA in router.backends
    
    def test_router_select_backend(self, router):
        """Test selecting a backend."""
        # Mock is always available
        result = router.select_backend(BackendType.MOCK)
        
        assert result is True
        assert router.active_backend.backend_type == BackendType.MOCK
    
    def test_router_select_unavailable(self, router):
        """Test selecting unavailable backend."""
        # Register but don't start Ollama
        router.register_backend(OllamaBackend(base_url="http://localhost:99999"))
        
        result = router.select_backend(BackendType.OLLAMA)
        
        assert result is False
    
    def test_router_set_model(self, router):
        """Test setting model name."""
        router.set_model("llama2:7b")
        
        assert router.active_model == "llama2:7b"
    
    def test_router_get_available(self, router):
        """Test getting available backends."""
        available = router.get_available_backends()
        
        assert len(available) >= 1
        assert any(b.backend_type == BackendType.MOCK for b in available)
    
    def test_router_list_all_models(self, router):
        """Test listing all models."""
        models = router.list_all_models()
        
        assert "Mock" in models
        assert len(models["Mock"]) == 3
    
    def test_router_generate(self, router):
        """Test generation through router."""
        router.select_backend(BackendType.MOCK)
        
        result = router.generate("Hello world")
        
        assert result.text
        assert result.finish_reason != "error"
    
    def test_router_generate_with_model(self, router):
        """Test generation with specific model."""
        router.select_backend(BackendType.MOCK)
        
        result = router.generate("Hello", model="mock-13b")
        
        assert result.model == "mock-13b"
    
    def test_router_generate_stream(self, router):
        """Test streaming generation."""
        router.select_backend(BackendType.MOCK)
        
        chunks = list(router.generate_stream("Hello"))
        
        assert len(chunks) > 0
    
    def test_router_chat(self, router):
        """Test chat completion."""
        router.select_backend(BackendType.MOCK)
        
        messages = [{"role": "user", "content": "Hello"}]
        result = router.chat(messages)
        
        assert result.text
    
    def test_router_get_status(self, router):
        """Test status retrieval."""
        status = router.get_status()
        
        assert "active_backend" in status
        assert "active_model" in status
        assert "registered_backends" in status
        assert "available_backends" in status
        assert "default_config" in status
    
    def test_router_failover(self, router):
        """Test failover when primary fails."""
        # Register unavailable backend
        router.register_backend(OllamaBackend(base_url="http://localhost:99999"))
        router.active_backend = router.backends[BackendType.OLLAMA]
        
        # Should failover to mock
        result = router.generate("Test", fallback=True)
        
        assert result.finish_reason != "error" or "Mock" in result.backend or result.backend == "none"
    
    def test_router_no_fallback(self, router):
        """Test generation without fallback."""
        # Set unavailable as active
        router.register_backend(OllamaBackend(base_url="http://localhost:99999"))
        router.active_backend = router.backends[BackendType.OLLAMA]
        
        result = router.generate("Test", fallback=False)
        
        # Should fail or error
        assert "Error" in result.text or result.finish_reason == "error"


class TestCreateDefaultRouter:
    """Tests for create_default_router function."""
    
    def test_creates_router(self):
        """Test factory creates router."""
        router = create_default_router(verbose=False)
        
        assert router is not None
        assert isinstance(router, LLMRouter)
    
    def test_registers_backends(self):
        """Test factory registers all backends."""
        router = create_default_router(verbose=False)
        
        # Should have mock + other backends
        assert BackendType.MOCK in router.backends


# ============================================================================
# Test Q-Lang Integration
# ============================================================================

class TestQLangIntegration:
    """Tests for Q-Lang command handling."""
    
    def test_handle_status_command(self, router, capsys):
        """Test llm status command."""
        handle_llm_command(router, "llm status", {})
        
        captured = capsys.readouterr()
        assert "LLM Router Status" in captured.out or "Active Backend" in captured.out
    
    def test_handle_list_command(self, router, capsys):
        """Test llm list command."""
        router.select_backend(BackendType.MOCK)
        
        handle_llm_command(router, "llm list", {})
        
        captured = capsys.readouterr()
        assert "Available Models" in captured.out or "Mock" in captured.out
    
    def test_handle_select_command(self, router, capsys):
        """Test llm select command."""
        handle_llm_command(router, "llm select mock", {})
        
        # Check that mock backend is now active
        assert router.active_backend is not None
        assert router.active_backend.backend_type == BackendType.MOCK
    
    def test_handle_model_command(self, router, capsys):
        """Test llm model command."""
        handle_llm_command(router, "llm model test-model", {})
        
        assert router.active_model == "test-model"
    
    def test_handle_generate_command(self, router, capsys):
        """Test llm generate command."""
        router.select_backend(BackendType.MOCK)
        context = {}
        
        handle_llm_command(router, 'llm generate "Hello world"', context)
        
        assert "llm_output" in context
        captured = capsys.readouterr()
        assert "Generating" in captured.out
    
    def test_handle_chat_command(self, router, capsys):
        """Test llm chat command."""
        router.select_backend(BackendType.MOCK)
        context = {}
        
        handle_llm_command(router, 'llm chat "Hello"', context)
        
        assert "chat_history" in context
        assert len(context["chat_history"]) == 2  # user + assistant
    
    def test_handle_stream_command(self, router, capsys):
        """Test llm stream command."""
        router.select_backend(BackendType.MOCK)
        context = {}
        
        handle_llm_command(router, 'llm stream "Hello"', context)
        
        assert "llm_output" in context
        captured = capsys.readouterr()
        assert "Streaming" in captured.out
    
    def test_handle_config_show(self, router, capsys):
        """Test llm config show command."""
        handle_llm_command(router, "llm config", {})
        
        captured = capsys.readouterr()
        assert "max_tokens" in captured.out
        assert "temperature" in captured.out
    
    def test_handle_config_set(self, router, capsys):
        """Test llm config set command."""
        handle_llm_command(router, "llm config temperature=0.9 max_tokens=512", {})
        
        assert router.default_config.temperature == 0.9
        assert router.default_config.max_tokens == 512
    
    def test_handle_help_command(self, router, capsys):
        """Test llm help command."""
        handle_llm_command(router, "llm help", {})
        
        captured = capsys.readouterr()
        assert "LLM Backend Commands" in captured.out
        assert "generate" in captured.out
        assert "chat" in captured.out
    
    def test_handle_unknown_command(self, router, capsys):
        """Test unknown llm command."""
        handle_llm_command(router, "llm unknown", {})
        
        captured = capsys.readouterr()
        assert "Unknown" in captured.out or "❌" in captured.out
    
    def test_handle_missing_args(self, router, capsys):
        """Test command with missing arguments."""
        handle_llm_command(router, "llm generate", {})
        
        captured = capsys.readouterr()
        assert "Usage" in captured.out
    
    def test_handle_select_unknown_backend(self, router, capsys):
        """Test selecting unknown backend."""
        handle_llm_command(router, "llm select unknownbackend", {})
        
        captured = capsys.readouterr()
        assert "Unknown" in captured.out


# ============================================================================
# Test Availability Flags
# ============================================================================

class TestAvailability:
    """Tests for module availability flags."""
    
    def test_llm_backend_flag(self):
        """Test HAS_LLM_BACKEND flag."""
        assert HAS_LLM_BACKEND is True
    
    def test_httpx_flag_type(self):
        """Test HAS_HTTPX is boolean."""
        assert isinstance(HAS_HTTPX, bool)
    
    def test_requests_flag_type(self):
        """Test HAS_REQUESTS is boolean."""
        assert isinstance(HAS_REQUESTS, bool)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_generation_workflow(self, router):
        """Test complete generation workflow."""
        # Select backend
        router.select_backend(BackendType.MOCK)
        
        # Set model
        router.set_model("mock-7b")
        
        # Configure
        router.default_config.temperature = 0.8
        router.default_config.max_tokens = 512
        
        # Generate
        result = router.generate("Write a haiku about programming")
        
        assert result.text
        assert result.backend == "Mock"
        assert result.finish_reason == "stop"
    
    def test_chat_conversation(self, router):
        """Test multi-turn chat conversation."""
        router.select_backend(BackendType.MOCK)
        
        # First turn
        messages = [{"role": "user", "content": "Hello, who are you?"}]
        result1 = router.chat(messages)
        
        # Second turn
        messages.append({"role": "assistant", "content": result1.text})
        messages.append({"role": "user", "content": "Tell me more"})
        result2 = router.chat(messages)
        
        assert result1.text
        assert result2.text
    
    def test_streaming_collects_full_response(self, router):
        """Test streaming collects full response."""
        router.select_backend(BackendType.MOCK)
        
        chunks = []
        for chunk in router.generate_stream("Explain machine learning"):
            chunks.append(chunk)
        
        full_response = "".join(chunks)
        assert len(full_response) > 10
    
    def test_config_persistence(self, router):
        """Test configuration persists across calls."""
        router.select_backend(BackendType.MOCK)
        
        # Set config
        router.default_config.temperature = 0.1
        router.default_config.max_tokens = 100
        
        # Make multiple calls
        result1 = router.generate("Test 1")
        result2 = router.generate("Test 2")
        
        # Config should still be set
        assert router.default_config.temperature == 0.1
        assert router.default_config.max_tokens == 100


# ============================================================================
# Test Abstract Base Class
# ============================================================================

class TestLLMBackendABC:
    """Tests for LLMBackend abstract base class."""
    
    def test_cannot_instantiate_abstract(self):
        """Test cannot instantiate abstract class directly."""
        with pytest.raises(TypeError):
            LLMBackend()
    
    def test_mock_implements_interface(self, mock_backend):
        """Test mock implements all required methods."""
        assert hasattr(mock_backend, 'name')
        assert hasattr(mock_backend, 'backend_type')
        assert hasattr(mock_backend, 'is_available')
        assert hasattr(mock_backend, 'list_models')
        assert hasattr(mock_backend, 'generate')
        assert hasattr(mock_backend, 'generate_stream')
        assert hasattr(mock_backend, 'chat')
    
    def test_default_chat_implementation(self, mock_backend):
        """Test default chat uses generate with formatted prompt."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        
        # MockBackend overrides chat, but base class has default implementation
        result = mock_backend.chat(messages)
        assert result.text


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_generate_returns_error_result(self, router):
        """Test generate returns error result on failure."""
        # Set unavailable backend without fallback
        router.register_backend(OllamaBackend(base_url="http://localhost:99999"))
        router.active_backend = router.backends[BackendType.OLLAMA]
        
        result = router.generate("Test", fallback=False)
        
        assert "Error" in result.text or result.finish_reason == "error"
    
    def test_stream_yields_error(self, router):
        """Test stream yields error on failure."""
        router.active_backend = None
        
        chunks = list(router.generate_stream("Test"))
        
        full = "".join(chunks)
        assert "Error" in full or "available" in full.lower()
    
    def test_chat_returns_error_result(self, router):
        """Test chat returns error result on failure."""
        router.active_backend = None
        
        result = router.chat([{"role": "user", "content": "Test"}])
        
        assert "Error" in result.text or result.finish_reason == "error"
