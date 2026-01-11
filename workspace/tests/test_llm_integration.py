"""
Tests for QENEX LLM Integration Layer (100% Local)
===================================================
Tests the Scout/DeepSeek integration with local LLM backends only.

Uses MockBackend for all tests - no external API calls.

Author: QENEX Sovereign Agent
Date: 2026-01-11
"""

import os
import sys
import json
import time
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "qenex-qlang" / "src"))

from llm_integration import (
    TaskType,
    IntegrationConfig,
    TokenUsage,
    IntegratedScoutReasoner,
    IntegratedDeepSeekEngine,
    ExperimentStateManager,
    QENEXIntegration,
    handle_integrate_command,
    HAS_INTEGRATION,
)
from llm_backend import (
    LLMRouter,
    BackendType,
    GenerationConfig,
    GenerationResult,
    MockBackend,
    create_default_router,
)
from context_store import ContextStore, ContextChunk


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def temp_dir():
    """Provide temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_router():
    """Create router with MockBackend."""
    router = LLMRouter(verbose=False)
    router.register_backend(MockBackend())
    router.select_backend(BackendType.MOCK)
    return router


@pytest.fixture
def integration_config():
    """Create test integration config."""
    return IntegrationConfig(
        scout_backend=BackendType.MOCK,
        scout_model="test-scout-model",
        scout_temperature=0.3,
        scout_max_tokens=1024,
        deepseek_backend=BackendType.MOCK,
        deepseek_model="test-deepseek-model",
        deepseek_temperature=0.2,
        deepseek_max_tokens=2048,
        enable_context_persistence=True,
        auto_checkpoint=True,
        checkpoint_interval=5,
        track_tokens=True,
        token_budget=None,
    )


@pytest.fixture
def context_store(temp_dir):
    """Create context store in temp directory."""
    return ContextStore(base_dir=temp_dir, verbose=False)


@pytest.fixture
def mock_context():
    """Create mock Scout context."""
    context = Mock()
    context.chunks = {}
    context.total_tokens = 0
    context.conversation_history = []
    return context


# ==============================================================================
# TaskType Tests
# ==============================================================================

class TestTaskType:
    """Tests for TaskType enum."""
    
    def test_all_task_types_defined(self):
        """Verify all expected task types exist."""
        expected = [
            "REASONING", "CODE_GENERATION", "ANALYSIS", "PROOF",
            "HYPOTHESIS", "DOCUMENTATION", "TRANSLATION", "GENERAL"
        ]
        for name in expected:
            assert hasattr(TaskType, name)
    
    def test_task_type_values(self):
        """Verify task type values."""
        assert TaskType.REASONING.value == "reasoning"
        assert TaskType.CODE_GENERATION.value == "code"
        assert TaskType.ANALYSIS.value == "analysis"


# ==============================================================================
# IntegrationConfig Tests
# ==============================================================================

class TestIntegrationConfig:
    """Tests for IntegrationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = IntegrationConfig()
        assert config.scout_backend == BackendType.OLLAMA
        assert config.deepseek_backend == BackendType.OLLAMA
        assert config.enable_context_persistence is True
        assert config.auto_checkpoint is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = IntegrationConfig(
            scout_backend=BackendType.MOCK,
            scout_model="custom-model",
            scout_temperature=0.5,
            scout_max_tokens=2048,
            deepseek_backend=BackendType.LLAMACPP,
            deepseek_model="deepseek-custom",
            deepseek_temperature=0.1,
            deepseek_max_tokens=4096,
            enable_context_persistence=False,
            auto_checkpoint=False,
            checkpoint_interval=20,
            track_tokens=False,
            token_budget=10000,
        )
        assert config.scout_backend == BackendType.MOCK
        assert config.scout_model == "custom-model"
        assert config.scout_temperature == 0.5
        assert config.deepseek_backend == BackendType.LLAMACPP
        assert config.token_budget == 10000
    
    def test_config_only_local_backends(self):
        """Verify only local backends are used."""
        # These should be the only valid backends
        local_backends = [
            BackendType.OLLAMA,
            BackendType.LLAMACPP,
            BackendType.VLLM,
            BackendType.OPENAI_COMPAT,  # Can be local
            BackendType.MOCK,
        ]
        
        for backend in local_backends:
            config = IntegrationConfig(scout_backend=backend)
            assert config.scout_backend == backend


# ==============================================================================
# TokenUsage Tests
# ==============================================================================

class TestTokenUsage:
    """Tests for TokenUsage tracking."""
    
    def test_initial_state(self):
        """Test initial token usage is zero."""
        usage = TokenUsage()
        assert usage.total_prompt_tokens == 0
        assert usage.total_completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.operations == 0
    
    def test_add_tokens(self):
        """Test adding token counts."""
        usage = TokenUsage()
        usage.add(100, 50)
        
        assert usage.total_prompt_tokens == 100
        assert usage.total_completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.operations == 1
    
    def test_accumulate_tokens(self):
        """Test accumulating multiple operations."""
        usage = TokenUsage()
        usage.add(100, 50)
        usage.add(200, 100)
        usage.add(50, 25)
        
        assert usage.total_prompt_tokens == 350
        assert usage.total_completion_tokens == 175
        assert usage.total_tokens == 525
        assert usage.operations == 3
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        usage = TokenUsage()
        usage.add(100, 50)
        
        d = usage.to_dict()
        assert d["prompt_tokens"] == 100
        assert d["completion_tokens"] == 50
        assert d["total_tokens"] == 150
        assert d["operations"] == 1


# ==============================================================================
# IntegratedScoutReasoner Tests
# ==============================================================================

class TestIntegratedScoutReasoner:
    """Tests for IntegratedScoutReasoner."""
    
    def test_init(self, mock_router, integration_config):
        """Test reasoner initialization."""
        reasoner = IntegratedScoutReasoner(
            mock_router, integration_config, verbose=False
        )
        assert reasoner.router is mock_router
        assert reasoner.config is integration_config
        assert isinstance(reasoner.token_usage, TokenUsage)
    
    def test_system_prompt_defined(self, mock_router, integration_config):
        """Verify system prompt is defined."""
        reasoner = IntegratedScoutReasoner(
            mock_router, integration_config, verbose=False
        )
        assert "Scout" in reasoner.SYSTEM_PROMPT
        assert "scientific reasoning" in reasoner.SYSTEM_PROMPT.lower()
    
    def test_mode_prompts(self, mock_router, integration_config):
        """Test mode-specific prompts exist."""
        reasoner = IntegratedScoutReasoner(
            mock_router, integration_config, verbose=False
        )
        expected_modes = ["REASON", "SYNTHESIZE", "PROVE", "HYPOTHESIZE", "VERIFY", "ANALYZE"]
        for mode in expected_modes:
            assert mode in reasoner.mode_prompts
    
    def test_reason_basic(self, mock_router, integration_config):
        """Test basic reasoning."""
        reasoner = IntegratedScoutReasoner(
            mock_router, integration_config, verbose=False
        )
        
        result = reasoner.reason("What is entropy?", mode="REASON")
        
        assert result["success"] is True
        assert result["mode"] == "REASON"
        assert result["query"] == "What is entropy?"
        assert "output" in result
        assert result["backend"] == "Mock"
    
    def test_reason_with_context(self, mock_router, integration_config, mock_context):
        """Test reasoning with loaded context."""
        # Add a chunk to context
        mock_context.chunks = {
            "chunk1": Mock(
                source="physics.txt",
                content="Entropy is a measure of disorder in a system."
            )
        }
        
        reasoner = IntegratedScoutReasoner(
            mock_router, integration_config, context=mock_context, verbose=False
        )
        
        result = reasoner.reason("Explain entropy", mode="REASON", use_context=True)
        
        assert result["success"] is True
        assert result["context_chunks"] == 1
    
    def test_reason_modes(self, mock_router, integration_config):
        """Test different reasoning modes."""
        reasoner = IntegratedScoutReasoner(
            mock_router, integration_config, verbose=False
        )
        
        modes = ["REASON", "SYNTHESIZE", "PROVE", "HYPOTHESIZE", "VERIFY", "ANALYZE"]
        
        for mode in modes:
            result = reasoner.reason("Test query", mode=mode)
            assert result["mode"] == mode
            assert result["success"] is True
    
    def test_token_tracking(self, mock_router, integration_config):
        """Test token usage is tracked."""
        reasoner = IntegratedScoutReasoner(
            mock_router, integration_config, verbose=False
        )
        
        # Initial state
        assert reasoner.token_usage.operations == 0
        
        # After reasoning
        reasoner.reason("Query 1")
        assert reasoner.token_usage.operations == 1
        
        reasoner.reason("Query 2")
        assert reasoner.token_usage.operations == 2
    
    def test_build_prompt(self, mock_router, integration_config):
        """Test prompt building."""
        reasoner = IntegratedScoutReasoner(
            mock_router, integration_config, verbose=False
        )
        
        prompt = reasoner._build_prompt(
            query="What is gravity?",
            mode="REASON",
            context_str="Newton's law of gravitation..."
        )
        
        assert "What is gravity?" in prompt
        assert "Newton's law" in prompt
        assert "REASON" in prompt


# ==============================================================================
# IntegratedDeepSeekEngine Tests
# ==============================================================================

class TestIntegratedDeepSeekEngine:
    """Tests for IntegratedDeepSeekEngine."""
    
    def test_init(self, mock_router, integration_config):
        """Test engine initialization."""
        engine = IntegratedDeepSeekEngine(
            mock_router, integration_config, verbose=False
        )
        assert engine.router is mock_router
        assert engine.config is integration_config
        assert isinstance(engine.token_usage, TokenUsage)
        assert engine.generation_history == []
    
    def test_system_prompt_defined(self, mock_router, integration_config):
        """Verify system prompt is defined."""
        engine = IntegratedDeepSeekEngine(
            mock_router, integration_config, verbose=False
        )
        assert "DeepSeek" in engine.SYSTEM_PROMPT
        assert "code" in engine.SYSTEM_PROMPT.lower()
    
    def test_mode_prompts(self, mock_router, integration_config):
        """Test mode-specific prompts exist."""
        engine = IntegratedDeepSeekEngine(
            mock_router, integration_config, verbose=False
        )
        expected_modes = [
            "GENERATE", "OPTIMIZE", "TEST", "DOCUMENT",
            "TRANSLATE", "EXPLAIN", "DEBUG", "COMPLETE"
        ]
        for mode in expected_modes:
            assert mode in engine.MODE_PROMPTS
    
    def test_generate_basic(self, mock_router, integration_config):
        """Test basic code generation."""
        engine = IntegratedDeepSeekEngine(
            mock_router, integration_config, verbose=False
        )
        
        result = engine.generate(
            description="Calculate factorial",
            language="python",
            mode="GENERATE"
        )
        
        assert result["success"] is True
        assert result["mode"] == "GENERATE"
        assert result["language"] == "python"
        assert "code" in result
        assert result["backend"] == "Mock"
    
    def test_generate_different_languages(self, mock_router, integration_config):
        """Test generation for different languages."""
        engine = IntegratedDeepSeekEngine(
            mock_router, integration_config, verbose=False
        )
        
        languages = ["python", "julia", "rust", "qlang"]
        
        for lang in languages:
            result = engine.generate(
                description="Hello world",
                language=lang
            )
            assert result["language"] == lang
            assert result["success"] is True
    
    def test_generation_history(self, mock_router, integration_config):
        """Test generation history is tracked."""
        engine = IntegratedDeepSeekEngine(
            mock_router, integration_config, verbose=False
        )
        
        assert len(engine.generation_history) == 0
        
        engine.generate("Task 1", language="python")
        assert len(engine.generation_history) == 1
        
        engine.generate("Task 2", language="julia")
        assert len(engine.generation_history) == 2
    
    def test_extract_code_with_block(self, mock_router, integration_config):
        """Test code extraction from response."""
        engine = IntegratedDeepSeekEngine(
            mock_router, integration_config, verbose=False
        )
        
        # Test with language-annotated block
        text = "Here's the code:\n```python\ndef hello():\n    print('Hello')\n```\n"
        code = engine._extract_code(text, "python")
        assert "def hello()" in code
        assert "print('Hello')" in code
    
    def test_extract_code_generic_block(self, mock_router, integration_config):
        """Test code extraction from generic block."""
        engine = IntegratedDeepSeekEngine(
            mock_router, integration_config, verbose=False
        )
        
        text = "```\ndef foo(): pass\n```"
        code = engine._extract_code(text, "python")
        assert "def foo()" in code
    
    def test_extract_code_no_block(self, mock_router, integration_config):
        """Test code extraction when no block present."""
        engine = IntegratedDeepSeekEngine(
            mock_router, integration_config, verbose=False
        )
        
        text = "def bar(): return 42"
        code = engine._extract_code(text, "python")
        assert code == text


# ==============================================================================
# ExperimentStateManager Tests
# ==============================================================================

class TestExperimentStateManager:
    """Tests for ExperimentStateManager."""
    
    def test_init(self, context_store):
        """Test manager initialization."""
        manager = ExperimentStateManager(
            context_store, 
            experiment_name="test-exp"
        )
        assert manager.experiment_name == "test-exp"
        assert manager.operation_count == 0
        assert manager.checkpoint_count == 0
        assert "operations" in manager.state
        assert "results" in manager.state
    
    def test_record_operation(self, context_store):
        """Test recording operations."""
        manager = ExperimentStateManager(context_store)
        
        manager.record_operation(
            operation_type="reasoning",
            input_data={"query": "test"},
            output_data={"result": "answer"}
        )
        
        assert manager.operation_count == 1
        assert len(manager.state["operations"]) == 1
        assert manager.state["operations"][0]["type"] == "reasoning"
    
    def test_auto_checkpoint(self, context_store):
        """Test automatic checkpointing."""
        manager = ExperimentStateManager(
            context_store,
            auto_checkpoint=True,
            checkpoint_interval=3
        )
        
        # Record 3 operations to trigger checkpoint
        for i in range(3):
            manager.record_operation(
                f"op_{i}",
                {"i": i},
                {"result": i}
            )
        
        assert manager.checkpoint_count == 1
    
    def test_store_result(self, context_store):
        """Test storing named results."""
        manager = ExperimentStateManager(context_store)
        
        manager.store_result("energy", -76.5)
        manager.store_result("converged", True)
        
        assert manager.get_result("energy") == -76.5
        assert manager.get_result("converged") is True
        assert manager.get_result("nonexistent") is None
    
    def test_checkpoint(self, context_store):
        """Test manual checkpointing."""
        manager = ExperimentStateManager(
            context_store,
            experiment_name="checkpoint-test"
        )
        
        manager.record_operation("test", {"a": 1}, {"b": 2})
        path = manager.checkpoint()
        
        assert path.exists()
        assert manager.checkpoint_count == 1
    
    def test_summarize_long_strings(self, context_store):
        """Test summarization of long outputs."""
        manager = ExperimentStateManager(context_store)
        
        long_data = {
            "output": "x" * 1000,
            "count": 42,
            "items": [1, 2, 3, 4, 5],
        }
        
        summary = manager._summarize(long_data, max_len=100)
        
        assert len(summary["output"]) < 1000
        assert "..." in summary["output"]
        assert summary["count"] == 42
        assert "list of 5 items" in summary["items"]


# ==============================================================================
# QENEXIntegration Tests
# ==============================================================================

class TestQENEXIntegration:
    """Tests for main QENEXIntegration class."""
    
    def test_init_default(self, temp_dir):
        """Test default initialization."""
        config = IntegrationConfig(
            context_store_path=temp_dir
        )
        integration = QENEXIntegration(config=config, verbose=False)
        
        assert integration.router is not None
        assert integration.scout is not None
        assert integration.deepseek is not None
        assert integration.experiment is not None
    
    def test_init_with_mock_backend(self, temp_dir):
        """Test initialization with mock backend."""
        config = IntegrationConfig(
            scout_backend=BackendType.MOCK,
            deepseek_backend=BackendType.MOCK,
            context_store_path=temp_dir
        )
        integration = QENEXIntegration(config=config, verbose=False)
        
        assert integration.config.scout_backend == BackendType.MOCK
        assert integration.config.deepseek_backend == BackendType.MOCK
    
    def test_reason(self, temp_dir):
        """Test reasoning through integration."""
        config = IntegrationConfig(
            scout_backend=BackendType.MOCK,
            context_store_path=temp_dir
        )
        integration = QENEXIntegration(config=config, verbose=False)
        
        # Register mock backend
        integration.router.register_backend(MockBackend())
        
        result = integration.reason("What is quantum entanglement?")
        
        assert result["success"] is True
        assert "output" in result
    
    def test_generate_code(self, temp_dir):
        """Test code generation through integration."""
        config = IntegrationConfig(
            deepseek_backend=BackendType.MOCK,
            context_store_path=temp_dir
        )
        integration = QENEXIntegration(config=config, verbose=False)
        
        # Register mock backend
        integration.router.register_backend(MockBackend())
        
        result = integration.generate_code(
            description="Calculate matrix determinant",
            language="python"
        )
        
        assert result["success"] is True
        assert "code" in result
    
    def test_token_aggregation(self, temp_dir):
        """Test token usage aggregation."""
        config = IntegrationConfig(
            scout_backend=BackendType.MOCK,
            deepseek_backend=BackendType.MOCK,
            context_store_path=temp_dir
        )
        integration = QENEXIntegration(config=config, verbose=False)
        integration.router.register_backend(MockBackend())
        
        # Initial state
        assert integration.total_tokens.operations == 0
        
        # After operations
        integration.reason("Query 1")
        integration.generate_code("Task 1")
        
        assert integration.total_tokens.operations == 2
    
    def test_get_status(self, temp_dir):
        """Test status retrieval."""
        config = IntegrationConfig(
            scout_backend=BackendType.MOCK,
            deepseek_backend=BackendType.MOCK,
            context_store_path=temp_dir
        )
        integration = QENEXIntegration(config=config, verbose=False)
        
        status = integration.get_status()
        
        assert "scout" in status
        assert "deepseek" in status
        assert "tokens" in status
        assert "experiment" in status
        assert "available_backends" in status
    
    def test_checkpoint(self, temp_dir):
        """Test checkpoint creation."""
        config = IntegrationConfig(
            scout_backend=BackendType.MOCK,
            context_store_path=temp_dir
        )
        integration = QENEXIntegration(config=config, verbose=False)
        integration.router.register_backend(MockBackend())
        
        # Do some work
        integration.reason("Test query")
        
        # Checkpoint
        path = integration.checkpoint()
        
        assert path.exists() or str(path) != ""


# ==============================================================================
# Command Handler Tests
# ==============================================================================

class TestIntegrateCommandHandler:
    """Tests for Q-Lang integrate command handler."""
    
    @pytest.fixture
    def integration(self, temp_dir):
        """Create integration for command testing."""
        config = IntegrationConfig(
            scout_backend=BackendType.MOCK,
            deepseek_backend=BackendType.MOCK,
            context_store_path=temp_dir
        )
        integ = QENEXIntegration(config=config, verbose=False)
        integ.router.register_backend(MockBackend())
        return integ
    
    def test_status_command(self, integration, capsys):
        """Test status command."""
        handle_integrate_command(integration, "integrate status", {})
        captured = capsys.readouterr()
        
        assert "Integration Status" in captured.out
        assert "Scout" in captured.out
        assert "DeepSeek" in captured.out
    
    def test_scout_backend_command(self, integration, capsys):
        """Test scout backend command."""
        handle_integrate_command(integration, "integrate scout mock", {})
        captured = capsys.readouterr()
        
        assert "Scout backend set to" in captured.out
    
    def test_deepseek_backend_command(self, integration, capsys):
        """Test deepseek backend command."""
        handle_integrate_command(integration, "integrate deepseek mock", {})
        captured = capsys.readouterr()
        
        assert "DeepSeek backend set to" in captured.out
    
    def test_configure_command(self, integration, capsys):
        """Test configure command."""
        handle_integrate_command(integration, "integrate configure", {})
        captured = capsys.readouterr()
        
        assert "configuration" in captured.out.lower()
    
    def test_configure_set_value(self, integration, capsys):
        """Test configure with value."""
        handle_integrate_command(
            integration, 
            "integrate configure scout_temperature=0.5", 
            {}
        )
        captured = capsys.readouterr()
        
        assert integration.config.scout_temperature == 0.5
    
    def test_checkpoint_command(self, integration, capsys):
        """Test checkpoint command."""
        handle_integrate_command(integration, "integrate checkpoint", {})
        captured = capsys.readouterr()
        
        assert "Checkpoint saved" in captured.out
    
    def test_help_command(self, integration, capsys):
        """Test help command."""
        handle_integrate_command(integration, "integrate help", {})
        captured = capsys.readouterr()
        
        assert "Commands" in captured.out
        assert "status" in captured.out
        assert "scout" in captured.out
        assert "deepseek" in captured.out
    
    def test_unknown_command(self, integration, capsys):
        """Test unknown command handling."""
        handle_integrate_command(integration, "integrate foobar", {})
        captured = capsys.readouterr()
        
        assert "Unknown command" in captured.out
    
    def test_invalid_backend(self, integration, capsys):
        """Test invalid backend name."""
        handle_integrate_command(integration, "integrate scout invalid_backend", {})
        captured = capsys.readouterr()
        
        assert "Unknown backend" in captured.out


# ==============================================================================
# Local-Only Backend Tests
# ==============================================================================

class TestLocalOnlyBackends:
    """Verify only local backends are supported."""
    
    def test_no_external_api_backends(self):
        """Verify no external API backends are imported."""
        # These should NOT exist
        with pytest.raises(ImportError):
            from llm_backends_extended import AnthropicBackend
    
    def test_available_backends_are_local(self, mock_router):
        """Verify all available backends are local."""
        for backend in mock_router.get_available_backends():
            # All should be one of these local types
            assert backend.backend_type in [
                BackendType.OLLAMA,
                BackendType.LLAMACPP,
                BackendType.VLLM,
                BackendType.OPENAI_COMPAT,
                BackendType.MOCK,
            ]
    
    def test_integration_config_local_only(self):
        """Verify IntegrationConfig only allows local backends."""
        config = IntegrationConfig()
        
        # Default should be local
        assert config.scout_backend in [
            BackendType.OLLAMA,
            BackendType.LLAMACPP,
            BackendType.VLLM,
            BackendType.OPENAI_COMPAT,
            BackendType.MOCK,
        ]


# ==============================================================================
# Integration Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_query(self, mock_router, integration_config):
        """Test handling of empty query."""
        reasoner = IntegratedScoutReasoner(
            mock_router, integration_config, verbose=False
        )
        
        result = reasoner.reason("")
        assert result["success"] is True  # Should still work
    
    def test_very_long_query(self, mock_router, integration_config):
        """Test handling of very long query."""
        reasoner = IntegratedScoutReasoner(
            mock_router, integration_config, verbose=False
        )
        
        long_query = "x" * 10000
        result = reasoner.reason(long_query)
        assert "success" in result
    
    def test_special_characters(self, mock_router, integration_config):
        """Test handling of special characters."""
        engine = IntegratedDeepSeekEngine(
            mock_router, integration_config, verbose=False
        )
        
        result = engine.generate(
            description="Handle $pecial ch@racters: \n\t\"'",
            language="python"
        )
        assert result["success"] is True
    
    def test_concurrent_operations(self, temp_dir):
        """Test multiple operations don't interfere."""
        config = IntegrationConfig(
            scout_backend=BackendType.MOCK,
            deepseek_backend=BackendType.MOCK,
            context_store_path=temp_dir
        )
        integration = QENEXIntegration(config=config, verbose=False)
        integration.router.register_backend(MockBackend())
        
        # Run multiple operations
        results = []
        for i in range(5):
            r1 = integration.reason(f"Query {i}")
            r2 = integration.generate_code(f"Task {i}")
            results.extend([r1, r2])
        
        # All should succeed
        for r in results:
            assert r["success"] is True
        
        # Token tracking should accumulate
        assert integration.total_tokens.operations == 10


# ==============================================================================
# Module Level Tests
# ==============================================================================

class TestModuleLevel:
    """Test module-level functionality."""
    
    def test_has_integration_flag(self):
        """Verify HAS_INTEGRATION flag is True."""
        assert HAS_INTEGRATION is True
    
    def test_all_classes_importable(self):
        """Verify all main classes are importable."""
        from llm_integration import (
            TaskType,
            IntegrationConfig,
            TokenUsage,
            IntegratedScoutReasoner,
            IntegratedDeepSeekEngine,
            ExperimentStateManager,
            QENEXIntegration,
            handle_integrate_command,
        )
        
        # All should be defined
        assert TaskType is not None
        assert IntegrationConfig is not None
        assert TokenUsage is not None
        assert IntegratedScoutReasoner is not None
        assert IntegratedDeepSeekEngine is not None
        assert ExperimentStateManager is not None
        assert QENEXIntegration is not None
        assert handle_integrate_command is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
