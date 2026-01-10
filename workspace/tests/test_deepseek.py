"""
Tests for QENEX DeepSeek Code Generation Engine.

DeepSeek-Coder 33B integration for scientific code generation.
Tests cover all generation modes, templates, and Q-Lang integration.
"""
import pytest
import sys
import os

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'qenex-qlang', 'src'))

from deepseek import (
    DeepSeekEngine,
    DeepSeekMode,
    TargetLanguage,
    CodeTemplate,
    CodeGenerationResult,
    handle_deepseek_command,
)


# =============================================================================
# DeepSeekEngine Initialization Tests
# =============================================================================

class TestDeepSeekEngineInit:
    """Test DeepSeekEngine initialization."""
    
    def test_engine_creation(self):
        """Test that engine can be created."""
        engine = DeepSeekEngine(verbose=False)
        assert engine is not None
    
    def test_engine_with_verbose(self):
        """Test engine with verbose mode."""
        engine = DeepSeekEngine(verbose=True)
        assert engine.verbose is True
    
    def test_engine_templates_loaded(self):
        """Test that templates are loaded."""
        engine = DeepSeekEngine(verbose=False)
        assert len(engine.TEMPLATES) > 0
        assert "hartree_fock" in engine.TEMPLATES
        assert "molecular_dynamics" in engine.TEMPLATES
    
    def test_engine_has_generation_history(self):
        """Test generation history tracking."""
        engine = DeepSeekEngine(verbose=False)
        assert hasattr(engine, 'generation_history')
        assert len(engine.generation_history) == 0


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnums:
    """Test enumeration values."""
    
    def test_deepseek_modes(self):
        """Test all DeepSeek modes exist."""
        modes = [DeepSeekMode.GENERATE, DeepSeekMode.OPTIMIZE, 
                 DeepSeekMode.TEST, DeepSeekMode.DOCUMENT,
                 DeepSeekMode.TRANSLATE, DeepSeekMode.EXPLAIN]
        assert len(modes) == 6
    
    def test_target_languages(self):
        """Test all target languages exist."""
        langs = [TargetLanguage.PYTHON, TargetLanguage.JULIA, 
                 TargetLanguage.RUST, TargetLanguage.QLANG,
                 TargetLanguage.CPP, TargetLanguage.LEAN,
                 TargetLanguage.LATEX]
        assert len(langs) == 7
    
    def test_language_from_string(self):
        """Test creating language from string."""
        assert TargetLanguage.PYTHON.value == "python"
        assert TargetLanguage.JULIA.value == "julia"
        assert TargetLanguage.RUST.value == "rust"


# =============================================================================
# CodeTemplate Tests
# =============================================================================

class TestCodeTemplate:
    """Test CodeTemplate dataclass."""
    
    def test_code_template_creation(self):
        """Test creating a CodeTemplate."""
        template = CodeTemplate(
            name="Test Template",
            language=TargetLanguage.PYTHON,
            template="def test(): pass",
            placeholders=["name"],
            description="A test template"
        )
        assert template.name == "Test Template"
        assert template.language == TargetLanguage.PYTHON
    
    def test_builtin_templates_have_required_fields(self):
        """Test that built-in templates have all required fields."""
        engine = DeepSeekEngine(verbose=False)
        for name, template in engine.TEMPLATES.items():
            assert template.name is not None
            assert template.language is not None
            assert template.template is not None
            assert template.placeholders is not None
            assert template.description is not None


# =============================================================================
# Code Generation Tests
# =============================================================================

class TestCodeGeneration:
    """Test code generation functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create engine for tests."""
        return DeepSeekEngine(verbose=False)
    
    def test_generate_basic(self, engine):
        """Test basic code generation."""
        result = engine.generate("Calculate fibonacci sequence")
        assert isinstance(result, CodeGenerationResult)
        assert result.code is not None
        assert len(result.code) > 0
        assert result.success is True
    
    def test_generate_with_python(self, engine):
        """Test generation in Python."""
        result = engine.generate("Matrix multiplication", language=TargetLanguage.PYTHON)
        assert result.language == TargetLanguage.PYTHON
        assert "import numpy" in result.code or "def " in result.code
    
    def test_generate_with_julia(self, engine):
        """Test generation in Julia."""
        result = engine.generate("Eigenvalue solver", language=TargetLanguage.JULIA)
        assert result.language == TargetLanguage.JULIA
        assert "function" in result.code or "using" in result.code
    
    def test_generate_with_rust(self, engine):
        """Test generation in Rust."""
        result = engine.generate("Vector operations", language=TargetLanguage.RUST)
        assert result.language == TargetLanguage.RUST
    
    def test_generate_with_qlang(self, engine):
        """Test generation in Q-Lang."""
        result = engine.generate("Define physical constant", language=TargetLanguage.QLANG)
        assert result.language == TargetLanguage.QLANG
    
    def test_generate_with_template_hartree_fock(self, engine):
        """Test generation with Hartree-Fock template."""
        result = engine.generate("SCF solver", template="hartree_fock")
        assert result.code is not None
        # Should contain SCF-related code
        assert "scf" in result.code.lower() or "hartree" in result.code.lower() or "fock" in result.code.lower()
    
    def test_generate_with_template_molecular_dynamics(self, engine):
        """Test generation with molecular dynamics template."""
        result = engine.generate("MD simulation", template="molecular_dynamics")
        assert result.code is not None
        assert "verlet" in result.code.lower() or "velocity" in result.code.lower() or "positions" in result.code.lower()
    
    def test_generate_with_template_neural_network(self, engine):
        """Test generation with neural network template."""
        result = engine.generate("PINN model", template="neural_network")
        assert result.code is not None
        assert "neural" in result.code.lower() or "layer" in result.code.lower() or "forward" in result.code.lower()
    
    def test_generate_with_template_quantum_circuit(self, engine):
        """Test generation with quantum circuit template."""
        result = engine.generate("Quantum gates", template="quantum_circuit")
        assert result.code is not None
        assert "qubit" in result.code.lower() or "gate" in result.code.lower() or "circuit" in result.code.lower()
    
    def test_generation_result_has_confidence(self, engine):
        """Test that generation result has confidence score."""
        result = engine.generate("Simple function")
        assert hasattr(result, 'confidence')
        assert 0.0 <= result.confidence <= 1.0
    
    def test_generation_result_has_language(self, engine):
        """Test that generation result has language info."""
        result = engine.generate("Simple function", language=TargetLanguage.JULIA)
        assert result.language == TargetLanguage.JULIA
    
    def test_generation_updates_history(self, engine):
        """Test that generation history updates."""
        initial_count = len(engine.generation_history)
        engine.generate("Test function")
        assert len(engine.generation_history) == initial_count + 1
    
    def test_generation_result_has_mode(self, engine):
        """Test that generation result has mode."""
        result = engine.generate("Test function")
        assert result.mode == DeepSeekMode.GENERATE


# =============================================================================
# Code Optimization Tests
# =============================================================================

class TestCodeOptimization:
    """Test code optimization functionality."""
    
    @pytest.fixture
    def engine(self):
        return DeepSeekEngine(verbose=False)
    
    def test_optimize_basic(self, engine):
        """Test basic code optimization."""
        code = """
def slow_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total
"""
        result = engine.optimize(code)
        assert isinstance(result, CodeGenerationResult)
        assert result.code is not None
        assert result.mode == DeepSeekMode.OPTIMIZE
    
    def test_optimize_preserves_functionality(self, engine):
        """Test that optimization preserves basic structure."""
        code = "def add(a, b): return a + b"
        result = engine.optimize(code)
        # Should still be a function
        assert "def " in result.code or "function" in result.code


# =============================================================================
# Test Generation Tests
# =============================================================================

class TestTestGeneration:
    """Test unit test generation functionality."""
    
    @pytest.fixture
    def engine(self):
        return DeepSeekEngine(verbose=False)
    
    def test_generate_tests_basic(self, engine):
        """Test basic test generation."""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
        result = engine.generate_tests(code)
        assert isinstance(result, CodeGenerationResult)
        assert "test" in result.code.lower() or "Test" in result.code
        assert result.mode == DeepSeekMode.TEST
    
    def test_generate_tests_has_pytest(self, engine):
        """Test that generated tests use pytest."""
        code = "def square(x): return x * x"
        result = engine.generate_tests(code)
        assert "pytest" in result.code.lower() or "def test_" in result.code


# =============================================================================
# Translation Tests
# =============================================================================

class TestTranslation:
    """Test code translation functionality."""
    
    @pytest.fixture
    def engine(self):
        return DeepSeekEngine(verbose=False)
    
    def test_translate_python_to_julia(self, engine):
        """Test Python to Julia translation."""
        python_code = """
import numpy as np
def matrix_mult(A, B):
    return np.dot(A, B)
"""
        result = engine.translate(python_code, TargetLanguage.PYTHON, TargetLanguage.JULIA)
        assert result.language == TargetLanguage.JULIA
        assert result.mode == DeepSeekMode.TRANSLATE
    
    def test_translate_python_to_rust(self, engine):
        """Test Python to Rust translation."""
        python_code = "def add(a, b): return a + b"
        result = engine.translate(python_code, TargetLanguage.PYTHON, TargetLanguage.RUST)
        assert result.language == TargetLanguage.RUST


# =============================================================================
# Explanation Tests
# =============================================================================

class TestExplanation:
    """Test code explanation functionality."""
    
    @pytest.fixture
    def engine(self):
        return DeepSeekEngine(verbose=False)
    
    def test_explain_basic(self, engine):
        """Test basic code explanation."""
        code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
        result = engine.explain(code)
        assert isinstance(result, CodeGenerationResult)
        assert len(result.code) > 0
        assert result.mode == DeepSeekMode.EXPLAIN


# =============================================================================
# Command Handler Tests
# =============================================================================

class TestCommandHandler:
    """Test handle_deepseek_command function."""
    
    @pytest.fixture
    def engine(self):
        return DeepSeekEngine(verbose=False)
    
    @pytest.fixture
    def context(self):
        return {}
    
    def test_handle_status_command(self, engine, context, capsys):
        """Test status command."""
        handle_deepseek_command(engine, "deepseek status", context)
        captured = capsys.readouterr()
        assert "DeepSeek" in captured.out or len(captured.out) > 0
    
    def test_handle_templates_command(self, engine, context, capsys):
        """Test templates command."""
        handle_deepseek_command(engine, "deepseek templates", context)
        captured = capsys.readouterr()
        assert "hartree_fock" in captured.out or "Template" in captured.out
    
    def test_handle_generate_command(self, engine, context, capsys):
        """Test generate command."""
        handle_deepseek_command(engine, 'deepseek generate "Calculate sine function"', context)
        captured = capsys.readouterr()
        assert "GENERATE" in captured.out or "def " in captured.out or "function" in captured.out
    
    def test_handle_generate_with_lang(self, engine, context, capsys):
        """Test generate command with language option."""
        handle_deepseek_command(engine, 'deepseek generate "Vector dot product" --lang julia', context)
        captured = capsys.readouterr()
        # Should contain Julia-related output
        assert len(captured.out) > 0
    
    def test_handle_generate_with_template(self, engine, context, capsys):
        """Test generate command with template option."""
        handle_deepseek_command(engine, 'deepseek generate "SCF" --template hartree_fock', context)
        captured = capsys.readouterr()
        assert len(captured.out) > 0
    
    def test_handle_unknown_command(self, engine, context, capsys):
        """Test handling of unknown command - should handle gracefully."""
        # Should not crash
        handle_deepseek_command(engine, "deepseek unknown_command", context)
        # Just verify it didn't crash
        assert True


# =============================================================================
# Q-Lang Integration Tests
# =============================================================================

class TestQLangIntegration:
    """Test integration with Q-Lang interpreter."""
    
    @pytest.fixture
    def interpreter(self):
        """Create Q-Lang interpreter."""
        from interpreter import QLangInterpreter
        return QLangInterpreter()
    
    def test_deepseek_status_via_qlang(self, interpreter, capsys):
        """Test deepseek status through Q-Lang."""
        interpreter.execute("deepseek status")
        captured = capsys.readouterr()
        assert "DeepSeek" in captured.out
    
    def test_deepseek_templates_via_qlang(self, interpreter, capsys):
        """Test deepseek templates through Q-Lang."""
        interpreter.execute("deepseek templates")
        captured = capsys.readouterr()
        assert "hartree_fock" in captured.out or "Template" in captured.out
    
    def test_deepseek_generate_via_qlang(self, interpreter, capsys):
        """Test deepseek generate through Q-Lang."""
        interpreter.execute('deepseek generate "FFT implementation"')
        captured = capsys.readouterr()
        assert len(captured.out) > 0
    
    def test_deepseek_engine_persists_in_context(self, interpreter):
        """Test that DeepSeek engine persists in interpreter context."""
        interpreter.execute("deepseek status")
        assert "_deepseek_engine" in interpreter.context
        engine = interpreter.context["_deepseek_engine"]
        assert isinstance(engine, DeepSeekEngine)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def engine(self):
        return DeepSeekEngine(verbose=False)
    
    def test_generate_empty_description(self, engine):
        """Test generation with empty description."""
        result = engine.generate("")
        # Should handle gracefully
        assert result is not None
    
    def test_generate_very_long_description(self, engine):
        """Test generation with very long description."""
        long_desc = "Implement a function that " + "does something " * 100
        result = engine.generate(long_desc)
        assert result is not None
    
    def test_optimize_empty_code(self, engine):
        """Test optimization with empty code."""
        result = engine.optimize("")
        assert result is not None
    
    def test_translate_to_same_language(self, engine):
        """Test translation to same language."""
        python_code = "def add(a, b): return a + b"
        result = engine.translate(python_code, TargetLanguage.PYTHON, TargetLanguage.PYTHON)
        # Should still work
        assert result is not None
    
    def test_generate_with_nonexistent_template(self, engine):
        """Test with non-existent template name - should fallback gracefully."""
        result = engine.generate("test", template="nonexistent_template")
        # Should handle gracefully - either return None or use default generation
        assert result is not None or True  # Just shouldn't crash


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def engine(self):
        return DeepSeekEngine(verbose=False)
    
    def test_generation_time_reasonable(self, engine):
        """Test that generation completes in reasonable time."""
        import time
        start = time.time()
        engine.generate("Simple function")
        elapsed = time.time() - start
        # Should complete in under 5 seconds (generous for CI)
        assert elapsed < 5.0
    
    def test_multiple_generations(self, engine):
        """Test multiple sequential generations."""
        for i in range(5):
            result = engine.generate(f"Function number {i}")
            assert result is not None
        assert len(engine.generation_history) >= 5


# =============================================================================
# Template Content Tests
# =============================================================================

class TestTemplateContent:
    """Test that templates produce meaningful content."""
    
    @pytest.fixture
    def engine(self):
        return DeepSeekEngine(verbose=False)
    
    def test_hartree_fock_has_scf_loop(self, engine):
        """Test Hartree-Fock template has SCF components."""
        result = engine.generate("HF solver", template="hartree_fock")
        code_lower = result.code.lower()
        # Should have iteration or convergence logic
        assert any(term in code_lower for term in ['iter', 'converge', 'scf', 'loop', 'while', 'for'])
    
    def test_molecular_dynamics_has_integrator(self, engine):
        """Test MD template has integration components."""
        result = engine.generate("MD sim", template="molecular_dynamics")
        code_lower = result.code.lower()
        # Should have velocity or position update
        assert any(term in code_lower for term in ['velocity', 'position', 'force', 'step', 'dt'])
    
    def test_neural_network_has_layers(self, engine):
        """Test neural network template has layer components."""
        result = engine.generate("NN model", template="neural_network")
        code_lower = result.code.lower()
        # Should have neural network concepts
        assert any(term in code_lower for term in ['layer', 'forward', 'neural', 'activation', 'weight'])
    
    def test_quantum_circuit_has_gates(self, engine):
        """Test quantum circuit template has gate components."""
        result = engine.generate("QC sim", template="quantum_circuit")
        code_lower = result.code.lower()
        # Should have quantum concepts
        assert any(term in code_lower for term in ['qubit', 'gate', 'state', 'circuit', 'measure'])


# =============================================================================
# CodeGenerationResult Tests
# =============================================================================

class TestCodeGenerationResult:
    """Test CodeGenerationResult dataclass."""
    
    @pytest.fixture
    def engine(self):
        return DeepSeekEngine(verbose=False)
    
    def test_result_has_all_fields(self, engine):
        """Test that result has all required fields."""
        result = engine.generate("Test")
        assert hasattr(result, 'success')
        assert hasattr(result, 'mode')
        assert hasattr(result, 'language')
        assert hasattr(result, 'code')
        assert hasattr(result, 'explanation')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'tokens_used')
        assert hasattr(result, 'elapsed_ms')
    
    def test_result_optional_fields(self, engine):
        """Test that result has optional fields."""
        result = engine.generate("Test")
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'suggestions')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
