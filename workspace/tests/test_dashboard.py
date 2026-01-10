"""
Tests for QENEX Lab Web Dashboard.

Tests cover:
- Dashboard state management
- Q-Lang REPL functions
- Research functions
- Scout functions
- DeepSeek functions
- Orchestrator functions
"""

import pytest
import sys
import os

# Add package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "qenex-qlang", "src"))

from dashboard import (
    QenexDashboardState,
    execute_qlang,
    get_qlang_context,
    search_papers,
    fetch_paper,
    scout_reason,
    load_paper_to_scout,
    generate_code,
    list_templates,
    run_research_pipeline,
    run_implementation_pipeline,
    get_orchestrator_status,
    get_system_status,
    _state,
    GRADIO_AVAILABLE,
    INTERPRETER_AVAILABLE,
    RESEARCH_AVAILABLE,
    SCOUT_AVAILABLE,
    DEEPSEEK_AVAILABLE,
    ORCHESTRATOR_AVAILABLE,
)


# ============================================================================
# Test Dashboard State
# ============================================================================

class TestDashboardState:
    """Tests for dashboard state management."""
    
    def test_state_initialization(self):
        """Test initial state."""
        state = QenexDashboardState()
        assert state.interpreter is None
        assert state.research is None
        assert state.scout is None
        assert state.deepseek is None
        assert state.orchestrator is None
        assert state.history == []
        assert state.papers == {}
    
    def test_init_interpreter(self):
        """Test interpreter initialization."""
        state = QenexDashboardState()
        interp = state.init_interpreter()
        
        if INTERPRETER_AVAILABLE:
            assert interp is not None
            # Second call should return same instance
            assert state.init_interpreter() is interp
        else:
            assert interp is None
    
    def test_init_research(self):
        """Test research engine initialization."""
        state = QenexDashboardState()
        research = state.init_research()
        
        if RESEARCH_AVAILABLE:
            assert research is not None
            assert state.init_research() is research
        else:
            assert research is None
    
    def test_init_scout(self):
        """Test scout initialization."""
        state = QenexDashboardState()
        scout = state.init_scout()
        
        if SCOUT_AVAILABLE:
            assert scout is not None
            assert state.init_scout() is scout
        else:
            assert scout is None
    
    def test_init_deepseek(self):
        """Test deepseek initialization."""
        state = QenexDashboardState()
        deepseek = state.init_deepseek()
        
        if DEEPSEEK_AVAILABLE:
            assert deepseek is not None
            assert state.init_deepseek() is deepseek
        else:
            assert deepseek is None
    
    def test_init_orchestrator(self):
        """Test orchestrator initialization."""
        state = QenexDashboardState()
        orch = state.init_orchestrator()
        
        if ORCHESTRATOR_AVAILABLE:
            assert orch is not None
            assert state.init_orchestrator() is orch
        else:
            assert orch is None


# ============================================================================
# Test Q-Lang REPL
# ============================================================================

class TestQLangREPL:
    """Tests for Q-Lang REPL functions."""
    
    def test_execute_simple_command(self):
        """Test executing simple Q-Lang command."""
        output, history = execute_qlang("x = 5", "")
        
        # Should succeed
        assert "Error" not in output or "x = 5" in history
    
    def test_execute_with_history(self):
        """Test history accumulation."""
        _, history1 = execute_qlang("a = 1", "")
        _, history2 = execute_qlang("b = 2", history1)
        
        assert "a = 1" in history2
        assert "b = 2" in history2
    
    def test_execute_error_handling(self):
        """Test error handling in execution."""
        output, _ = execute_qlang("invalid_syntax {{{{", "")
        
        # Should handle gracefully
        assert output is not None
    
    def test_get_context(self):
        """Test getting context variables."""
        # Execute something first
        execute_qlang("test_var = 42", "")
        
        context = get_qlang_context()
        assert "Variables" in context or "not available" in context


# ============================================================================
# Test Research Functions
# ============================================================================

class TestResearchFunctions:
    """Tests for research functions."""
    
    def test_search_papers(self):
        """Test paper search."""
        results, summary = search_papers("test query", max_results=2)
        
        # Should return something (either results or error)
        assert results is not None
        assert isinstance(results, str)
    
    def test_search_papers_empty_query(self):
        """Test search with empty query."""
        results, _ = search_papers("", max_results=1)
        assert results is not None
    
    def test_fetch_paper_invalid(self):
        """Test fetching invalid paper."""
        result = fetch_paper("invalid:12345")
        
        # Should handle gracefully
        assert result is not None
        assert isinstance(result, str)


# ============================================================================
# Test Scout Functions
# ============================================================================

class TestScoutFunctions:
    """Tests for Scout functions."""
    
    def test_scout_reason(self):
        """Test Scout reasoning."""
        result = scout_reason("What is 2+2?", mode="reason")
        
        assert result is not None
        assert isinstance(result, str)
    
    def test_scout_reason_modes(self):
        """Test different Scout modes."""
        modes = ["reason", "hypothesize", "prove", "synthesize", "critique"]
        
        for mode in modes:
            result = scout_reason("Test prompt", mode=mode)
            assert result is not None
    
    def test_load_paper_to_scout_not_found(self):
        """Test loading non-existent paper."""
        result = load_paper_to_scout("nonexistent_paper_id")
        
        assert "not found" in result.lower() or "error" in result.lower()


# ============================================================================
# Test DeepSeek Functions
# ============================================================================

class TestDeepSeekFunctions:
    """Tests for DeepSeek functions."""
    
    def test_generate_code(self):
        """Test code generation."""
        result = generate_code("Hello world function", language="python")
        
        assert result is not None
        assert isinstance(result, str)
    
    def test_generate_code_languages(self):
        """Test different target languages."""
        languages = ["python", "julia", "rust"]
        
        for lang in languages:
            result = generate_code("Simple function", language=lang)
            assert result is not None
    
    def test_list_templates(self):
        """Test listing templates."""
        result = list_templates()
        
        assert result is not None
        assert isinstance(result, str)


# ============================================================================
# Test Orchestrator Functions
# ============================================================================

class TestOrchestratorFunctions:
    """Tests for orchestrator functions."""
    
    def test_run_research_pipeline(self):
        """Test research pipeline."""
        result = run_research_pipeline("test topic", max_papers=2)
        
        assert result is not None
        assert isinstance(result, str)
    
    def test_run_implementation_pipeline(self):
        """Test implementation pipeline."""
        result = run_implementation_pipeline("simple function", language="python")
        
        assert result is not None
        assert isinstance(result, str)
    
    def test_get_orchestrator_status(self):
        """Test getting orchestrator status."""
        result = get_orchestrator_status()
        
        assert result is not None
        assert isinstance(result, str)


# ============================================================================
# Test System Status
# ============================================================================

class TestSystemStatus:
    """Tests for system status."""
    
    def test_get_system_status(self):
        """Test system status output."""
        status = get_system_status()
        
        assert "QENEX" in status
        assert "Status" in status
    
    def test_status_shows_components(self):
        """Test that status shows all components."""
        status = get_system_status()
        
        components = ["Interpreter", "Research", "Scout", "DeepSeek", "Orchestrator"]
        for comp in components:
            assert comp in status


# ============================================================================
# Test Availability Flags
# ============================================================================

class TestAvailabilityFlags:
    """Tests for module availability."""
    
    def test_interpreter_available(self):
        """Test interpreter availability flag."""
        assert isinstance(INTERPRETER_AVAILABLE, bool)
    
    def test_research_available(self):
        """Test research availability flag."""
        assert isinstance(RESEARCH_AVAILABLE, bool)
    
    def test_scout_available(self):
        """Test scout availability flag."""
        assert isinstance(SCOUT_AVAILABLE, bool)
    
    def test_deepseek_available(self):
        """Test deepseek availability flag."""
        assert isinstance(DEEPSEEK_AVAILABLE, bool)
    
    def test_orchestrator_available(self):
        """Test orchestrator availability flag."""
        assert isinstance(ORCHESTRATOR_AVAILABLE, bool)
    
    def test_gradio_availability(self):
        """Test Gradio availability flag."""
        assert isinstance(GRADIO_AVAILABLE, bool)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self):
        """Test a complete workflow through the dashboard."""
        # 1. Execute Q-Lang
        output1, history = execute_qlang("x = 10", "")
        
        # 2. Search papers
        results, _ = search_papers("quantum", max_results=1)
        
        # 3. Use Scout
        scout_result = scout_reason("Explain quantum computing")
        
        # 4. Generate code
        code_result = generate_code("quantum simulator")
        
        # 5. Check orchestrator status
        status = get_orchestrator_status()
        
        # All should complete without crashing
        assert all([output1, results, scout_result, code_result, status])
    
    def test_state_persistence(self):
        """Test that global state persists across calls."""
        # Initialize once
        _state.init_interpreter()
        
        # Should reuse same instance
        interp1 = _state.interpreter
        _state.init_interpreter()
        interp2 = _state.interpreter
        
        if INTERPRETER_AVAILABLE:
            assert interp1 is interp2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
