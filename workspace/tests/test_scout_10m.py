"""
QENEX Scout 10M Context Tests
==============================
Tests for the Scout 10M scientific reasoning system.

Author: QENEX Sovereign Agent
Date: 2026-01-10
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'qenex-qlang', 'src'))

from scout_10m import (
    Scout10MContext,
    ScoutReasoner,
    ScoutResult,
    ScoutMode,
    ContextChunk
)


class TestScout10MContext:
    """Test Scout context management."""
    
    @pytest.fixture
    def context(self):
        """Create a context manager for tests."""
        return Scout10MContext(verbose=False)
    
    def test_context_creation(self, context):
        """Test context initialization."""
        assert context.MAX_TOKENS == 10_000_000
        assert context.total_tokens == 0
        assert len(context.chunks) == 0
    
    def test_token_estimation(self, context):
        """Test token count estimation."""
        text = "Hello world"  # 11 chars
        tokens = context.estimate_tokens(text)
        assert tokens == 11 // 4  # 2 tokens (conservative)
    
    def test_add_chunk(self, context):
        """Test adding a chunk to context."""
        content = "This is test content for the context."
        chunk_id = context.add_chunk(
            content=content,
            source="test.py",
            chunk_type="code"
        )
        
        assert chunk_id is not None
        assert len(context.chunks) == 1
        assert context.total_tokens > 0
    
    def test_add_multiple_chunks(self, context):
        """Test adding multiple chunks."""
        for i in range(5):
            context.add_chunk(
                content=f"Content {i} " * 100,
                source=f"file_{i}.py",
                chunk_type="code"
            )
        
        assert len(context.chunks) == 5
    
    def test_remove_chunk(self, context):
        """Test removing a chunk."""
        chunk_id = context.add_chunk(
            content="Test content",
            source="test.py",
            chunk_type="code"
        )
        
        tokens_before = context.total_tokens
        removed = context.remove_chunk(chunk_id)
        
        assert removed is True
        assert context.total_tokens < tokens_before
        assert len(context.chunks) == 0
    
    def test_clear_context(self, context):
        """Test clearing all context."""
        for i in range(3):
            context.add_chunk(f"Content {i}", f"file{i}.py", "code")
        
        context.clear()
        
        assert len(context.chunks) == 0
        assert context.total_tokens == 0
    
    def test_get_stats(self, context):
        """Test context statistics."""
        context.add_chunk("Code content", "code.py", "code")
        context.add_chunk("Paper content", "paper.md", "paper")
        
        stats = context.get_stats()
        
        assert stats['total_chunks'] == 2
        assert 'code' in stats['chunks_by_type']
        assert 'paper' in stats['chunks_by_type']
        assert stats['usage_percent'] < 1.0  # Very small usage


class TestScoutContextFileLoading:
    """Test file loading into context."""
    
    @pytest.fixture
    def context(self):
        """Create a context manager for tests."""
        return Scout10MContext(verbose=False)
    
    def test_load_file(self, context):
        """Test loading a Python file."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Test Python file\ndef hello():\n    print('Hello')\n")
            temp_path = f.name
        
        try:
            chunk_id = context.load_file(temp_path)
            assert chunk_id is not None
            assert len(context.chunks) == 1
            
            # Check chunk type detection
            chunk = context.chunks[chunk_id]
            assert chunk.chunk_type == 'code'
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self, context):
        """Test loading a file that doesn't exist."""
        chunk_id = context.load_file("/nonexistent/path/file.py")
        assert chunk_id is None
    
    def test_load_directory(self, context):
        """Test loading files from directory."""
        # Create temp directory with files
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                Path(tmpdir, f"file{i}.py").write_text(f"# File {i}\n")
            
            loaded = context.load_directory(tmpdir, "*.py")
            assert loaded == 3
            assert len(context.chunks) == 3


class TestScoutReasoner:
    """Test Scout reasoning capabilities."""
    
    @pytest.fixture
    def reasoner(self):
        """Create a reasoner for tests."""
        return ScoutReasoner(verbose=False)
    
    def test_reasoner_creation(self, reasoner):
        """Test reasoner initialization."""
        assert reasoner.context is not None
        assert len(reasoner.templates) == 6  # All modes have templates
    
    def test_reason_mode(self, reasoner):
        """Test basic reasoning."""
        result = reasoner.reason("Test query", mode=ScoutMode.REASON)
        
        assert result.success
        assert result.mode == ScoutMode.REASON
        assert result.confidence > 0
        assert "## Reasoning" in result.output
    
    def test_hypothesize_mode(self, reasoner):
        """Test hypothesis generation."""
        result = reasoner.reason("quantum computing", mode=ScoutMode.HYPOTHESIZE)
        
        assert result.success
        assert result.mode == ScoutMode.HYPOTHESIZE
        assert "Hypothesis" in result.output
    
    def test_prove_mode(self, reasoner):
        """Test proof construction."""
        result = reasoner.reason("P implies Q", mode=ScoutMode.PROVE)
        
        assert result.success
        assert result.mode == ScoutMode.PROVE
        assert "Proof" in result.output
    
    def test_verify_mode(self, reasoner):
        """Test verification."""
        result = reasoner.reason("Energy is conserved", mode=ScoutMode.VERIFY)
        
        assert result.success
        assert result.mode == ScoutMode.VERIFY
        assert "Verification" in result.output
    
    def test_synthesize_mode(self, reasoner):
        """Test synthesis."""
        result = reasoner.reason("Compare approaches", mode=ScoutMode.SYNTHESIZE)
        
        assert result.success
        assert result.mode == ScoutMode.SYNTHESIZE
    
    def test_analyze_mode(self, reasoner):
        """Test analysis."""
        result = reasoner.reason("Code architecture", mode=ScoutMode.ANALYZE)
        
        assert result.success
        assert result.mode == ScoutMode.ANALYZE
        assert "Analysis" in result.output


class TestScoutResult:
    """Test ScoutResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a ScoutResult."""
        result = ScoutResult(
            success=True,
            mode=ScoutMode.REASON,
            output="Test output",
            reasoning_trace=["Step 1", "Step 2"],
            confidence=0.95,
            tokens_used=1000,
            elapsed_ms=50.0,
            citations=["source1.py", "source2.py"]
        )
        
        assert result.success
        assert result.mode == ScoutMode.REASON
        assert result.confidence == 0.95
        assert len(result.reasoning_trace) == 2
        assert len(result.citations) == 2


class TestScoutQLangIntegration:
    """Test Scout integration with Q-Lang interpreter."""
    
    @pytest.fixture
    def interpreter(self):
        """Create Q-Lang interpreter."""
        from interpreter import QLangInterpreter
        return QLangInterpreter()
    
    def test_scout_status_command(self, interpreter):
        """Test scout status command."""
        code = "scout status"
        # Should not raise
        interpreter.execute(code)
    
    def test_scout_context_stats(self, interpreter):
        """Test scout context stats command."""
        code = "scout context stats"
        interpreter.execute(code)
    
    def test_scout_reason_command(self, interpreter):
        """Test scout reason command."""
        code = 'scout reason "Test scientific query"'
        interpreter.execute(code)
        
        # Result should be stored in context
        assert "scout_result" in interpreter.context
        assert interpreter.context["scout_result"].success
    
    def test_scout_hypothesize_command(self, interpreter):
        """Test scout hypothesize command."""
        code = 'scout hypothesize "novel materials"'
        interpreter.execute(code)
        
        assert "scout_result" in interpreter.context
        assert interpreter.context["scout_result"].mode == ScoutMode.HYPOTHESIZE
    
    def test_scout_context_load(self, interpreter):
        """Test loading file into Scout context."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Test file\nprint('hello')\n")
            temp_path = f.name
        
        try:
            code = f'scout context load {temp_path}'
            interpreter.execute(code)
            
            # Verify context was loaded
            reasoner = interpreter.context.get("_scout_reasoner")
            assert reasoner is not None
            assert len(reasoner.context.chunks) == 1
        finally:
            os.unlink(temp_path)


class TestContextChunk:
    """Test ContextChunk dataclass."""
    
    def test_chunk_creation(self):
        """Test creating a ContextChunk."""
        chunk = ContextChunk(
            id="abc123",
            content="Test content",
            token_count=100,
            source="test.py",
            chunk_type="code",
            metadata={"key": "value"}
        )
        
        assert chunk.id == "abc123"
        assert chunk.token_count == 100
        assert chunk.chunk_type == "code"
        assert chunk.metadata["key"] == "value"


class TestScoutModeEnum:
    """Test ScoutMode enumeration."""
    
    def test_all_modes_exist(self):
        """Test that all modes are defined."""
        assert ScoutMode.REASON is not None
        assert ScoutMode.SYNTHESIZE is not None
        assert ScoutMode.PROVE is not None
        assert ScoutMode.HYPOTHESIZE is not None
        assert ScoutMode.VERIFY is not None
        assert ScoutMode.ANALYZE is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
