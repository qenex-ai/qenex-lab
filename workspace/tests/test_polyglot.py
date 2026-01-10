"""
QENEX Polyglot System Tests
===========================
Tests for the synergistic multi-language computing system.

Author: QENEX Sovereign Agent
Date: 2026-01-10
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'qenex-qlang', 'src'))

from polyglot import (
    PolyglotDispatcher, 
    PolyglotResult, 
    BenchmarkResult,
    Backend
)


class TestBackendEnum:
    """Test Backend enumeration."""
    
    def test_backend_values(self):
        """Test that all backend values exist."""
        assert Backend.PYTHON is not None
        assert Backend.JULIA is not None
        assert Backend.RUST is not None
        assert Backend.AUTO is not None


class TestPolyglotResult:
    """Test PolyglotResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a PolyglotResult."""
        result = PolyglotResult(
            success=True,
            backend_used=Backend.PYTHON,
            output=np.array([1, 2, 3]),
            time_ms=10.5,
            message="Test message"
        )
        assert result.success is True
        assert result.backend_used == Backend.PYTHON
        assert result.time_ms == 10.5
        assert result.message == "Test message"


class TestPolyglotDispatcher:
    """Test PolyglotDispatcher class."""
    
    @pytest.fixture
    def dispatcher(self):
        """Create a dispatcher for tests."""
        return PolyglotDispatcher(verbose=False)
    
    def test_dispatcher_creation(self, dispatcher):
        """Test dispatcher initialization."""
        assert dispatcher._python_available is True
        # Julia and Rust availability depends on system
    
    def test_select_backend_small_matrix(self, dispatcher):
        """Test backend selection for small matrices."""
        backend = dispatcher.select_backend('matmul', 50)
        assert backend == Backend.PYTHON
    
    def test_select_backend_optimization(self, dispatcher):
        """Test backend selection for optimization tasks."""
        backend = dispatcher.select_backend('optimize', 1000)
        assert backend == Backend.PYTHON
    
    def test_select_backend_symbolic(self, dispatcher):
        """Test backend selection for symbolic tasks."""
        backend = dispatcher.select_backend('symbolic', 100)
        assert backend == Backend.PYTHON


class TestMatrixOperations:
    """Test matrix operations via polyglot system."""
    
    @pytest.fixture
    def dispatcher(self):
        """Create a dispatcher for tests."""
        return PolyglotDispatcher(verbose=False)
    
    def test_matmul_small_python(self, dispatcher):
        """Test small matrix multiplication uses Python."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        
        result = dispatcher.matmul(A, B)
        
        assert result.success
        assert result.backend_used == Backend.PYTHON
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(result.output, expected)
    
    def test_matmul_explicit_python(self, dispatcher):
        """Test explicit Python backend selection."""
        A = np.random.rand(100, 100)
        B = np.random.rand(100, 100)
        
        result = dispatcher.matmul(A, B, backend=Backend.PYTHON)
        
        assert result.success
        assert result.backend_used == Backend.PYTHON
        expected = A @ B
        np.testing.assert_array_almost_equal(result.output, expected)


class TestLinearAlgebra:
    """Test linear algebra operations."""
    
    @pytest.fixture
    def dispatcher(self):
        """Create a dispatcher for tests."""
        return PolyglotDispatcher(verbose=False)
    
    def test_eigensolve(self, dispatcher):
        """Test eigenvalue decomposition."""
        M = np.array([[4.0, 1.0], [1.0, 3.0]])
        
        result = dispatcher.eigensolve(M)
        
        assert result.success
        vals, vecs = result.output
        
        # Verify eigenvalue equation: M @ v = λ * v
        for i in range(len(vals)):
            lhs = M @ vecs[:, i]
            rhs = vals[i] * vecs[:, i]
            np.testing.assert_array_almost_equal(lhs, rhs, decimal=10)
    
    def test_svd(self, dispatcher):
        """Test SVD decomposition."""
        # Use square matrix for simpler reconstruction
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = dispatcher.svd(M)
        
        assert result.success
        U, S, Vt = result.output
        
        # Reconstruct and verify: M ≈ U @ diag(S) @ Vt
        # For NumPy SVD, Vt is already transposed
        reconstructed = U[:, :len(S)] @ np.diag(S) @ Vt[:len(S), :]
        np.testing.assert_array_almost_equal(M, reconstructed, decimal=10)
    
    def test_solve(self, dispatcher):
        """Test linear system solve."""
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([9.0, 8.0])
        
        result = dispatcher.solve(A, b)
        
        assert result.success
        expected = np.array([2.0, 3.0])
        np.testing.assert_array_almost_equal(result.output, expected)
    
    def test_fft(self, dispatcher):
        """Test FFT computation."""
        x = np.array([1.0, 0.0, -1.0, 0.0])
        
        result = dispatcher.fft(x)
        
        assert result.success
        expected = np.fft.fft(x)
        np.testing.assert_array_almost_equal(result.output, expected)


class TestQLangIntegration:
    """Test polyglot integration with Q-Lang."""
    
    @pytest.fixture
    def interpreter(self):
        """Create Q-Lang interpreter."""
        from interpreter import QLangInterpreter
        return QLangInterpreter()
    
    def test_polyglot_status_command(self, interpreter):
        """Test polyglot status command."""
        code = "polyglot status"
        # Should not raise
        interpreter.execute(code)
    
    def test_polyglot_matmul_command(self, interpreter):
        """Test polyglot matmul through Q-Lang."""
        code = '''
A = [[1.0, 2.0], [3.0, 4.0]]
B = [[5.0, 6.0], [7.0, 8.0]]
polyglot matmul A B result
'''
        interpreter.execute(code)
        
        assert "result" in interpreter.context
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(
            interpreter.context["result"], 
            expected,
            decimal=8
        )
    
    def test_polyglot_eigen_command(self, interpreter):
        """Test polyglot eigen through Q-Lang."""
        code = '''
M = [[4.0, 1.0], [1.0, 3.0]]
polyglot eigen M
'''
        interpreter.execute(code)
        
        assert "eigenvalues" in interpreter.context
        assert "eigenvectors" in interpreter.context
    
    def test_polyglot_solve_command(self, interpreter):
        """Test polyglot solve through Q-Lang."""
        code = '''
A = [[3.0, 1.0], [1.0, 2.0]]
b = [9.0, 8.0]
polyglot solve A b x
'''
        interpreter.execute(code)
        
        assert "x" in interpreter.context
        expected = np.array([2.0, 3.0])
        np.testing.assert_array_almost_equal(
            interpreter.context["x"],
            expected,
            decimal=8
        )
    
    def test_polyglot_explicit_backend(self, interpreter):
        """Test explicit backend selection through Q-Lang."""
        code = '''
A = [[1.0, 2.0], [3.0, 4.0]]
B = [[5.0, 6.0], [7.0, 8.0]]
polyglot.python matmul A B C_python
'''
        interpreter.execute(code)
        
        assert "C_python" in interpreter.context


class TestBenchmarks:
    """Test benchmark functionality."""
    
    @pytest.fixture
    def dispatcher(self):
        """Create a dispatcher for tests."""
        return PolyglotDispatcher(verbose=False)
    
    def test_benchmark_returns_results(self, dispatcher):
        """Test that benchmarks return valid results."""
        results = dispatcher.benchmark_backends(sizes=[50, 100])
        
        assert 'matmul' in results
        assert 'eigensolve' in results
        assert 'svd' in results
        
        # Should have results for each size
        assert len(results['matmul']) >= 2  # At least Python for both sizes


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
