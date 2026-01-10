"""
QENEX Q-Lang Julia Bridge Tests
================================
Tests for the Julia integration module.

Author: QENEX Sovereign Agent
Date: 2026-01-10
"""

import pytest
import numpy as np
import sys
import os

# Add the Q-Lang source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'qenex-qlang', 'src'))

from julia_bridge import JuliaBridge, JuliaResult


class TestJuliaBridge:
    """Test suite for JuliaBridge class."""
    
    @pytest.fixture(scope="class")
    def bridge(self):
        """Create a shared JuliaBridge instance for tests."""
        return JuliaBridge(verbose=False)
    
    def test_julia_available(self, bridge):
        """Test that Julia is available on the system."""
        assert bridge.available, "Julia should be available"
    
    def test_julia_result_dataclass(self):
        """Test JuliaResult dataclass structure."""
        result = JuliaResult(
            success=True,
            output=[1, 2, 3],
            stdout="test output",
            stderr="",
            elapsed_ms=100.0
        )
        assert result.success is True
        assert result.output == [1, 2, 3]
        assert result.stdout == "test output"
        assert result.stderr == ""
        assert result.elapsed_ms == 100.0
    

class TestMatrixOperations:
    """Test matrix operations via Julia."""
    
    @pytest.fixture(scope="class")
    def bridge(self):
        """Create a shared JuliaBridge instance."""
        return JuliaBridge(verbose=False)
    
    @pytest.mark.skipif(
        not JuliaBridge(verbose=False).available,
        reason="Julia not available"
    )
    def test_matrix_multiply_2x2(self, bridge):
        """Test 2x2 matrix multiplication."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        
        result = bridge.matrix_multiply(A, B)
        
        assert result.success, f"Matrix multiply failed: {result.stderr}"
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(result.output, expected, decimal=10)
    
    @pytest.mark.skipif(
        not JuliaBridge(verbose=False).available,
        reason="Julia not available"
    )
    def test_matrix_multiply_dimension_mismatch(self, bridge):
        """Test that dimension mismatch is caught."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2
        B = np.array([[1.0, 2.0, 3.0]])         # 1x3
        
        result = bridge.matrix_multiply(A, B)
        
        assert not result.success
        assert "don't match" in result.stderr.lower()
    
    @pytest.mark.skipif(
        not JuliaBridge(verbose=False).available,
        reason="Julia not available"
    )
    def test_matrix_multiply_larger(self, bridge):
        """Test larger matrix multiplication."""
        np.random.seed(42)
        A = np.random.rand(50, 50)
        B = np.random.rand(50, 50)
        
        result = bridge.matrix_multiply(A, B)
        
        assert result.success, f"Matrix multiply failed: {result.stderr}"
        expected = A @ B
        np.testing.assert_array_almost_equal(result.output, expected, decimal=8)


class TestLinearAlgebra:
    """Test linear algebra operations via Julia."""
    
    @pytest.fixture(scope="class")
    def bridge(self):
        """Create a shared JuliaBridge instance."""
        return JuliaBridge(verbose=False)
    
    @pytest.mark.skipif(
        not JuliaBridge(verbose=False).available,
        reason="Julia not available"
    )
    def test_solve_linear_system(self, bridge):
        """Test solving Ax = b."""
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([9.0, 8.0])
        
        result = bridge.solve_linear(A, b)
        
        assert result.success, f"Linear solve failed: {result.stderr}"
        expected = np.array([2.0, 3.0])
        np.testing.assert_array_almost_equal(result.output, expected, decimal=10)
    
    @pytest.mark.skipif(
        not JuliaBridge(verbose=False).available,
        reason="Julia not available"
    )
    def test_eigenvalues_symmetric(self, bridge):
        """Test eigenvalue computation for symmetric matrix."""
        M = np.array([[4.0, 1.0], [1.0, 3.0]])  # Symmetric
        
        result = bridge.eigenvalues(M)
        
        assert result.success, f"Eigen failed: {result.stderr}"
        vals, vecs = result.output
        
        # Check eigenvalues match NumPy
        np_vals = np.linalg.eigvals(M)
        np.testing.assert_array_almost_equal(
            np.sort(np.real(vals)), 
            np.sort(np.real(np_vals)), 
            decimal=10
        )
    
    @pytest.mark.skipif(
        not JuliaBridge(verbose=False).available,
        reason="Julia not available"
    )
    def test_eigenvalues_non_square_fails(self, bridge):
        """Test that non-square matrix fails for eigenvalues."""
        M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
        
        result = bridge.eigenvalues(M)
        
        assert not result.success
        assert "square" in result.stderr.lower()
    
    @pytest.mark.skipif(
        not JuliaBridge(verbose=False).available,
        reason="Julia not available"
    )
    def test_svd(self, bridge):
        """Test SVD decomposition."""
        M = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2
        
        result = bridge.svd(M)
        
        assert result.success, f"SVD failed: {result.stderr}"
        U, S, V = result.output
        
        # Check reconstruction: M ≈ U @ diag(S) @ V.T
        reconstructed = U @ np.diag(S) @ V.T
        np.testing.assert_array_almost_equal(M, reconstructed, decimal=10)


class TestBenchmark:
    """Test Julia benchmark functionality."""
    
    @pytest.fixture(scope="class")
    def bridge(self):
        """Create a shared JuliaBridge instance."""
        return JuliaBridge(verbose=False)
    
    @pytest.mark.skipif(
        not JuliaBridge(verbose=False).available,
        reason="Julia not available"
    )
    def test_benchmark_runs(self, bridge):
        """Test that benchmark completes successfully."""
        result = bridge.benchmark(size=100)
        
        assert result.success, f"Benchmark failed: {result.stderr}"
        assert "GFLOPS" in result.stdout
        assert "Mean time" in result.stdout
    
    @pytest.mark.skipif(
        not JuliaBridge(verbose=False).available,
        reason="Julia not available"
    )
    def test_benchmark_timing(self, bridge):
        """Test that benchmark returns valid timing."""
        result = bridge.benchmark(size=50)
        
        assert result.success
        assert result.elapsed_ms > 0
        assert result.elapsed_ms < 60000  # Should complete in under 60s


class TestCustomCode:
    """Test custom Julia code execution."""
    
    @pytest.fixture(scope="class")
    def bridge(self):
        """Create a shared JuliaBridge instance."""
        return JuliaBridge(verbose=False)
    
    @pytest.mark.skipif(
        not JuliaBridge(verbose=False).available,
        reason="Julia not available"
    )
    def test_run_custom_code(self, bridge):
        """Test running custom Julia code."""
        code = 'println("Hello from Julia"); println(2 + 2)'
        result = bridge.run_custom(code)
        
        assert result.success
        assert "Hello from Julia" in result.stdout
        assert "4" in result.stdout
    
    @pytest.mark.skipif(
        not JuliaBridge(verbose=False).available,
        reason="Julia not available"
    )
    def test_run_custom_code_math(self, bridge):
        """Test mathematical computation in custom code."""
        code = '''
        using LinearAlgebra
        A = [1 2; 3 4]
        println("Determinant: ", det(A))
        '''
        result = bridge.run_custom(code)
        
        assert result.success
        assert "Determinant: -2.0" in result.stdout


class TestQLangIntegration:
    """Test Julia integration through Q-Lang interpreter."""
    
    @pytest.fixture(scope="class")
    def interpreter(self):
        """Create a Q-Lang interpreter instance."""
        from interpreter import QLangInterpreter
        return QLangInterpreter()
    
    @pytest.mark.skipif(
        not JuliaBridge(verbose=False).available,
        reason="Julia not available"
    )
    def test_julia_command_in_qlang(self, interpreter):
        """Test Julia benchmark command through Q-Lang."""
        code = "julia benchmark 50"
        # Should not raise
        interpreter.execute(code)
    
    @pytest.mark.skipif(
        not JuliaBridge(verbose=False).available,
        reason="Julia not available"
    )
    def test_julia_matmul_stores_result(self, interpreter):
        """Test Julia matrix multiply stores result in context."""
        code = '''
A = [[1.0, 2.0], [3.0, 4.0]]
B = [[5.0, 6.0], [7.0, 8.0]]
julia matmul A B result_matrix
'''
        interpreter.execute(code)
        
        assert "result_matrix" in interpreter.context
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(
            interpreter.context["result_matrix"], 
            expected, 
            decimal=8
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
