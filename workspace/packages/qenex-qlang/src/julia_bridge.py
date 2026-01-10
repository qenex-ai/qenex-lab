"""
QENEX Q-Lang Julia Bridge
=========================
High-performance numerical computing via Julia integration.

Provides Q-Lang commands for:
- Matrix operations (BLAS-accelerated)
- Tensor contractions
- Scientific computing
- Attention mechanisms (for ML)

Usage in Q-Lang:
    julia matrix_multiply A B       # C = A * B
    julia eigenvalues M             # Eigendecomposition
    julia solve A b                 # Solve Ax = b
    julia attention Q K V scale     # Scaled dot-product attention
    julia benchmark N               # Matrix multiply benchmark

Author: QENEX Sovereign Agent
Date: 2026-01-10
"""

import subprocess
import numpy as np
import json
import tempfile
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class JuliaResult:
    """Result from a Julia computation."""
    success: bool
    output: Any
    stdout: str
    stderr: str
    elapsed_ms: float


class JuliaBridge:
    """
    Bridge between Q-Lang and Julia for high-performance numerics.
    
    Uses subprocess calls to Julia for maximum compatibility.
    For production, consider embedded Julia via PyJulia or FFI.
    """
    
    JULIA_PATH = "/usr/local/bin/julia"
    JULIA_MATH_DIR = "/opt/qenex/brain/scout/julia_math"
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.available = self._check_julia()
        
        if verbose:
            if self.available:
                print("✅ Julia bridge initialized")
                print(f"   Path: {self.JULIA_PATH}")
                print(f"   Math ops: {self.JULIA_MATH_DIR}")
            else:
                print("❌ Julia not available")
    
    def _check_julia(self) -> bool:
        """Check if Julia is available."""
        try:
            result = subprocess.run(
                [self.JULIA_PATH, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _run_julia(self, code: str, timeout: int = 60) -> JuliaResult:
        """Run Julia code and return result."""
        import time
        
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
            f.write(code)
            script_path = f.name
        
        try:
            start = time.time()
            result = subprocess.run(
                [self.JULIA_PATH, script_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            elapsed = (time.time() - start) * 1000
            
            return JuliaResult(
                success=result.returncode == 0,
                output=None,
                stdout=result.stdout,
                stderr=result.stderr,
                elapsed_ms=elapsed
            )
        except subprocess.TimeoutExpired:
            return JuliaResult(
                success=False,
                output=None,
                stdout="",
                stderr="Julia execution timed out",
                elapsed_ms=timeout * 1000
            )
        finally:
            os.unlink(script_path)
    
    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> JuliaResult:
        """
        Perform matrix multiplication C = A * B using Julia's BLAS.
        
        Args:
            A: First matrix (m x k)
            B: Second matrix (k x n)
        
        Returns:
            JuliaResult with C matrix in output field
        """
        m, k = A.shape
        k2, n = B.shape
        
        if k != k2:
            return JuliaResult(
                success=False,
                output=None,
                stdout="",
                stderr=f"Inner dimensions don't match: {k} != {k2}",
                elapsed_ms=0
            )
        
        # Save matrices to temp files
        with tempfile.NamedTemporaryFile(mode='w', suffix='_A.csv', delete=False) as f:
            np.savetxt(f.name, A, delimiter=',')
            a_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_B.csv', delete=False) as f:
            np.savetxt(f.name, B, delimiter=',')
            b_path = f.name
        
        c_path = tempfile.mktemp(suffix='_C.csv')
        
        code = f'''
using LinearAlgebra
using DelimitedFiles

A = readdlm("{a_path}", ',', Float64)
B = readdlm("{b_path}", ',', Float64)

C = A * B

writedlm("{c_path}", C, ',')
println("SUCCESS")
println("Shape: ", size(C))
'''
        
        result = self._run_julia(code)
        
        # Load result
        if result.success and os.path.exists(c_path):
            C = np.loadtxt(c_path, delimiter=',')
            result.output = C
        
        # Cleanup
        for path in [a_path, b_path, c_path]:
            if os.path.exists(path):
                os.unlink(path)
        
        return result
    
    def eigenvalues(self, M: np.ndarray) -> JuliaResult:
        """
        Compute eigenvalues and eigenvectors of a matrix.
        
        Args:
            M: Square matrix
        
        Returns:
            JuliaResult with (eigenvalues, eigenvectors) tuple
        """
        if M.shape[0] != M.shape[1]:
            return JuliaResult(
                success=False,
                output=None,
                stdout="",
                stderr="Matrix must be square",
                elapsed_ms=0
            )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_M.csv', delete=False) as f:
            np.savetxt(f.name, M, delimiter=',')
            m_path = f.name
        
        vals_path = tempfile.mktemp(suffix='_vals.csv')
        vecs_path = tempfile.mktemp(suffix='_vecs.csv')
        
        code = f'''
using LinearAlgebra
using DelimitedFiles

M = readdlm("{m_path}", ',', Float64)

F = eigen(M)

writedlm("{vals_path}", F.values, ',')
writedlm("{vecs_path}", F.vectors, ',')
println("SUCCESS")
println("Eigenvalues computed: ", length(F.values))
'''
        
        result = self._run_julia(code)
        
        if result.success:
            if os.path.exists(vals_path) and os.path.exists(vecs_path):
                vals = np.loadtxt(vals_path, delimiter=',', dtype=complex)
                vecs = np.loadtxt(vecs_path, delimiter=',', dtype=complex)
                result.output = (vals, vecs)
        
        for path in [m_path, vals_path, vecs_path]:
            if os.path.exists(path):
                os.unlink(path)
        
        return result
    
    def solve_linear(self, A: np.ndarray, b: np.ndarray) -> JuliaResult:
        """
        Solve linear system Ax = b.
        
        Args:
            A: Coefficient matrix (n x n)
            b: Right-hand side vector (n,)
        
        Returns:
            JuliaResult with solution vector x
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='_A.csv', delete=False) as f:
            np.savetxt(f.name, A, delimiter=',')
            a_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_b.csv', delete=False) as f:
            np.savetxt(f.name, b, delimiter=',')
            b_path = f.name
        
        x_path = tempfile.mktemp(suffix='_x.csv')
        
        code = f'''
using LinearAlgebra
using DelimitedFiles

A = readdlm("{a_path}", ',', Float64)
b = vec(readdlm("{b_path}", ',', Float64))

x = A \\ b

writedlm("{x_path}", x, ',')
println("SUCCESS")
println("Residual norm: ", norm(A*x - b))
'''
        
        result = self._run_julia(code)
        
        if result.success and os.path.exists(x_path):
            x = np.loadtxt(x_path, delimiter=',')
            result.output = x
        
        for path in [a_path, b_path, x_path]:
            if os.path.exists(path):
                os.unlink(path)
        
        return result
    
    def svd(self, M: np.ndarray) -> JuliaResult:
        """
        Compute Singular Value Decomposition: M = U * S * V'.
        
        Args:
            M: Input matrix
        
        Returns:
            JuliaResult with (U, S, V) tuple
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='_M.csv', delete=False) as f:
            np.savetxt(f.name, M, delimiter=',')
            m_path = f.name
        
        u_path = tempfile.mktemp(suffix='_U.csv')
        s_path = tempfile.mktemp(suffix='_S.csv')
        v_path = tempfile.mktemp(suffix='_V.csv')
        
        code = f'''
using LinearAlgebra
using DelimitedFiles

M = readdlm("{m_path}", ',', Float64)

F = svd(M)

writedlm("{u_path}", F.U, ',')
writedlm("{s_path}", F.S, ',')
writedlm("{v_path}", F.V, ',')
println("SUCCESS")
println("Singular values: ", length(F.S))
'''
        
        result = self._run_julia(code)
        
        if result.success:
            U = np.loadtxt(u_path, delimiter=',') if os.path.exists(u_path) else None
            S = np.loadtxt(s_path, delimiter=',') if os.path.exists(s_path) else None
            V = np.loadtxt(v_path, delimiter=',') if os.path.exists(v_path) else None
            result.output = (U, S, V)
        
        for path in [m_path, u_path, s_path, v_path]:
            if os.path.exists(path):
                os.unlink(path)
        
        return result
    
    def benchmark(self, size: int = 1000) -> JuliaResult:
        """
        Run a matrix multiplication benchmark.
        
        Args:
            size: Matrix dimension (N x N)
        
        Returns:
            JuliaResult with benchmark statistics
        """
        code = f'''
using LinearAlgebra
using Random
using Statistics

N = {size}

println("=" ^ 60)
println("QENEX Julia Benchmark")
println("=" ^ 60)
println("Matrix size: $N x $N")
println("BLAS vendor: $(BLAS.vendor())")
println("BLAS threads: $(BLAS.get_num_threads())")
println("-" ^ 60)

# Warmup
Random.seed!(42)
A = rand(Float64, N, N)
B = rand(Float64, N, N)
C = A * B

# Benchmark
times = Float64[]
for i in 1:5
    local A = rand(Float64, N, N)
    local B = rand(Float64, N, N)
    local t = @elapsed local C = A * B
    push!(times, t)
end

mean_time = mean(times)
gflops = (2 * N^3 * 1e-9) / mean_time

println("Mean time: $(round(mean_time * 1000, digits=2)) ms")
println("Std dev: $(round(std(times) * 1000, digits=2)) ms")
println("GFLOPS: $(round(gflops, digits=2))")
println("Checksum (trace): $(tr(C))")
println("=" ^ 60)
'''
        
        return self._run_julia(code, timeout=120)
    
    def attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                  scale: float = None) -> JuliaResult:
        """
        Compute scaled dot-product attention.
        
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
        
        Args:
            Q: Query matrix (seq_len x head_dim)
            K: Key matrix (seq_len x head_dim)
            V: Value matrix (seq_len x head_dim)
            scale: Scaling factor (default: sqrt(head_dim))
        
        Returns:
            JuliaResult with attention output
        """
        if scale is None:
            scale = np.sqrt(Q.shape[1])
        
        # Save matrices
        with tempfile.NamedTemporaryFile(mode='w', suffix='_Q.csv', delete=False) as f:
            np.savetxt(f.name, Q, delimiter=',')
            q_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_K.csv', delete=False) as f:
            np.savetxt(f.name, K, delimiter=',')
            k_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_V.csv', delete=False) as f:
            np.savetxt(f.name, V, delimiter=',')
            v_path = f.name
        
        out_path = tempfile.mktemp(suffix='_out.csv')
        
        code = f'''
using LinearAlgebra
using DelimitedFiles

Q = readdlm("{q_path}", ',', Float64)
K = readdlm("{k_path}", ',', Float64)
V = readdlm("{v_path}", ',', Float64)
scale = {scale}

# Scaled dot-product attention
scores = Q * transpose(K)
scores ./= scale

# Softmax (row-wise)
scores_exp = exp.(scores .- maximum(scores, dims=2))
attn_weights = scores_exp ./ sum(scores_exp, dims=2)

# Apply to values
output = attn_weights * V

writedlm("{out_path}", output, ',')
println("SUCCESS")
println("Output shape: ", size(output))
'''
        
        result = self._run_julia(code)
        
        if result.success and os.path.exists(out_path):
            output = np.loadtxt(out_path, delimiter=',')
            result.output = output
        
        for path in [q_path, k_path, v_path, out_path]:
            if os.path.exists(path):
                os.unlink(path)
        
        return result
    
    def fft(self, x: np.ndarray) -> JuliaResult:
        """
        Compute Fast Fourier Transform.
        
        Args:
            x: Input array
        
        Returns:
            JuliaResult with FFT output
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='_x.csv', delete=False) as f:
            np.savetxt(f.name, x, delimiter=',')
            x_path = f.name
        
        out_path = tempfile.mktemp(suffix='_fft.csv')
        
        code = f'''
using FFTW
using DelimitedFiles

x = vec(readdlm("{x_path}", ',', Float64))

y = fft(x)

# Save real and imaginary parts
writedlm("{out_path}", [real.(y) imag.(y)], ',')
println("SUCCESS")
println("FFT length: ", length(y))
'''
        
        result = self._run_julia(code)
        
        if result.success and os.path.exists(out_path):
            data = np.loadtxt(out_path, delimiter=',')
            result.output = data[:, 0] + 1j * data[:, 1]
        
        for path in [x_path, out_path]:
            if os.path.exists(path):
                os.unlink(path)
        
        return result
    
    def run_custom(self, julia_code: str) -> JuliaResult:
        """
        Run custom Julia code.
        
        Args:
            julia_code: Julia code to execute
        
        Returns:
            JuliaResult with stdout/stderr
        """
        return self._run_julia(julia_code)
    
    def tensor_contract(self, size: int) -> JuliaResult:
        """
        Run tensor contraction demo from julia_math.
        
        Args:
            size: Tensor dimension
        
        Returns:
            JuliaResult with benchmark info
        """
        script_path = os.path.join(self.JULIA_MATH_DIR, "tensor_ops.jl")
        
        if not os.path.exists(script_path):
            return JuliaResult(
                success=False,
                output=None,
                stdout="",
                stderr=f"tensor_ops.jl not found at {script_path}",
                elapsed_ms=0
            )
        
        import time
        start = time.time()
        
        result = subprocess.run(
            [self.JULIA_PATH, script_path, str(size)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        elapsed = (time.time() - start) * 1000
        
        return JuliaResult(
            success=result.returncode == 0,
            output=None,
            stdout=result.stdout,
            stderr=result.stderr,
            elapsed_ms=elapsed
        )


def handle_julia_command(bridge: JuliaBridge, line: str, context: dict) -> None:
    """
    Handle Julia commands from Q-Lang interpreter.
    
    Args:
        bridge: JuliaBridge instance
        line: Command line starting with 'julia'
        context: Q-Lang context dictionary
    """
    parts = line.split()
    
    if len(parts) < 2:
        print("❌ Usage: julia <command> [args...]")
        print("   Commands: benchmark, matmul, eigen, solve, svd, attention, fft, tensor, run")
        return
    
    cmd = parts[1]
    
    if not bridge.available:
        print("❌ Julia is not available on this system")
        return
    
    try:
        if cmd == "benchmark":
            size = int(parts[2]) if len(parts) > 2 else 1000
            print(f"   [Julia] Running benchmark with {size}x{size} matrices...")
            result = bridge.benchmark(size)
            print(result.stdout)
            if result.stderr:
                print(f"   Errors: {result.stderr}")
            print(f"   Total time: {result.elapsed_ms:.1f} ms")
        
        elif cmd == "tensor":
            size = int(parts[2]) if len(parts) > 2 else 500
            print(f"   [Julia] Running tensor contraction ({size}x{size})...")
            result = bridge.tensor_contract(size)
            print(result.stdout)
            if result.stderr:
                print(f"   Errors: {result.stderr}")
        
        elif cmd == "matmul":
            # julia matmul <A_var> <B_var> [result_var]
            if len(parts) < 4:
                print("❌ Usage: julia matmul <A_var> <B_var> [result_var]")
                return
            
            A_name, B_name = parts[2], parts[3]
            result_name = parts[4] if len(parts) > 4 else "julia_result"
            
            if A_name not in context or B_name not in context:
                print(f"❌ Variables not found: {A_name}, {B_name}")
                return
            
            A = np.array(context[A_name])
            B = np.array(context[B_name])
            
            print(f"   [Julia] Computing {A_name} * {B_name}...")
            result = bridge.matrix_multiply(A, B)
            
            if result.success:
                context[result_name] = result.output
                print(f"✅ Result stored in '{result_name}' ({result.output.shape})")
                print(f"   Time: {result.elapsed_ms:.1f} ms")
            else:
                print(f"❌ {result.stderr}")
        
        elif cmd == "eigen":
            # julia eigen <M_var> [vals_var] [vecs_var]
            if len(parts) < 3:
                print("❌ Usage: julia eigen <M_var> [vals_var] [vecs_var]")
                return
            
            M_name = parts[2]
            vals_name = parts[3] if len(parts) > 3 else "eigenvalues"
            vecs_name = parts[4] if len(parts) > 4 else "eigenvectors"
            
            if M_name not in context:
                print(f"❌ Variable not found: {M_name}")
                return
            
            M = np.array(context[M_name])
            
            print(f"   [Julia] Computing eigendecomposition of {M_name}...")
            result = bridge.eigenvalues(M)
            
            if result.success:
                vals, vecs = result.output
                context[vals_name] = vals
                context[vecs_name] = vecs
                print(f"✅ Eigenvalues stored in '{vals_name}'")
                print(f"✅ Eigenvectors stored in '{vecs_name}'")
                print(f"   Time: {result.elapsed_ms:.1f} ms")
            else:
                print(f"❌ {result.stderr}")
        
        elif cmd == "solve":
            # julia solve <A_var> <b_var> [x_var]
            if len(parts) < 4:
                print("❌ Usage: julia solve <A_var> <b_var> [x_var]")
                return
            
            A_name, b_name = parts[2], parts[3]
            x_name = parts[4] if len(parts) > 4 else "x"
            
            if A_name not in context or b_name not in context:
                print(f"❌ Variables not found")
                return
            
            A = np.array(context[A_name])
            b = np.array(context[b_name])
            
            print(f"   [Julia] Solving {A_name} * x = {b_name}...")
            result = bridge.solve_linear(A, b)
            
            if result.success:
                context[x_name] = result.output
                print(f"✅ Solution stored in '{x_name}'")
                print(f"   Time: {result.elapsed_ms:.1f} ms")
                print(result.stdout)
            else:
                print(f"❌ {result.stderr}")
        
        elif cmd == "svd":
            # julia svd <M_var>
            if len(parts) < 3:
                print("❌ Usage: julia svd <M_var>")
                return
            
            M_name = parts[2]
            
            if M_name not in context:
                print(f"❌ Variable not found: {M_name}")
                return
            
            M = np.array(context[M_name])
            
            print(f"   [Julia] Computing SVD of {M_name}...")
            result = bridge.svd(M)
            
            if result.success:
                U, S, V = result.output
                context["U"] = U
                context["S"] = S
                context["V"] = V
                print(f"✅ U, S, V stored in context")
                print(f"   Singular values: {S[:5]}..." if len(S) > 5 else f"   Singular values: {S}")
                print(f"   Time: {result.elapsed_ms:.1f} ms")
            else:
                print(f"❌ {result.stderr}")
        
        elif cmd == "attention":
            # julia attention <Q_var> <K_var> <V_var> [scale]
            if len(parts) < 5:
                print("❌ Usage: julia attention <Q_var> <K_var> <V_var> [scale]")
                return
            
            Q_name, K_name, V_name = parts[2], parts[3], parts[4]
            scale = float(parts[5]) if len(parts) > 5 else None
            
            for name in [Q_name, K_name, V_name]:
                if name not in context:
                    print(f"❌ Variable not found: {name}")
                    return
            
            Q = np.array(context[Q_name])
            K = np.array(context[K_name])
            V = np.array(context[V_name])
            
            print(f"   [Julia] Computing attention...")
            result = bridge.attention(Q, K, V, scale)
            
            if result.success:
                context["attention_output"] = result.output
                print(f"✅ Result stored in 'attention_output' ({result.output.shape})")
                print(f"   Time: {result.elapsed_ms:.1f} ms")
            else:
                print(f"❌ {result.stderr}")
        
        elif cmd == "fft":
            # julia fft <x_var>
            if len(parts) < 3:
                print("❌ Usage: julia fft <x_var>")
                return
            
            x_name = parts[2]
            
            if x_name not in context:
                print(f"❌ Variable not found: {x_name}")
                return
            
            x = np.array(context[x_name])
            
            print(f"   [Julia] Computing FFT...")
            result = bridge.fft(x)
            
            if result.success:
                context["fft_result"] = result.output
                print(f"✅ Result stored in 'fft_result'")
                print(f"   Time: {result.elapsed_ms:.1f} ms")
            else:
                print(f"❌ {result.stderr}")
        
        elif cmd == "run":
            # julia run "<code>"
            code = " ".join(parts[2:]).strip('"\'')
            
            if not code:
                print("❌ Usage: julia run \"<julia_code>\"")
                return
            
            print(f"   [Julia] Running custom code...")
            result = bridge.run_custom(code)
            print(result.stdout)
            if result.stderr:
                print(f"   Stderr: {result.stderr}")
            print(f"   Time: {result.elapsed_ms:.1f} ms")
        
        else:
            print(f"❌ Unknown Julia command: {cmd}")
            print("   Available: benchmark, tensor, matmul, eigen, solve, svd, attention, fft, run")
    
    except Exception as e:
        print(f"❌ Julia Error: {e}")


# Test the bridge
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           QENEX Q-Lang Julia Bridge                          ║
    ║           High-Performance Numerical Computing               ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    bridge = JuliaBridge(verbose=True)
    
    if bridge.available:
        print("\n--- Running Benchmark ---")
        result = bridge.benchmark(500)
        print(result.stdout)
        
        print("\n--- Matrix Multiply Test ---")
        A = np.random.rand(100, 100)
        B = np.random.rand(100, 100)
        result = bridge.matrix_multiply(A, B)
        if result.success:
            print(f"✅ Matrix multiply successful: {result.output.shape}")
            print(f"   Time: {result.elapsed_ms:.1f} ms")
        
        print("\n--- Tensor Contraction ---")
        result = bridge.tensor_contract(300)
        print(result.stdout)
