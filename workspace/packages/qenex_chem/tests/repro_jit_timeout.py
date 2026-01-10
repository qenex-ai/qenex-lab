
import numpy as np
import time
import sys
import os

# Ensure imports work by adding current directory to path
sys.path.append(os.getcwd())

try:
    from packages.qenex_chem.src.integrals import eri_primitive
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def run_case(label, la, ma, na, lb, mb, nb, lc, mc, nc, ld, md, nd):
    print(f"--- Running Case: {label} ---")
    sys.stdout.flush()
    
    alphaA, alphaB, alphaC, alphaD = 1.0, 1.0, 1.0, 1.0
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([1.0, 0.0, 0.0])
    C = np.array([0.0, 1.0, 0.0])
    D = np.array([0.0, 0.0, 1.0])
    normA, normB, normC, normD = 1.0, 1.0, 1.0, 1.0

    start_time = time.time()
    # This call triggers JIT if not cached
    val = eri_primitive(
        alphaA, alphaB, alphaC, alphaD,
        A, B, C, D,
        la, ma, na, lb, mb, nb, 
        lc, mc, nc, ld, md, nd,
        normA, normB, normC, normD
    )
    end_time = time.time()
    print(f"  Result: {val}")
    print(f"  Time: {end_time - start_time:.4f}s")
    sys.stdout.flush()

def test_jit_compilation():
    print("Starting JIT compilation test...")
    sys.stdout.flush()
    
    # Warmup with simplest case (ss|ss) - L=0
    # This should compile the base path and basic overhead
    run_case("(ss|ss)", 0,0,0, 0,0,0, 0,0,0, 0,0,0)

    # Mild complexity (ps|ss)
    run_case("(ps|ss)", 1,0,0, 0,0,0, 0,0,0, 0,0,0)

    # Target complexity (pp|pp)
    # This was crashing before due to stack overflow
    run_case("(pp|pp)", 1,0,0, 0,1,0, 0,0,1, 1,0,0)

    print("All tests passed.")

if __name__ == "__main__":
    test_jit_compilation()
