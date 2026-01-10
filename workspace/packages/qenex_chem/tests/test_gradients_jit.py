
import numpy as np
import time
import sys
import os

# Ensure imports work by adding current directory to path
sys.path.append(os.getcwd())

try:
    from packages.qenex_chem.src.integrals import grad_rhf_2e_jit
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_gradient_jit():
    print("Starting Gradient JIT test...")
    
    # Setup dummy system: H2 molecule-like
    # 2 atoms, 2 primitives total (minimal)
    
    atom_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.74, 0.0, 0.0]
    ]) # 2 atoms
    
    # 2 primitives (one on each atom)
    atom_indices = np.array([0, 1], dtype=np.int64)
    basis_indices = np.array([0, 1], dtype=np.int64) # Each primitive is its own basis function here
    
    exponents = np.array([1.0, 1.0])
    norms = np.array([1.0, 1.0])
    
    # s-orbitals (0,0,0)
    lmns = np.array([
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=np.int64)
    
    # Density matrix (2x2)
    D_matrix = np.array([
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    
    start_time = time.time()
    print("Calling grad_rhf_2e_jit (first time - compiling)...")
    
    grads = grad_rhf_2e_jit(
        atom_coords,
        atom_indices,
        basis_indices,
        exponents,
        norms,
        lmns,
        D_matrix
    )
    
    end_time = time.time()
    print(f"Gradient computation finished in {end_time - start_time:.4f} seconds.")
    print("Gradients:")
    print(grads)
    
    assert grads.shape == (2, 3)
    # Check for NaNs
    assert not np.isnan(grads).any()
    
    print("Gradient Test PASSED.")

if __name__ == "__main__":
    test_gradient_jit()
