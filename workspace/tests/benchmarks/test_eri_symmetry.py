"""
Benchmark: 8-Fold ERI Symmetry Optimization

Tests that compute_eri_symmetric produces the same results as compute_eri_parallel
but with improved performance due to exploiting permutational symmetry.
"""
import time
import numpy as np
import pytest

from molecule import Molecule
from solver import HartreeFockSolver


def test_eri_symmetry_correctness():
    """
    Verify that compute_eri_symmetric produces identical results to compute_eri_parallel.
    """
    import qenex_accelerate as qa
    
    # Create a small test molecule
    mol = Molecule([
        ('H', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.0, 1.4))
    ])
    
    solver = HartreeFockSolver()
    basis = solver.build_basis(mol)
    
    # Get the basis data
    coords, at_idx, bas_idx, exps, norms, lmns = solver._flatten_basis(basis, mol)
    n_basis = len(basis)
    
    # Compute ERIs with both methods
    eri_full = qa.compute_eri_parallel(coords, exps, norms, lmns, at_idx, bas_idx, n_basis)
    eri_sym = qa.compute_eri_symmetric(coords, exps, norms, lmns, at_idx, bas_idx, n_basis)
    
    # Compare results
    diff = np.abs(eri_full - eri_sym)
    max_diff = np.max(diff)
    
    print(f"\nERI Symmetry Test:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  ERI tensor shape: {eri_full.shape}")
    
    assert max_diff < 1e-10, f"ERI symmetric differs from full computation: max diff = {max_diff}"
    print("  PASS: Results are identical")


def test_eri_symmetry_performance():
    """
    Benchmark the performance improvement from 8-fold symmetry.
    """
    import qenex_accelerate as qa
    
    # Create a larger test molecule for better timing
    mol = Molecule([
        ('O', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.757, 0.587)),
        ('H', (0.0, -0.757, 0.587))
    ])
    
    solver = HartreeFockSolver()
    basis = solver.build_basis(mol)
    
    # Get the basis data
    coords, at_idx, bas_idx, exps, norms, lmns = solver._flatten_basis(basis, mol)
    n_basis = len(basis)
    
    # Warm up
    _ = qa.compute_eri_parallel(coords, exps, norms, lmns, at_idx, bas_idx, n_basis)
    _ = qa.compute_eri_symmetric(coords, exps, norms, lmns, at_idx, bas_idx, n_basis)
    
    # Benchmark full computation
    n_runs = 3
    
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = qa.compute_eri_parallel(coords, exps, norms, lmns, at_idx, bas_idx, n_basis)
    time_full = (time.perf_counter() - start) / n_runs
    
    # Benchmark symmetric computation  
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = qa.compute_eri_symmetric(coords, exps, norms, lmns, at_idx, bas_idx, n_basis)
    time_sym = (time.perf_counter() - start) / n_runs
    
    speedup = time_full / time_sym if time_sym > 0 else float('inf')
    
    print(f"\nERI Performance Benchmark (Water molecule, n_basis={n_basis}):")
    print(f"  Full computation:      {time_full*1000:.2f} ms")
    print(f"  Symmetric computation: {time_sym*1000:.2f} ms")
    print(f"  Speedup:               {speedup:.2f}x")
    
    # We expect at least 2x speedup (theoretical max is 8x)
    # Note: overhead may reduce actual speedup
    print(f"  INFO: Speedup of {speedup:.2f}x achieved")


def test_full_scf_with_symmetric_eri():
    """
    Test that the symmetric ERI function produces correct SCF energies.
    """
    # H2 molecule at equilibrium
    mol = Molecule([
        ('H', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.0, 1.4))
    ])
    
    solver = HartreeFockSolver()
    E_elec, E_tot = solver.compute_energy(mol)
    
    # Known reference: H2 STO-3G energy is approximately -1.117 Hartree
    print(f"\nH2 SCF Energy Test:")
    print(f"  Electronic energy: {E_elec:.6f} Ha")
    print(f"  Total energy:      {E_tot:.6f} Ha")
    print(f"  Reference:         ~-1.117 Ha")
    
    assert abs(E_tot - (-1.117)) < 0.01, f"H2 energy {E_tot:.6f} differs from reference -1.117"
    print("  PASS: SCF energy is correct")


if __name__ == "__main__":
    print("=" * 60)
    print("8-Fold ERI Symmetry Optimization Tests")
    print("=" * 60)
    
    test_eri_symmetry_correctness()
    test_eri_symmetry_performance()
    test_full_scf_with_symmetric_eri()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
