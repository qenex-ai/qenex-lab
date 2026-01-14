#!/usr/bin/env python3
"""
PROMETHEUS + QENEX CHEM Integration Test
=========================================

Tests that the PROMETHEUS backend properly accelerates Hartree-Fock calculations.

Usage:
    PROMETHEUS_LIB_PATH=/path/to/libprometheus_c.so python test_prometheus_chem.py
"""

import sys
import os
import time
import numpy as np

# Add paths
workspace_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, workspace_path)
sys.path.insert(0, os.path.join(workspace_path, "packages"))

# Test PROMETHEUS availability
print("=" * 60)
print("PROMETHEUS + QENEX CHEM Integration Test")
print("=" * 60)

# Check PROMETHEUS
try:
    from qenex_accelerate.prometheus import is_available, get_info, dgemm

    prometheus_available = is_available()
    if prometheus_available:
        info = get_info()
        print(f"\n✅ PROMETHEUS available: {info['library_path']}")
    else:
        print("\n⚠️  PROMETHEUS library not found")
        print("   Set PROMETHEUS_LIB_PATH environment variable")
except ImportError as e:
    print(f"\n❌ PROMETHEUS import failed: {e}")
    prometheus_available = False

# Test qenex_chem import
print("\n--- Testing qenex_chem import ---")
try:
    from qenex_chem.src.solver import HartreeFockSolver, PROMETHEUS_AVAILABLE
    from qenex_chem.src.molecule import Molecule

    print(f"✅ qenex_chem imported successfully")
    print(f"   PROMETHEUS_AVAILABLE in solver: {PROMETHEUS_AVAILABLE}")
except ImportError as e:
    print(f"❌ qenex_chem import failed: {e}")
    sys.exit(1)

# Test basic DGEMM if PROMETHEUS available
if prometheus_available:
    print("\n--- Testing PROMETHEUS DGEMM ---")
    N = 64
    A = np.random.randn(N, N).astype(np.float64)
    B = np.random.randn(N, N).astype(np.float64)

    C_numpy = A @ B
    C_prometheus = dgemm(A, B)

    error = np.max(np.abs(C_numpy - C_prometheus))
    if error < 1e-10:
        print(f"✅ DGEMM correctness: PASS (max error: {error:.2e})")
    else:
        print(f"❌ DGEMM correctness: FAIL (max error: {error:.2e})")

# Test H2 molecule
print("\n--- Testing H2 Hartree-Fock ---")
h2 = Molecule(
    [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.4)),  # ~0.74 Angstrom in Bohr
    ]
)

solver = HartreeFockSolver()

# Time the calculation
start = time.perf_counter()
energy, _ = solver.compute_energy(h2, verbose=False)
elapsed = time.perf_counter() - start

print(f"   H2 Energy: {energy:.8f} Hartree")
print(f"   Time: {elapsed * 1000:.2f} ms")

# Reference value (STO-3G): approximately -1.117 Hartree
expected = -1.117
if abs(energy - expected) < 0.1:
    print(f"✅ H2 energy reasonable (expected ~{expected:.3f})")
else:
    print(f"⚠️  H2 energy differs from expected ({expected:.3f})")

# Test HeH+ (if we have time)
print("\n--- Testing HeH+ ---")
heh_plus = Molecule([("He", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5))], charge=1)

start = time.perf_counter()
energy_heh, _ = solver.compute_energy(heh_plus, verbose=False)
elapsed = time.perf_counter() - start

print(f"   HeH+ Energy: {energy_heh:.8f} Hartree")
print(f"   Time: {elapsed * 1000:.2f} ms")

# Benchmark if PROMETHEUS available
if prometheus_available:
    print("\n--- Performance Benchmark ---")
    print("Running 5 iterations each...")

    # Create fresh solver for fair comparison
    times_with_prometheus = []
    for _ in range(5):
        solver = HartreeFockSolver()
        start = time.perf_counter()
        solver.compute_energy(h2, verbose=False)
        times_with_prometheus.append(time.perf_counter() - start)

    avg_time = np.mean(times_with_prometheus) * 1000
    std_time = np.std(times_with_prometheus) * 1000

    print(f"   Average time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   Backend: {'PROMETHEUS' if PROMETHEUS_AVAILABLE else 'NumPy'}")

print("\n" + "=" * 60)
print("Integration test complete!")
print("=" * 60)
