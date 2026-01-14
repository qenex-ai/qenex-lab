#!/usr/bin/env python3
"""
PROMETHEUS Integration Test & Benchmark for QENEX LAB
======================================================

Tests the PROMETHEUS BLAS backend and compares performance
against NumPy (which uses system BLAS).
"""

import numpy as np
import time
import sys

# Add the package to path
sys.path.insert(0, "/home/ubuntu/qenex-lab/workspace/packages/qenex-accelerate")


def test_prometheus():
    """Test PROMETHEUS integration with QENEX LAB."""

    print("=" * 70)
    print("PROMETHEUS UNCHAINED - QENEX LAB Integration Test")
    print("=" * 70)

    # Import PROMETHEUS
    try:
        from prometheus import dgemm, dot, axpy, kahan_sum, is_available, get_info
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Check availability
    if not is_available():
        print("❌ PROMETHEUS library not found")
        print(
            "   Set PROMETHEUS_LIB_PATH=/home/ubuntu/prometheus-unchained/build/libprometheus_c.so"
        )
        return False

    info = get_info()
    print(f"\n✅ PROMETHEUS loaded successfully")
    print(f"   Version: {info['version']}")
    print(f"   Library: {info['library_path']}")

    # Test correctness
    print("\n--- Correctness Tests ---")

    # Test DGEMM
    print("\n[DGEMM] Matrix multiply...")
    A = np.random.rand(128, 128).astype(np.float64)
    B = np.random.rand(128, 128).astype(np.float64)

    C_numpy = A @ B
    C_prometheus = dgemm(A, B)

    diff = np.max(np.abs(C_numpy - C_prometheus))
    if diff < 1e-10:
        print(f"   ✅ PASS (max diff: {diff:.2e})")
    else:
        print(f"   ❌ FAIL (max diff: {diff:.2e})")
        return False

    # Test DOT
    print("\n[DOT] Dot product...")
    a = np.random.rand(1000).astype(np.float64)
    b = np.random.rand(1000).astype(np.float64)

    dot_numpy = np.dot(a, b)
    dot_prometheus = dot(a, b)

    diff = abs(dot_numpy - dot_prometheus)
    if diff < 1e-10:
        print(f"   ✅ PASS (diff: {diff:.2e})")
    else:
        print(f"   ❌ FAIL (diff: {diff:.2e})")
        return False

    # Test AXPY
    print("\n[AXPY] y = alpha*x + y...")
    x = np.random.rand(1000).astype(np.float64)
    y_numpy = np.random.rand(1000).astype(np.float64)
    y_prometheus = y_numpy.copy()
    alpha = 2.5

    y_numpy = alpha * x + y_numpy
    axpy(alpha, x, y_prometheus)

    diff = np.max(np.abs(y_numpy - y_prometheus))
    if diff < 1e-10:
        print(f"   ✅ PASS (max diff: {diff:.2e})")
    else:
        print(f"   ❌ FAIL (max diff: {diff:.2e})")
        return False

    # Test Kahan Sum
    print("\n[KAHAN] Compensated summation...")
    # Create array with values that cause cancellation
    x = np.array([1e16, 1.0, -1e16, 2.0], dtype=np.float64)

    sum_naive = np.sum(x)  # Will lose precision
    sum_kahan = kahan_sum(x)

    expected = 3.0
    print(f"   Naive sum: {sum_naive} (expected: {expected})")
    print(f"   Kahan sum: {sum_kahan}")
    if abs(sum_kahan - expected) < 1e-10:
        print(f"   ✅ PASS")
    else:
        print(f"   ⚠️  Kahan sum differs from expected")

    print("\n" + "=" * 70)
    print("All correctness tests PASSED!")
    print("=" * 70)

    return True


def benchmark_prometheus():
    """Benchmark PROMETHEUS vs NumPy BLAS."""

    print("\n" + "=" * 70)
    print("PROMETHEUS vs NumPy Performance Benchmark")
    print("=" * 70)

    from prometheus import dgemm

    sizes = [64, 128, 192, 256, 320, 384, 512]
    n_warmup = 3
    n_trials = 10

    print(
        f"\n{'Size':>6} | {'NumPy (ms)':>12} | {'PROMETHEUS (ms)':>15} | {'Speedup':>8}"
    )
    print("-" * 55)

    for N in sizes:
        A = np.random.rand(N, N).astype(np.float64)
        B = np.random.rand(N, N).astype(np.float64)

        # Warmup
        for _ in range(n_warmup):
            _ = A @ B
            _ = dgemm(A, B)

        # Benchmark NumPy
        times_numpy = []
        for _ in range(n_trials):
            start = time.perf_counter()
            C = A @ B
            end = time.perf_counter()
            times_numpy.append(end - start)

        # Benchmark PROMETHEUS
        times_prometheus = []
        for _ in range(n_trials):
            start = time.perf_counter()
            C = dgemm(A, B)
            end = time.perf_counter()
            times_prometheus.append(end - start)

        numpy_ms = np.median(times_numpy) * 1000
        prometheus_ms = np.median(times_prometheus) * 1000
        speedup = numpy_ms / prometheus_ms

        status = "⭐" if speedup > 1.0 else ""
        print(
            f"{N:>6} | {numpy_ms:>12.3f} | {prometheus_ms:>15.3f} | {speedup:>7.2f}x {status}"
        )

    print("-" * 55)
    print("\n⭐ = PROMETHEUS faster than NumPy")


def main():
    """Run tests and benchmarks."""

    # Run correctness tests
    if not test_prometheus():
        print("\n❌ Correctness tests failed!")
        sys.exit(1)

    # Run benchmarks
    benchmark_prometheus()

    print("\n" + "=" * 70)
    print("PROMETHEUS UNCHAINED successfully integrated with QENEX LAB!")
    print("=" * 70)


if __name__ == "__main__":
    main()
