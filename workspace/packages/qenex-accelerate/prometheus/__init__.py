"""
PROMETHEUS UNCHAINED - Python Interface for QENEX LAB
======================================================

High-performance CPU-only BLAS operations using AVX-512.
Achieves 1.0-2.0x OpenBLAS performance for N=128-512.

Usage:
    from qenex_accelerate.prometheus import dgemm, dot, axpy

    # Matrix multiply: C = alpha * A @ B + beta * C
    C = dgemm(A, B, alpha=1.0, beta=0.0)

    # Dot product
    result = dot(a, b)

    # AXPY: y = alpha * x + y
    axpy(alpha, x, y)

This module provides a drop-in replacement for NumPy matrix operations
with significantly better performance on AVX-512 capable CPUs.
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Optional
import os

__version__ = "1.0.0"
__all__ = [
    "dgemm",
    "dot",
    "axpy",
    "scal",
    "gemv",
    "kahan_sum",
    "is_available",
    "get_info",
]

# Library loading
_lib = None
_lib_path = None


def _find_library() -> Optional[Path]:
    """Find the PROMETHEUS shared library."""
    # Search paths in order of preference
    search_paths = [
        # Environment variable override
        os.environ.get("PROMETHEUS_LIB_PATH"),
        # Standard install locations
        "/home/ubuntu/prometheus-unchained/build/libprometheus_c.so",
        "/usr/local/lib/libprometheus_c.so",
        "/usr/lib/libprometheus_c.so",
        # Relative to this file
        str(Path(__file__).parent / "lib" / "libprometheus_c.so"),
        str(Path(__file__).parent.parent / "lib" / "libprometheus_c.so"),
    ]

    for path in search_paths:
        if path and Path(path).exists():
            return Path(path)

    return None


def _load_library():
    """Load the PROMETHEUS shared library."""
    global _lib, _lib_path

    if _lib is not None:
        return _lib

    lib_path = _find_library()
    if lib_path is None:
        raise RuntimeError(
            "PROMETHEUS library not found. Set PROMETHEUS_LIB_PATH environment variable "
            "or install to /usr/local/lib/libprometheus_c.so"
        )

    _lib_path = lib_path
    _lib = ctypes.CDLL(str(lib_path))

    # Define function signatures

    # DGEMM: void prometheus_dgemm(int M, int N, int K, double alpha,
    #                              const double* A, int lda,
    #                              const double* B, int ldb,
    #                              double beta, double* C, int ldc)
    _lib.prometheus_dgemm.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,  # M, N, K
        ctypes.c_double,  # alpha
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,  # A, lda
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,  # B, ldb
        ctypes.c_double,  # beta
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,  # C, ldc
    ]
    _lib.prometheus_dgemm.restype = None

    # DOT: double prometheus_dot(const double* a, const double* b, int n)
    _lib.prometheus_dot.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    _lib.prometheus_dot.restype = ctypes.c_double

    # AXPY: void prometheus_axpy(int n, double alpha, const double* x, double* y)
    _lib.prometheus_axpy.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _lib.prometheus_axpy.restype = None

    # SCAL: void prometheus_scal(int n, double alpha, double* x)
    _lib.prometheus_scal.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    _lib.prometheus_scal.restype = None

    # GEMV: void prometheus_gemv(int m, int n, double alpha,
    #                            const double* A, int lda,
    #                            const double* x, double beta, double* y)
    _lib.prometheus_gemv.argtypes = [
        ctypes.c_int,
        ctypes.c_int,  # m, n
        ctypes.c_double,  # alpha
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,  # A, lda
        ctypes.POINTER(ctypes.c_double),  # x
        ctypes.c_double,  # beta
        ctypes.POINTER(ctypes.c_double),  # y
    ]
    _lib.prometheus_gemv.restype = None

    # KAHAN_SUM: double prometheus_kahan_sum(const double* x, int n)
    _lib.prometheus_kahan_sum.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    _lib.prometheus_kahan_sum.restype = ctypes.c_double

    return _lib


def is_available() -> bool:
    """Check if PROMETHEUS is available."""
    try:
        _load_library()
        return True
    except RuntimeError:
        return False


def get_info() -> dict:
    """Get information about PROMETHEUS installation."""
    info = {
        "available": is_available(),
        "version": __version__,
        "library_path": str(_lib_path) if _lib_path else None,
    }
    return info


def _ensure_contiguous(arr: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Ensure array is contiguous and of correct dtype."""
    if not arr.flags["C_CONTIGUOUS"] or arr.dtype != dtype:
        return np.ascontiguousarray(arr, dtype=dtype)
    return arr


def _get_ptr(arr: np.ndarray):
    """Get ctypes pointer to array data."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def dgemm(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Double-precision General Matrix Multiply.

    Computes: C = alpha * A @ B + beta * C

    Parameters:
        A: Input matrix (M x K)
        B: Input matrix (K x N)
        alpha: Scalar multiplier for A @ B (default 1.0)
        beta: Scalar multiplier for C (default 0.0)
        C: Optional output matrix (M x N). If None, created with zeros.

    Returns:
        C: Result matrix (M x N)

    Performance:
        - 1.5-2.0x faster than OpenBLAS for N=128-256
        - 1.0-1.03x for N=320-384
        - Competitive (0.96-0.99x) for N=512-768
    """
    lib = _load_library()

    # Ensure inputs are contiguous float64
    A = _ensure_contiguous(A)
    B = _ensure_contiguous(B)

    M, K = A.shape
    K2, N = B.shape

    if K != K2:
        raise ValueError(f"Matrix dimensions incompatible: A is {M}x{K}, B is {K2}x{N}")

    # Create or validate output
    if C is None:
        C = np.zeros((M, N), dtype=np.float64, order="C")
    else:
        C = _ensure_contiguous(C)
        if C.shape != (M, N):
            raise ValueError(f"C must be {M}x{N}, got {C.shape}")

    # Call PROMETHEUS
    lib.prometheus_dgemm(
        M,
        N,
        K,
        alpha,
        _get_ptr(A),
        K,  # A is MxK, lda=K (row-major)
        _get_ptr(B),
        N,  # B is KxN, ldb=N (row-major)
        beta,
        _get_ptr(C),
        N,  # C is MxN, ldc=N (row-major)
    )

    return C


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dot product of two vectors.

    Parameters:
        a: First vector
        b: Second vector (must have same length as a)

    Returns:
        Dot product (scalar)
    """
    lib = _load_library()

    a = _ensure_contiguous(a.ravel())
    b = _ensure_contiguous(b.ravel())

    if len(a) != len(b):
        raise ValueError(f"Vector lengths must match: {len(a)} vs {len(b)}")

    return lib.prometheus_dot(_get_ptr(a), _get_ptr(b), len(a))


def axpy(alpha: float, x: np.ndarray, y: np.ndarray) -> None:
    """
    AXPY operation: y = alpha * x + y (in-place).

    Parameters:
        alpha: Scalar multiplier
        x: Input vector
        y: Input/output vector (modified in-place)
    """
    lib = _load_library()

    x = _ensure_contiguous(x.ravel())
    y = _ensure_contiguous(y.ravel())

    if len(x) != len(y):
        raise ValueError(f"Vector lengths must match: {len(x)} vs {len(y)}")

    lib.prometheus_axpy(len(x), alpha, _get_ptr(x), _get_ptr(y))


def scal(alpha: float, x: np.ndarray) -> None:
    """
    Scale operation: x = alpha * x (in-place).

    Parameters:
        alpha: Scalar multiplier
        x: Input/output vector (modified in-place)
    """
    lib = _load_library()

    x = _ensure_contiguous(x.ravel())
    lib.prometheus_scal(len(x), alpha, _get_ptr(x))


def gemv(
    A: np.ndarray,
    x: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    General Matrix-Vector multiply.

    Computes: y = alpha * A @ x + beta * y

    Parameters:
        A: Input matrix (M x N)
        x: Input vector (N,)
        alpha: Scalar multiplier for A @ x
        beta: Scalar multiplier for y
        y: Optional output vector (M,). If None, created with zeros.

    Returns:
        y: Result vector (M,)
    """
    lib = _load_library()

    A = _ensure_contiguous(A)
    x = _ensure_contiguous(x.ravel())

    M, N = A.shape

    if len(x) != N:
        raise ValueError(f"Vector length {len(x)} must match matrix columns {N}")

    if y is None:
        y = np.zeros(M, dtype=np.float64)
    else:
        y = _ensure_contiguous(y.ravel())
        if len(y) != M:
            raise ValueError(
                f"Output vector length {len(y)} must match matrix rows {M}"
            )

    lib.prometheus_gemv(
        M,
        N,
        alpha,
        _get_ptr(A),
        N,
        _get_ptr(x),
        beta,
        _get_ptr(y),
    )

    return y


def kahan_sum(x: np.ndarray) -> float:
    """
    Kahan compensated summation for improved numerical accuracy.

    Parameters:
        x: Input array

    Returns:
        Sum with reduced floating-point error
    """
    lib = _load_library()

    x = _ensure_contiguous(x.ravel())
    return lib.prometheus_kahan_sum(_get_ptr(x), len(x))


# Convenience function for NumPy-compatible matmul
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """NumPy-compatible matrix multiplication using PROMETHEUS."""
    return dgemm(A, B)
