#!/usr/bin/env python3
"""
QENEX Synergistic Polyglot Demo
===============================

Demonstrates the power of Q-Lang's polyglot architecture:
- Python for flexibility and prototyping
- Julia for raw numerical speed  
- Rust for memory-safe parallelism

Run with:
    cd /opt/qenex_lab/workspace
    source venv/bin/activate
    python experiments/polyglot_demo.py
"""

import sys
sys.path.insert(0, 'packages/qenex-qlang/src')
from interpreter import QLangInterpreter

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       QENEX Synergistic Polyglot Demo                        ║
    ║       Multi-Language Scientific Computing                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    ql = QLangInterpreter()
    
    # Demo Q-Lang code
    code = '''
# 1. BACKEND STATUS
polyglot status

# 2. QUANTUM CHEMISTRY SIMULATION
# Define molecular overlap matrix
S_matrix = [[1.0, 0.66], [0.66, 1.0]]

# Build Fock matrix 
F_matrix = [[-1.252, -0.477], [-0.477, -0.597]]

# Solve eigenvalue problem (small matrix -> Python backend)
polyglot eigen F_matrix

# 3. SIGNAL PROCESSING
signal = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]
polyglot fft signal

# 4. LINEAR ALGEBRA
# Symmetric positive definite matrix
M_sym = [[4.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 2.0]]

# SVD decomposition
polyglot svd M_sym

# Solve Mx = b
b_vec = [6.0, 5.0, 4.0]
polyglot solve M_sym b_vec x_solution
print x_solution

# 5. MATRIX OPERATIONS
A_test = [[1.0, 2.0], [3.0, 4.0]]
B_test = [[5.0, 6.0], [7.0, 8.0]]

# Auto-dispatch (small -> Python)
polyglot matmul A_test B_test C_result
print C_result

# 6. EXPLICIT PYTHON BACKEND
polyglot.python matmul A_test B_test C_python
'''
    
    print("=" * 60)
    print("Executing Q-Lang Polyglot Code")
    print("=" * 60)
    
    ql.execute(code)
    
    print("\n" + "=" * 60)
    print("Polyglot Architecture Summary")
    print("=" * 60)
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    Q-Lang DSL Layer                          │
    │         (Domain-specific scientific syntax)                  │
    └─────────────────┬───────────────┬───────────────┬───────────┘
                      │               │               │
    ┌─────────────────▼───┐ ┌────────▼────────┐ ┌────▼─────────┐
    │     Python Core     │ │   Julia Bridge  │ │ Rust Accel   │
    │  ─────────────────  │ │ ──────────────  │ │ ───────────  │
    │  • NumPy arrays     │ │ • BLAS L3 ops   │ │ • ERI O(N⁴)  │
    │  • SciPy optimize   │ │ • Eigensolvers  │ │ • Parallel   │
    │  • Symbolic math    │ │ • FFT/FFTW      │ │ • SIMD       │
    │  • Visualization    │ │ • Type stable   │ │ • Zero-copy  │
    └─────────────────────┘ └─────────────────┘ └──────────────┘
    
    Benefits:
    ✓ PERFORMANCE: Each language handles what it does best
    ✓ SAFETY: Rust for memory-critical operations
    ✓ FLEXIBILITY: Python ecosystem for everything else
    ✓ SPEED: Julia for numerical heavy-lifting
    ✓ UNIFIED: Single Q-Lang syntax for all backends
    """)


if __name__ == "__main__":
    main()
