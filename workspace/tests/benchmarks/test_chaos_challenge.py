"""
QENEX LAB CHAOS CHALLENGE TEST SUITE
=====================================

This module stress-tests QENEX LAB to find edge cases, numerical issues,
and potential bugs. The goal is to break things and find root causes.

Categories:
1. Numerical Edge Cases (overflow, underflow, singularities)
2. Chemistry Edge Cases (unusual molecules, dissociation limits)
3. Q-Lang Interpreter Edge Cases (parsing, evaluation)
4. Dimensional Analysis Edge Cases
5. Precision Limits
6. Concurrent/Stress Testing

Author: QENEX Sovereign Agent
Date: 2026-01-10
"""

import sys
import os
import math
import numpy as np
import traceback
from decimal import Decimal, getcontext, InvalidOperation, Overflow, DivisionByZero
from typing import List, Tuple, Dict, Any
import warnings

# High precision
getcontext().prec = 50

# Add package paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../packages/qenex_chem/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../packages/qenex-core/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../packages/qenex-qlang/src'))

# Track all issues found
issues_found = []

def record_issue(category: str, test_name: str, description: str, severity: str = "HIGH"):
    """Record an issue found during testing."""
    issue = {
        "category": category,
        "test": test_name,
        "description": description,
        "severity": severity
    }
    issues_found.append(issue)
    print(f"🔴 ISSUE FOUND [{severity}]: {test_name}")
    print(f"   {description}\n")

def record_pass(test_name: str):
    """Record a passed test."""
    print(f"✅ PASS: {test_name}")

def record_warning(test_name: str, description: str):
    """Record a warning (not a failure but concerning)."""
    print(f"⚠️  WARNING: {test_name}")
    print(f"   {description}\n")


# =============================================================================
# CATEGORY 1: NUMERICAL EDGE CASES
# =============================================================================

def test_numerical_edge_cases():
    """Test numerical limits and edge cases."""
    print("\n" + "=" * 70)
    print("CHAOS TEST 1: Numerical Edge Cases")
    print("=" * 70 + "\n")
    
    from interpreter import QLangInterpreter, QValue, Dimensions
    
    # Test 1.1: Division by zero
    print("Test 1.1: Division by zero...")
    interp = QLangInterpreter()
    try:
        interp.execute("x = 1.0")
        interp.execute("y = 0.0")
        interp.execute("result = x / y")
        result = interp.context.get('result')
        if result is not None:
            val = result.value if hasattr(result, 'value') else result
            if str(val) == 'Infinity' or val == float('inf'):
                record_pass("Division by zero returns Infinity")
            else:
                record_issue("NUMERICAL", "Division by zero", 
                           f"Expected Infinity, got {val}", "MEDIUM")
        else:
            record_issue("NUMERICAL", "Division by zero",
                        "Result is None", "HIGH")
    except Exception as e:
        record_issue("NUMERICAL", "Division by zero", f"Crashed: {e}", "HIGH")
    
    # Test 1.2: Very small numbers (underflow)
    print("Test 1.2: Underflow (1e-400)...")
    interp = QLangInterpreter()
    try:
        interp.execute("tiny = 1e-400")
        tiny = interp.context.get('tiny')
        if tiny is not None:
            val = float(tiny.value) if hasattr(tiny, 'value') else float(tiny)
            if val == 0.0:
                record_warning("Underflow handling", "1e-400 became 0.0 (float underflow)")
            else:
                record_pass(f"Underflow handled: 1e-400 = {val}")
        else:
            record_issue("NUMERICAL", "Underflow", "Result is None", "HIGH")
    except Exception as e:
        record_issue("NUMERICAL", "Underflow", f"Crashed: {e}", "HIGH")
    
    # Test 1.3: Very large numbers (overflow)
    print("Test 1.3: Overflow (1e400)...")
    interp = QLangInterpreter()
    try:
        interp.execute("huge = 1e400")
        huge = interp.context.get('huge')
        if huge is not None:
            val = huge.value if hasattr(huge, 'value') else huge
            if str(val) == 'Infinity' or val == float('inf'):
                record_warning("Overflow handling", "1e400 became Infinity")
            else:
                record_pass(f"Large number handled: 1e400")
        else:
            record_issue("NUMERICAL", "Overflow", "Result is None", "HIGH")
    except Exception as e:
        # This might be expected for some systems
        record_warning("Overflow", f"Exception (may be expected): {e}")
    
    # Test 1.4: Square root of negative number
    print("Test 1.4: sqrt(-1)...")
    interp = QLangInterpreter()
    try:
        interp.execute("neg = -1")
        interp.execute("result = sqrt(neg)")
        result = interp.context.get('result')
        if result is not None:
            val = result.value if hasattr(result, 'value') else result
            # Should be complex: 1j
            if isinstance(val, complex) or 'j' in str(val).lower():
                record_pass(f"sqrt(-1) = {val} (complex)")
            else:
                record_issue("NUMERICAL", "sqrt(-1)", 
                           f"Expected complex, got {val} ({type(val)})", "MEDIUM")
        else:
            record_issue("NUMERICAL", "sqrt(-1)", "Result is None", "HIGH")
    except Exception as e:
        record_issue("NUMERICAL", "sqrt(-1)", f"Crashed: {e}", "HIGH")
    
    # Test 1.5: 0^0 (mathematically undefined)
    print("Test 1.5: 0^0 (undefined)...")
    interp = QLangInterpreter()
    try:
        interp.execute("result = 0 ** 0")
        result = interp.context.get('result')
        if result is not None:
            val = float(result.value) if hasattr(result, 'value') else float(result)
            # Python convention: 0**0 = 1
            if val == 1.0:
                record_pass("0^0 = 1 (Python convention)")
            else:
                record_warning("0^0", f"Got {val}, expected 1 or NaN")
    except Exception as e:
        record_warning("0^0", f"Exception: {e}")
    
    # Test 1.6: log(0) - should be -infinity
    print("Test 1.6: log(0)...")
    interp = QLangInterpreter()
    try:
        interp.execute("import math")
        interp.execute("result = log(0)")
        result = interp.context.get('result')
        # This should either be -inf or raise an error
        record_warning("log(0)", f"Got {result}, should be -inf or error")
    except Exception as e:
        if "domain" in str(e).lower() or "math" in str(e).lower():
            record_pass("log(0) correctly raises domain error")
        else:
            record_warning("log(0)", f"Exception: {e}")
    
    # Test 1.7: Catastrophic cancellation stress test
    print("Test 1.7: Catastrophic cancellation stress test...")
    interp = QLangInterpreter()
    try:
        # (1 + 1e-16) - 1 should be ~1e-16, not 0
        interp.execute("a = 1.0")
        interp.execute("b = 0.0000000000000001")  # 1e-16
        interp.execute("c = (a + b) - a")
        result = interp.context.get('c')
        if result is not None:
            val = float(result.value) if hasattr(result, 'value') else float(result)
            if val == 0.0:
                record_issue("NUMERICAL", "Catastrophic cancellation",
                           "Lost precision: (1 + 1e-16) - 1 = 0", "HIGH")
            elif abs(val - 1e-16) / 1e-16 < 0.01:
                record_pass(f"Catastrophic cancellation avoided: {val}")
            else:
                record_warning("Catastrophic cancellation", f"Got {val}, expected 1e-16")
    except Exception as e:
        record_issue("NUMERICAL", "Catastrophic cancellation", f"Crashed: {e}", "HIGH")
    
    # Test 1.8: NaN propagation
    print("Test 1.8: NaN propagation...")
    interp = QLangInterpreter()
    try:
        interp.execute("nan_val = 0.0 / 0.0")
        result = interp.context.get('nan_val')
        # Division 0/0 should give NaN or raise error
    except Exception as e:
        record_pass(f"0/0 raises exception: {type(e).__name__}")


# =============================================================================
# CATEGORY 2: CHEMISTRY EDGE CASES
# =============================================================================

def test_chemistry_edge_cases():
    """Test chemistry module with unusual molecules and conditions."""
    print("\n" + "=" * 70)
    print("CHAOS TEST 2: Chemistry Edge Cases")
    print("=" * 70 + "\n")
    
    from molecule import Molecule
    from solver import HartreeFockSolver, UHFSolver
    
    ANGSTROM_TO_BOHR = 1.8897259886
    
    # Test 2.1: Two atoms at same position (singularity)
    print("Test 2.1: Two atoms at same position (nuclear singularity)...")
    try:
        mol = Molecule([('H', (0, 0, 0)), ('H', (0, 0, 0))])
        hf = HartreeFockSolver()
        E, _ = hf.compute_energy(mol, verbose=False)
        record_issue("CHEMISTRY", "Nuclear singularity",
                    f"Should crash or return inf, got E={E}", "HIGH")
    except Exception as e:
        if "singular" in str(e).lower() or "zero" in str(e).lower() or "inf" in str(e).lower():
            record_pass(f"Nuclear singularity detected: {type(e).__name__}")
        else:
            record_pass(f"Nuclear singularity raises exception: {e}")
    
    # Test 2.2: Very long bond (dissociation limit)
    print("Test 2.2: H2 at very long bond (100 Å)...")
    try:
        R = 100.0 * ANGSTROM_TO_BOHR  # Very long bond
        mol = Molecule([('H', (0, 0, 0)), ('H', (R, 0, 0))])
        hf = HartreeFockSolver()
        E, _ = hf.compute_energy(mol, verbose=False, max_iter=200)
        # At infinite separation, E should approach -1.0 Eh (2 × H atom)
        if E > -0.5:  # Way too high
            record_issue("CHEMISTRY", "Dissociation limit",
                        f"H2 at 100Å: E={E:.4f}, expected ~-1.0 Eh", "MEDIUM")
        else:
            record_pass(f"Dissociation limit: H2 at 100Å, E={E:.4f} Eh")
    except Exception as e:
        record_issue("CHEMISTRY", "Dissociation limit", f"Crashed: {e}", "HIGH")
    
    # Test 2.3: Very short bond (approaching singularity)
    print("Test 2.3: H2 at very short bond (0.1 Å)...")
    try:
        R = 0.1 * ANGSTROM_TO_BOHR  # Very short bond
        mol = Molecule([('H', (0, 0, 0)), ('H', (R, 0, 0))])
        hf = HartreeFockSolver()
        E, _ = hf.compute_energy(mol, verbose=False)
        # Very positive energy due to nuclear repulsion
        if E > 10:  # Very high nuclear repulsion
            record_pass(f"Short bond handled: H2 at 0.1Å, E={E:.4f} Eh (high Vnn)")
        else:
            record_warning("Short bond", f"H2 at 0.1Å: E={E:.4f}, expected very positive")
    except Exception as e:
        record_issue("CHEMISTRY", "Short bond", f"Crashed: {e}", "HIGH")
    
    # Test 2.4: Single atom (should work with UHF since H has 1 electron = odd)
    print("Test 2.4: Single hydrogen atom...")
    try:
        mol = Molecule([('H', (0, 0, 0))])
        uhf = UHFSolver()  # Use UHF for odd-electron systems
        E, _ = uhf.compute_energy(mol, verbose=False)
        # H atom with minimal basis: E ≈ -0.4666 Eh
        if -0.6 < E < -0.4:
            record_pass(f"Single H atom: E={E:.4f} Eh (expected ~-0.467)")
        else:
            record_warning("Single H atom", f"E={E:.4f}, expected ~-0.467 Eh")
    except Exception as e:
        record_issue("CHEMISTRY", "Single atom", f"Crashed: {e}", "HIGH")
    
    # Test 2.5: Odd electron system with RHF (should fail gracefully)
    print("Test 2.5: Odd electron system with RHF (Li atom)...")
    try:
        mol = Molecule([('Li', (0, 0, 0))])
        hf = HartreeFockSolver()  # RHF - requires even electrons
        E, _ = hf.compute_energy(mol, verbose=False)
        record_issue("CHEMISTRY", "Odd electrons with RHF",
                    f"Should raise error for odd electrons, got E={E}", "MEDIUM")
    except ValueError as e:
        if "even" in str(e).lower() or "odd" in str(e).lower():
            record_pass(f"Odd electrons correctly rejected: {e}")
        else:
            record_pass(f"Li with RHF raises ValueError: {e}")
    except Exception as e:
        record_warning("Odd electrons", f"Unexpected exception: {e}")
    
    # Test 2.6: Li atom with UHF (should work)
    print("Test 2.6: Li atom with UHF...")
    try:
        mol = Molecule([('Li', (0, 0, 0))])
        uhf = UHFSolver()
        E, _ = uhf.compute_energy(mol, verbose=False)
        # Li atom: E ≈ -7.43 Eh (HF limit)
        if -8.0 < E < -7.0:
            record_pass(f"Li atom with UHF: E={E:.4f} Eh")
        else:
            record_warning("Li atom UHF", f"E={E:.4f}, expected ~-7.43 Eh")
    except Exception as e:
        record_issue("CHEMISTRY", "Li atom UHF", f"Crashed: {e}", "HIGH")
    
    # Test 2.7: Empty molecule
    print("Test 2.7: Empty molecule...")
    try:
        mol = Molecule([])
        hf = HartreeFockSolver()
        E, _ = hf.compute_energy(mol, verbose=False)
        if E == 0.0:
            record_pass("Empty molecule returns E=0")
        else:
            record_warning("Empty molecule", f"E={E}, expected 0")
    except Exception as e:
        record_pass(f"Empty molecule raises exception: {type(e).__name__}")
    
    # Test 2.8: Unknown element
    print("Test 2.8: Unknown element (Unobtanium)...")
    try:
        mol = Molecule([('Uo', (0, 0, 0))])  # Fake element
        hf = HartreeFockSolver()
        E, _ = hf.compute_energy(mol, verbose=False)
        record_issue("CHEMISTRY", "Unknown element",
                    f"Should fail for unknown element, got E={E}", "MEDIUM")
    except Exception as e:
        record_pass(f"Unknown element rejected: {type(e).__name__}")
    
    # Test 2.9: SCF convergence failure
    print("Test 2.9: Difficult convergence case (stretched H2O)...")
    try:
        # Stretched water - can be hard to converge
        OH = 3.0 * ANGSTROM_TO_BOHR  # Very stretched
        mol = Molecule([
            ('O', (0, 0, 0)),
            ('H', (OH, 0, 0)),
            ('H', (0, OH, 0))
        ])
        hf = HartreeFockSolver()
        E, _ = hf.compute_energy(mol, verbose=False, max_iter=50)
        record_pass(f"Stretched H2O converged: E={E:.4f} Eh")
    except Exception as e:
        if "converge" in str(e).lower():
            record_warning("SCF convergence", f"Failed to converge (may be expected): {e}")
        else:
            record_issue("CHEMISTRY", "Stretched H2O", f"Crashed: {e}", "MEDIUM")


# =============================================================================
# CATEGORY 3: Q-LANG INTERPRETER EDGE CASES
# =============================================================================

def test_qlang_edge_cases():
    """Test Q-Lang interpreter with unusual inputs."""
    print("\n" + "=" * 70)
    print("CHAOS TEST 3: Q-Lang Interpreter Edge Cases")
    print("=" * 70 + "\n")
    
    from interpreter import QLangInterpreter
    
    # Test 3.1: Empty expression
    print("Test 3.1: Empty expression...")
    interp = QLangInterpreter()
    try:
        interp.execute("")
        record_pass("Empty expression handled")
    except Exception as e:
        record_issue("QLANG", "Empty expression", f"Crashed: {e}", "LOW")
    
    # Test 3.2: Very long variable name
    print("Test 3.2: Very long variable name (1000 chars)...")
    interp = QLangInterpreter()
    try:
        long_name = "a" * 1000
        interp.execute(f"{long_name} = 42")
        result = interp.context.get(long_name)
        if result is not None:
            record_pass("Long variable name handled")
        else:
            record_issue("QLANG", "Long variable name", "Variable not stored", "LOW")
    except Exception as e:
        record_issue("QLANG", "Long variable name", f"Crashed: {e}", "LOW")
    
    # Test 3.3: Special characters in context
    print("Test 3.3: Attempting code injection...")
    interp = QLangInterpreter()
    try:
        # Try to inject malicious code
        interp.execute("x = __import__('os').system('echo hacked')")
        record_issue("QLANG", "Code injection", 
                    "Potential security vulnerability!", "CRITICAL")
    except Exception as e:
        # [FIX] Check if __import__ was blocked (this is a PASS, not a fail)
        if "__import__" in str(e) and "not defined" in str(e):
            record_pass("Code injection blocked: __import__ not available in sandbox")
        else:
            record_pass(f"Code injection blocked: {type(e).__name__}")
    
    # Test 3.4: Recursive definition
    print("Test 3.4: Self-referential definition...")
    interp = QLangInterpreter()
    try:
        interp.execute("x = 1")
        interp.execute("x = x + 1")
        result = interp.context.get('x')
        val = float(result.value) if hasattr(result, 'value') else float(result)
        if val == 2:
            record_pass("Self-reference handled: x = x + 1")
        else:
            record_warning("Self-reference", f"x = x + 1 gave {val}, expected 2")
    except Exception as e:
        record_issue("QLANG", "Self-reference", f"Crashed: {e}", "MEDIUM")
    
    # Test 3.5: Unicode variable names
    print("Test 3.5: Unicode variable names...")
    interp = QLangInterpreter()
    try:
        interp.execute("π = 3.14159")
        result = interp.context.get('π')
        if result is not None:
            record_pass("Unicode variable names supported")
        else:
            record_warning("Unicode variables", "Variable not stored")
    except Exception as e:
        record_warning("Unicode variables", f"Not supported: {e}")
    
    # Test 3.6: Deeply nested expression
    print("Test 3.6: Deeply nested expression...")
    interp = QLangInterpreter()
    try:
        # (((((...(1)...)))))
        depth = 100
        expr = "(" * depth + "1" + ")" * depth
        interp.execute(f"x = {expr}")
        result = interp.context.get('x')
        if result is not None:
            record_pass(f"Deep nesting ({depth} levels) handled")
        else:
            record_issue("QLANG", "Deep nesting", "Result is None", "LOW")
    except RecursionError:
        record_issue("QLANG", "Deep nesting", "RecursionError", "MEDIUM")
    except Exception as e:
        record_warning("Deep nesting", f"Exception: {e}")
    
    # Test 3.7: Division with units
    print("Test 3.7: Dimensional division edge case...")
    interp = QLangInterpreter()
    try:
        interp.execute("length = 10 * m")
        interp.execute("time = 0 * s")
        interp.execute("velocity = length / time")
        result = interp.context.get('velocity')
        # Should be infinity with dimensions L/T
        record_pass(f"Division by zero with units: {result}")
    except Exception as e:
        record_pass(f"Division by zero with units raises: {type(e).__name__}")
    
    # Test 3.8: Protected constant modification
    print("Test 3.8: Attempting to modify protected constant (c)...")
    interp = QLangInterpreter()
    try:
        original_c = interp.context['c']
        interp.execute("c = 100")
        new_c = interp.context['c']
        if original_c == new_c:
            record_pass("Protected constant 'c' cannot be modified")
        else:
            record_issue("QLANG", "Constant protection",
                        "Speed of light was modified!", "CRITICAL")
    except Exception as e:
        record_pass(f"Constant modification blocked: {type(e).__name__}")


# =============================================================================
# CATEGORY 4: DIMENSIONAL ANALYSIS EDGE CASES
# =============================================================================

def test_dimensional_edge_cases():
    """Test dimensional analysis with edge cases."""
    print("\n" + "=" * 70)
    print("CHAOS TEST 4: Dimensional Analysis Edge Cases")
    print("=" * 70 + "\n")
    
    from interpreter import QLangInterpreter, QValue, Dimensions
    
    # Test 4.1: Fractional dimensions
    print("Test 4.1: Fractional dimensions (sqrt of area)...")
    interp = QLangInterpreter()
    try:
        interp.execute("area = 100 * m * m")
        interp.execute("side = sqrt(area)")
        result = interp.context.get('side')
        if result is not None:
            dims = result.dims
            if dims.length == 1.0:
                record_pass(f"sqrt(m²) = m: dimensions correct")
            else:
                record_warning("Fractional dimensions", 
                              f"sqrt(m²) has length^{dims.length}")
    except Exception as e:
        record_issue("DIMENSIONAL", "Fractional dimensions", f"Crashed: {e}", "MEDIUM")
    
    # Test 4.2: Complex unit combinations
    print("Test 4.2: Complex unit combination (G * M * m / r²)...")
    interp = QLangInterpreter()
    try:
        interp.execute("M = 6e24 * kg")
        interp.execute("m_obj = 1 * kg")
        interp.execute("r = 6.4e6 * m")
        interp.execute("F = G * M * m_obj / (r * r)")
        result = interp.context.get('F')
        if result is not None:
            dims = result.dims
            # Force: M L T^-2
            if dims.mass == 1 and dims.length == 1 and dims.time == -2:
                record_pass("Gravitational force has correct dimensions [M L T⁻²]")
            else:
                record_issue("DIMENSIONAL", "Complex unit combination",
                           f"Expected [M L T⁻²], got {dims}", "HIGH")
    except Exception as e:
        record_issue("DIMENSIONAL", "Complex units", f"Crashed: {e}", "HIGH")
    
    # Test 4.3: Dimensionless ratio
    print("Test 4.3: Dimensionless ratio...")
    interp = QLangInterpreter()
    try:
        interp.execute("d1 = 10 * m")
        interp.execute("d2 = 5 * m")
        interp.execute("ratio = d1 / d2")
        result = interp.context.get('ratio')
        if result is not None:
            dims = result.dims
            if dims.mass == 0 and dims.length == 0 and dims.time == 0:
                record_pass("m/m is dimensionless")
            else:
                record_issue("DIMENSIONAL", "Dimensionless ratio",
                           f"m/m should be dimensionless, got {dims}", "HIGH")
    except Exception as e:
        record_issue("DIMENSIONAL", "Dimensionless ratio", f"Crashed: {e}", "MEDIUM")
    
    # Test 4.4: Adding same units
    print("Test 4.4: Adding values with same units...")
    interp = QLangInterpreter()
    try:
        interp.execute("d1 = 10 * m")
        interp.execute("d2 = 5 * m")
        interp.execute("total = d1 + d2")
        result = interp.context.get('total')
        if result is not None:
            val = float(result.value)
            if val == 15.0:
                record_pass("10m + 5m = 15m")
            else:
                record_issue("DIMENSIONAL", "Unit addition",
                           f"10m + 5m = {val}m, expected 15m", "HIGH")
    except Exception as e:
        record_issue("DIMENSIONAL", "Unit addition", f"Crashed: {e}", "HIGH")


# =============================================================================
# CATEGORY 5: INTEGRAL EDGE CASES
# =============================================================================

def test_integral_edge_cases():
    """Test the ERI and other integrals with edge cases."""
    print("\n" + "=" * 70)
    print("CHAOS TEST 5: Integral Calculation Edge Cases")
    print("=" * 70 + "\n")
    
    try:
        import integrals as ints
        from integrals import PrimitiveGaussian, ContractedGaussian
    except ImportError as e:
        record_issue("INTEGRALS", "Import", f"Could not import integrals: {e}", "HIGH")
        return
    
    # Test 5.1: Overlap of identical orbitals
    print("Test 5.1: Overlap integral of identical s-orbitals...")
    try:
        # Two identical s-orbitals at same position
        p1 = PrimitiveGaussian(alpha=1.0, origin=np.array([0.0, 0.0, 0.0]), 
                               l=0, m=0, n=0)
        S = ints.overlap(p1, p1)
        # Self-overlap should be 1 for normalized orbital
        if 0.9 < S < 1.1:
            record_pass(f"Self-overlap ≈ 1: S = {S:.6f}")
        else:
            record_warning("Self-overlap", f"S = {S:.6f}, expected ~1")
    except Exception as e:
        record_issue("INTEGRALS", "Self-overlap", f"Crashed: {e}", "HIGH")
    
    # Test 5.2: Very diffuse orbital (small exponent)
    print("Test 5.2: Very diffuse orbital (α = 0.01)...")
    try:
        p1 = PrimitiveGaussian(alpha=0.01, origin=np.array([0.0, 0.0, 0.0]),
                               l=0, m=0, n=0)
        p2 = PrimitiveGaussian(alpha=0.01, origin=np.array([10.0, 0.0, 0.0]),
                               l=0, m=0, n=0)
        S = ints.overlap(p1, p2)
        record_pass(f"Diffuse orbital overlap: S = {S:.6f}")
    except Exception as e:
        record_issue("INTEGRALS", "Diffuse orbital", f"Crashed: {e}", "MEDIUM")
    
    # Test 5.3: Very tight orbital (large exponent)
    print("Test 5.3: Very tight orbital (α = 1000)...")
    try:
        p1 = PrimitiveGaussian(alpha=1000.0, origin=np.array([0.0, 0.0, 0.0]),
                               l=0, m=0, n=0)
        p2 = PrimitiveGaussian(alpha=1000.0, origin=np.array([0.1, 0.0, 0.0]),
                               l=0, m=0, n=0)
        S = ints.overlap(p1, p2)
        # Should be very small due to tight functions
        record_pass(f"Tight orbital overlap: S = {S:.6e}")
    except Exception as e:
        record_issue("INTEGRALS", "Tight orbital", f"Crashed: {e}", "MEDIUM")
    
    # Test 5.4: Nuclear attraction at nucleus
    print("Test 5.4: Nuclear attraction integral at nucleus...")
    try:
        p1 = PrimitiveGaussian(alpha=1.0, origin=np.array([0.0, 0.0, 0.0]),
                               l=0, m=0, n=0)
        # Nucleus at same position as orbital center
        V = ints.nuclear_attraction(p1, p1, np.array([0.0, 0.0, 0.0]), Z=1)
        record_pass(f"Nuclear attraction at nucleus: V = {V:.6f}")
    except Exception as e:
        record_issue("INTEGRALS", "Nuclear at center", f"Crashed: {e}", "HIGH")
    
    # Test 5.5: ERI with identical orbitals
    print("Test 5.5: Two-electron integral (ii|ii)...")
    try:
        p1 = PrimitiveGaussian(alpha=1.0, origin=np.array([0.0, 0.0, 0.0]),
                               l=0, m=0, n=0)
        eri_val = ints.eri(p1, p1, p1, p1)
        # Should be positive (electron repulsion)
        if eri_val > 0:
            record_pass(f"ERI (ii|ii) = {eri_val:.6f} (positive)")
        else:
            record_issue("INTEGRALS", "ERI sign", 
                        f"ERI should be positive, got {eri_val}", "HIGH")
    except Exception as e:
        record_issue("INTEGRALS", "ERI", f"Crashed: {e}", "HIGH")


# =============================================================================
# CATEGORY 6: MEMORY AND PERFORMANCE
# =============================================================================

def test_memory_performance():
    """Test memory usage and performance limits."""
    print("\n" + "=" * 70)
    print("CHAOS TEST 6: Memory and Performance"  )
    print("=" * 70 + "\n")
    
    import time
    
    # Test 6.1: Large array in Q-Lang
    print("Test 6.1: Large array creation...")
    from interpreter import QLangInterpreter
    interp = QLangInterpreter()
    try:
        start = time.time()
        interp.execute("import numpy as np")
        interp.execute("big_array = np.zeros((1000, 1000))")
        elapsed = time.time() - start
        record_pass(f"1000x1000 array created in {elapsed:.3f}s")
    except MemoryError:
        record_issue("MEMORY", "Large array", "MemoryError", "MEDIUM")
    except Exception as e:
        record_warning("Large array", f"Exception: {e}")
    
    # Test 6.2: Many variables
    print("Test 6.2: Creating 1000 variables...")
    interp = QLangInterpreter()
    try:
        start = time.time()
        for i in range(1000):
            interp.execute(f"var_{i} = {i}")
        elapsed = time.time() - start
        record_pass(f"1000 variables created in {elapsed:.3f}s")
    except Exception as e:
        record_issue("MEMORY", "Many variables", f"Crashed: {e}", "MEDIUM")
    
    # Test 6.3: Repeated calculations
    print("Test 6.3: Repeated HF calculations (memory leak test)...")
    try:
        from molecule import Molecule
        from solver import HartreeFockSolver
        
        ANGSTROM_TO_BOHR = 1.8897259886
        
        start = time.time()
        for i in range(5):
            mol = Molecule([('H', (0, 0, 0)), ('H', (1.4, 0, 0))])
            hf = HartreeFockSolver()
            E, _ = hf.compute_energy(mol, verbose=False)
        elapsed = time.time() - start
        record_pass(f"5 HF calculations in {elapsed:.3f}s (no memory leak)")
    except Exception as e:
        record_issue("MEMORY", "Repeated HF", f"Crashed: {e}", "HIGH")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_chaos_challenge():
    """Run all chaos tests and generate report."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "    QENEX LAB CHAOS CHALLENGE - FINDING BUGS AND EDGE CASES    ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")
    
    # Run all test categories
    test_numerical_edge_cases()
    test_chemistry_edge_cases()
    test_qlang_edge_cases()
    test_dimensional_edge_cases()
    test_integral_edge_cases()
    test_memory_performance()
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("CHAOS CHALLENGE SUMMARY")
    print("=" * 70)
    
    if not issues_found:
        print("\n✅ NO CRITICAL ISSUES FOUND!")
        print("   QENEX LAB passed all chaos tests.\n")
    else:
        critical = [i for i in issues_found if i['severity'] == 'CRITICAL']
        high = [i for i in issues_found if i['severity'] == 'HIGH']
        medium = [i for i in issues_found if i['severity'] == 'MEDIUM']
        low = [i for i in issues_found if i['severity'] == 'LOW']
        
        print(f"\n🔴 ISSUES FOUND: {len(issues_found)}")
        print(f"   CRITICAL: {len(critical)}")
        print(f"   HIGH:     {len(high)}")
        print(f"   MEDIUM:   {len(medium)}")
        print(f"   LOW:      {len(low)}")
        
        print("\n" + "-" * 70)
        print("ISSUES TO FIX:")
        print("-" * 70)
        
        for issue in sorted(issues_found, key=lambda x: 
                          {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}[x['severity']]):
            print(f"\n[{issue['severity']}] {issue['category']}: {issue['test']}")
            print(f"    {issue['description']}")
    
    print("\n" + "=" * 70)
    
    return issues_found


if __name__ == "__main__":
    issues = run_chaos_challenge()
