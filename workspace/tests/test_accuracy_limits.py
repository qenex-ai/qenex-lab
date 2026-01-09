
import pytest
import subprocess
import os
import sys

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INTERPRETER_PATH = os.path.join(PROJECT_ROOT, "packages/qenex-qlang/src/interpreter.py")

def run_qlang_code(code_str):
    temp_file = os.path.join(PROJECT_ROOT, "tests", "temp_precision_test.ql")
    with open(temp_file, "w") as f:
        f.write(code_str)
    
    result = subprocess.run(
        [sys.executable, INTERPRETER_PATH, temp_file],
        capture_output=True,
        text=True
    )
    if os.path.exists(temp_file):
        os.remove(temp_file)
    return result

def test_catastrophic_cancellation():
    """
    Test 1: Floating Point Precision Loss.
    Subtracting two very large, nearly identical numbers should result in 
    significant precision loss if not handled by arbitrary precision types.
    """
    code = """
    # Two numbers that differ by 1e-15
    define a = 1.000000000000001
    define b = 1.0
    # Expected: 1e-15. 
    # In standard float64, this is at the edge of machine epsilon (2e-16).
    # If we go smaller, we lose it.
    
    define diff = a - b
    print diff
    """
    result = run_qlang_code(code)
    # If Q-Lang uses standard Python float (64-bit), it might work for 1e-15 but fail for 1e-17.
    # Let's try deeper.
    
    code_deep = """
    # Differ by 1e-20 (below float64 epsilon ~1e-16)
    define large = 1.0
    define small = 1.0e-20
    define sum = large + small
    # sum should be > 1.0. But in float64, 1 + 1e-20 == 1.
    define check = sum - large
    print check
    """
    result = run_qlang_code(code_deep)
    # If 'check' is 0.0, we have inaccurate science (loss of mass/energy).
    # We expect this to FAIL (i.e., return 0.0) demonstrating inaccuracy.
    assert "0.0" not in result.stdout

def test_relativistic_limits():
    """
    Test 2: Classical Physics Kernel breakdown.
    Calculating Kinetic Energy at v=0.99c using classical formula vs relativistic.
    Q-Lang's simple calculator uses E = 0.5mv^2 (Classical).
    This is INACCURATE for v ~ c.
    """
    code = """
    define v = 0.99 * c
    define m = 1.0 * kg
    
    # Classical Energy: 0.5 * m * v^2
    define E_classical = 0.5 * m * v**2
    
    # Relativistic Energy: (gamma - 1) * m * c^2
    # gamma = 1 / sqrt(1 - v^2/c^2)
    # v/c = 0.99
    # 1 - 0.99^2 = 1 - 0.9801 = 0.0199
    # sqrt(0.0199) ~ 0.141067
    # gamma ~ 7.088
    # E_rel ~ (6.088) * m * c^2
    # Classical E ~ 0.5 * (0.99)^2 * m * c^2 ~ 0.49 * m * c^2
    
    # The difference is massive (factor of ~12x).
    # Does Q-Lang warn about this?
    
    print E_classical
    """
    result = run_qlang_code(code)
    # We assert that it does NOT warn, proving it allows inaccurate science.
    assert "Warning" not in result.stdout and "Relativistic" not in result.stdout

if __name__ == "__main__":
    try:
        test_catastrophic_cancellation()
        print("Precision Check: PASSED (System handled small numbers?)")
    except AssertionError:
        print("Precision Check: FAILED (System lost information < 1e-16)")
        
    try:
        test_relativistic_limits()
        print("Relativistic Check: FAILED TO WARN (System allowed classical formula at relativistic speeds)")
    except AssertionError:
        print("Relativistic Check: PASSED (System warned user)")
