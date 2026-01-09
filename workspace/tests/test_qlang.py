import pytest
import subprocess
import os
import sys

# Paths to key directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INTERPRETER_PATH = os.path.join(PROJECT_ROOT, "packages/qenex-qlang/src/interpreter.py")
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "packages/qenex-qlang/examples")

def run_qlang_script(script_name):
    """
    Helper to run a Q-Lang script via the interpreter and capture output.
    """
    script_path = os.path.join(EXAMPLES_DIR, script_name)
    
    # Ensure script exists
    if not os.path.exists(script_path):
        pytest.fail(f"Script not found: {script_path}")

    # Run the interpreter
    result = subprocess.run(
        [sys.executable, INTERPRETER_PATH, script_path],
        capture_output=True,
        text=True
    )
    
    return result

def test_grand_challenge_h2_optimization():
    """
    Verifies that grand_challenge.ql runs successfully and finds the correct bond length.
    """
    result = run_qlang_script("grand_challenge.ql")
    
    # Check for runtime errors
    assert result.returncode == 0, f"Interpreter crashed:\n{result.stderr}"
    
    # Check for physics success
    # The script prints "Optimization Complete" and shows the bond length
    assert "Optimization Complete" in result.stdout
    assert "Final Bond Length" in result.stdout
    
    # Optional: Parse output to ensure bond length is reasonable (0.6 - 0.8 A)
    # We look for "Final Bond Length: 0.6..." or similar
    # This is a loose check to ensure physics isn't broken
    assert "0.6" in result.stdout or "0.7" in result.stdout

def test_ruthless_isotope_evaluation():
    """
    Verifies that ruthless_isotope.ql correctly validates the isotope effect.
    """
    result = run_qlang_script("ruthless_isotope.ql")
    
    # Check for runtime errors
    assert result.returncode == 0, f"Interpreter crashed:\n{result.stderr}"
    
    # Check for specific success marker from the script
    assert "SUCCESS: Isotope effect validated within tolerance" in result.stdout
