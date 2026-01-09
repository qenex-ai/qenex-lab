import sys
import os
import numpy as np

# Add src to path
# Go up 2 levels from tests/validation to workspace root, then into packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../packages/qenex_chem/src')))

from molecule import Molecule
from solver import HartreeFockSolver
from optimizer import GeometryOptimizer

def test_h2_optimization():
    print("\n--- Testing Geometry Optimization: H2 ---")
    
    # Start H2 slightly stretched (Equilibrium is ~0.74 A = ~1.4 Bohr)
    # 1.0 Angstrom = 1.8897 Bohr
    # Start at 1.0 A (1.89 Bohr) to see if it shrinks
    
    # Using Bohr directly for simplicity
    # Atom 1 at origin, Atom 2 at 1.8 (stretched)
    mol = Molecule([
        ('H', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.0, 1.8))
    ])
    
    solver = HartreeFockSolver()
    # Note: Basis set handling was refactored. The solver typically uses STO-3G by default internally,
    # or takes it in compute_energy via molecule. 
    # Current implementation in solver.py doesn't take init args.
    opt = GeometryOptimizer(solver)
    
    # Run Optimization
    optimized_mol, history = opt.optimize(mol, max_steps=10, tolerance=1e-3, learning_rate=0.5)
    
    # Verification
    final_pos = optimized_mol.atoms[1][1]
    final_dist = np.linalg.norm(np.array(final_pos) - np.array(optimized_mol.atoms[0][1]))
    
    print(f"Final Bond Length: {final_dist:.4f} Bohr")
    
    # STO-3G H2 Equilibrium is usually around 1.34 - 1.39 Bohr (0.71-0.73 A)
    if 1.30 < final_dist < 1.45:
        print("PASS: H2 bond length converged to reasonable range.")
    else:
        print(f"FAIL: H2 bond length {final_dist:.4f} is outside expected range (1.30-1.45).")

if __name__ == "__main__":
    test_h2_optimization()
