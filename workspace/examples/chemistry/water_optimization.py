import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../packages/qenex_chem/src')))

from molecule import Molecule
from solver import HartreeFockSolver
from optimizer import GeometryOptimizer

def optimize_water():
    print("\n--- Water (H2O) Optimization: Analytical Gradients ---")
    print("Goal: Find equilibrium bond angle and length.")
    print("Initial Guess: O at origin, H1 at (1.0, 0, 0), H2 at (0, 1.0, 0)")
    print("               (90 degree angle, 1.0 Bohr bond lengths ~ 0.529 A)")
    print("               Typical H-O is ~0.96 A (~1.81 Bohr), Angle ~104.5 deg")
    
    # Setup Molecule (C2v symmetry roughly maintained if we start symmetric)
    # Using Bohr directly
    # O: (0, 0, 0)
    # H1: (1.5, 0, 0) - Stretched out
    # H2: (0, 1.5, 0) - 90 deg angle
    
    mol = Molecule([
        ('O', (0.0, 0.0, 0.0)),
        ('H', (1.5, 0.0, 0.0)),
        ('H', (0.0, 1.5, 0.0))
    ], charge=0, multiplicity=1)

    solver = HartreeFockSolver()
    opt = GeometryOptimizer(solver)
    
    print("Starting Optimization...")
    # Use a slightly smaller learning rate for polyatomics to avoid wild oscillations
    optimized_mol, history = opt.optimize(mol, max_steps=30, learning_rate=0.2, tolerance=2e-3)
    
    # Analysis
    atoms = optimized_mol.atoms
    O_pos = np.array(atoms[0][1])
    H1_pos = np.array(atoms[1][1])
    H2_pos = np.array(atoms[2][1])
    
    vec1 = H1_pos - O_pos
    vec2 = H2_pos - O_pos
    
    dist1 = np.linalg.norm(vec1)
    dist2 = np.linalg.norm(vec2)
    
    # Angle
    # cos(theta) = (v1 . v2) / (|v1| |v2|)
    cos_theta = np.dot(vec1, vec2) / (dist1 * dist2)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_deg = np.degrees(theta_rad)
    
    print("\n--- Optimization Results ---")
    print(f"Final Energy: {history[-1]:.6f} Ha")
    print(f"Bond Length O-H1: {dist1:.4f} Bohr")
    print(f"Bond Length O-H2: {dist2:.4f} Bohr")
    print(f"Bond Angle H-O-H: {theta_deg:.2f} degrees")
    
    # STO-3G Water usually gives ~1.8-1.9 Bohr and ~100 degrees
    # 104.5 is exp, but minimal basis sets underestimate angle usually.
    
    if 1.7 < dist1 < 2.0 and 95.0 < theta_deg < 110.0:
        print("SUCCESS: Water geometry is chemically plausible for STO-3G.")
    else:
        print("WARNING: Geometry might be distorted or not converged.")

if __name__ == "__main__":
    optimize_water()
