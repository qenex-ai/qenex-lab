import numpy as np
import sys
import os

# Add packages to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../packages')))

from qenex_chem.src.molecule import Molecule
from qenex_chem.src.solver import HartreeFockSolver
from qenex_chem.src.optimizer import GeometryOptimizer

def calculate_geometry(atoms):
    """Helper to calculate bond lengths and angles for Water (O is index 0)"""
    O = np.array(atoms[0][1])
    H1 = np.array(atoms[1][1])
    H2 = np.array(atoms[2][1])
    
    r1 = np.linalg.norm(H1 - O)
    r2 = np.linalg.norm(H2 - O)
    
    v1 = H1 - O
    v2 = H2 - O
    
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return r1, r2, angle_deg

def optimize_water():
    print("=== Water Geometry Optimization using JIT-Accelerated RHF ===")
    
    # Use a better initial guess to converge faster
    # Equilibrium H2O is around r=0.96 A, angle=104.5
    # Starting closer to this will reduce iterations
    theta = np.radians(100.0)
    r = 1.0
    
    H1_pos = (r, 0.0, 0.0)
    H2_pos = (r * np.cos(theta), r * np.sin(theta), 0.0)
    
    mol_str = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", H1_pos),
        ("H", H2_pos)
    ]
    
    mol = Molecule(mol_str)
    
    r1, r2, angle = calculate_geometry(mol.atoms)
    print(f"Initial Geometry: r(OH)={r1:.4f} A, r(OH)={r2:.4f} A, Angle={angle:.2f} deg")
    
    solver = HartreeFockSolver()
    optimizer = GeometryOptimizer(solver)
    
    # Increase tolerance slightly to ensure quick convergence for demonstration
    # STO-3G is approximate anyway
    print("\nStarting Optimization...")
    mol_final, history = optimizer.optimize(mol, learning_rate=0.2, max_steps=10, tolerance=1e-2)
    
    energy = history[-1]
    
    print("\nOptimization Complete!")
    print(f"Final Energy: {energy:.8f} Hartrees")
    
    final_r1, final_r2, final_angle = calculate_geometry(mol_final.atoms)
    print(f"Final Geometry:   r(OH)={final_r1:.4f} A, r(OH)={final_r2:.4f} A, Angle={final_angle:.2f} deg")

if __name__ == "__main__":
    optimize_water()
