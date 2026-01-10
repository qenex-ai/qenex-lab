import time
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../packages/qenex_chem/src')))

from molecule import Molecule
from solver import HartreeFockSolver

def test_speed():
    print("Setting up Water molecule...")
    mol = Molecule([
        ('O', (0.0, 0.0, 0.0)),
        ('H', (1.5, 0.0, 0.0)),
        ('H', (0.0, 1.5, 0.0))
    ], charge=0, multiplicity=1)

    solver = HartreeFockSolver()
    
    print("Computing SCF (Energy)...")
    start = time.time()
    energy = solver.compute_energy(mol)
    end = time.time()
    print(f"Energy computed in {end - start:.4f} seconds. Energy: {energy}")

    print("Computing Gradient...")
    start = time.time()
    grad = solver.compute_gradient(mol)
    end = time.time()
    print(f"Gradient computed in {end - start:.4f} seconds.")
    print("Gradient norm:", np.linalg.norm(grad))

if __name__ == "__main__":
    test_speed()
