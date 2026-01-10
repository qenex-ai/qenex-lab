
import sys
import os
import numpy as np

# Adjust path to find the packages
sys.path.append(os.path.abspath("packages"))

from qenex_chem.src.molecule import Molecule
from qenex_chem.src.solver import HartreeFockSolver

def test_gradient_jit():
    print("Initializing Molecule...")
    # Simple H2 molecule
    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.74, 0.0, 0.0))]
    mol = Molecule(atoms, charge=0)

    print("Initializing Solver...")
    solver = HartreeFockSolver()

    print("Computing Energy (Pre-requisite)...")
    solver.compute_energy(mol, verbose=True)

    print("Computing Gradient (Triggers JIT)...")
    # This calls ints.grad_rhf_2e_jit which we modified
    grads = solver.compute_gradient(mol)
    
    print("Gradient Computed Successfully!")
    print("Gradients:", grads)

if __name__ == "__main__":
    test_gradient_jit()
