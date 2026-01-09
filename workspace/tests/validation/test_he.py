
import numpy as np
import sys
sys.path.append('packages/qenex-chem/src')
from solver import HartreeFockSolver
from molecule import Molecule

mol = Molecule([('He', (0.0, 0.0, 0.0))])
solver = HartreeFockSolver(basis='sto-3g')
E_elec, E_tot = solver.compute_energy(mol)
print(f"He Atom: E_tot={E_tot:.6f} (Ref: -2.807)")
