
import numpy as np
import sys
sys.path.append('packages/qenex-chem/src')
from solver import HartreeFockSolver
from molecule import Molecule

# Fluorine Atom (9 e, doublet)
mol = Molecule([('F', (0.0, 0.0, 0.0))], multiplicity=2)
solver = HartreeFockSolver(basis='sto-3g')
E_elec, E_tot = solver.compute_energy(mol)
print(f"F Atom: E_tot={E_tot:.6f} (Ref: ~ -98)")
