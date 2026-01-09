
import numpy as np
import sys
sys.path.append('packages/qenex-chem/src')
import integrals as ints
from solver import ContractedGaussian, HartreeFockSolver
from molecule import Molecule

# Define 2 H atoms far apart (20 Bohr)
# Energy should be 2 * E(H) = 2 * (-0.4666) = -0.9332
# Nuclear repulsion ~ 0.
mol = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (20.0, 0.0, 0.0))])

# Run SCF
solver = HartreeFockSolver(basis='sto-3g')
E_nuc = solver.compute_nuclear_repulsion(mol)
E_elec, E_tot = solver.compute_energy(mol)

print(f"H2 (R=20): E_tot={E_tot:.6f} (Ref: -0.93316)")
