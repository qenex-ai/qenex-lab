
import numpy as np
import sys
sys.path.append('packages/qenex-chem/src')
import integrals as ints
from solver import HartreeFockSolver, ContractedGaussian
from molecule import Molecule

# Water Geometry (Standard STO-3G)
# O (0,0,0)
# H (1.43, 0, 0) -> x axis
# H (-0.44, 1.36, 0) -> roughly?
# Let's use the coords from prompt or standard.
# "Geometry is correct in Bohr"
# "Nuclear Repulsion is 9.196"

# Let's try to find coords that give 9.196
# O (0,0,0)
# H1 (x1, y1, z1)
# H2 (x2, y2, z2)
# Z_O=8, Z_H=1.
# 8/r1 + 8/r2 + 1/r12 = 9.196
# Assuming r1=r2=1.809 (standard O-H bond length 0.957A)
# 16/1.809 = 8.844. 
# 1/r12 = 9.196 - 8.844 = 0.352.
# r12 = 1/0.352 = 2.84.
# angle: 2 r sin(theta/2) = r12 => sin(theta/2) = 2.84 / 3.618 = 0.785.
# theta/2 = 51.7 deg. theta = 103.4. (Close to 104.5).

# Coords:
# O: 0, 0, 0
# H1: 1.809, 0, 0
# H2: 1.809 * cos(103.4), 1.809 * sin(103.4), 0
#     -0.419, 1.760, 0
mol = Molecule([
    ('O', (0.0, 0.0, 0.0)),
    ('H', (1.809, 0.0, 0.0)),
    ('H', (-0.419, 1.760, 0.0))
])

solver = HartreeFockSolver(basis='sto-3g')
E_nuc = solver.compute_nuclear_repulsion(mol)
print(f"Nuclear Repulsion: {E_nuc:.3f} (Target 9.196)")

# Run Full Water
E_elec, E_tot = solver.compute_energy(mol)
print(f"Water E_tot: {E_tot:.6f} (Ref: -74.96)")
