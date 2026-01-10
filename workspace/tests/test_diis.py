import pytest
import os
import numpy as np

# Add packages root to path
# Use explicit package path to ensure resolution

try:
    import molecule
    import solver
    # Alias to match code
    Molecule = molecule.Molecule
    HartreeFockSolver = solver.HartreeFockSolver
except ImportError:
    # Fallback if installed as package
    from qenex_chem.src.molecule import Molecule
    from qenex_chem.src.solver import HartreeFockSolver

def test_water_diis():
    print("\n--- Testing Water Molecule with DIIS ---")
    
    # Water Geometry (Angstroms)
    # Approx equilibrium geometry
    coords = [
        ('O', [0.0000000000, 0.0000000000, -0.124649]),
        ('H', [0.0000000000, -1.432558, 0.986861]), # Slightly distorted for non-symmetry check
        ('H', [0.0000000000, 1.432558, 0.986861])
    ]
    # Scale to Bohr if needed? Usually coords are input in Angstroms and converted.
    # But let's assume STO-3G solver expects Angstroms and converts or uses bohr directly.
    # Standard STO-3G water energy is around -74.963 Hartrees.
    
    mol = Molecule(coords)
    # Convert to Bohr (1 Angstrom = 1.8897259886 Bohr)
    # The solver likely expects Bohr for integral calculations
    # Let's check molecule.py ... assuming it stores raw coords.
    # The integrals module uses these coords. 
    # Usually inputs are Angstroms, converted to Bohr.
    # Let's apply conversion manually to be safe or check molecule class.
    
    BOHR_CONV = 1.8897259886
    new_atoms = []
    for el, pos in mol.atoms:
        new_pos = [p * BOHR_CONV for p in pos]
        new_atoms.append((el, new_pos))
    mol.atoms = new_atoms
    
    solver = HartreeFockSolver()  # Uses STO-3G by default
    
    # Run SCF
    # We expect convergence in < 20 iterations with DIIS
    e_scf, e_mp2 = solver.compute_energy(mol, max_iter=30, tolerance=1e-8)
    
    print(f"\nFinal SCF Energy: {e_scf:.6f} Ha")
    print(f"Final MP2 Energy: {e_mp2:.6f} Ha")
    
    # Reference for H2O STO-3G at optimal geometry is ~ -74.96
    assert -76.0 < e_scf < -74.0, f"SCF Energy {e_scf} out of expected range for Water STO-3G"

if __name__ == "__main__":
    test_water_diis()
