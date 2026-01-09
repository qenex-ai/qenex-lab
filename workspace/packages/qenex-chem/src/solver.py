"""
Solver Module
Implements quantum chemistry methods (Hartree-Fock).
"""

from molecule import Molecule
import random
import math

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class MatrixHartreeFock:
    """
    Enhanced Solver using Linear Algebra (Diagonalization).
    """
    def __init__(self):
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy required for Matrix Solver.")
            
    def compute_energy(self, molecule: Molecule) -> float:
        # 1. Generate Mock Core Hamiltonian (H_core)
        # In real life: H_uv = <u| -0.5 del^2 - sum(Z/r) |v>
        # We simulate this with a symmetric matrix based on distance
        N = len(molecule.atoms)
        if N == 0: return 0.0
        
        # Build Interaction Matrix (H_core)
        H_core = np.zeros((N, N))
        for i in range(N):
            H_core[i, i] = -1.5 # Diagonal energy (1s orbital approx)
            for j in range(i + 1, N):
                # Distance-dependent hopping
                p1 = np.array(molecule.atoms[i][1])
                p2 = np.array(molecule.atoms[j][1])
                dist = np.linalg.norm(p1 - p2)
                coupling = -1.0 * np.exp(-dist) # Hopping integral
                H_core[i, j] = coupling
                H_core[j, i] = coupling

        # 2. Mock Overlap Matrix (S) - Assume nearly orthogonal for sto-3g
        S = np.eye(N) + 0.1 * np.eye(N, k=1) + 0.1 * np.eye(N, k=-1)
        
        # 3. Solve Generalized Eigenvalue Problem: FC = ESC
        # For simplicity, we just diagonalize H_core (Hückel Method approximation)
        eigenvalues, _ = np.linalg.eigh(H_core)
        
        # 4. Sum occupied orbitals (2 electrons per orbital)
        n_electrons = N - molecule.charge
        n_occupied = n_electrons // 2
        
        electronic_energy = 2 * np.sum(eigenvalues[:n_occupied])
        
        # 5. Add Nuclear Repulsion
        nuclear_energy = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                p1 = np.array(molecule.atoms[i][1])
                p2 = np.array(molecule.atoms[j][1])
                dist = np.linalg.norm(p1 - p2)
                if dist > 0:
                    nuclear_energy += 1.0 / dist # Z*Z/r (assuming H atoms)
        
        total_energy = electronic_energy + nuclear_energy
        return total_energy

class HartreeFockSolver:
    """
    Restricted Hartree-Fock (RHF) Solver.
    Uses Matrix Mechanics (Extended Hückel Approximation) for geometry-sensitive energy.
    """
    
    def __init__(self, basis_set: str = "sto-3g"):
        self.basis_set = basis_set
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy required for Matrix Solver.")
        
    def compute_energy(self, molecule: Molecule) -> float:
        """
        Computes the ground state energy in Hartrees using diagonalization.
        """
        # [SECURITY PATCH] Validate Empty Molecule
        if not molecule.atoms:
            raise ValueError("Vacuum Error: Molecule has no atoms.")

        # [SECURITY PATCH] Validate Electron Count
        total_protons = len(molecule.atoms) # Assume H (Z=1) for now
        total_electrons = total_protons - molecule.charge
        if total_electrons < 0:
            raise ValueError(f"Ionization Error: Negative electron count ({total_electrons}).")

        # [SECURITY PATCH] Validate Geometry for Singularities
        for i in range(len(molecule.atoms)):
            for j in range(i + 1, len(molecule.atoms)):
                pos_i = molecule.atoms[i][1]
                pos_j = molecule.atoms[j][1]
                dist_sq = sum((pi - pj)**2 for pi, pj in zip(pos_i, pos_j))
                
                if dist_sq < 1e-12: # Tolerance for floating point
                    raise ValueError(f"Nuclear Singularity Detected: Atoms {i} and {j} overlap!")

        N = len(molecule.atoms)
        print(f"   [RHF] Diagonalizing {N}x{N} Hamiltonian for {molecule}...")
        
        # 1. Build Hamiltonian (H_core)
        # H_ii = Site Energy (Ionization Potential)
        # H_ij = Hopping Integral (interaction strength decaying with distance)
        H_core = np.zeros((N, N))
        
        # Parameters for Hydrogen-like 1s orbitals
        E_site = -0.5 # Energy of 1s electron in H atom (Hartrees)
        Beta_0 = -1.0 # Max hopping strength
        Decay = 1.0   # Decay factor for exponential overlap
        
        for i in range(N):
            H_core[i, i] = E_site
            for j in range(i + 1, N):
                p1 = np.array(molecule.atoms[i][1])
                p2 = np.array(molecule.atoms[j][1])
                dist = np.linalg.norm(p1 - p2)
                
                # Approximate hopping integral: proportional to overlap S_ij
                # S_ij ~ exp(-dist)
                coupling = Beta_0 * np.exp(-(dist - 0.74)) # Normalized near equilibrium
                
                H_core[i, j] = coupling
                H_core[j, i] = coupling

        # 2. Solve Eigenvalue Problem (H C = E C)
        # In Extended Hückel, we usually solve H C = E S C, but assuming orthogonality (S=I) for simplicity here
        eigenvalues, eigenvectors = np.linalg.eigh(H_core)
        
        # 3. Fill Orbitals (Aufbau Principle)
        n_occupied = total_electrons // 2
        remainder = total_electrons % 2
        
        # Sort eigenvalues just in case (eigh typically returns sorted)
        eigenvalues.sort()
        
        orbital_energies = eigenvalues[:n_occupied]
        electronic_energy = 2 * np.sum(orbital_energies)
        
        if remainder:
            electronic_energy += eigenvalues[n_occupied]
            
        print(f"   [RHF] Electronic Energy: {electronic_energy:.4f} Eh")

        # 4. Add Nuclear Repulsion (Classical electrostatics)
        nuclear_energy = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                p1 = np.array(molecule.atoms[i][1])
                p2 = np.array(molecule.atoms[j][1])
                dist = np.linalg.norm(p1 - p2)
                if dist > 0:
                    # E_nuc = Z_i * Z_j / r_ij
                    # Assuming Z=1 (Hydrogen) for all atoms in this simple model
                    nuclear_energy += 1.0 / dist 
        
        print(f"   [RHF] Nuclear Repulsion: {nuclear_energy:.4f} Eh")
        
        total_energy = electronic_energy + nuclear_energy
        return total_energy
