"""
Geometry Optimizer Module
Provides routines to optimize molecular geometry using Hartree-Fock energy gradients.
Currently implements Numerical Gradients (Finite Difference) for robustness.
"""

import numpy as np
from copy import deepcopy
from solver import HartreeFockSolver
from molecule import Molecule

class GeometryOptimizer:
    def __init__(self, solver: HartreeFockSolver):
        self.solver = solver
        self.history = []

    def compute_gradient_numerical(self, molecule: Molecule, step_size=0.005):
        """
        Computes the nuclear gradient via Central Finite Difference.
        Grad_A = [E(R_A + h) - E(R_A - h)] / (2h)
        """
        atoms = molecule.atoms
        gradients = []
        
        # Base energy not strictly needed for gradient, but good for tracking
        # base_energy, _ = self.solver.compute_energy(molecule)
        
        print(f"  Computing Numerical Gradient (step={step_size})...")
        
        for i in range(len(atoms)):
            atom_grad = []
            element, original_pos = atoms[i]
            original_pos = np.array(original_pos)
            
            # For each Cartesian component (x, y, z)
            for axis in range(3):
                # Shift +h
                pos_plus = original_pos.copy()
                pos_plus[axis] += step_size
                
                mol_plus = deepcopy(molecule)
                mol_plus.atoms[i] = (element, tuple(pos_plus))
                
                # We need to suppress printing during gradient calc to keep logs clean
                # or just accept the noise. For now, we'll let it print but maybe we should add a 'verbose' flag to solver later.
                # Assuming solver is reasonably fast for small systems.
                e_plus, _ = self.solver.compute_energy(mol_plus, max_iter=40, tolerance=1e-6)
                
                # Shift -h
                pos_minus = original_pos.copy()
                pos_minus[axis] -= step_size
                
                mol_minus = deepcopy(molecule)
                mol_minus.atoms[i] = (element, tuple(pos_minus))
                
                e_minus, _ = self.solver.compute_energy(mol_minus, max_iter=40, tolerance=1e-6)
                
                # Central Difference
                grad_component = (e_plus - e_minus) / (2.0 * step_size)
                atom_grad.append(grad_component)
                
            gradients.append(tuple(atom_grad))
            
        return gradients

    def compute_gradient_analytical(self, molecule: Molecule):
        """
        Computes the nuclear gradient using Analytical Derivatives.
        """
        print("  Computing Analytical Gradient...")
        return self.solver.compute_gradient(molecule)

    def optimize(self, molecule: Molecule, max_steps=20, learning_rate=0.5, tolerance=1e-3, method='analytical'):
        """
        Performs geometry optimization using Steepest Descent.
        R_new = R_old - learning_rate * Gradient
        
        Args:
            method: 'analytical' (default) or 'numerical'
        """
        print("========================================")
        print("Geometry Optimization Initiated")
        print("========================================")
        
        current_mol = deepcopy(molecule)
        
        for step in range(max_steps):
            print(f"\n[Step {step+1}/{max_steps}]")
            
            # 1. Compute Energy
            energy, _ = self.solver.compute_energy(current_mol)
            self.history.append(energy)
            print(f"Energy: {energy:.6f} Ha")
            
            # 2. Compute Gradient
            if method == 'analytical':
                gradients = self.compute_gradient_analytical(current_mol)
            else:
                gradients = self.compute_gradient_numerical(current_mol)
            
            # 3. Check Convergence (RMS Force)
            grad_array = np.array(gradients)
            rms_force = np.sqrt(np.mean(grad_array**2))
            max_force = np.max(np.abs(grad_array))
            
            print(f"Forces (RMS: {rms_force:.6f}, Max: {max_force:.6f}):")
            for i, g in enumerate(gradients):
                print(f"  Atom {i} ({current_mol.atoms[i][0]}): {g}")
                
            if rms_force < tolerance:
                print("========================================")
                print("Geometry Converged!")
                print("========================================")
                return current_mol, self.history
                
            # 4. Update Coordinates
            new_atoms = []
            for i, (el, pos) in enumerate(current_mol.atoms):
                g = np.array(gradients[i])
                new_pos = np.array(pos) - learning_rate * g
                new_atoms.append((el, tuple(new_pos)))
                
            current_mol.atoms = new_atoms
            
        print("Warning: Maximum steps reached without full convergence.")
        return current_mol, self.history
