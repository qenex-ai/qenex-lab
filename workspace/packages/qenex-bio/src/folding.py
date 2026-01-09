"""
Protein Folding Module
Implements a toy HP (Hydrophobic-Polar) lattice model for protein folding.
This serves as a benchmark for discrete optimization in Q-Lang.
"""

import random
import math
import numpy as np
from typing import List, Tuple, Dict, Optional

class ProteinFolder:
    """
    Simulates protein folding using the HP Model on a 2D lattice.
    Hydrophobic (H) monomers want to be compact (neighbors).
    Polar (P) monomers are indifferent.
    Energy = -1 for every non-covalent H-H neighbor.
    
    Optimized with NumPy for vectorized energy calculation and pivot moves.
    """
    
    def __init__(self, engine: str = "hp-lattice"):
        self.engine = engine
        
    def fold_sequence(self, sequence: str, conditions: Optional[Dict] = None) -> Dict:
        """
        Runs a Monte Carlo simulation to fold the HP sequence.
        """
        # Validate input
        valid_chars = set("HP")
        if not all(c in valid_chars for c in sequence):
            raise ValueError("HP Model only accepts 'H' and 'P' residues.")
            
        n = len(sequence)
        if n < 2:
            raise ValueError("Sequence too short to fold.")
            
        # Simulation parameters
        steps = 10000
        temp = 2.0
        cooling_rate = 0.9995
        
        if conditions:
            temp = float(conditions.get("temperature", 2.0))
            steps = int(conditions.get("steps", 10000))
            
        # Initialize conformation: Straight line along X axis
        # Shape (N, 2), int32
        coords = np.zeros((n, 2), dtype=np.int32)
        coords[:, 0] = np.arange(n)
        
        # Precompute H-mask for energy calc
        # 1 where H, 0 where P
        h_mask = np.array([1 if c == 'H' else 0 for c in sequence], dtype=np.int32)
        # Indices of H residues
        h_indices = np.where(h_mask == 1)[0]
        
        best_coords = coords.copy()
        best_energy = self._calculate_energy_vectorized(coords, h_indices)
        current_energy = best_energy
        
        # Rotation matrices for 2D lattice: 90 and -90 degrees
        # R90 = [[0, -1], [1, 0]], R-90 = [[0, 1], [-1, 0]]
        rot_matrices = [
            np.array([[0, -1], [1, 0]], dtype=np.int32),
            np.array([[0, 1], [-1, 0]], dtype=np.int32)
        ]
        
        # Monte Carlo annealing
        for step in range(steps):
            # Propose move: Pivot
            # Pick a random pivot point (not ends)
            pivot = random.randint(1, n - 2)
            
            # Select rotation
            rot_idx = random.randint(0, 1)
            rot_mat = rot_matrices[rot_idx]
            
            # Create proposal
            new_coords = coords.copy()
            
            # Pivot point
            pivot_pt = coords[pivot]
            
            # Vectorized Rotation of tail
            # 1. Shift tail to origin relative to pivot
            tail_view = new_coords[pivot+1:]
            tail_view -= pivot_pt
            
            # 2. Rotate
            # (N, 2) dot (2, 2) -> (N, 2)
            # Use matmul @
            tail_rotated = tail_view @ rot_mat.T 
            # Note: A @ R.T is standard for row vectors v' = vR^T
            
            # 3. Update tail
            tail_view[:] = tail_rotated
            
            # 4. Shift back
            tail_view += pivot_pt
            
            # Collision Check
            # Fast check: len(unique) == len(coords)
            # For small N (<100), this is reasonably fast in NumPy
            # For larger N, a set of tuples is often faster than np.unique
            # Let's use the hybrid approach: convert to set only if needed
            # But np.unique with axis=0 is standard.
            
            if len(np.unique(new_coords, axis=0)) < n:
                continue # Self-intersection, invalid move
                
            # Energy Calculation
            new_energy = self._calculate_energy_vectorized(new_coords, h_indices)
            
            # Metropolis Criterion
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / temp):
                coords = new_coords
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_coords = coords.copy()
                    
            # Cool down
            temp *= cooling_rate
            if temp < 0.01: temp = 0.01
            
        return {
            "sequence": sequence,
            "coordinates": best_coords.tolist(), # Convert back for serialization
            "energy": best_energy,
            "structure_visual": self._visualize(best_coords, sequence)
        }
        
    def _calculate_energy_vectorized(self, coords: np.ndarray, h_indices: np.ndarray) -> float:
        """
        Vectorized Energy Calculation:
        -1 for every topological neighbor pair of H-H that is NOT sequential.
        """
        if len(h_indices) < 2:
            return 0.0
            
        # Get coordinates of only H residues
        h_coords = coords[h_indices]
        
        # Compute pairwise distances (Manhattan or Squared Euclidean)
        # Since it's a grid, Squared Euclidean distance == 1 implies Manhattan == 1
        # (dx^2 + dy^2 = 1) only if (1,0) or (0,1)
        
        # Broadcasting: (M, 1, 2) - (1, M, 2) -> (M, M, 2)
        diff = h_coords[:, np.newaxis, :] - h_coords[np.newaxis, :, :]
        
        # Squared distances: (M, M)
        dist_sq = np.sum(diff**2, axis=2)
        
        # Find neighbors: dist_sq == 1
        # This creates a boolean adjacency matrix of H-H contacts
        contacts = (dist_sq == 1)
        
        # We need to exclude sequential neighbors.
        # In the full sequence, sequential neighbors have indices i, i+1.
        # But h_indices are not necessarily sequential integers.
        # We need to check the original indices.
        
        # Original indices matrix: (M, 1) - (1, M) -> (M, M) difference of indices
        idx_diff = np.abs(h_indices[:, np.newaxis] - h_indices[np.newaxis, :])
        
        # Non-sequential means index difference > 1
        non_sequential = (idx_diff > 1)
        
        # Valid contacts: (Distance == 1) AND (Index Diff > 1)
        valid_contacts = contacts & non_sequential
        
        # Count valid contacts. 
        # Since the matrix is symmetric, we divide by 2.
        num_contacts = np.sum(valid_contacts) / 2
        
        return -1.0 * float(num_contacts)

    def _visualize(self, conf: np.ndarray, seq: str) -> str:
        """ASCII Art of protein structure"""
        xs = conf[:, 0]
        ys = conf[:, 1]
        
        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        
        w = max_x - min_x + 1
        h = max_y - min_y + 1
        
        # Guard against huge grids if protein unfolds
        if w > 100 or h > 100:
            return "(Structure too large to visualize)"
            
        grid = [[' ' for _ in range(w)] for _ in range(h)]
        
        for i in range(len(conf)):
            x, y = conf[i]
            char = seq[i]
            # H is bold (#), P is light (O)
            marker = '#' if char == 'H' else 'O'
            # Y-axis inverted for printing
            grid[max_y - y][x - min_x] = marker
            
        return "\n".join("".join(row) for row in grid)
