"""
Lattice Simulator Module
Handles Monte Carlo simulations for spin systems and lattice QCD.
"""

from typing import Dict, Any, List
import random
import math

class LatticeSimulator:
    """
    Simulates lattice systems using Monte Carlo methods (Ising Model).
    """
    
    def __init__(self, dimensions: int, size: int):
        if dimensions < 1:
            raise ValueError("Lattice must have at least 1 dimension.")
        if size < 2:
            raise ValueError("Lattice size must be at least 2 for connectivity.")
        self.dimensions = dimensions
        self.size = size
        self.grid = self._initialize_grid()
        self.J = 1.0  # Coupling constant (Ferromagnetic)
        
    def _initialize_grid(self) -> List[int]:
        """Initialize spins randomly to -1 or 1."""
        total_sites = self.size ** self.dimensions
        return [random.choice([-1, 1]) for _ in range(total_sites)]

    def _get_neighbors(self, index: int) -> List[int]:
        """Get periodic boundary neighbors for a 1D or 2D lattice."""
        neighbors = []
        
        # 2D logic (assuming 2D for the semiconductor candidates)
        if self.dimensions == 2:
            x = index % self.size
            y = index // self.size
            
            # Left, Right, Up, Down (Periodic)
            neighbors.append(((y) * self.size) + ((x - 1) % self.size))
            neighbors.append(((y) * self.size) + ((x + 1) % self.size))
            neighbors.append(((y - 1) % self.size) * self.size + x)
            neighbors.append(((y + 1) % self.size) * self.size + x)
            
        # 1D logic fallback
        elif self.dimensions == 1:
            neighbors.append((index - 1) % self.size)
            neighbors.append((index + 1) % self.size)
            
        # Higher dims not implemented for this demo
        return neighbors

    def _calculate_total_energy(self) -> float:
        """Calculate total energy H = -J * sum(s_i * s_j)."""
        energy = 0
        for i, spin in enumerate(self.grid):
            neighbors = self._get_neighbors(i)
            for n in neighbors:
                energy += -self.J * spin * self.grid[n]
        # Divide by 2 because each pair is counted twice
        return energy / 2.0
    
    def run_simulation(self, steps: int, temperature: float) -> Dict[str, Any]:
        """
        Runs the Metropolis-Hastings algorithm.
        """
        # [SECURITY PATCH] Thermodynamic Constraints
        if temperature <= 0:
            raise ValueError(f"Thermodynamic Violation: Temperature must be > 0 K (Got {temperature}).")

        # [SECURITY PATCH] Causality Constraints
        if steps < 0:
            raise ValueError("Causality Violation: Simulation steps cannot be negative.")

        # Monte Carlo Loop
        for _ in range(steps):
            # Pick random site
            site_idx = random.randint(0, len(self.grid) - 1)
            current_spin = self.grid[site_idx]
            
            # Calculate energy change if flipped
            # dE = E_new - E_old = -J * (-s) * sum(neighbors) - (-J * s * sum(neighbors))
            # dE = 2 * J * s * sum(neighbors)
            neighbor_sum = sum(self.grid[n] for n in self._get_neighbors(site_idx))
            dE = 2 * self.J * current_spin * neighbor_sum
            
            # Metropolis Criterion
            if dE < 0 or random.random() < math.exp(-dE / temperature):
                self.grid[site_idx] *= -1 # Flip spin
        
        # Calculate final stats
        final_energy = self._calculate_total_energy()
        magnetization = sum(self.grid) / len(self.grid)
        avg_energy_per_site = final_energy / len(self.grid)

        return {
            "steps": steps,
            "temperature": temperature,
            "magnetization": magnetization,
            "energy": avg_energy_per_site
        }
