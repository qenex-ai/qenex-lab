
import math
import random

class OptimizedLattice:
    """
    Optimized Lattice Simulator using Checkerboard Decomposition.
    Faster and more stable than naive Metropolis.
    """
    def __init__(self, size):
        self.size = size
        # Checkerboard setup: 0=Black, 1=White
        self.grid = [[1 if (i+j)%2==0 else -1 for j in range(size)] for i in range(size)]
    
    def step(self, beta):
        # Update Black sites (independent of each other, depend only on White)
        self._update_subgrid(0, beta)
        # Update White sites (independent of each other, depend only on Black)
        self._update_subgrid(1, beta)
        
    def _update_subgrid(self, parity, beta):
        for i in range(self.size):
            for j in range(self.size):
                if (i + j) % 2 == parity:
                    # Calculate local field from 4 neighbors (Periodic Boundary)
                    neighbors = (
                        self.grid[(i+1)%self.size][j] +
                        self.grid[(i-1)%self.size][j] +
                        self.grid[i][(j+1)%self.size] +
                        self.grid[i][(j-1)%self.size]
                    )
                    # dE = E_new - E_old
                    # E = -J * S_i * Sum(S_neighbors)
                    # Flip S_i -> -S_i
                    # dE = (-J * -S_i * neighbors) - (-J * S_i * neighbors)
                    # dE = 2 * J * S_i * neighbors. Let J=1.
                    dE = 2 * self.grid[i][j] * neighbors
                    
                    # Metropolis acceptance
                    if dE <= 0:
                        self.grid[i][j] *= -1 # Accept flip
                    else:
                        if random.random() < math.exp(-dE * beta):
                             self.grid[i][j] *= -1 # Accept flip
                        
    def get_magnetization(self):
        return abs(sum(sum(row) for row in self.grid)) / (self.size**2)

    def run_simulation(self, sweeps: int, temp: float):
        """
        Runs the simulation for a number of sweeps at a given temperature.
        Returns the final magnetization.
        """
        beta = 1.0 / temp if temp > 0 else 1e9
        for _ in range(sweeps):
            self.step(beta)
        
        return self.get_magnetization()
