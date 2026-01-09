
import numpy as np

class OptimizedLattice:
    """
    Optimized Lattice Simulator using NumPy Vectorization and Checkerboard Decomposition.
    Faster and more stable than naive Metropolis.
    """
    def __init__(self, size: int):
        self.size = size
        # Random initialization -1 or 1
        self.grid = np.random.choice([-1, 1], size=(size, size)).astype(np.int8)
        
        # Pre-compute masks for checkerboard update
        # indices are (row, col)
        # Parity 0 (Black): (row + col) % 2 == 0
        # Parity 1 (White): (row + col) % 2 == 1
        x, y = np.indices((size, size))
        self.mask0 = (x + y) % 2 == 0
        self.mask1 = (x + y) % 2 == 1
    
    def step(self, beta: float):
        # Checkerboard update:
        # Update Black sites (Parity 0) - they only depend on White neighbors
        self._update_subgrid(self.mask0, beta)
        
        # Update White sites (Parity 1) - they only depend on Black neighbors
        self._update_subgrid(self.mask1, beta)
        
    def _update_subgrid(self, mask, beta):
        # 1. Calculate sum of neighbors for the whole grid
        # Optimized using np.roll for periodic boundary conditions
        # Neighbors: up, down, left, right
        
        # Note: We compute rolling on the full grid. 
        # This is vectorized and fast, even though we only need half the values.
        up    = np.roll(self.grid, -1, axis=0)
        down  = np.roll(self.grid,  1, axis=0)
        left  = np.roll(self.grid, -1, axis=1)
        right = np.roll(self.grid,  1, axis=1)
        
        # Neighbor sum matrix
        neighbors = up + down + left + right
        
        # 2. Extract values for the current subgrid (checkerboard color)
        S = self.grid[mask]
        N = neighbors[mask]
        
        # 3. Calculate Energy Change (dE)
        # Hamiltonian H = -J * sum(S_i * S_j). 
        # Change in energy if we flip spin S_i -> -S_i:
        # dE = E_new - E_old
        #    = (-J * (-S_i) * N) - (-J * (S_i) * N)
        #    = J * S_i * N + J * S_i * N
        #    = 2 * J * S_i * N
        # Let J = 1
        dE = 2 * S * N
        
        # 4. Metropolis Acceptance Criterion
        # Flip if dE <= 0  OR  random < exp(-beta * dE)
        
        # We can combine these checks.
        # If dE <= 0, exp(-beta*dE) >= 1, so random < exp is always True.
        # So we just need to check: random < exp(-beta * dE)
        
        # Calculate transition probabilities
        # We calculate exp only where needed could be an optimization, but vector exp is fast.
        # Note: dE is int8/int, beta is float.
        prob = np.exp(-dE * beta)
        
        # Generate random numbers for this subgrid
        rand_vals = np.random.random(S.shape)
        
        # Determine which sites to flip
        should_flip = rand_vals < prob
        
        # 5. Apply Flips
        # We need to update the original grid.
        # grid[mask] returns a copy, so we must modify and assign back.
        
        # Efficiently flip: -S is equivalent to S * -1
        # We only flip where should_flip is True.
        
        # S[should_flip] *= -1 
        # But S is a copy.
        
        # Let's construct the new values for the masked area
        new_S = S.copy()
        new_S[should_flip] *= -1
        
        # Assign back to main grid
        self.grid[mask] = new_S
                        
    def get_magnetization(self) -> float:
        return abs(np.sum(self.grid)) / (self.size**2)

    def run_simulation(self, sweeps: int, temp: float) -> float:
        """
        Runs the simulation for a number of sweeps at a given temperature.
        Returns the final magnetization.
        """
        # Avoid division by zero
        beta = 1.0 / temp if temp > 1e-9 else 1e9
        
        for _ in range(sweeps):
            self.step(beta)
        
        return float(self.get_magnetization())
