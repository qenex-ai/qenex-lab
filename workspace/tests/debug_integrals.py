
import os
import numpy as np

# Add packages to path

import integrals
from integrals import BasisFunction, nuclear_attraction

def debug_nuclear():
    print("Checking Symmetry of Nuclear Attraction Integrals...")
    
    C = np.array([0.0, 0.0, 0.0])
    Z = 1.0
    
    # Case 1: Py along Y axis
    a_y = BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (0, 1, 0))
    b_y = BasisFunction([0.0, 1.0, 0.0], 1.0, 1.0, (0, 1, 0))
    val_y = nuclear_attraction(a_y, b_y, C, Z)
    print(f"(py|V|py) along Y: {val_y}")
    
    # Case 2: Pz along Z axis
    a_z = BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (0, 0, 1))
    b_z = BasisFunction([0.0, 0.0, 1.0], 1.0, 1.0, (0, 0, 1))
    val_z = nuclear_attraction(a_z, b_z, C, Z)
    print(f"(pz|V|pz) along Z: {val_z}")
    
    # Case 3: Px along X axis
    a_x = BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (1, 0, 0))
    b_x = BasisFunction([1.0, 0.0, 0.0], 1.0, 1.0, (1, 0, 0))
    val_x = nuclear_attraction(a_x, b_x, C, Z)
    print(f"(px|V|px) along X: {val_x}")

if __name__ == "__main__":
    debug_nuclear()
