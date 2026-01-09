
import numpy as np
import sys
sys.path.append('packages/qenex-chem/src')
import integrals as ints

# Check (pp|pp) ERI
bf = ints.BasisFunction([0.0,0.0,0.0], 1.0, 1.0, (1,0,0)) # px
val = ints.eri(bf, bf, bf, bf)
print(f"ERI (px px | px px): {val:.6f}")

# Check (px px | py py)
bf_y = ints.BasisFunction([0.0,0.0,0.0], 1.0, 1.0, (0,1,0)) # py
val_xy = ints.eri(bf, bf, bf_y, bf_y)
print(f"ERI (px px | py py): {val_xy:.6f}")

# Check (px py | px py) -> Exchange integral K
val_k = ints.eri(bf, bf_y, bf, bf_y)
print(f"ERI (px py | px py): {val_k:.6f}")

# Theory:
# For p-orbitals, J_xx > J_xy.
# J_xx = J_yy = J_zz.
# J_xy = J_xz.
# K_xy > 0.
# J_xx - K_xx ? K_xx = J_xx.
# J_xy = J_xx - 2 K_xy ?
# For atomic orbitals.
