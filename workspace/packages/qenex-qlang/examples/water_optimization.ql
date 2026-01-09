# water_optimization.ql
# This script attempts to optimize the geometry of a Water molecule (H2O).
# It defines variables for bond length ($r) and bond angle ($a).
# It uses the 'optimize' command to find the structure with minimum energy.

# Initial Guess
# r ~ 1.8 Bohr (approx 0.95 Angstrom)
# a ~ 1.8 Radians (approx 104.5 degrees)
# Note: Angle is tricky in cartesian.
# We will fix Oxygen at origin (0,0,0).
# H1 at (r, 0, 0).
# H2 at (r*cos(a), r*sin(a), 0).

define r = 1.8
define theta = 1.8

print "--- Initial State ---"
print "Bond Length r:"
print $r
print "Bond Angle theta:"
print $theta

# Calculate coordinates using Q-Lang math
# Note: variables $r and $theta are interpolated by the interpreter before passing to kernel
# But we need to use 'optimize geometry' which varies them.

print "--- Starting Optimization ---"
# Syntax: optimize geometry <Type> <Coords> <Type> <Coords> ... [basis]
# Variables $r and $theta will be varied by the optimizer.
# We express H2 coordinates using the variables.

optimize geometry O 0,0,0 H $r,0,0 H $r*cos($theta),$r*sin($theta),0

print "--- Optimization Complete ---"
print "Final Bond Length:"
print $r
print "Final Bond Angle:"
print $theta
print "Final Energy:"
print $last_energy

# Verification
# Equilibrium bond length for H2O is approx 1.809 Bohr (0.958 A)
# Equilibrium angle is approx 104.5 deg = 1.82 rad
# Our semi-empirical model might differ, but should be close.

if $last_energy < -70*kg*m*m/s/s:
    print "✅ Energy is reasonable for H2O (Hartree-Fock ~ -75 Eh)"
else:
    print "⚠️  Energy is too high. Model might be inaccurate."
