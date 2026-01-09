# Q-Lang Example: Dark Matter Halo Density Profile
# Domain: Astrophysics

# 1. Constants
# G is pre-defined in Q-Lang
define M_sun = 1.989e30 [kg]
define kpc = 3.086e19 [m]

# 2. Parameters for Milky Way
define rho_0 = 0.008 * M_sun / (1000 * kpc)**3 # Central density approx
define r_s = 20.0 * kpc                        # Scale radius

# 3. NFW Profile Definition
# rho(r) = rho_0 / ( (r/r_s) * (1 + r/r_s)^2 )

# 4. Query
# Calculate density at Solar radius (r = 8 kpc)
define r_sun = 8.0 * kpc
define x = r_sun / r_s
define rho_local = rho_0 / (x * (1 + x)**2) [kg/m^3]
