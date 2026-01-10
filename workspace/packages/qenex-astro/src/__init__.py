"""
QENEX Astrophysics Package
==========================
Comprehensive astrophysics simulations covering stellar physics,
cosmology, exoplanets, gravitational waves, and high-energy phenomena.
"""

from .astrophysics import (
    # Constants
    G, c, h, hbar, k_B, sigma_SB,
    M_sun, R_sun, L_sun, T_sun,
    AU, pc, ly, H_0, T_CMB,
    
    # Enums
    SpectralType,
    
    # Classes
    Star, StellarStructure,
    CosmologicalModel, DarkMatterHalo,
    Exoplanet, BinarySystem,
    AstrophysicsSimulator,
)

__version__ = "0.1.0"
__all__ = [
    # Constants
    'G', 'c', 'h', 'hbar', 'k_B', 'sigma_SB',
    'M_sun', 'R_sun', 'L_sun', 'T_sun',
    'AU', 'pc', 'ly', 'H_0', 'T_CMB',
    
    # Enums
    'SpectralType',
    
    # Classes
    'Star', 'StellarStructure',
    'CosmologicalModel', 'DarkMatterHalo',
    'Exoplanet', 'BinarySystem',
    'AstrophysicsSimulator',
]
