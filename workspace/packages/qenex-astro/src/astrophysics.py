"""
QENEX Astrophysics Module
=========================
Comprehensive astrophysics simulations covering:
- Stellar structure and evolution
- Cosmology and dark energy
- Exoplanet detection and habitability
- Gravitational dynamics
- High-energy astrophysics

Physical Constants (CODATA 2018 + IAU 2015):
- G = 6.67430e-11 m³/(kg·s²)
- c = 299792458 m/s
- M_sun = 1.98892e30 kg
- R_sun = 6.9634e8 m
- L_sun = 3.828e26 W
- AU = 1.495978707e11 m
- pc = 3.0857e16 m
- H_0 = 67.4 km/s/Mpc (Planck 2018)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum, auto
import json

# =============================================================================
# PHYSICAL CONSTANTS (CODATA 2018 + IAU 2015)
# =============================================================================

# Fundamental
G = 6.67430e-11          # Gravitational constant [m³/(kg·s²)]
c = 299792458.0          # Speed of light [m/s]
h = 6.62607015e-34       # Planck constant [J·s]
hbar = 1.054571817e-34   # Reduced Planck constant [J·s]
k_B = 1.380649e-23       # Boltzmann constant [J/K]
sigma_SB = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]

# Solar
M_sun = 1.98892e30       # Solar mass [kg]
R_sun = 6.9634e8         # Solar radius [m]
L_sun = 3.828e26         # Solar luminosity [W]
T_sun = 5772.0           # Solar effective temperature [K]

# Astronomical distances
AU = 1.495978707e11      # Astronomical unit [m]
pc = 3.0857e16           # Parsec [m]
ly = 9.4607e15           # Light year [m]

# Cosmological (Planck 2018)
H_0 = 67.4               # Hubble constant [km/s/Mpc]
H_0_SI = H_0 * 1000 / (1e6 * pc)  # [s⁻¹]
Omega_m = 0.315          # Matter density parameter
Omega_Lambda = 0.685     # Dark energy density parameter
Omega_b = 0.0493         # Baryon density parameter
T_CMB = 2.7255           # CMB temperature [K]

# Nuclear
m_p = 1.67262192369e-27  # Proton mass [kg]
m_e = 9.1093837015e-31   # Electron mass [kg]


# =============================================================================
# STELLAR PHYSICS
# =============================================================================

class SpectralType(Enum):
    """Morgan-Keenan spectral classification."""
    O = auto()  # >30,000 K, blue
    B = auto()  # 10,000-30,000 K, blue-white
    A = auto()  # 7,500-10,000 K, white
    F = auto()  # 6,000-7,500 K, yellow-white
    G = auto()  # 5,200-6,000 K, yellow (Sun)
    K = auto()  # 3,700-5,200 K, orange
    M = auto()  # 2,400-3,700 K, red


@dataclass
class Star:
    """
    Stellar model with fundamental properties.
    
    Attributes:
        mass: Mass in solar masses
        radius: Radius in solar radii
        luminosity: Luminosity in solar luminosities
        temperature: Effective temperature [K]
        age: Age in Gyr
        metallicity: [Fe/H] in dex
    """
    mass: float  # M_sun
    radius: float = None  # R_sun
    luminosity: float = None  # L_sun
    temperature: float = None  # K
    age: float = 0.0  # Gyr
    metallicity: float = 0.0  # [Fe/H]
    name: str = ""
    
    def __post_init__(self):
        """Apply mass-luminosity and mass-radius relations if not specified."""
        if self.luminosity is None:
            self.luminosity = self._mass_luminosity_relation()
        if self.radius is None:
            self.radius = self._mass_radius_relation()
        if self.temperature is None:
            self.temperature = self._effective_temperature()
    
    def _mass_luminosity_relation(self) -> float:
        """
        Main sequence mass-luminosity relation.
        L/L_sun ≈ (M/M_sun)^α where α depends on mass range.
        """
        M = self.mass
        if M < 0.43:
            return 0.23 * M**2.3
        elif M < 2.0:
            return M**4.0
        elif M < 55:
            return 1.4 * M**3.5
        else:
            return 32000 * M  # Very massive stars
    
    def _mass_radius_relation(self) -> float:
        """Main sequence mass-radius relation: R/R_sun ≈ (M/M_sun)^0.8"""
        return self.mass**0.8
    
    def _effective_temperature(self) -> float:
        """Calculate effective temperature from L and R using Stefan-Boltzmann."""
        L_SI = self.luminosity * L_sun
        R_SI = self.radius * R_sun
        T4 = L_SI / (4 * np.pi * R_SI**2 * sigma_SB)
        return T4**0.25
    
    @property
    def spectral_type(self) -> SpectralType:
        """Determine spectral type from temperature."""
        T = self.temperature
        if T > 30000:
            return SpectralType.O
        elif T > 10000:
            return SpectralType.B
        elif T > 7500:
            return SpectralType.A
        elif T > 6000:
            return SpectralType.F
        elif T > 5200:
            return SpectralType.G
        elif T > 3700:
            return SpectralType.K
        else:
            return SpectralType.M
    
    @property
    def main_sequence_lifetime(self) -> float:
        """Estimate main sequence lifetime in Gyr."""
        # t_MS ≈ 10 * (M/M_sun) / (L/L_sun) Gyr
        return 10.0 * self.mass / self.luminosity
    
    @property
    def absolute_magnitude(self) -> float:
        """Absolute visual magnitude."""
        return 4.83 - 2.5 * np.log10(self.luminosity)
    
    def habitable_zone(self) -> Tuple[float, float]:
        """
        Calculate habitable zone boundaries in AU.
        
        Uses conservative estimates based on stellar flux.
        Inner edge: runaway greenhouse (~1.1 S_eff)
        Outer edge: maximum greenhouse (~0.35 S_eff)
        """
        L = self.luminosity
        inner = np.sqrt(L / 1.1)  # AU
        outer = np.sqrt(L / 0.35)  # AU
        return (inner, outer)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "mass_Msun": self.mass,
            "radius_Rsun": self.radius,
            "luminosity_Lsun": self.luminosity,
            "temperature_K": self.temperature,
            "spectral_type": self.spectral_type.name,
            "age_Gyr": self.age,
            "metallicity_FeH": self.metallicity,
            "main_sequence_lifetime_Gyr": self.main_sequence_lifetime,
            "habitable_zone_AU": self.habitable_zone()
        }


class StellarStructure:
    """
    Solve stellar structure equations for a given mass.
    
    Lane-Emden equation for polytropic stars:
    (1/ξ²) d/dξ (ξ² dθ/dξ) = -θⁿ
    
    where:
    - ξ is dimensionless radius
    - θ is dimensionless density/temperature
    - n is polytropic index
    """
    
    def __init__(self, mass: float, polytropic_index: float = 3.0):
        """
        Initialize stellar structure solver.
        
        Args:
            mass: Stellar mass in solar masses
            polytropic_index: n (3 for radiation-dominated, 1.5 for convective)
        """
        self.mass = mass
        self.n = polytropic_index
        self.xi = None
        self.theta = None
        self.dtheta = None
    
    def solve_lane_emden(self, xi_max: float = 10.0, n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve Lane-Emden equation using 4th-order Runge-Kutta.
        
        Returns:
            xi: Dimensionless radius array
            theta: Dimensionless density/temperature array
        """
        n = self.n
        
        def derivatives(xi, y):
            theta, dtheta = y
            if xi < 1e-10:
                # Handle singularity at origin
                d2theta = -1/3 if n == 0 else -(1/3) * theta**n
            else:
                d2theta = -2/xi * dtheta - theta**n if theta > 0 else 0
            return np.array([dtheta, d2theta])
        
        # Initial conditions: θ(0) = 1, θ'(0) = 0
        xi = np.linspace(1e-10, xi_max, n_points)
        h = xi[1] - xi[0]
        
        theta = np.zeros(n_points)
        dtheta = np.zeros(n_points)
        theta[0] = 1.0
        dtheta[0] = 0.0
        
        # RK4 integration
        for i in range(n_points - 1):
            y = np.array([theta[i], dtheta[i]])
            
            k1 = derivatives(xi[i], y)
            k2 = derivatives(xi[i] + h/2, y + h*k1/2)
            k3 = derivatives(xi[i] + h/2, y + h*k2/2)
            k4 = derivatives(xi[i] + h, y + h*k3)
            
            y_new = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
            
            theta[i+1] = max(y_new[0], 0)  # θ can't be negative
            dtheta[i+1] = y_new[1]
            
            if theta[i+1] <= 0:
                # Found the surface (first zero of θ)
                theta[i+1:] = 0
                break
        
        self.xi = xi
        self.theta = theta
        self.dtheta = dtheta
        
        return xi, theta
    
    def get_surface_radius(self) -> float:
        """Get the surface radius ξ₁ where θ first becomes zero."""
        if self.theta is None:
            self.solve_lane_emden()
        
        # Find first zero crossing
        for i in range(len(self.theta) - 1):
            if self.theta[i] > 0 and self.theta[i+1] <= 0:
                # Linear interpolation
                return self.xi[i] - self.theta[i] * (self.xi[i+1] - self.xi[i]) / (self.theta[i+1] - self.theta[i])
        
        return self.xi[-1]
    
    def density_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get density profile ρ(r) normalized to central density."""
        if self.theta is None:
            self.solve_lane_emden()
        return self.xi, self.theta**self.n


# =============================================================================
# COSMOLOGY
# =============================================================================

@dataclass
class CosmologicalModel:
    """
    ΛCDM Cosmological Model with Friedmann equations.
    
    Parameters (Planck 2018 defaults):
        H_0: Hubble constant [km/s/Mpc]
        Omega_m: Matter density parameter
        Omega_Lambda: Dark energy density parameter
        Omega_r: Radiation density parameter
        Omega_k: Curvature parameter (1 - Ω_total)
    """
    H_0: float = 67.4  # km/s/Mpc
    Omega_m: float = 0.315
    Omega_Lambda: float = 0.685
    Omega_r: float = 9.2e-5  # Radiation (including neutrinos)
    
    @property
    def Omega_k(self) -> float:
        """Curvature parameter."""
        return 1.0 - self.Omega_m - self.Omega_Lambda - self.Omega_r
    
    @property
    def H_0_SI(self) -> float:
        """Hubble constant in SI units [s⁻¹]."""
        return self.H_0 * 1000 / (1e6 * pc)
    
    @property
    def hubble_time(self) -> float:
        """Hubble time 1/H_0 in Gyr."""
        return 1.0 / self.H_0_SI / (1e9 * 365.25 * 24 * 3600)
    
    def E(self, z: float) -> float:
        """
        Dimensionless Hubble parameter E(z) = H(z)/H_0.
        
        E²(z) = Ω_r(1+z)⁴ + Ω_m(1+z)³ + Ω_k(1+z)² + Ω_Λ
        """
        zp1 = 1 + z
        return np.sqrt(
            self.Omega_r * zp1**4 +
            self.Omega_m * zp1**3 +
            self.Omega_k * zp1**2 +
            self.Omega_Lambda
        )
    
    def H(self, z: float) -> float:
        """Hubble parameter H(z) in km/s/Mpc."""
        return self.H_0 * self.E(z)
    
    def comoving_distance(self, z: float, n_steps: int = 1000) -> float:
        """
        Comoving distance to redshift z in Mpc.
        
        D_c = (c/H_0) ∫₀ᶻ dz'/E(z')
        """
        if z == 0:
            return 0.0
        
        z_arr = np.linspace(0, z, n_steps)
        integrand = 1.0 / self.E(z_arr)
        integral = np.trapz(integrand, z_arr)
        
        return (c / 1000) / self.H_0 * integral  # Mpc
    
    def luminosity_distance(self, z: float) -> float:
        """Luminosity distance in Mpc: D_L = (1+z) * D_c for flat universe."""
        return (1 + z) * self.comoving_distance(z)
    
    def angular_diameter_distance(self, z: float) -> float:
        """Angular diameter distance in Mpc: D_A = D_c / (1+z)."""
        return self.comoving_distance(z) / (1 + z)
    
    def lookback_time(self, z: float, n_steps: int = 1000) -> float:
        """
        Lookback time to redshift z in Gyr.
        
        t_L = (1/H_0) ∫₀ᶻ dz'/[(1+z')E(z')]
        """
        if z == 0:
            return 0.0
        
        z_arr = np.linspace(0, z, n_steps)
        integrand = 1.0 / ((1 + z_arr) * self.E(z_arr))
        integral = np.trapz(integrand, z_arr)
        
        return self.hubble_time * integral  # Gyr
    
    def age_of_universe(self, n_steps: int = 1000) -> float:
        """Age of universe in Gyr (lookback time to z=∞)."""
        # Integrate to high z (effectively infinity for ΛCDM)
        z_arr = np.logspace(-10, 4, n_steps)
        integrand = 1.0 / ((1 + z_arr) * self.E(z_arr))
        integral = np.trapz(integrand, z_arr)
        return self.hubble_time * integral
    
    def critical_density(self, z: float = 0) -> float:
        """Critical density at redshift z in kg/m³."""
        H_z = self.H(z) * 1000 / (1e6 * pc)  # Convert to SI
        return 3 * H_z**2 / (8 * np.pi * G)
    
    def distance_modulus(self, z: float) -> float:
        """Distance modulus μ = m - M = 5*log10(D_L/10pc)."""
        D_L_pc = self.luminosity_distance(z) * 1e6  # Convert Mpc to pc
        return 5 * np.log10(D_L_pc / 10)
    
    def to_dict(self) -> Dict:
        return {
            "H_0_km_s_Mpc": self.H_0,
            "Omega_matter": self.Omega_m,
            "Omega_Lambda": self.Omega_Lambda,
            "Omega_radiation": self.Omega_r,
            "Omega_curvature": self.Omega_k,
            "hubble_time_Gyr": self.hubble_time,
            "age_of_universe_Gyr": self.age_of_universe(),
            "critical_density_kg_m3": self.critical_density()
        }


class DarkMatterHalo:
    """
    NFW (Navarro-Frenk-White) Dark Matter Halo Profile.
    
    ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
    
    Parameters:
        M_vir: Virial mass in solar masses
        c: Concentration parameter (typically 5-20)
        z: Redshift
    """
    
    def __init__(self, M_vir: float, c: float = 10.0, z: float = 0.0):
        self.M_vir = M_vir  # M_sun
        self.c = c
        self.z = z
        self.cosmo = CosmologicalModel()
        
        # Calculate derived parameters
        self._calculate_parameters()
    
    def _calculate_parameters(self):
        """Calculate scale radius and characteristic density."""
        # Virial radius (r_200 where ρ = 200 * ρ_crit)
        rho_crit = self.cosmo.critical_density(self.z)
        M_vir_kg = self.M_vir * M_sun
        
        # r_vir from M_vir = (4π/3) * 200 * ρ_crit * r_vir³
        self.r_vir = (3 * M_vir_kg / (4 * np.pi * 200 * rho_crit))**(1/3)  # meters
        self.r_vir_kpc = self.r_vir / (1000 * pc)  # kpc
        
        # Scale radius
        self.r_s = self.r_vir / self.c  # meters
        self.r_s_kpc = self.r_s / (1000 * pc)  # kpc
        
        # Characteristic density
        delta_c = (200/3) * self.c**3 / (np.log(1 + self.c) - self.c/(1 + self.c))
        self.rho_s = delta_c * rho_crit
    
    def density(self, r: float) -> float:
        """
        NFW density at radius r (in meters).
        
        Returns density in kg/m³.
        """
        x = r / self.r_s
        if x < 1e-10:
            x = 1e-10  # Avoid singularity
        return self.rho_s / (x * (1 + x)**2)
    
    def enclosed_mass(self, r: float) -> float:
        """
        Mass enclosed within radius r.
        
        M(<r) = 4π ρ_s r_s³ [ln(1+x) - x/(1+x)]
        
        Returns mass in kg.
        """
        x = r / self.r_s
        return 4 * np.pi * self.rho_s * self.r_s**3 * (np.log(1 + x) - x/(1 + x))
    
    def circular_velocity(self, r: float) -> float:
        """
        Circular velocity at radius r.
        
        v_c = sqrt(G * M(<r) / r)
        
        Returns velocity in m/s.
        """
        M_enc = self.enclosed_mass(r)
        return np.sqrt(G * M_enc / r)
    
    def rotation_curve(self, r_array: np.ndarray) -> np.ndarray:
        """Calculate rotation curve for array of radii (in kpc)."""
        r_m = r_array * 1000 * pc  # Convert kpc to meters
        return np.array([self.circular_velocity(r) / 1000 for r in r_m])  # km/s
    
    def to_dict(self) -> Dict:
        return {
            "M_vir_Msun": self.M_vir,
            "concentration": self.c,
            "redshift": self.z,
            "r_vir_kpc": self.r_vir_kpc,
            "r_s_kpc": self.r_s_kpc,
            "rho_s_kg_m3": self.rho_s
        }


# =============================================================================
# EXOPLANETS
# =============================================================================

@dataclass
class Exoplanet:
    """
    Exoplanet model with detection and habitability analysis.
    
    Attributes:
        mass: Planet mass in Earth masses
        radius: Planet radius in Earth radii
        semi_major_axis: Orbital distance in AU
        eccentricity: Orbital eccentricity
        period: Orbital period in days
        host_star: Host star properties
    """
    mass: float  # M_Earth
    radius: float  # R_Earth
    semi_major_axis: float  # AU
    eccentricity: float = 0.0
    period: float = None  # days
    host_star: Star = None
    name: str = ""
    
    # Earth reference values
    M_Earth = 5.972e24  # kg
    R_Earth = 6.371e6   # m
    
    def __post_init__(self):
        if self.period is None and self.host_star is not None:
            self.period = self._kepler_period()
    
    def _kepler_period(self) -> float:
        """Calculate orbital period from Kepler's 3rd law."""
        if self.host_star is None:
            M_star = 1.0  # Assume solar mass
        else:
            M_star = self.host_star.mass
        
        # P² = (4π²/GM) a³
        a_m = self.semi_major_axis * AU
        M_kg = M_star * M_sun
        P_s = 2 * np.pi * np.sqrt(a_m**3 / (G * M_kg))
        return P_s / (24 * 3600)  # Convert to days
    
    @property
    def density(self) -> float:
        """Bulk density in g/cm³."""
        M_kg = self.mass * self.M_Earth
        R_m = self.radius * self.R_Earth
        rho = M_kg / (4/3 * np.pi * R_m**3)
        return rho / 1000  # g/cm³
    
    @property
    def surface_gravity(self) -> float:
        """Surface gravity in m/s² (Earth = 9.81)."""
        return G * self.mass * self.M_Earth / (self.radius * self.R_Earth)**2
    
    @property
    def escape_velocity(self) -> float:
        """Escape velocity in km/s."""
        return np.sqrt(2 * G * self.mass * self.M_Earth / (self.radius * self.R_Earth)) / 1000
    
    def equilibrium_temperature(self, albedo: float = 0.3) -> float:
        """
        Equilibrium temperature assuming blackbody.
        
        T_eq = T_star * sqrt(R_star / 2a) * (1 - A)^0.25
        """
        if self.host_star is None:
            T_star = T_sun
            R_star = R_sun
        else:
            T_star = self.host_star.temperature
            R_star = self.host_star.radius * R_sun
        
        a_m = self.semi_major_axis * AU
        return T_star * np.sqrt(R_star / (2 * a_m)) * (1 - albedo)**0.25
    
    def is_in_habitable_zone(self) -> bool:
        """Check if planet is in the habitable zone of its host star."""
        if self.host_star is None:
            return False
        
        hz_inner, hz_outer = self.host_star.habitable_zone()
        return hz_inner <= self.semi_major_axis <= hz_outer
    
    @property
    def planet_type(self) -> str:
        """Classify planet based on mass and radius."""
        M, R = self.mass, self.radius
        
        if M < 0.1:
            return "Asteroid"
        elif R < 1.25 and M < 2:
            return "Terrestrial"
        elif R < 2.0 and M < 10:
            return "Super-Earth"
        elif R < 4.0 and M < 20:
            return "Mini-Neptune"
        elif R < 6.0 and M < 100:
            return "Neptune-like"
        elif M < 300:
            return "Saturn-like"
        else:
            return "Jupiter-like"
    
    def transit_depth(self) -> float:
        """Transit depth (R_p/R_star)² for detection."""
        if self.host_star is None:
            R_star = 1.0  # Solar radii
        else:
            R_star = self.host_star.radius
        
        R_p_Rsun = self.radius * self.R_Earth / R_sun
        return (R_p_Rsun / R_star)**2
    
    def radial_velocity_amplitude(self) -> float:
        """
        Radial velocity semi-amplitude K in m/s.
        
        K = (2πG/P)^(1/3) * M_p*sin(i) / (M_star + M_p)^(2/3) * 1/sqrt(1-e²)
        """
        if self.host_star is None:
            M_star = M_sun
        else:
            M_star = self.host_star.mass * M_sun
        
        M_p = self.mass * self.M_Earth
        P = self.period * 24 * 3600  # seconds
        e = self.eccentricity
        
        # Assume sin(i) = 1 (edge-on)
        K = (2 * np.pi * G / P)**(1/3) * M_p / (M_star + M_p)**(2/3)
        K /= np.sqrt(1 - e**2)
        
        return K
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "mass_MEarth": self.mass,
            "radius_REarth": self.radius,
            "semi_major_axis_AU": self.semi_major_axis,
            "eccentricity": self.eccentricity,
            "period_days": self.period,
            "density_g_cm3": self.density,
            "surface_gravity_m_s2": self.surface_gravity,
            "escape_velocity_km_s": self.escape_velocity,
            "equilibrium_temp_K": self.equilibrium_temperature(),
            "planet_type": self.planet_type,
            "in_habitable_zone": self.is_in_habitable_zone(),
            "transit_depth": self.transit_depth(),
            "RV_amplitude_m_s": self.radial_velocity_amplitude()
        }


# =============================================================================
# GRAVITATIONAL WAVES
# =============================================================================

class BinarySystem:
    """
    Compact binary system for gravitational wave calculations.
    
    Uses post-Newtonian approximation for inspiral phase.
    """
    
    def __init__(self, m1: float, m2: float, separation: float = None, frequency: float = None):
        """
        Initialize binary system.
        
        Args:
            m1, m2: Component masses in solar masses
            separation: Initial separation in meters (or)
            frequency: Gravitational wave frequency in Hz
        """
        self.m1 = m1 * M_sun  # kg
        self.m2 = m2 * M_sun  # kg
        self.M_total = self.m1 + self.m2
        self.mu = self.m1 * self.m2 / self.M_total  # Reduced mass
        self.eta = self.mu / self.M_total  # Symmetric mass ratio
        self.M_chirp = self.M_total * self.eta**(3/5)  # Chirp mass
        
        if separation is not None:
            self.a = separation
            self.f_gw = self._frequency_from_separation()
        elif frequency is not None:
            self.f_gw = frequency
            self.a = self._separation_from_frequency()
        else:
            raise ValueError("Must specify either separation or frequency")
    
    def _frequency_from_separation(self) -> float:
        """GW frequency from orbital separation (f_gw = 2 * f_orb)."""
        f_orb = np.sqrt(G * self.M_total / self.a**3) / (2 * np.pi)
        return 2 * f_orb
    
    def _separation_from_frequency(self) -> float:
        """Orbital separation from GW frequency."""
        f_orb = self.f_gw / 2
        return (G * self.M_total / (2 * np.pi * f_orb)**2)**(1/3)
    
    def gw_luminosity(self) -> float:
        """
        Gravitational wave luminosity (Peters formula).
        
        L_GW = (32/5) * (G⁴/c⁵) * (m1*m2)² * (m1+m2) / a⁵
        """
        return (32/5) * G**4 / c**5 * (self.m1 * self.m2)**2 * self.M_total / self.a**5
    
    def inspiral_time(self) -> float:
        """
        Time to merger from current separation (Peters formula).
        
        τ = (5/256) * (c⁵/G³) * a⁴ / (m1*m2*(m1+m2))
        
        Returns time in seconds.
        """
        return (5/256) * c**5 / G**3 * self.a**4 / (self.m1 * self.m2 * self.M_total)
    
    def strain_amplitude(self, distance: float) -> float:
        """
        GW strain amplitude at distance (in meters).
        
        h = (4/D) * (G*M_chirp/c²)^(5/3) * (π*f_gw/c)^(2/3)
        """
        return (4/distance) * (G * self.M_chirp / c**2)**(5/3) * (np.pi * self.f_gw / c)**(2/3)
    
    def characteristic_strain(self, distance: float, observation_time: float) -> float:
        """
        Characteristic strain for detection.
        
        h_c = h * sqrt(f_gw * T_obs)
        """
        h = self.strain_amplitude(distance)
        return h * np.sqrt(self.f_gw * observation_time)
    
    def chirp_rate(self) -> float:
        """
        Frequency derivative df/dt.
        
        df/dt = (96/5) * π^(8/3) * (G*M_chirp/c³)^(5/3) * f^(11/3)
        """
        return (96/5) * np.pi**(8/3) * (G * self.M_chirp / c**3)**(5/3) * self.f_gw**(11/3)
    
    def to_dict(self) -> Dict:
        return {
            "m1_Msun": self.m1 / M_sun,
            "m2_Msun": self.m2 / M_sun,
            "M_total_Msun": self.M_total / M_sun,
            "M_chirp_Msun": self.M_chirp / M_sun,
            "separation_m": self.a,
            "f_gw_Hz": self.f_gw,
            "inspiral_time_s": self.inspiral_time(),
            "gw_luminosity_W": self.gw_luminosity()
        }


# =============================================================================
# HIGH-LEVEL SIMULATION INTERFACE
# =============================================================================

class AstrophysicsSimulator:
    """
    High-level interface for astrophysics simulations.
    
    Integrates with QENEX discovery pipeline.
    """
    
    def __init__(self):
        self.results = {}
        self.cosmo = CosmologicalModel()
    
    def simulate_star(self, mass: float, **kwargs) -> Dict:
        """Simulate stellar properties."""
        star = Star(mass=mass, **kwargs)
        result = star.to_dict()
        
        # Add stellar structure if requested
        if kwargs.get('structure', False):
            struct = StellarStructure(mass)
            xi, theta = struct.solve_lane_emden()
            result['structure'] = {
                'xi': xi.tolist(),
                'theta': theta.tolist(),
                'surface_xi': struct.get_surface_radius()
            }
        
        self.results['star'] = result
        return result
    
    def simulate_cosmology(self, z_max: float = 10.0, n_points: int = 100) -> Dict:
        """Simulate cosmological distances and times."""
        z_arr = np.linspace(0, z_max, n_points)
        
        result = {
            'model': self.cosmo.to_dict(),
            'z': z_arr.tolist(),
            'comoving_distance_Mpc': [self.cosmo.comoving_distance(z) for z in z_arr],
            'luminosity_distance_Mpc': [self.cosmo.luminosity_distance(z) for z in z_arr],
            'lookback_time_Gyr': [self.cosmo.lookback_time(z) for z in z_arr],
            'H_km_s_Mpc': [self.cosmo.H(z) for z in z_arr]
        }
        
        self.results['cosmology'] = result
        return result
    
    def simulate_dark_matter_halo(self, M_vir: float, c: float = 10.0) -> Dict:
        """Simulate NFW dark matter halo."""
        halo = DarkMatterHalo(M_vir, c)
        
        # Calculate rotation curve
        r_kpc = np.logspace(-1, 2, 100)  # 0.1 to 100 kpc
        v_circ = halo.rotation_curve(r_kpc)
        
        result = halo.to_dict()
        result['rotation_curve'] = {
            'r_kpc': r_kpc.tolist(),
            'v_km_s': v_circ.tolist()
        }
        
        self.results['dark_matter'] = result
        return result
    
    def simulate_exoplanet(self, mass: float, radius: float, 
                           semi_major_axis: float, host_mass: float = 1.0) -> Dict:
        """Simulate exoplanet properties."""
        host = Star(mass=host_mass)
        planet = Exoplanet(
            mass=mass, 
            radius=radius, 
            semi_major_axis=semi_major_axis,
            host_star=host
        )
        
        result = planet.to_dict()
        result['host_star'] = host.to_dict()
        
        self.results['exoplanet'] = result
        return result
    
    def simulate_binary_gw(self, m1: float, m2: float, f_gw: float, 
                           distance_Mpc: float = 100.0) -> Dict:
        """Simulate gravitational wave source."""
        binary = BinarySystem(m1, m2, frequency=f_gw)
        
        distance_m = distance_Mpc * 1e6 * pc
        
        result = binary.to_dict()
        result['distance_Mpc'] = distance_Mpc
        result['strain_amplitude'] = binary.strain_amplitude(distance_m)
        result['chirp_rate_Hz_s'] = binary.chirp_rate()
        
        self.results['gravitational_waves'] = result
        return result
    
    def get_all_results(self) -> Dict:
        """Get all simulation results."""
        return self.results


# =============================================================================
# MAIN / DEMO
# =============================================================================

def main():
    """Demonstrate astrophysics module capabilities."""
    
    print("=" * 60)
    print("QENEX ASTROPHYSICS MODULE")
    print("=" * 60)
    
    sim = AstrophysicsSimulator()
    
    # 1. Stellar simulation
    print("\n1. STELLAR PHYSICS")
    print("-" * 40)
    star_result = sim.simulate_star(mass=1.0, name="Sun-like")
    print(f"Star: {star_result['name']}")
    print(f"  Spectral Type: {star_result['spectral_type']}")
    print(f"  Temperature: {star_result['temperature_K']:.0f} K")
    print(f"  Luminosity: {star_result['luminosity_Lsun']:.2f} L_sun")
    print(f"  Main Sequence Lifetime: {star_result['main_sequence_lifetime_Gyr']:.2f} Gyr")
    print(f"  Habitable Zone: {star_result['habitable_zone_AU'][0]:.2f} - {star_result['habitable_zone_AU'][1]:.2f} AU")
    
    # 2. Cosmology
    print("\n2. COSMOLOGY (ΛCDM)")
    print("-" * 40)
    cosmo_result = sim.simulate_cosmology(z_max=3.0)
    print(f"H_0: {cosmo_result['model']['H_0_km_s_Mpc']} km/s/Mpc")
    print(f"Ω_m: {cosmo_result['model']['Omega_matter']}")
    print(f"Ω_Λ: {cosmo_result['model']['Omega_Lambda']}")
    print(f"Age of Universe: {cosmo_result['model']['age_of_universe_Gyr']:.2f} Gyr")
    print(f"Lookback time to z=1: {sim.cosmo.lookback_time(1.0):.2f} Gyr")
    print(f"Comoving distance to z=1: {sim.cosmo.comoving_distance(1.0):.0f} Mpc")
    
    # 3. Dark Matter Halo
    print("\n3. DARK MATTER (NFW Halo)")
    print("-" * 40)
    dm_result = sim.simulate_dark_matter_halo(M_vir=1e12, c=10)
    print(f"Virial Mass: {dm_result['M_vir_Msun']:.2e} M_sun")
    print(f"Concentration: {dm_result['concentration']}")
    print(f"Virial Radius: {dm_result['r_vir_kpc']:.1f} kpc")
    print(f"Scale Radius: {dm_result['r_s_kpc']:.1f} kpc")
    
    # 4. Exoplanet
    print("\n4. EXOPLANET")
    print("-" * 40)
    planet_result = sim.simulate_exoplanet(
        mass=1.0, radius=1.0, semi_major_axis=1.0, host_mass=1.0
    )
    print(f"Planet Type: {planet_result['planet_type']}")
    print(f"Surface Gravity: {planet_result['surface_gravity_m_s2']:.2f} m/s²")
    print(f"Equilibrium Temp: {planet_result['equilibrium_temp_K']:.0f} K")
    print(f"In Habitable Zone: {planet_result['in_habitable_zone']}")
    print(f"Transit Depth: {planet_result['transit_depth']:.2e}")
    print(f"RV Amplitude: {planet_result['RV_amplitude_m_s']:.2f} m/s")
    
    # 5. Gravitational Waves
    print("\n5. GRAVITATIONAL WAVES")
    print("-" * 40)
    gw_result = sim.simulate_binary_gw(m1=30, m2=30, f_gw=100, distance_Mpc=400)
    print(f"Binary: {gw_result['m1_Msun']:.0f} + {gw_result['m2_Msun']:.0f} M_sun")
    print(f"Chirp Mass: {gw_result['M_chirp_Msun']:.1f} M_sun")
    print(f"GW Frequency: {gw_result['f_gw_Hz']:.1f} Hz")
    print(f"Inspiral Time: {gw_result['inspiral_time_s']:.2f} s")
    print(f"Strain at {gw_result['distance_Mpc']} Mpc: {gw_result['strain_amplitude']:.2e}")
    
    print("\n" + "=" * 60)
    print("Astrophysics simulation complete!")
    print("=" * 60)
    
    return sim


if __name__ == "__main__":
    main()
