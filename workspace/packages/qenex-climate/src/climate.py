"""
QENEX Climate Science Module
============================
Comprehensive climate modeling covering:
- Energy balance models (0D, 1D, 2D)
- Carbon cycle dynamics
- Ice sheet and glacier dynamics
- Ocean circulation (thermohaline)
- Atmospheric chemistry (ozone, aerosols)
- Climate feedback mechanisms

Physical Constants:
- Solar constant S₀ = 1361 W/m²
- Stefan-Boltzmann σ = 5.67e-8 W/(m²·K⁴)
- Earth radius R_E = 6.371e6 m
- Earth albedo α ≈ 0.30

References:
- IPCC AR6 (2021)
- Trenberth & Fasullo (2012) - Earth's Energy Budget
- Pierrehumbert (2010) - Principles of Planetary Climate
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Union
from enum import Enum, auto
import json
from abc import ABC, abstractmethod

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Solar and radiative
S_0 = 1361.0               # Solar constant [W/m²] (TSI at 1 AU)
SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]

# Earth parameters
R_EARTH = 6.371e6          # Earth radius [m]
A_EARTH = 5.1e14           # Earth surface area [m²]
M_ATM = 5.15e18            # Mass of atmosphere [kg]
M_OCEAN = 1.4e21           # Mass of ocean [kg]
C_P_AIR = 1004.0           # Specific heat of air [J/(kg·K)]
C_P_OCEAN = 3985.0         # Specific heat of seawater [J/(kg·K)]
RHO_OCEAN = 1025.0         # Density of seawater [kg/m³]

# Greenhouse parameters
ALPHA_EARTH = 0.30         # Earth's average albedo
EMISSIVITY_EARTH = 0.612   # Effective atmospheric emissivity (1-layer)

# Carbon cycle
PGC_TO_PPM = 2.124         # Conversion: 1 PgC ≈ 2.124 ppm CO2
CO2_PREINDUSTRIAL = 280.0  # Pre-industrial CO2 [ppm]
CO2_CURRENT = 420.0        # Current CO2 (2024) [ppm]

# Ice sheet parameters
RHO_ICE = 917.0            # Ice density [kg/m³]
L_FUSION = 3.34e5          # Latent heat of fusion [J/kg]
GREENLAND_VOLUME = 2.85e15  # Greenland ice volume [m³]
ANTARCTICA_VOLUME = 26.5e15 # Antarctica ice volume [m³]

# Time constants
SECONDS_PER_YEAR = 365.25 * 24 * 3600


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ForcingType(Enum):
    """Types of radiative forcing."""
    CO2 = auto()
    CH4 = auto()
    N2O = auto()
    AEROSOL = auto()
    SOLAR = auto()
    VOLCANIC = auto()
    LAND_USE = auto()


class FeedbackType(Enum):
    """Climate feedback mechanisms."""
    WATER_VAPOR = auto()      # Positive: warming → more water vapor → more warming
    ICE_ALBEDO = auto()       # Positive: warming → less ice → lower albedo → more warming
    CLOUD = auto()            # Complex: can be positive or negative
    LAPSE_RATE = auto()       # Negative: warming → steeper lapse rate → more radiation
    PLANCK = auto()           # Negative: warming → more outgoing radiation
    VEGETATION = auto()       # Complex: depends on region


class OceanBasin(Enum):
    """Major ocean basins."""
    ATLANTIC = auto()
    PACIFIC = auto()
    INDIAN = auto()
    SOUTHERN = auto()
    ARCTIC = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RadiativeForcing:
    """Radiative forcing component.
    
    Attributes:
        type: Type of forcing
        value: Forcing magnitude [W/m²]
        uncertainty: 1-sigma uncertainty [W/m²]
        year: Reference year
    """
    type: ForcingType
    value: float  # W/m²
    uncertainty: float = 0.0
    year: int = 2020


@dataclass
class ClimateState:
    """Global climate state at a point in time.
    
    Attributes:
        year: Calendar year
        temperature: Global mean surface temperature anomaly [K]
        co2: Atmospheric CO2 concentration [ppm]
        sea_level: Sea level anomaly [m]
        ice_mass_greenland: Greenland ice mass anomaly [Gt]
        ice_mass_antarctica: Antarctica ice mass anomaly [Gt]
        ocean_heat_content: Ocean heat content anomaly [ZJ]
    """
    year: float
    temperature: float = 0.0          # K anomaly
    co2: float = CO2_PREINDUSTRIAL    # ppm
    sea_level: float = 0.0            # m
    ice_mass_greenland: float = 0.0   # Gt anomaly
    ice_mass_antarctica: float = 0.0  # Gt anomaly
    ocean_heat_content: float = 0.0   # ZJ (10²¹ J)


@dataclass
class CarbonReservoir:
    """Carbon reservoir in the global carbon cycle.
    
    Attributes:
        name: Reservoir name
        mass: Carbon mass [PgC]
        residence_time: Mean residence time [years]
    """
    name: str
    mass: float  # PgC
    residence_time: float  # years
    

@dataclass
class OceanLayer:
    """Ocean layer for box model.
    
    Attributes:
        name: Layer name
        depth: Layer thickness [m]
        temperature: Temperature [K]
        salinity: Salinity [PSU]
        volume: Layer volume [m³]
    """
    name: str
    depth: float
    temperature: float
    salinity: float = 35.0
    volume: float = 0.0


# =============================================================================
# ENERGY BALANCE MODELS
# =============================================================================

class ZeroDimensionalEBM:
    """Zero-dimensional Energy Balance Model.
    
    The simplest climate model treating Earth as a uniform sphere.
    
    Energy balance: (1-α)S₀/4 = εσT⁴
    
    Includes greenhouse effect through effective emissivity.
    
    Example:
        >>> ebm = ZeroDimensionalEBM()
        >>> T_eq = ebm.equilibrium_temperature()
        >>> print(f"Equilibrium temperature: {T_eq:.1f} K")
    """
    
    def __init__(
        self,
        solar_constant: float = S_0,
        albedo: float = ALPHA_EARTH,
        emissivity: float = EMISSIVITY_EARTH
    ):
        """Initialize 0D EBM.
        
        Args:
            solar_constant: Solar irradiance at Earth [W/m²]
            albedo: Planetary albedo [0-1]
            emissivity: Effective atmospheric emissivity [0-1]
        """
        self.S_0 = solar_constant
        self.alpha = albedo
        self.epsilon = emissivity
        
    def equilibrium_temperature(self) -> float:
        """Calculate equilibrium surface temperature.
        
        Returns:
            Equilibrium temperature [K]
        """
        # Incoming solar radiation (averaged over sphere)
        S_in = self.S_0 * (1 - self.alpha) / 4
        
        # With greenhouse effect: T_s⁴ = S_in / (εσ)
        # But surface emits σT_s⁴, atmosphere absorbs fraction and re-emits
        # Effective: T_s = (S_in / (εσ))^(1/4) for simple 1-layer model
        
        # More accurate: T_s = ((1 + τ/2) * S_in / σ)^(1/4)
        # where τ is optical depth. For ε = 0.612, τ ≈ 0.78
        tau = -np.log(1 - self.epsilon)
        T_surface = ((1 + tau/2) * S_in / SIGMA_SB) ** 0.25
        
        return T_surface
    
    def no_atmosphere_temperature(self) -> float:
        """Calculate temperature without greenhouse effect.
        
        Returns:
            Bare rock temperature [K]
        """
        S_in = self.S_0 * (1 - self.alpha) / 4
        return (S_in / SIGMA_SB) ** 0.25
    
    def greenhouse_warming(self) -> float:
        """Calculate greenhouse warming contribution.
        
        Returns:
            Temperature increase due to greenhouse effect [K]
        """
        return self.equilibrium_temperature() - self.no_atmosphere_temperature()
    
    def climate_sensitivity(self, delta_forcing: float = 3.7) -> float:
        """Estimate climate sensitivity from forcing.
        
        Uses linearized feedback parameter.
        
        Args:
            delta_forcing: Radiative forcing change [W/m²]
                          (3.7 W/m² for CO2 doubling)
        
        Returns:
            Equilibrium temperature change [K]
        """
        # Planck feedback parameter: λ_0 = 4εσT³
        T_eq = self.equilibrium_temperature()
        lambda_0 = 4 * self.epsilon * SIGMA_SB * T_eq**3
        
        # No-feedback sensitivity
        delta_T_0 = delta_forcing / lambda_0
        
        # With feedbacks (empirical factor ~2-4x amplification)
        # Using central estimate of ECS ~3K per doubling
        feedback_factor = 2.5
        
        return delta_T_0 * feedback_factor


class OneDimensionalEBM:
    """One-dimensional (latitudinal) Energy Balance Model.
    
    Resolves temperature as function of latitude, including:
    - Latitudinal variation of insolation
    - Ice-albedo feedback
    - Meridional heat transport
    
    dT/dt = (1/C) * [S(φ)(1-α(T)) - εσT⁴ + D∇²T]
    
    where φ is latitude, D is diffusion coefficient.
    """
    
    def __init__(
        self,
        n_latitudes: int = 90,
        diffusion_coeff: float = 0.44,  # W/(m²·K)
        ice_albedo: float = 0.6,
        ocean_albedo: float = 0.1,
        ice_threshold: float = 263.0  # K
    ):
        """Initialize 1D EBM.
        
        Args:
            n_latitudes: Number of latitude bands
            diffusion_coeff: Meridional heat diffusion [W/(m²·K)]
            ice_albedo: Albedo of ice-covered regions
            ocean_albedo: Albedo of ice-free regions
            ice_threshold: Temperature below which ice forms [K]
        """
        self.n_lat = n_latitudes
        self.D = diffusion_coeff
        self.alpha_ice = ice_albedo
        self.alpha_ocean = ocean_albedo
        self.T_ice = ice_threshold
        
        # Latitude grid (cell centers)
        self.phi = np.linspace(-90, 90, n_latitudes)  # degrees
        self.phi_rad = np.radians(self.phi)
        
        # Grid spacing
        self.dphi = np.radians(180.0 / n_latitudes)
        
        # Heat capacity (mixed layer ocean ~50m)
        self.C = C_P_OCEAN * RHO_OCEAN * 50  # J/(m²·K)
        
        # Initialize temperature profile
        self.T = self._initial_temperature()
        
    def _initial_temperature(self) -> np.ndarray:
        """Initialize temperature with realistic meridional profile.
        
        Returns:
            Temperature array [K]
        """
        # Approximate: T = T_eq - ΔT * sin²(φ)
        T_eq = 288.0  # Equatorial temperature
        delta_T = 40.0  # Equator-to-pole difference
        return T_eq - delta_T * np.sin(self.phi_rad)**2
    
    def _insolation(self, day_of_year: int = 172) -> np.ndarray:
        """Calculate daily-mean insolation at each latitude.
        
        Args:
            day_of_year: Day (172 = summer solstice NH)
        
        Returns:
            Insolation array [W/m²]
        """
        # Solar declination
        delta = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        delta_rad = np.radians(delta)
        
        # Hour angle at sunrise/sunset
        cos_h0 = -np.tan(self.phi_rad) * np.tan(delta_rad)
        cos_h0 = np.clip(cos_h0, -1, 1)
        h0 = np.arccos(cos_h0)
        
        # Daily mean insolation
        S = (S_0 / np.pi) * (h0 * np.sin(self.phi_rad) * np.sin(delta_rad) +
                             np.cos(self.phi_rad) * np.cos(delta_rad) * np.sin(h0))
        
        return np.maximum(S, 0)
    
    def _annual_mean_insolation(self) -> np.ndarray:
        """Calculate annual mean insolation.
        
        Returns:
            Annual mean insolation [W/m²]
        """
        S_annual = np.zeros(self.n_lat)
        for day in range(365):
            S_annual += self._insolation(day)
        return S_annual / 365
    
    def _albedo(self, T: np.ndarray) -> np.ndarray:
        """Calculate albedo with ice-albedo feedback.
        
        Args:
            T: Temperature array [K]
        
        Returns:
            Albedo array
        """
        # Smooth transition around ice threshold
        transition_width = 10.0  # K
        ice_fraction = 0.5 * (1 - np.tanh((T - self.T_ice) / transition_width))
        return self.alpha_ocean + (self.alpha_ice - self.alpha_ocean) * ice_fraction
    
    def _diffusion_operator(self, T: np.ndarray) -> np.ndarray:
        """Apply meridional diffusion.
        
        ∇²T in spherical coordinates for latitude only.
        
        Args:
            T: Temperature array [K]
        
        Returns:
            Diffusion term [W/m²]
        """
        # Second derivative in latitude with spherical geometry
        cos_phi = np.cos(self.phi_rad)
        
        diff = np.zeros_like(T)
        for i in range(1, self.n_lat - 1):
            # (1/cos(φ)) d/dφ [cos(φ) dT/dφ]
            dT_north = (T[i+1] - T[i]) / self.dphi
            dT_south = (T[i] - T[i-1]) / self.dphi
            
            cos_n = np.cos(self.phi_rad[i] + self.dphi/2)
            cos_s = np.cos(self.phi_rad[i] - self.dphi/2)
            
            diff[i] = (cos_n * dT_north - cos_s * dT_south) / (cos_phi[i] * self.dphi)
        
        # Boundary conditions (no flux at poles)
        diff[0] = diff[1]
        diff[-1] = diff[-2]
        
        return self.D * diff
    
    def step(self, dt: float = 86400.0) -> None:
        """Advance model one timestep.
        
        Args:
            dt: Timestep [seconds]
        """
        # Incoming solar
        S = self._annual_mean_insolation()
        alpha = self._albedo(self.T)
        Q_in = S * (1 - alpha)
        
        # Outgoing longwave
        Q_out = EMISSIVITY_EARTH * SIGMA_SB * self.T**4
        
        # Diffusion
        Q_diff = self._diffusion_operator(self.T)
        
        # Update temperature
        dT_dt = (Q_in - Q_out + Q_diff) / self.C
        self.T += dT_dt * dt
    
    def run_to_equilibrium(
        self,
        max_years: int = 100,
        tolerance: float = 0.01
    ) -> int:
        """Run model to equilibrium.
        
        Args:
            max_years: Maximum simulation years
            tolerance: Convergence criterion [K/year]
        
        Returns:
            Number of years to convergence
        """
        dt = 86400.0  # 1 day
        steps_per_year = 365
        
        for year in range(max_years):
            T_old = self.T.copy()
            
            for _ in range(steps_per_year):
                self.step(dt)
            
            # Check convergence
            dT = np.max(np.abs(self.T - T_old))
            if dT < tolerance:
                return year + 1
        
        return max_years
    
    def ice_line_latitude(self) -> float:
        """Find the ice line latitude.
        
        Returns:
            Latitude where ice begins [degrees]
        """
        ice_mask = self.T < self.T_ice
        if not np.any(ice_mask):
            return 90.0  # No ice
        if np.all(ice_mask):
            return 0.0  # Snowball
            
        # Find transition
        idx = np.where(ice_mask)[0][0]
        if idx > 0:
            # Linear interpolation
            return np.interp(self.T_ice, 
                           [self.T[idx], self.T[idx-1]],
                           [self.phi[idx], self.phi[idx-1]])
        return self.phi[idx]
    
    def global_mean_temperature(self) -> float:
        """Calculate area-weighted global mean temperature.
        
        Returns:
            Global mean temperature [K]
        """
        cos_phi = np.cos(self.phi_rad)
        return np.sum(self.T * cos_phi) / np.sum(cos_phi)


# =============================================================================
# CARBON CYCLE
# =============================================================================

class CarbonCycleModel:
    """Global carbon cycle box model.
    
    Four-reservoir model:
    - Atmosphere
    - Surface ocean
    - Deep ocean
    - Terrestrial biosphere
    
    Based on IPCC AR6 carbon cycle assessment.
    """
    
    def __init__(self):
        """Initialize carbon cycle with pre-industrial state."""
        # Reservoir masses [PgC]
        self.atmosphere = CarbonReservoir("Atmosphere", 590, 4.0)
        self.surface_ocean = CarbonReservoir("Surface Ocean", 900, 10.0)
        self.deep_ocean = CarbonReservoir("Deep Ocean", 37100, 1000.0)
        self.biosphere = CarbonReservoir("Terrestrial", 2000, 20.0)
        
        # Exchange coefficients [PgC/year]
        self.k_atm_bio = 120.0      # Atmosphere ↔ Biosphere
        self.k_atm_ocean = 80.0     # Atmosphere ↔ Surface ocean
        self.k_surface_deep = 40.0  # Surface ↔ Deep ocean
        
        # CO2 fertilization parameter
        self.beta_co2 = 0.4  # NPP enhancement per CO2 doubling
        
        # Ocean chemistry (Revelle factor)
        self.revelle_factor = 10.0  # Buffer factor
        
        # State history
        self.history: List[Dict] = []
        
    def co2_ppm(self) -> float:
        """Convert atmospheric carbon to CO2 concentration.
        
        Returns:
            CO2 concentration [ppm]
        """
        return self.atmosphere.mass * PGC_TO_PPM
    
    def _co2_fertilization(self) -> float:
        """Calculate CO2 fertilization factor for biosphere uptake.
        
        Returns:
            Fertilization multiplier
        """
        co2_ratio = self.co2_ppm() / CO2_PREINDUSTRIAL
        return 1 + self.beta_co2 * np.log(co2_ratio)
    
    def _ocean_uptake_factor(self) -> float:
        """Calculate ocean uptake efficiency.
        
        Decreases as ocean acidifies (Revelle factor increases).
        
        Returns:
            Uptake efficiency multiplier
        """
        # Simplified: efficiency decreases with CO2
        co2_ratio = self.co2_ppm() / CO2_PREINDUSTRIAL
        return 1.0 / (1 + 0.1 * (co2_ratio - 1))
    
    def step(self, dt: float, emissions: float = 0.0) -> None:
        """Advance carbon cycle one timestep.
        
        Args:
            dt: Timestep [years]
            emissions: Anthropogenic emissions [PgC/year]
        """
        # Atmosphere-Biosphere exchange
        fertilization = self._co2_fertilization()
        flux_bio_uptake = self.k_atm_bio * fertilization
        flux_bio_release = self.k_atm_bio * (self.biosphere.mass / 2000)
        
        # Atmosphere-Ocean exchange
        ocean_factor = self._ocean_uptake_factor()
        co2_ratio = self.co2_ppm() / CO2_PREINDUSTRIAL
        flux_ocean_uptake = self.k_atm_ocean * co2_ratio * ocean_factor
        flux_ocean_release = self.k_atm_ocean * (self.surface_ocean.mass / 900)
        
        # Surface-Deep ocean exchange
        flux_to_deep = self.k_surface_deep * (self.surface_ocean.mass / 900)
        flux_from_deep = self.k_surface_deep * (self.deep_ocean.mass / 37100)
        
        # Net fluxes
        d_atm = (emissions 
                 + flux_bio_release - flux_bio_uptake
                 + flux_ocean_release - flux_ocean_uptake) * dt
        
        d_bio = (flux_bio_uptake - flux_bio_release) * dt
        
        d_surface = (flux_ocean_uptake - flux_ocean_release
                     + flux_from_deep - flux_to_deep) * dt
        
        d_deep = (flux_to_deep - flux_from_deep) * dt
        
        # Update reservoirs
        self.atmosphere.mass += d_atm
        self.biosphere.mass += d_bio
        self.surface_ocean.mass += d_surface
        self.deep_ocean.mass += d_deep
        
    def run_scenario(
        self,
        years: int,
        emission_profile: Callable[[float], float],
        dt: float = 0.1
    ) -> List[Dict]:
        """Run emission scenario.
        
        Args:
            years: Simulation duration
            emission_profile: Function returning emissions [PgC/year] for given year
            dt: Timestep [years]
        
        Returns:
            History of climate states
        """
        self.history = []
        
        n_steps = int(years / dt)
        for i in range(n_steps):
            t = i * dt
            emissions = emission_profile(t)
            self.step(dt, emissions)
            
            if i % int(1/dt) == 0:  # Record annually
                self.history.append({
                    'year': t,
                    'co2_ppm': self.co2_ppm(),
                    'atmosphere': self.atmosphere.mass,
                    'biosphere': self.biosphere.mass,
                    'surface_ocean': self.surface_ocean.mass,
                    'deep_ocean': self.deep_ocean.mass,
                    'emissions': emissions
                })
        
        return self.history
    
    def airborne_fraction(self) -> float:
        """Calculate fraction of emissions remaining in atmosphere.
        
        Returns:
            Airborne fraction [0-1]
        """
        if not self.history:
            return 0.0
            
        initial_atm = 590.0  # Pre-industrial
        current_atm = self.atmosphere.mass
        
        total_emissions = sum(h['emissions'] for h in self.history)
        if total_emissions == 0:
            return 0.0
            
        return (current_atm - initial_atm) / total_emissions


# =============================================================================
# RADIATIVE FORCING
# =============================================================================

def co2_forcing(co2_ppm: float, reference: float = CO2_PREINDUSTRIAL) -> float:
    """Calculate radiative forcing from CO2.
    
    Formula: ΔF = 5.35 * ln(C/C₀) W/m²
    
    Args:
        co2_ppm: CO2 concentration [ppm]
        reference: Reference concentration [ppm]
    
    Returns:
        Radiative forcing [W/m²]
    """
    return 5.35 * np.log(co2_ppm / reference)


def ch4_forcing(ch4_ppb: float, reference: float = 722.0) -> float:
    """Calculate radiative forcing from CH4.
    
    Args:
        ch4_ppb: CH4 concentration [ppb]
        reference: Pre-industrial CH4 [ppb]
    
    Returns:
        Radiative forcing [W/m²]
    """
    return 0.036 * (np.sqrt(ch4_ppb) - np.sqrt(reference))


def n2o_forcing(n2o_ppb: float, reference: float = 270.0) -> float:
    """Calculate radiative forcing from N2O.
    
    Args:
        n2o_ppb: N2O concentration [ppb]
        reference: Pre-industrial N2O [ppb]
    
    Returns:
        Radiative forcing [W/m²]
    """
    return 0.12 * (np.sqrt(n2o_ppb) - np.sqrt(reference))


def total_anthropogenic_forcing(
    co2: float = CO2_CURRENT,
    ch4: float = 1900.0,
    n2o: float = 332.0,
    aerosol: float = -1.1,
    other: float = 0.1
) -> Tuple[float, Dict[str, float]]:
    """Calculate total anthropogenic radiative forcing.
    
    Args:
        co2: CO2 concentration [ppm]
        ch4: CH4 concentration [ppb]
        n2o: N2O concentration [ppb]
        aerosol: Aerosol forcing [W/m²] (negative, cooling)
        other: Other forcings [W/m²]
    
    Returns:
        Tuple of (total forcing, breakdown by component)
    """
    components = {
        'CO2': co2_forcing(co2),
        'CH4': ch4_forcing(ch4),
        'N2O': n2o_forcing(n2o),
        'Aerosol': aerosol,
        'Other': other
    }
    
    total = sum(components.values())
    return total, components


# =============================================================================
# ICE SHEET DYNAMICS
# =============================================================================

@dataclass
class IceSheet:
    """Ice sheet model with mass balance.
    
    Attributes:
        name: Ice sheet name
        volume: Current volume [m³]
        area: Surface area [m²]
        accumulation_rate: Snowfall rate [m/year ice equivalent]
        equilibrium_line: Equilibrium line altitude [m]
    """
    name: str
    volume: float
    area: float
    accumulation_rate: float = 0.3
    equilibrium_line: float = 1500.0
    
    def mass_gt(self) -> float:
        """Return mass in gigatons.
        
        Returns:
            Ice mass [Gt]
        """
        return self.volume * RHO_ICE / 1e12
    
    def sea_level_equivalent(self) -> float:
        """Calculate sea level rise if completely melted.
        
        Returns:
            Sea level equivalent [m]
        """
        # Ice on land only (exclude floating ice)
        water_volume = self.volume * RHO_ICE / 1000  # m³ water
        ocean_area = 3.61e14  # m²
        return water_volume / ocean_area


class IceSheetModel:
    """Simple ice sheet mass balance model.
    
    Mass balance: dM/dt = Accumulation - (Surface melt + Calving + Basal melt)
    
    Temperature sensitivity ~7.1 mm/year per °C for Greenland.
    """
    
    def __init__(self, ice_sheet: IceSheet):
        """Initialize ice sheet model.
        
        Args:
            ice_sheet: Initial ice sheet state
        """
        self.ice = ice_sheet
        self.history: List[Dict] = []
        
        # Sensitivity parameters (Greenland-like)
        self.melt_sensitivity = 200e9  # m³/year per °C
        self.calving_rate = 50e9  # m³/year base rate
        
    def mass_balance(self, temperature_anomaly: float) -> float:
        """Calculate annual mass balance.
        
        Args:
            temperature_anomaly: Temperature above pre-industrial [K]
        
        Returns:
            Mass balance [m³/year]
        """
        # Accumulation (slightly increases with warming due to more moisture)
        accumulation = self.ice.accumulation_rate * self.ice.area
        accumulation *= (1 + 0.05 * temperature_anomaly)
        
        # Surface melt (increases strongly with warming)
        surface_melt = self.melt_sensitivity * max(0, temperature_anomaly)
        
        # Dynamic response (calving increases with warming)
        calving = self.calving_rate * (1 + 0.1 * temperature_anomaly)
        
        return accumulation - surface_melt - calving
    
    def step(self, dt: float, temperature_anomaly: float) -> float:
        """Advance model one timestep.
        
        Args:
            dt: Timestep [years]
            temperature_anomaly: Temperature anomaly [K]
        
        Returns:
            Sea level contribution [m]
        """
        mb = self.mass_balance(temperature_anomaly)
        dV = mb * dt
        
        old_volume = self.ice.volume
        self.ice.volume = max(0, self.ice.volume + dV)
        
        # Sea level contribution
        dV_actual = old_volume - self.ice.volume
        water_volume = dV_actual * RHO_ICE / 1000
        ocean_area = 3.61e14
        
        return water_volume / ocean_area
    
    def run_projection(
        self,
        years: int,
        temperature_scenario: Callable[[float], float],
        dt: float = 1.0
    ) -> List[Dict]:
        """Run projection with temperature scenario.
        
        Args:
            years: Simulation duration
            temperature_scenario: Function returning T anomaly for year
            dt: Timestep [years]
        
        Returns:
            Projection history
        """
        self.history = []
        cumulative_sl = 0.0
        
        for year in range(int(years / dt)):
            t = year * dt
            T_anom = temperature_scenario(t)
            sl_contrib = self.step(dt, T_anom)
            cumulative_sl += sl_contrib
            
            self.history.append({
                'year': t,
                'volume_m3': self.ice.volume,
                'mass_gt': self.ice.mass_gt(),
                'temperature_anomaly': T_anom,
                'sea_level_contribution_m': cumulative_sl
            })
        
        return self.history


# =============================================================================
# OCEAN CIRCULATION
# =============================================================================

class ThermohalineCirculation:
    """Simplified thermohaline circulation (AMOC) model.
    
    Two-box model representing:
    - North Atlantic surface
    - Deep ocean
    
    Circulation driven by density differences (temperature & salinity).
    """
    
    def __init__(self):
        """Initialize THC model."""
        # North Atlantic surface box
        self.T_surface = 10.0  # °C
        self.S_surface = 35.0  # PSU
        
        # Deep ocean box
        self.T_deep = 2.0  # °C
        self.S_deep = 34.9  # PSU
        
        # AMOC strength [Sv] (1 Sv = 10⁶ m³/s)
        self.psi = 18.0  # Sverdrups
        
        # Parameters
        self.k_T = 1.0e-4  # Temperature relaxation [/year]
        self.k_S = 1.0e-5  # Salinity relaxation [/year]
        self.alpha = 1.5e-4  # Thermal expansion [/°C]
        self.beta = 8.0e-4  # Haline contraction [/PSU]
        
        self.history: List[Dict] = []
        
    def density_difference(self) -> float:
        """Calculate density difference driving circulation.
        
        Returns:
            Density difference [kg/m³]
        """
        # ρ = ρ₀(1 - α(T-T₀) + β(S-S₀))
        rho_0 = 1025.0  # Reference density
        
        delta_rho = rho_0 * (
            -self.alpha * (self.T_surface - self.T_deep) +
            self.beta * (self.S_surface - self.S_deep)
        )
        return delta_rho
    
    def amoc_strength(self) -> float:
        """Calculate AMOC strength from density difference.
        
        Returns:
            AMOC strength [Sv]
        """
        delta_rho = self.density_difference()
        
        # Stommel-type circulation strength
        k_psi = 3.0  # Sv per kg/m³
        psi = k_psi * max(0, delta_rho)
        
        return min(psi, 25.0)  # Cap at reasonable maximum
    
    def step(
        self,
        dt: float,
        freshwater_forcing: float = 0.0,
        heat_forcing: float = 0.0
    ) -> None:
        """Advance model one timestep.
        
        Args:
            dt: Timestep [years]
            freshwater_forcing: Freshwater input [Sv] (positive = fresher)
            heat_forcing: Surface heat flux anomaly [W/m²]
        """
        # Update AMOC strength
        self.psi = self.amoc_strength()
        
        # Temperature evolution
        # Surface box gains heat, deep box gains from overturning
        self.T_surface += dt * (heat_forcing / (C_P_OCEAN * RHO_OCEAN * 100) +
                                self.psi * (self.T_deep - self.T_surface) * 1e-3)
        
        # Salinity evolution
        # Freshwater forcing reduces surface salinity
        self.S_surface += dt * (-freshwater_forcing * 0.1 +
                                self.psi * (self.S_deep - self.S_surface) * 1e-3)
        
    def run_freshwater_hosing(
        self,
        years: int,
        hosing_rate: float = 0.1,  # Sv
        dt: float = 0.1
    ) -> List[Dict]:
        """Run freshwater hosing experiment.
        
        Simulates ice sheet melt adding freshwater to North Atlantic.
        
        Args:
            years: Duration [years]
            hosing_rate: Freshwater input [Sv]
            dt: Timestep [years]
        
        Returns:
            History of AMOC state
        """
        self.history = []
        
        for i in range(int(years / dt)):
            t = i * dt
            self.step(dt, freshwater_forcing=hosing_rate)
            
            if i % int(1/dt) == 0:
                self.history.append({
                    'year': t,
                    'amoc_sv': self.psi,
                    'T_surface': self.T_surface,
                    'S_surface': self.S_surface,
                    'density_diff': self.density_difference()
                })
        
        return self.history


# =============================================================================
# CLIMATE SCENARIOS
# =============================================================================

class ClimateScenario:
    """Climate projection scenario (SSP-based).
    
    Scenarios:
    - SSP1-1.9: Very low emissions, 1.5°C target
    - SSP1-2.6: Low emissions, 2°C target  
    - SSP2-4.5: Intermediate emissions
    - SSP3-7.0: High emissions
    - SSP5-8.5: Very high emissions
    """
    
    # Scenario parameters: (peak year, peak emissions PgC/y, 2100 forcing W/m²)
    SCENARIOS = {
        'SSP1-1.9': (2020, 10, 1.9),
        'SSP1-2.6': (2025, 11, 2.6),
        'SSP2-4.5': (2040, 12, 4.5),
        'SSP3-7.0': (2070, 15, 7.0),
        'SSP5-8.5': (2080, 20, 8.5),
    }
    
    def __init__(self, scenario: str = 'SSP2-4.5'):
        """Initialize scenario.
        
        Args:
            scenario: Scenario name
        """
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        self.name = scenario
        self.peak_year, self.peak_emissions, self.forcing_2100 = self.SCENARIOS[scenario]
        
    def emissions(self, year: float) -> float:
        """Get emissions for given year.
        
        Args:
            year: Calendar year (relative to 2020)
        
        Returns:
            Emissions [PgC/year]
        """
        # Current (2020) emissions ~10 PgC/year
        E_2020 = 10.0
        
        if year <= 0:
            return E_2020
        
        # Sigmoid approach to peak, then decline
        peak_year_rel = self.peak_year - 2020
        
        if year < peak_year_rel:
            # Rising phase
            return E_2020 + (self.peak_emissions - E_2020) * (year / peak_year_rel)
        else:
            # Declining phase
            decay_rate = 0.02  # Approximate
            if 'SSP5' in self.name:
                decay_rate = 0.01  # Slower decline
            elif 'SSP1' in self.name:
                decay_rate = 0.05  # Faster decline
            
            return self.peak_emissions * np.exp(-decay_rate * (year - peak_year_rel))
    
    def temperature_projection(self, year: float) -> float:
        """Get projected temperature anomaly.
        
        Based on IPCC AR6 assessed warming levels.
        
        Args:
            year: Calendar year (relative to 2020)
        
        Returns:
            Temperature anomaly above pre-industrial [K]
        """
        # Current warming ~1.2K
        T_2020 = 1.2
        
        # 2100 warming by scenario (central estimates)
        T_2100 = {
            'SSP1-1.9': 1.4,
            'SSP1-2.6': 1.8,
            'SSP2-4.5': 2.7,
            'SSP3-7.0': 3.6,
            'SSP5-8.5': 4.4,
        }[self.name]
        
        if year <= 0:
            return T_2020
        if year >= 80:  # 2100
            return T_2100
        
        # Smooth interpolation
        return T_2020 + (T_2100 - T_2020) * (year / 80) ** 0.7


# =============================================================================
# INTEGRATED CLIMATE MODEL
# =============================================================================

class SimpleClimateModel:
    """Integrated simple climate model.
    
    Combines:
    - Energy balance
    - Carbon cycle
    - Ice sheet contribution
    - Sea level rise
    
    For quick scenario analysis and education.
    """
    
    def __init__(self, scenario: str = 'SSP2-4.5'):
        """Initialize integrated model.
        
        Args:
            scenario: Emission scenario
        """
        self.scenario = ClimateScenario(scenario)
        self.ebm = ZeroDimensionalEBM()
        self.carbon = CarbonCycleModel()
        
        # Initialize Greenland ice sheet
        greenland = IceSheet(
            name="Greenland",
            volume=GREENLAND_VOLUME,
            area=1.7e12  # m²
        )
        self.greenland = IceSheetModel(greenland)
        
        # State
        self.year = 0.0
        self.temperature = 1.2  # Current warming
        self.sea_level = 0.0
        
        self.history: List[ClimateState] = []
        
    def step(self, dt: float = 1.0) -> ClimateState:
        """Advance model one year.
        
        Args:
            dt: Timestep [years]
        
        Returns:
            Updated climate state
        """
        # Get emissions
        emissions = self.scenario.emissions(self.year)
        
        # Update carbon cycle
        self.carbon.step(dt, emissions)
        
        # Calculate forcing and temperature
        forcing = co2_forcing(self.carbon.co2_ppm())
        sensitivity = 3.0 / 3.7  # K per W/m² (ECS = 3K per doubling)
        
        # Simple temperature response with inertia
        T_eq = 1.2 + forcing * sensitivity
        tau = 10.0  # Response timescale [years]
        self.temperature += (T_eq - self.temperature) * dt / tau
        
        # Ice sheet and sea level
        sl_ice = self.greenland.step(dt, self.temperature)
        sl_thermal = 0.002 * self.temperature * dt  # Thermal expansion ~2mm/year per K
        self.sea_level += sl_ice + sl_thermal
        
        self.year += dt
        
        # Record state
        state = ClimateState(
            year=self.year + 2020,
            temperature=self.temperature,
            co2=self.carbon.co2_ppm(),
            sea_level=self.sea_level,
            ice_mass_greenland=-(GREENLAND_VOLUME - self.greenland.ice.volume) * RHO_ICE / 1e12
        )
        self.history.append(state)
        
        return state
    
    def run_projection(self, years: int = 80) -> List[ClimateState]:
        """Run projection to 2100.
        
        Args:
            years: Projection duration [years]
        
        Returns:
            List of climate states
        """
        self.history = []
        
        for _ in range(years):
            self.step()
        
        return self.history
    
    def summary(self) -> Dict:
        """Generate projection summary.
        
        Returns:
            Summary statistics
        """
        if not self.history:
            self.run_projection()
        
        final = self.history[-1]
        
        return {
            'scenario': self.scenario.name,
            'year_2100': {
                'temperature_anomaly_K': round(final.temperature, 2),
                'co2_ppm': round(final.co2, 1),
                'sea_level_rise_m': round(final.sea_level, 3),
                'greenland_mass_loss_Gt': round(-final.ice_mass_greenland, 0)
            },
            'peak_warming_year': max(self.history, key=lambda s: s.temperature).year,
            'cumulative_emissions_PgC': round(sum(
                self.scenario.emissions(y) for y in range(80)
            ), 1)
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def estimate_remaining_carbon_budget(
    target_warming: float = 1.5,
    current_warming: float = 1.2,
    tcre: float = 0.45
) -> float:
    """Estimate remaining carbon budget for temperature target.
    
    TCRE: Transient Climate Response to Cumulative Emissions
    ~0.45°C per 1000 PgC (IPCC AR6 central estimate)
    
    Args:
        target_warming: Target warming [K]
        current_warming: Current warming [K]
        tcre: Climate sensitivity [K per 1000 PgC]
    
    Returns:
        Remaining carbon budget [PgC]
    """
    remaining_warming = target_warming - current_warming
    return remaining_warming / tcre * 1000


def warming_from_emissions(
    cumulative_emissions: float,
    tcre: float = 0.45
) -> float:
    """Calculate warming from cumulative emissions.
    
    Args:
        cumulative_emissions: Total emissions [PgC]
        tcre: Climate sensitivity [K per 1000 PgC]
    
    Returns:
        Warming [K]
    """
    return cumulative_emissions * tcre / 1000


def years_to_budget_exhaustion(
    budget: float,
    current_emissions: float = 10.0,
    decline_rate: float = 0.0
) -> float:
    """Calculate years until carbon budget exhausted.
    
    Args:
        budget: Remaining budget [PgC]
        current_emissions: Current annual emissions [PgC/year]
        decline_rate: Annual decline rate [fraction]
    
    Returns:
        Years until exhaustion
    """
    if decline_rate == 0:
        return budget / current_emissions
    
    # Integral of E₀ * exp(-r*t) from 0 to T = E₀/r * (1 - exp(-rT))
    # Solving for T when integral = budget
    # T = -ln(1 - budget * r / E₀) / r
    
    ratio = budget * decline_rate / current_emissions
    if ratio >= 1:
        return float('inf')  # Budget never exhausted
    
    return -np.log(1 - ratio) / decline_rate


# =============================================================================
# TIPPING POINTS
# =============================================================================

@dataclass
class TippingElement:
    """Climate tipping element.
    
    Attributes:
        name: Tipping element name
        threshold_low: Lower bound of threshold [K]
        threshold_high: Upper bound of threshold [K]
        timescale: Transition timescale [years]
        impact: Global impact description
    """
    name: str
    threshold_low: float
    threshold_high: float
    timescale: Tuple[float, float]  # (min, max) years
    impact: str


# Major tipping elements (IPCC AR6, Armstrong McKay et al. 2022)
TIPPING_ELEMENTS = [
    TippingElement(
        "Greenland Ice Sheet",
        threshold_low=0.8,
        threshold_high=3.0,
        timescale=(1000, 15000),
        impact="~7m sea level rise"
    ),
    TippingElement(
        "West Antarctic Ice Sheet",
        threshold_low=1.0,
        threshold_high=3.0,
        timescale=(500, 13000),
        impact="~3m sea level rise"
    ),
    TippingElement(
        "Amazon Rainforest Dieback",
        threshold_low=2.0,
        threshold_high=6.0,
        timescale=(50, 200),
        impact="~30 PgC release, biodiversity loss"
    ),
    TippingElement(
        "Atlantic Circulation Collapse",
        threshold_low=1.4,
        threshold_high=8.0,
        timescale=(15, 300),
        impact="European cooling, shifted monsoons"
    ),
    TippingElement(
        "Permafrost Collapse",
        threshold_low=1.0,
        threshold_high=2.3,
        timescale=(10, 300),
        impact="~60-100 PgC release"
    ),
    TippingElement(
        "Coral Reef Die-off",
        threshold_low=1.0,
        threshold_high=1.5,
        timescale=(10, 50),
        impact="Ecosystem collapse, fisheries loss"
    ),
]


def assess_tipping_risk(warming: float) -> List[Dict]:
    """Assess tipping point risks for given warming level.
    
    Args:
        warming: Global warming above pre-industrial [K]
    
    Returns:
        List of tipping elements and their risk status
    """
    results = []
    
    for element in TIPPING_ELEMENTS:
        if warming < element.threshold_low:
            risk = "Low"
            probability = 0.1
        elif warming < element.threshold_high:
            # Linear interpolation
            frac = (warming - element.threshold_low) / (element.threshold_high - element.threshold_low)
            risk = "Moderate" if frac < 0.5 else "High"
            probability = 0.1 + 0.8 * frac
        else:
            risk = "Very High"
            probability = 0.9
        
        results.append({
            'name': element.name,
            'risk_level': risk,
            'probability': round(probability, 2),
            'threshold_range': f"{element.threshold_low}-{element.threshold_high}°C",
            'timescale': f"{element.timescale[0]}-{element.timescale[1]} years",
            'impact': element.impact
        })
    
    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'S_0', 'SIGMA_SB', 'R_EARTH', 'CO2_PREINDUSTRIAL', 'CO2_CURRENT',
    
    # Enums
    'ForcingType', 'FeedbackType', 'OceanBasin',
    
    # Data classes
    'RadiativeForcing', 'ClimateState', 'CarbonReservoir', 'OceanLayer',
    'IceSheet', 'TippingElement',
    
    # Models
    'ZeroDimensionalEBM', 'OneDimensionalEBM',
    'CarbonCycleModel',
    'IceSheetModel',
    'ThermohalineCirculation',
    'ClimateScenario', 'SimpleClimateModel',
    
    # Functions
    'co2_forcing', 'ch4_forcing', 'n2o_forcing', 'total_anthropogenic_forcing',
    'estimate_remaining_carbon_budget', 'warming_from_emissions',
    'years_to_budget_exhaustion', 'assess_tipping_risk',
    
    # Data
    'TIPPING_ELEMENTS',
]
