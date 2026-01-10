"""
QENEX Climate Science Package
=============================
Comprehensive climate modeling covering energy balance, carbon cycle,
ice sheets, ocean circulation, and climate projections.
"""

from .climate import (
    # Constants
    S_0, SIGMA_SB, R_EARTH, CO2_PREINDUSTRIAL, CO2_CURRENT,
    
    # Enums
    ForcingType, FeedbackType, OceanBasin,
    
    # Data classes
    RadiativeForcing, ClimateState, CarbonReservoir, OceanLayer,
    IceSheet, TippingElement,
    
    # Models
    ZeroDimensionalEBM, OneDimensionalEBM,
    CarbonCycleModel,
    IceSheetModel,
    ThermohalineCirculation,
    ClimateScenario, SimpleClimateModel,
    
    # Functions
    co2_forcing, ch4_forcing, n2o_forcing, total_anthropogenic_forcing,
    estimate_remaining_carbon_budget, warming_from_emissions,
    years_to_budget_exhaustion, assess_tipping_risk,
    
    # Data
    TIPPING_ELEMENTS,
)

__version__ = "0.1.0"
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
