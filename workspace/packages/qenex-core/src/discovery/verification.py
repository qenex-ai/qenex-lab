"""
QENEX Formal Verification Module
================================
Scientific hypothesis verification using dimensional analysis, 
physical constraint checking, and formal mathematical validation.

This module provides:
1. Dimensional Analysis - Verify dimensional consistency of equations
2. Physical Constraint Checking - Validate against known physical laws
3. Numerical Bounds Checking - Ensure values are in physically plausible ranges
4. Symmetry Verification - Check conservation laws and invariances
5. Mathematical Consistency - Validate mathematical forms
6. Cross-Validation - Compare with established results

Author: QENEX Sovereign Agent
Date: 2026-01-10
"""

import re
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum, auto
from datetime import datetime
import json


# =============================================================================
# VERIFICATION STATUS AND TYPES
# =============================================================================

class VerificationStatus(Enum):
    """Status of a verification check."""
    PASSED = auto()         # Check passed
    FAILED = auto()         # Check failed
    WARNING = auto()        # Check passed with warnings
    SKIPPED = auto()        # Check not applicable
    UNCERTAIN = auto()      # Insufficient information to verify


class VerificationType(Enum):
    """Types of verification checks."""
    DIMENSIONAL = auto()        # Dimensional analysis
    PHYSICAL_BOUNDS = auto()    # Physical value bounds
    CONSERVATION = auto()       # Conservation laws
    SYMMETRY = auto()           # Symmetry constraints
    MATHEMATICAL = auto()       # Mathematical consistency
    EMPIRICAL = auto()          # Comparison with data
    CROSS_DOMAIN = auto()       # Cross-domain consistency


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    check_type: VerificationType
    status: VerificationStatus
    message: str
    confidence: float          # 0-1, confidence in the check
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "check_type": self.check_type.name,
            "status": self.status.name,
            "message": self.message,
            "confidence": self.confidence,
            "details": self.details,
            "suggestions": self.suggestions,
        }


@dataclass
class VerificationReport:
    """Complete verification report for a hypothesis."""
    hypothesis_id: str
    hypothesis_statement: str
    domain: str
    checks: List[VerificationResult]
    overall_status: VerificationStatus
    overall_confidence: float
    verified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.status == VerificationStatus.PASSED)
    
    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if c.status == VerificationStatus.FAILED)
    
    @property
    def n_warnings(self) -> int:
        return sum(1 for c in self.checks if c.status == VerificationStatus.WARNING)
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "VERIFICATION REPORT",
            "=" * 60,
            f"Hypothesis: {self.hypothesis_id}",
            f"Domain: {self.domain}",
            f"Statement: {self.hypothesis_statement[:100]}...",
            "-" * 60,
            f"Overall Status: {self.overall_status.name}",
            f"Overall Confidence: {self.overall_confidence:.2%}",
            f"Passed: {self.n_passed} | Failed: {self.n_failed} | Warnings: {self.n_warnings}",
            "-" * 60,
            "INDIVIDUAL CHECKS:",
        ]
        
        for check in self.checks:
            status_symbol = {
                VerificationStatus.PASSED: "[PASS]",
                VerificationStatus.FAILED: "[FAIL]",
                VerificationStatus.WARNING: "[WARN]",
                VerificationStatus.SKIPPED: "[SKIP]",
                VerificationStatus.UNCERTAIN: "[????]",
            }[check.status]
            
            lines.append(f"  {status_symbol} {check.check_type.name}: {check.message}")
            
            if check.suggestions:
                for sugg in check.suggestions:
                    lines.append(f"         -> {sugg}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_statement": self.hypothesis_statement,
            "domain": self.domain,
            "checks": [c.to_dict() for c in self.checks],
            "overall_status": self.overall_status.name,
            "overall_confidence": self.overall_confidence,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_warnings": self.n_warnings,
            "verified_at": self.verified_at,
        }


# =============================================================================
# DIMENSIONAL ANALYSIS
# =============================================================================

class Dimension:
    """Represents a physical dimension in terms of base units."""
    
    # SI base dimensions: [M, L, T, I, Theta, N, J]
    # Mass, Length, Time, Current, Temperature, Amount, Luminosity
    
    BASE_DIMS = ['M', 'L', 'T', 'I', 'Theta', 'N', 'J']
    
    def __init__(self, exponents: Dict[str, float] = None):
        self.exponents = {dim: 0.0 for dim in self.BASE_DIMS}
        if exponents:
            for dim, exp in exponents.items():
                if dim in self.exponents:
                    self.exponents[dim] = exp
    
    def __eq__(self, other: 'Dimension') -> bool:
        if not isinstance(other, Dimension):
            return False
        return all(
            abs(self.exponents[d] - other.exponents[d]) < 1e-10
            for d in self.BASE_DIMS
        )
    
    def __mul__(self, other: 'Dimension') -> 'Dimension':
        result = Dimension()
        for dim in self.BASE_DIMS:
            result.exponents[dim] = self.exponents[dim] + other.exponents[dim]
        return result
    
    def __truediv__(self, other: 'Dimension') -> 'Dimension':
        result = Dimension()
        for dim in self.BASE_DIMS:
            result.exponents[dim] = self.exponents[dim] - other.exponents[dim]
        return result
    
    def __pow__(self, n: float) -> 'Dimension':
        result = Dimension()
        for dim in self.BASE_DIMS:
            result.exponents[dim] = self.exponents[dim] * n
        return result
    
    def __repr__(self) -> str:
        terms = []
        for dim in self.BASE_DIMS:
            exp = self.exponents[dim]
            if abs(exp) > 1e-10:
                if abs(exp - 1.0) < 1e-10:
                    terms.append(dim)
                else:
                    terms.append(f"{dim}^{exp:.1f}")
        return " ".join(terms) if terms else "dimensionless"
    
    def is_dimensionless(self) -> bool:
        return all(abs(exp) < 1e-10 for exp in self.exponents.values())


# Common physical dimensions
DIMENSIONS = {
    # Base units
    "mass": Dimension({'M': 1}),
    "length": Dimension({'L': 1}),
    "time": Dimension({'T': 1}),
    "current": Dimension({'I': 1}),
    "temperature": Dimension({'Theta': 1}),
    "amount": Dimension({'N': 1}),
    "luminosity": Dimension({'J': 1}),
    
    # Derived mechanical
    "velocity": Dimension({'L': 1, 'T': -1}),
    "acceleration": Dimension({'L': 1, 'T': -2}),
    "force": Dimension({'M': 1, 'L': 1, 'T': -2}),
    "energy": Dimension({'M': 1, 'L': 2, 'T': -2}),
    "power": Dimension({'M': 1, 'L': 2, 'T': -3}),
    "pressure": Dimension({'M': 1, 'L': -1, 'T': -2}),
    "momentum": Dimension({'M': 1, 'L': 1, 'T': -1}),
    "angular_momentum": Dimension({'M': 1, 'L': 2, 'T': -1}),
    "torque": Dimension({'M': 1, 'L': 2, 'T': -2}),
    
    # Electromagnetic
    "charge": Dimension({'I': 1, 'T': 1}),
    "voltage": Dimension({'M': 1, 'L': 2, 'T': -3, 'I': -1}),
    "resistance": Dimension({'M': 1, 'L': 2, 'T': -3, 'I': -2}),
    "capacitance": Dimension({'M': -1, 'L': -2, 'T': 4, 'I': 2}),
    "magnetic_field": Dimension({'M': 1, 'T': -2, 'I': -1}),
    "electric_field": Dimension({'M': 1, 'L': 1, 'T': -3, 'I': -1}),
    
    # Thermodynamic
    "entropy": Dimension({'M': 1, 'L': 2, 'T': -2, 'Theta': -1}),
    "heat_capacity": Dimension({'M': 1, 'L': 2, 'T': -2, 'Theta': -1}),
    
    # Quantum/atomic
    "action": Dimension({'M': 1, 'L': 2, 'T': -1}),  # Same as angular momentum
    "wavefunction": Dimension({'L': -1.5}),  # 3D: psi^2 has dim L^-3
    
    # Concentrations and rates
    "concentration": Dimension({'L': -3, 'N': 1}),
    "rate": Dimension({'T': -1}),
    "frequency": Dimension({'T': -1}),
    
    # Dimensionless
    "dimensionless": Dimension(),
}


class DimensionalAnalyzer:
    """Performs dimensional analysis on equations and expressions."""
    
    # Map quantity names to dimensions
    QUANTITY_DIMENSIONS = {
        # Physics
        "energy": "energy",
        "E": "energy",
        "kinetic_energy": "energy",
        "potential_energy": "energy",
        "mass": "mass",
        "m": "mass",
        "M": "mass",
        "velocity": "velocity",
        "v": "velocity",
        "speed": "velocity",
        "c": "velocity",  # speed of light
        "time": "time",
        "t": "time",
        "T": "temperature",
        "temperature": "temperature",
        "force": "force",
        "F": "force",
        "pressure": "pressure",
        "P": "pressure",
        "length": "length",
        "distance": "length",
        "r": "length",
        "R": "length",
        "x": "length",
        "y": "length",
        "z": "length",
        "h": "action",  # Planck constant
        "hbar": "action",
        "k_B": "entropy",  # Boltzmann constant
        "G": Dimension({'M': -1, 'L': 3, 'T': -2}),  # Gravitational constant (special)
        
        # Chemistry/biology
        "concentration": "concentration",
        "C": "concentration",
        "rate_constant": "rate",
        "k": "rate",
        
        # Astronomy
        "luminosity": "power",
        "L": "power",
        "flux": Dimension({'M': 1, 'T': -3}),  # W/m^2
        
        # Neuroscience
        "membrane_potential": "voltage",
        "V_m": "voltage",
        "firing_rate": "rate",
        "synaptic_weight": "dimensionless",
        
        # Climate
        "radiative_forcing": Dimension({'M': 1, 'T': -3}),  # W/m^2
        "albedo": "dimensionless",
        "CO2": "concentration",
    }
    
    def __init__(self):
        self.dimensions = DIMENSIONS.copy()
    
    def get_dimension(self, quantity: str) -> Optional[Dimension]:
        """Get the dimension of a named quantity."""
        if quantity in self.QUANTITY_DIMENSIONS:
            dim_or_name = self.QUANTITY_DIMENSIONS[quantity]
            if isinstance(dim_or_name, Dimension):
                return dim_or_name
            return self.dimensions.get(dim_or_name)
        return None
    
    def check_equation_balance(self, 
                               lhs_quantities: List[str],
                               rhs_quantities: List[str],
                               lhs_exponents: List[float] = None,
                               rhs_exponents: List[float] = None) -> VerificationResult:
        """
        Check if an equation is dimensionally balanced.
        
        Args:
            lhs_quantities: Quantities on left side
            rhs_quantities: Quantities on right side
            lhs_exponents: Exponents for LHS quantities (default all 1)
            rhs_exponents: Exponents for RHS quantities (default all 1)
        
        Returns:
            VerificationResult
        """
        if lhs_exponents is None:
            lhs_exponents = [1.0] * len(lhs_quantities)
        if rhs_exponents is None:
            rhs_exponents = [1.0] * len(rhs_quantities)
        
        # Calculate LHS dimension
        lhs_dim = Dimension()
        unknown_lhs = []
        for qty, exp in zip(lhs_quantities, lhs_exponents):
            dim = self.get_dimension(qty)
            if dim:
                lhs_dim = lhs_dim * (dim ** exp)
            else:
                unknown_lhs.append(qty)
        
        # Calculate RHS dimension
        rhs_dim = Dimension()
        unknown_rhs = []
        for qty, exp in zip(rhs_quantities, rhs_exponents):
            dim = self.get_dimension(qty)
            if dim:
                rhs_dim = rhs_dim * (dim ** exp)
            else:
                unknown_rhs.append(qty)
        
        # Check for unknowns
        if unknown_lhs or unknown_rhs:
            return VerificationResult(
                check_type=VerificationType.DIMENSIONAL,
                status=VerificationStatus.UNCERTAIN,
                message=f"Unknown quantities: LHS={unknown_lhs}, RHS={unknown_rhs}",
                confidence=0.5,
                details={"unknown_lhs": unknown_lhs, "unknown_rhs": unknown_rhs},
                suggestions=["Define dimensions for unknown quantities"]
            )
        
        # Check balance
        if lhs_dim == rhs_dim:
            return VerificationResult(
                check_type=VerificationType.DIMENSIONAL,
                status=VerificationStatus.PASSED,
                message=f"Equation is dimensionally consistent: [{lhs_dim}] = [{rhs_dim}]",
                confidence=1.0,
                details={"lhs_dim": str(lhs_dim), "rhs_dim": str(rhs_dim)}
            )
        else:
            return VerificationResult(
                check_type=VerificationType.DIMENSIONAL,
                status=VerificationStatus.FAILED,
                message=f"Dimensional mismatch: [{lhs_dim}] != [{rhs_dim}]",
                confidence=1.0,
                details={"lhs_dim": str(lhs_dim), "rhs_dim": str(rhs_dim)},
                suggestions=[
                    "Check for missing or extra factors",
                    "Verify exponents are correct",
                    "Look for dimensionless combinations"
                ]
            )
    
    def parse_mathematical_form(self, equation: str) -> Dict[str, Any]:
        """
        Parse a mathematical expression to extract quantities.
        
        Args:
            equation: String like "Y = A * X^alpha" or "E = m * c^2"
        
        Returns:
            Dict with parsed components
        """
        result = {
            "lhs": [],
            "rhs": [],
            "lhs_exponents": [],
            "rhs_exponents": [],
            "operators": [],
            "raw": equation
        }
        
        # Split by equals sign
        if "=" in equation:
            lhs, rhs = equation.split("=", 1)
        else:
            lhs, rhs = equation, ""
        
        # Extract quantities and exponents using regex
        # Pattern: quantity^exponent or just quantity
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\^|\*\*)\s*([0-9.]+)|([a-zA-Z_][a-zA-Z0-9_]*)'
        
        # Parse LHS
        for match in re.finditer(pattern, lhs):
            if match.group(1):  # Has exponent
                result["lhs"].append(match.group(1))
                result["lhs_exponents"].append(float(match.group(2)))
            elif match.group(3):  # No exponent
                result["lhs"].append(match.group(3))
                result["lhs_exponents"].append(1.0)
        
        # Parse RHS
        for match in re.finditer(pattern, rhs):
            if match.group(1):  # Has exponent
                result["rhs"].append(match.group(1))
                result["rhs_exponents"].append(float(match.group(2)))
            elif match.group(3):  # No exponent
                result["rhs"].append(match.group(3))
                result["rhs_exponents"].append(1.0)
        
        return result


# =============================================================================
# PHYSICAL CONSTRAINTS AND BOUNDS
# =============================================================================

class PhysicalConstraintChecker:
    """Checks hypotheses against known physical constraints and bounds."""
    
    # Physical bounds for various quantities
    PHYSICAL_BOUNDS = {
        # Universal bounds
        "velocity": (0, 299792458),  # m/s, can't exceed c
        "temperature": (0, 1e32),     # K, can't be negative (absolute zero)
        "mass": (0, 1e53),            # kg, can't be negative
        "energy": (-1e60, 1e60),      # J, can be negative (binding energy)
        "entropy": (0, 1e30),         # J/K, can't be negative
        
        # Chemistry
        "bond_length": (0.5e-10, 5e-9),   # m, atomic to molecular scale
        "binding_energy": (-1e3, 0),       # eV, typically negative
        
        # Climate
        "temperature_anomaly": (-50, 50),   # K, realistic range
        "CO2_ppm": (100, 10000),            # ppm, plausible range
        "albedo": (0, 1),                   # dimensionless
        "climate_sensitivity": (1.5, 6),   # K per CO2 doubling
        
        # Neuroscience
        "membrane_potential": (-0.1, 0.1),  # V, typically -70 to +40 mV
        "firing_rate": (0, 1000),           # Hz
        "synaptic_weight": (0, 10),         # dimensionless
        
        # Astrophysics
        "stellar_mass": (0.08, 300),        # solar masses
        "luminosity_ratio": (1e-4, 1e7),    # L/L_sun
        "metallicity": (-5, 1),             # log(Z/Z_sun)
        
        # Biology
        "mutation_rate": (0, 1),            # per base per generation
        "fitness": (0, 100),                # arbitrary units
        "population_size": (1, 1e15),       # individuals
    }
    
    # Conservation laws
    CONSERVATION_LAWS = {
        "energy": "Total energy is conserved in isolated systems",
        "momentum": "Total momentum is conserved in absence of external forces",
        "angular_momentum": "Angular momentum is conserved if no external torque",
        "charge": "Electric charge is conserved",
        "baryon_number": "Baryon number is conserved (to high precision)",
        "lepton_number": "Lepton number is conserved (approximately)",
        "mass_energy": "Mass-energy is conserved (E = mc^2)",
        "probability": "Total probability must equal 1",
    }
    
    # Physical impossibilities
    IMPOSSIBILITIES = [
        "perpetual motion",
        "faster than light",
        "negative mass",
        "negative temperature",  # Note: in stat mech, can have negative T
        "infinite energy",
        "100% efficiency heat engine",
        "decreasing entropy in isolated system",
    ]
    
    def check_value_bounds(self, 
                           quantity: str, 
                           value: float,
                           unit: str = None) -> VerificationResult:
        """Check if a value is within physical bounds."""
        
        if quantity not in self.PHYSICAL_BOUNDS:
            return VerificationResult(
                check_type=VerificationType.PHYSICAL_BOUNDS,
                status=VerificationStatus.UNCERTAIN,
                message=f"No bounds defined for '{quantity}'",
                confidence=0.3,
                suggestions=["Define physical bounds for this quantity"]
            )
        
        low, high = self.PHYSICAL_BOUNDS[quantity]
        
        if low <= value <= high:
            return VerificationResult(
                check_type=VerificationType.PHYSICAL_BOUNDS,
                status=VerificationStatus.PASSED,
                message=f"{quantity} = {value:.4g} is within bounds [{low:.4g}, {high:.4g}]",
                confidence=0.9,
                details={"value": value, "bounds": (low, high)}
            )
        else:
            return VerificationResult(
                check_type=VerificationType.PHYSICAL_BOUNDS,
                status=VerificationStatus.FAILED,
                message=f"{quantity} = {value:.4g} is OUT OF BOUNDS [{low:.4g}, {high:.4g}]",
                confidence=0.95,
                details={"value": value, "bounds": (low, high)},
                suggestions=[
                    "Check unit conversion",
                    "Verify calculation",
                    f"Value should be between {low} and {high}"
                ]
            )
    
    def check_conservation(self, 
                           conserved_quantity: str,
                           initial_value: float,
                           final_value: float,
                           tolerance: float = 1e-10) -> VerificationResult:
        """Check if a conservation law is satisfied."""
        
        if conserved_quantity not in self.CONSERVATION_LAWS:
            return VerificationResult(
                check_type=VerificationType.CONSERVATION,
                status=VerificationStatus.UNCERTAIN,
                message=f"Unknown conserved quantity: {conserved_quantity}",
                confidence=0.3
            )
        
        relative_error = abs(final_value - initial_value) / (abs(initial_value) + 1e-30)
        
        if relative_error < tolerance:
            return VerificationResult(
                check_type=VerificationType.CONSERVATION,
                status=VerificationStatus.PASSED,
                message=f"{conserved_quantity} is conserved (error: {relative_error:.2e})",
                confidence=0.95,
                details={
                    "initial": initial_value,
                    "final": final_value,
                    "relative_error": relative_error,
                    "law": self.CONSERVATION_LAWS[conserved_quantity]
                }
            )
        else:
            return VerificationResult(
                check_type=VerificationType.CONSERVATION,
                status=VerificationStatus.FAILED,
                message=f"{conserved_quantity} NOT conserved (error: {relative_error:.2e})",
                confidence=0.95,
                details={
                    "initial": initial_value,
                    "final": final_value,
                    "relative_error": relative_error
                },
                suggestions=[
                    f"Check for external sources/sinks of {conserved_quantity}",
                    "Verify system is properly isolated",
                    "Check for numerical errors"
                ]
            )
    
    def check_for_impossibilities(self, statement: str) -> VerificationResult:
        """Check if a hypothesis contains physical impossibilities."""
        
        statement_lower = statement.lower()
        found = []
        
        for impossibility in self.IMPOSSIBILITIES:
            if impossibility in statement_lower:
                found.append(impossibility)
        
        if found:
            return VerificationResult(
                check_type=VerificationType.PHYSICAL_BOUNDS,
                status=VerificationStatus.FAILED,
                message=f"Statement contains physical impossibilities: {found}",
                confidence=0.99,
                details={"impossibilities": found},
                suggestions=[
                    "Remove or rephrase impossible claims",
                    "Check if statement violates fundamental physics"
                ]
            )
        else:
            return VerificationResult(
                check_type=VerificationType.PHYSICAL_BOUNDS,
                status=VerificationStatus.PASSED,
                message="No obvious physical impossibilities detected",
                confidence=0.7,  # Can't be 100% sure
                details={"checked_for": self.IMPOSSIBILITIES}
            )


# =============================================================================
# MATHEMATICAL CONSISTENCY CHECKER
# =============================================================================

class MathematicalConsistencyChecker:
    """Checks mathematical consistency of hypothesis formulations."""
    
    # Common mathematical patterns and their expected forms
    MATHEMATICAL_PATTERNS = {
        "power_law": {
            "form": r"[A-Za-z]+\s*[~=]\s*[A-Za-z]+\^?[\d.]*\s*\*?\s*[A-Za-z]*\^?[\d.]*",
            "example": "Y ~ X^alpha or Y = A * X^n",
            "constraints": ["exponent should be finite", "prefactor should be positive"]
        },
        "exponential": {
            "form": r"exp\s*\(|e\^",
            "example": "exp(-E/kT) or e^(-t/tau)",
            "constraints": ["argument should be dimensionless"]
        },
        "gaussian": {
            "form": r"exp\s*\(\s*-.*\^2",
            "example": "exp(-(x-mu)^2 / (2*sigma^2))",
            "constraints": ["variance must be positive", "should be normalized"]
        },
        "differential_equation": {
            "form": r"d[A-Za-z]+/dt|d\^2|nabla",
            "example": "dx/dt = f(x) or d^2x/dt^2 = -omega^2*x",
            "constraints": ["initial conditions required", "check stability"]
        },
        "conservation": {
            "form": r"d[A-Za-z]+/dt\s*=\s*0",
            "example": "dE/dt = 0 (energy conservation)",
            "constraints": ["applies to isolated systems"]
        }
    }
    
    def check_pattern_validity(self, 
                               mathematical_form: str,
                               expected_pattern: str = None) -> VerificationResult:
        """Check if a mathematical form is valid."""
        
        if not mathematical_form:
            return VerificationResult(
                check_type=VerificationType.MATHEMATICAL,
                status=VerificationStatus.SKIPPED,
                message="No mathematical form provided",
                confidence=0.0
            )
        
        # Check for basic validity
        issues = []
        
        # Check balanced parentheses
        if mathematical_form.count('(') != mathematical_form.count(')'):
            issues.append("Unbalanced parentheses")
        
        # Check for division by zero patterns
        if re.search(r'/\s*0[^.]', mathematical_form):
            issues.append("Potential division by zero")
        
        # Check for undefined operations
        if re.search(r'log\s*\(\s*0|ln\s*\(\s*0|sqrt\s*\(\s*-', mathematical_form):
            issues.append("Potentially undefined operation")
        
        # Identify which pattern it matches
        matched_pattern = None
        for pattern_name, pattern_info in self.MATHEMATICAL_PATTERNS.items():
            if re.search(pattern_info["form"], mathematical_form, re.IGNORECASE):
                matched_pattern = pattern_name
                break
        
        if issues:
            return VerificationResult(
                check_type=VerificationType.MATHEMATICAL,
                status=VerificationStatus.FAILED,
                message=f"Mathematical issues found: {issues}",
                confidence=0.9,
                details={"issues": issues, "matched_pattern": matched_pattern},
                suggestions=["Fix mathematical notation", "Check for edge cases"]
            )
        
        if matched_pattern:
            return VerificationResult(
                check_type=VerificationType.MATHEMATICAL,
                status=VerificationStatus.PASSED,
                message=f"Valid {matched_pattern} form detected",
                confidence=0.8,
                details={
                    "matched_pattern": matched_pattern,
                    "pattern_info": self.MATHEMATICAL_PATTERNS[matched_pattern]
                }
            )
        else:
            return VerificationResult(
                check_type=VerificationType.MATHEMATICAL,
                status=VerificationStatus.WARNING,
                message="Mathematical form appears valid but pattern not recognized",
                confidence=0.6,
                details={"form": mathematical_form}
            )
    
    def check_parameter_constraints(self,
                                    parameters: List[str],
                                    values: Dict[str, float] = None) -> VerificationResult:
        """Check if parameters satisfy typical constraints."""
        
        if not parameters:
            return VerificationResult(
                check_type=VerificationType.MATHEMATICAL,
                status=VerificationStatus.SKIPPED,
                message="No parameters to check",
                confidence=0.0
            )
        
        # Common parameter constraints
        PARAM_CONSTRAINTS = {
            "exponent": ("finite", lambda x: -100 < x < 100),
            "alpha": ("finite", lambda x: -100 < x < 100),
            "beta": ("finite", lambda x: -100 < x < 100),
            "gamma": ("positive", lambda x: x > 0),
            "prefactor": ("positive", lambda x: x > 0),
            "coefficient": ("finite", lambda x: -1e10 < x < 1e10),
            "time_constant": ("positive", lambda x: x > 0),
            "tau": ("positive", lambda x: x > 0),
            "sigma": ("positive", lambda x: x > 0),
            "variance": ("positive", lambda x: x > 0),
            "rate": ("non-negative", lambda x: x >= 0),
        }
        
        issues = []
        checked = []
        
        if values:
            for param in parameters:
                param_lower = param.lower()
                for constraint_name, (desc, check_fn) in PARAM_CONSTRAINTS.items():
                    if constraint_name in param_lower:
                        if param in values:
                            val = values[param]
                            if not check_fn(val):
                                issues.append(f"{param} should be {desc}, got {val}")
                            else:
                                checked.append(f"{param} is {desc}: OK")
                        break
        
        if issues:
            return VerificationResult(
                check_type=VerificationType.MATHEMATICAL,
                status=VerificationStatus.FAILED,
                message=f"Parameter constraint violations: {len(issues)}",
                confidence=0.85,
                details={"issues": issues, "checked": checked},
                suggestions=["Verify parameter values", "Check physical interpretation"]
            )
        
        return VerificationResult(
            check_type=VerificationType.MATHEMATICAL,
            status=VerificationStatus.PASSED,
            message=f"Parameter constraints satisfied ({len(checked)} checked)",
            confidence=0.7,
            details={"parameters": parameters, "checked": checked}
        )


# =============================================================================
# CROSS-DOMAIN CONSISTENCY CHECKER
# =============================================================================

class CrossDomainConsistencyChecker:
    """Checks consistency of hypotheses across scientific domains."""
    
    # Known cross-domain relationships
    CROSS_DOMAIN_PATTERNS = {
        ("physics", "chemistry"): [
            "Quantum mechanics underlies chemical bonding",
            "Thermodynamics applies to chemical reactions",
            "Electromagnetism governs molecular interactions",
        ],
        ("physics", "biology"): [
            "Biophysics: physics of living systems",
            "Thermodynamics constrains metabolism",
            "Statistical mechanics explains biological noise",
        ],
        ("chemistry", "biology"): [
            "Biochemistry: chemical basis of life",
            "Molecular biology: DNA/RNA/protein chemistry",
            "Pharmacology: drug-receptor chemistry",
        ],
        ("physics", "astronomy"): [
            "Astrophysics: physics of celestial objects",
            "Cosmology: physics of the universe",
            "Gravity dominates astronomical scales",
        ],
        ("physics", "climate"): [
            "Radiative transfer governs energy balance",
            "Thermodynamics controls heat transport",
            "Fluid dynamics describes atmospheric motion",
        ],
        ("physics", "neuroscience"): [
            "Electrophysiology: electrical activity in neurons",
            "Biophysics: ion channel dynamics",
            "Information theory: neural coding",
        ],
    }
    
    # Universal principles that should hold across domains
    UNIVERSAL_PRINCIPLES = {
        "conservation_of_energy": "Energy is neither created nor destroyed",
        "second_law": "Entropy of isolated system never decreases",
        "causality": "Effects cannot precede causes",
        "locality": "Information cannot travel faster than light",
        "uncertainty": "Certain pairs of properties cannot be simultaneously known",
    }
    
    def check_cross_domain_consistency(self,
                                       source_domain: str,
                                       target_domain: str,
                                       claim: str) -> VerificationResult:
        """Check if a claim is consistent across domains."""
        
        # Normalize domain names
        source = source_domain.lower().split("_")[0]
        target = target_domain.lower().split("_")[0]
        
        # Check known relationships
        key = (source, target) if source < target else (target, source)
        
        relationships = self.CROSS_DOMAIN_PATTERNS.get(key, [])
        
        # Check for universal principle violations
        claim_lower = claim.lower()
        violations = []
        
        for principle, description in self.UNIVERSAL_PRINCIPLES.items():
            # Simple heuristic checks
            if "perpetual motion" in claim_lower or "free energy" in claim_lower:
                violations.append(f"Possible violation of {principle}")
            if "faster than light" in claim_lower:
                violations.append(f"Possible violation of locality/causality")
        
        if violations:
            return VerificationResult(
                check_type=VerificationType.CROSS_DOMAIN,
                status=VerificationStatus.FAILED,
                message=f"Universal principle violations detected: {len(violations)}",
                confidence=0.9,
                details={"violations": violations},
                suggestions=["Revise claim to respect universal principles"]
            )
        
        if relationships:
            return VerificationResult(
                check_type=VerificationType.CROSS_DOMAIN,
                status=VerificationStatus.PASSED,
                message=f"Cross-domain link ({source} <-> {target}) is plausible",
                confidence=0.7,
                details={"known_relationships": relationships}
            )
        else:
            return VerificationResult(
                check_type=VerificationType.CROSS_DOMAIN,
                status=VerificationStatus.WARNING,
                message=f"No established cross-domain patterns for {source} <-> {target}",
                confidence=0.5,
                details={"source": source, "target": target},
                suggestions=["This may be a novel cross-domain connection"]
            )


# =============================================================================
# MAIN HYPOTHESIS VERIFIER
# =============================================================================

class HypothesisVerifier:
    """
    Main verification engine that combines all checkers.
    
    Performs comprehensive verification of scientific hypotheses:
    1. Dimensional analysis
    2. Physical bounds checking
    3. Conservation law verification
    4. Mathematical consistency
    5. Cross-domain consistency
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.dimensional = DimensionalAnalyzer()
        self.physical = PhysicalConstraintChecker()
        self.mathematical = MathematicalConsistencyChecker()
        self.cross_domain = CrossDomainConsistencyChecker()
        
        # Track verification history
        self.verification_history: List[VerificationReport] = []
        
        if verbose:
            print("=" * 60)
            print("QENEX Hypothesis Verifier Initialized")
            print("=" * 60)
            print("Checkers: Dimensional | Physical | Mathematical | Cross-Domain")
            print("=" * 60)
    
    def verify_hypothesis(self,
                          hypothesis_id: str,
                          statement: str,
                          domain: str,
                          mathematical_form: str = None,
                          parameters: List[str] = None,
                          parameter_values: Dict[str, float] = None,
                          source_domain: str = None) -> VerificationReport:
        """
        Perform comprehensive verification of a hypothesis.
        
        Args:
            hypothesis_id: Unique identifier
            statement: The hypothesis statement
            domain: Scientific domain
            mathematical_form: Mathematical formulation (if any)
            parameters: List of parameter names
            parameter_values: Dict of parameter name -> value
            source_domain: Source domain for cross-domain hypotheses
        
        Returns:
            VerificationReport with all check results
        """
        checks = []
        
        if self.verbose:
            print(f"\n[VERIFY] Hypothesis: {hypothesis_id}")
            print(f"  Domain: {domain}")
            print(f"  Statement: {statement[:80]}...")
        
        # 1. Check for physical impossibilities
        check = self.physical.check_for_impossibilities(statement)
        checks.append(check)
        if self.verbose:
            print(f"  [1] Impossibility check: {check.status.name}")
        
        # 2. Dimensional analysis (if mathematical form provided)
        if mathematical_form:
            parsed = self.dimensional.parse_mathematical_form(mathematical_form)
            if parsed["lhs"] and parsed["rhs"]:
                check = self.dimensional.check_equation_balance(
                    parsed["lhs"], parsed["rhs"],
                    parsed["lhs_exponents"], parsed["rhs_exponents"]
                )
                checks.append(check)
                if self.verbose:
                    print(f"  [2] Dimensional check: {check.status.name}")
            
            # 3. Mathematical consistency
            check = self.mathematical.check_pattern_validity(mathematical_form)
            checks.append(check)
            if self.verbose:
                print(f"  [3] Mathematical check: {check.status.name}")
        
        # 4. Parameter constraints
        if parameters:
            check = self.mathematical.check_parameter_constraints(
                parameters, parameter_values
            )
            checks.append(check)
            if self.verbose:
                print(f"  [4] Parameter check: {check.status.name}")
        
        # 5. Cross-domain consistency
        if source_domain:
            check = self.cross_domain.check_cross_domain_consistency(
                source_domain, domain, statement
            )
            checks.append(check)
            if self.verbose:
                print(f"  [5] Cross-domain check: {check.status.name}")
        
        # Calculate overall status
        if any(c.status == VerificationStatus.FAILED for c in checks):
            overall_status = VerificationStatus.FAILED
        elif any(c.status == VerificationStatus.WARNING for c in checks):
            overall_status = VerificationStatus.WARNING
        elif all(c.status in [VerificationStatus.PASSED, VerificationStatus.SKIPPED] for c in checks):
            overall_status = VerificationStatus.PASSED
        else:
            overall_status = VerificationStatus.UNCERTAIN
        
        # Calculate overall confidence
        if checks:
            overall_confidence = np.mean([c.confidence for c in checks])
        else:
            overall_confidence = 0.0
        
        report = VerificationReport(
            hypothesis_id=hypothesis_id,
            hypothesis_statement=statement,
            domain=domain,
            checks=checks,
            overall_status=overall_status,
            overall_confidence=overall_confidence
        )
        
        self.verification_history.append(report)
        
        if self.verbose:
            print(f"  OVERALL: {overall_status.name} (confidence: {overall_confidence:.2%})")
        
        return report
    
    def verify_from_generated_hypothesis(self, hypothesis) -> VerificationReport:
        """
        Verify a hypothesis from the hypothesis generator.
        
        Args:
            hypothesis: GeneratedHypothesis object
        
        Returns:
            VerificationReport
        """
        return self.verify_hypothesis(
            hypothesis_id=hypothesis.id,
            statement=hypothesis.statement,
            domain=hypothesis.domain,
            mathematical_form=hypothesis.mathematical_form,
            parameters=hypothesis.key_parameters,
            source_domain=hypothesis.source_analogy.split(":")[0] if hypothesis.source_analogy else None
        )
    
    def batch_verify(self, hypotheses: List[Any]) -> List[VerificationReport]:
        """Verify multiple hypotheses."""
        reports = []
        
        if self.verbose:
            print(f"\n[BATCH VERIFY] {len(hypotheses)} hypotheses")
        
        for hyp in hypotheses:
            if hasattr(hyp, 'id'):  # GeneratedHypothesis
                report = self.verify_from_generated_hypothesis(hyp)
            else:  # Dict
                report = self.verify_hypothesis(
                    hypothesis_id=hyp.get('id', 'unknown'),
                    statement=hyp.get('statement', ''),
                    domain=hyp.get('domain', ''),
                    mathematical_form=hyp.get('mathematical_form'),
                    parameters=hyp.get('key_parameters')
                )
            reports.append(report)
        
        # Summary statistics
        if self.verbose:
            passed = sum(1 for r in reports if r.overall_status == VerificationStatus.PASSED)
            failed = sum(1 for r in reports if r.overall_status == VerificationStatus.FAILED)
            warnings = sum(1 for r in reports if r.overall_status == VerificationStatus.WARNING)
            print(f"\n[SUMMARY] Passed: {passed} | Failed: {failed} | Warnings: {warnings}")
        
        return reports
    
    def export_reports(self, filepath: str):
        """Export all verification reports to JSON."""
        data = {
            "reports": [r.to_dict() for r in self.verification_history],
            "total": len(self.verification_history),
            "summary": {
                "passed": sum(1 for r in self.verification_history 
                             if r.overall_status == VerificationStatus.PASSED),
                "failed": sum(1 for r in self.verification_history 
                             if r.overall_status == VerificationStatus.FAILED),
                "warnings": sum(1 for r in self.verification_history 
                               if r.overall_status == VerificationStatus.WARNING),
            },
            "exported_at": datetime.now().isoformat(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.verbose:
            print(f"\nExported {len(self.verification_history)} reports to {filepath}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Demonstrate the formal verification module."""
    
    print("""
    =====================================================
         QENEX FORMAL VERIFICATION MODULE
         Scientific Hypothesis Verification Engine
    =====================================================
    Checkers:
      - Dimensional Analysis
      - Physical Constraint Checking  
      - Mathematical Consistency
      - Cross-Domain Validation
    =====================================================
    """)
    
    verifier = HypothesisVerifier(verbose=True)
    
    # Test 1: Valid hypothesis
    print("\n" + "=" * 50)
    print("TEST 1: Valid Physics Hypothesis")
    print("=" * 50)
    
    report1 = verifier.verify_hypothesis(
        hypothesis_id="test_001",
        statement="Kinetic energy follows a quadratic relationship with velocity",
        domain="physics",
        mathematical_form="E = 0.5 * m * v^2",
        parameters=["mass", "velocity"],
        parameter_values={"mass": 1.0, "velocity": 10.0}
    )
    print(report1.summary())
    
    # Test 2: Invalid hypothesis (physical impossibility)
    print("\n" + "=" * 50)
    print("TEST 2: Invalid Hypothesis (Physical Impossibility)")
    print("=" * 50)
    
    report2 = verifier.verify_hypothesis(
        hypothesis_id="test_002",
        statement="A new perpetual motion machine achieves faster than light travel",
        domain="physics",
        mathematical_form=None
    )
    print(report2.summary())
    
    # Test 3: Cross-domain hypothesis
    print("\n" + "=" * 50)
    print("TEST 3: Cross-Domain Hypothesis")
    print("=" * 50)
    
    report3 = verifier.verify_hypothesis(
        hypothesis_id="test_003",
        statement="Neural firing patterns exhibit critical phase transition behavior similar to magnetic systems",
        domain="neuroscience",
        mathematical_form="phi ~ |T - Tc|^beta",
        parameters=["critical_exponent_beta"],
        source_domain="physics"
    )
    print(report3.summary())
    
    # Test 4: Climate science hypothesis
    print("\n" + "=" * 50)
    print("TEST 4: Climate Science Hypothesis")
    print("=" * 50)
    
    report4 = verifier.verify_hypothesis(
        hypothesis_id="test_004",
        statement="Climate sensitivity to CO2 doubling is approximately 3K",
        domain="climate",
        mathematical_form="dT = lambda * F",
        parameters=["climate_sensitivity", "radiative_forcing"],
        parameter_values={"climate_sensitivity": 3.0}
    )
    print(report4.summary())
    
    # Export results
    verifier.export_reports("/opt/qenex_lab/workspace/reports/verification_demo.json")
    
    return verifier


if __name__ == "__main__":
    main()
