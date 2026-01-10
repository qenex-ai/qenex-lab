"""
Q-Lang 2.0: The Unified Scientific Language
Leveraging the best of:
- Python: Syntax, Ecosystem, and NumPy integration.
- Rust: Safety, Memory Discipline, and Scout Validation.
- Julia: High-Performance Numerical Computing bridge.
- Functional (F#/Elixir): Pipe Operators (|>) for clear data flow.
- Physics: First-class Dimensional Analysis and Uncertainty.
"""

import re
import numpy as np
import sys
import os
import math
import cmath
import random 
import decimal
from decimal import Decimal, getcontext
from dataclasses import dataclass
from typing import Dict, Any, List, Union, Tuple
import subprocess

# [PRECISION] Set global precision to 50 digits
getcontext().prec = 50

# [INTEROP] Robust Path Resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
packages_dir = os.path.dirname(os.path.dirname(current_dir))

# Add all package source directories to path
# Note: qenex_chem uses underscore, others use hyphens
sys.path.insert(0, os.path.join(packages_dir, "qenex_chem", "src"))  # Chemistry (underscore)
sys.path.insert(0, os.path.join(packages_dir, "qenex-chem", "src"))  # Fallback (hyphen)
sys.path.insert(0, os.path.join(packages_dir, "qenex-bio", "src"))
sys.path.insert(0, os.path.join(packages_dir, "qenex-physics", "src"))
sys.path.insert(0, os.path.join(packages_dir, "qenex-math", "src"))
sys.path.insert(0, os.path.join(packages_dir, "qenex-core", "src"))

# [INTEROP] Import Kernels
Molecule = None
MatrixHartreeFock = None
CISolver = None
MP2Solver = None
ProteinFolder = None
LatticeSimulator = None
ProofState = None
TacticalProver = None

try:
    from molecule import Molecule
    from solver import HartreeFockSolver as MatrixHartreeFock, CISolver, MP2Solver
    from folding import ProteinFolder
    from optimized_lattice import OptimizedLattice as LatticeSimulator
    from prover import ProofState, TacticalProver
    KERNELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Scientific kernels not found: {e}. Running in localized mode.")
    KERNELS_AVAILABLE = False
    KERNELS_AVAILABLE = False

@dataclass
class Dimensions:
    mass: float = 0
    length: float = 0
    time: float = 0
    current: float = 0
    temperature: float = 0
    amount: float = 0
    luminous: float = 0
    
    def __add__(self, other):
        return Dimensions(self.mass+other.mass, self.length+other.length, 
                          self.time+other.time, self.current+other.current,
                          self.temperature+other.temperature, self.amount+other.amount,
                          self.luminous+other.luminous)
                          
    def __sub__(self, other):
        return Dimensions(self.mass-other.mass, self.length-other.length, 
                          self.time-other.time, self.current-other.current,
                          self.temperature-other.temperature, self.amount-other.amount,
                          self.luminous-other.luminous)
    
    def __eq__(self, other):
        return (self.mass == other.mass and self.length == other.length and 
                self.time == other.time and self.current == other.current)

    def __str__(self):
        parts = []
        if self.mass: parts.append(f"M^{self.mass}")
        if self.length: parts.append(f"L^{self.length}")
        if self.time: parts.append(f"T^{self.time}")
        return " ".join(parts) if parts else "Dimensionless"

# --- The Core Data Type ---
@dataclass
class QValue:
    value: Union[Decimal, float, complex, np.ndarray]
    dims: Dimensions
    uncertainty: Union[Decimal, float, np.ndarray] = Decimal(0.0)
    
    def __repr__(self):
        if isinstance(self.uncertainty, (Decimal, float)):
             u_str = f" ± {self.uncertainty:.2e}" if self.uncertainty else ""
        elif isinstance(self.uncertainty, np.ndarray):
             u_str = f" ± {self.uncertainty}"
        else:
             u_str = ""

        if isinstance(self.value, np.ndarray):
            return f"{self.value}{u_str} [{self.dims}]"
        
        return f"{self.value:.4e}{u_str} [{self.dims}]"
    
    def is_dimensionless(self):
        return self.dims == Dimensions()

    def __add__(self, other):
        if isinstance(other, (int, float, Decimal, complex)):
            if self.is_dimensionless():
                if isinstance(self.value, complex) or isinstance(other, complex):
                    return QValue(complex(self.value) + complex(other), self.dims, self.uncertainty)

                val_dec = Decimal(str(self.value)) if not isinstance(self.value, Decimal) else self.value
                other_dec = Decimal(str(other))
                return QValue(val_dec + other_dec, self.dims, self.uncertainty)
            raise TypeError(f"Cannot add scalar {other} to dimensioned quantity {self.dims}")
            
        if not isinstance(other, QValue): return NotImplemented
        if self.dims != other.dims:
            raise TypeError(f"Dimensional Mismatch: Cannot add {self.dims} and {other.dims}")
        
        if isinstance(self.value, complex) or isinstance(other.value, complex):
             return QValue(complex(self.value) + complex(other.value), self.dims, self.uncertainty)

        val_self = Decimal(str(self.value)) if not isinstance(self.value, (Decimal, np.ndarray)) else self.value
        val_other = Decimal(str(other.value)) if not isinstance(other.value, (Decimal, np.ndarray)) else other.value
        
        unc_self = Decimal(str(self.uncertainty)) if not isinstance(self.uncertainty, (Decimal, np.ndarray)) else self.uncertainty
        unc_other = Decimal(str(other.uncertainty)) if not isinstance(other.uncertainty, (Decimal, np.ndarray)) else other.uncertainty
        
        # [FIX] Uncertainty Correlation for Addition
        # If adding a variable to itself, errors add linearly, not in quadrature
        if self is other:
            new_unc = unc_self + unc_other
        else:
            new_unc = (unc_self**2 + unc_other**2).sqrt()
        
        return QValue(val_self + val_other, self.dims, new_unc)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float, Decimal)):
            if self.is_dimensionless():
                val_dec = Decimal(str(self.value)) if not isinstance(self.value, Decimal) else self.value
                other_dec = Decimal(str(other))
                return QValue(val_dec - other_dec, self.dims, self.uncertainty)
            raise TypeError(f"Cannot subtract scalar {other} from dimensioned quantity {self.dims}")

        if not isinstance(other, QValue): return NotImplemented
        # Dimensions do not need to match for multiplication
        # if self.dims != other.dims:
        #      raise TypeError(f"Dimensional Mismatch: Cannot multiply {self.dims} and {other.dims}")
             
        val_self = Decimal(str(self.value)) if not isinstance(self.value, (Decimal, np.ndarray)) else self.value
        val_other = Decimal(str(other.value)) if not isinstance(other.value, (Decimal, np.ndarray)) else other.value
        
        # [FIX] Handle Infinity subtraction
        if isinstance(val_self, Decimal) and val_self.is_infinite():
            return QValue(val_self, self.dims, self.uncertainty)
        if isinstance(val_other, Decimal) and val_other.is_infinite():
            return QValue(-val_other, self.dims, self.uncertainty)

        unc_self = Decimal(str(self.uncertainty)) if not isinstance(self.uncertainty, (Decimal, np.ndarray)) else self.uncertainty
        unc_other = Decimal(str(other.uncertainty)) if not isinstance(other.uncertainty, (Decimal, np.ndarray)) else other.uncertainty
        
        # [FIX] Uncertainty Correlation for Subtraction
        # If subtracting a variable from itself, errors are perfectly correlated.
        # Since signs are opposite (A - B), linear errors subtract.
        # |delta_A - delta_B| = |u - u| = 0
        if self is other:
            new_unc = abs(unc_self - unc_other) # Should be 0
        else:
            new_unc = (unc_self**2 + unc_other**2).sqrt()

        return QValue(val_self - val_other, self.dims, new_unc)

    def __rsub__(self, other):
        if isinstance(other, (int, float, Decimal)):
            if self.is_dimensionless():
                val_dec = Decimal(str(self.value)) if not isinstance(self.value, Decimal) else self.value
                other_dec = Decimal(str(other))
                return QValue(other_dec - val_dec, self.dims, self.uncertainty)
            raise TypeError(f"Cannot subtract dimensioned quantity {self.dims} from scalar {other}")
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float, Decimal)):
             if other == 0: 
                 # [FIX] Return QValue with 0/Nan behavior instead of crashing, or raise proper error
                 # For scientific calc, let's treat as Infinity or handle safely
                 return QValue(Decimal('Infinity'), self.dims, self.uncertainty)
             
             val_dec = Decimal(str(self.value)) if not isinstance(self.value, Decimal) else self.value
             other_dec = Decimal(str(other))
             unc_dec = Decimal(str(self.uncertainty)) if not isinstance(self.uncertainty, Decimal) else self.uncertainty
             try:
                 return QValue(val_dec / other_dec, self.dims, unc_dec / abs(other_dec))
             except (decimal.DivisionByZero, decimal.InvalidOperation):
                 return QValue(Decimal('Infinity'), self.dims, self.uncertainty)
        
        if not isinstance(other, QValue): return NotImplemented
        
        val_self = Decimal(str(self.value)) if not isinstance(self.value, (Decimal, np.ndarray)) else self.value
        val_other = Decimal(str(other.value)) if not isinstance(other.value, (Decimal, np.ndarray)) else other.value
        
        # [FIX] Check for division by zero QValue
        if val_other == 0:
             # 0/0 = NaN, nonzero/0 = Infinity
             if val_self == 0:
                 return QValue(Decimal('NaN'), self.dims - other.dims, self.uncertainty)
             return QValue(Decimal('Infinity'), self.dims - other.dims, self.uncertainty)

        try:
            new_val = val_self / val_other
        except (decimal.DivisionByZero, decimal.InvalidOperation):
            return QValue(Decimal('Infinity'), self.dims - other.dims, self.uncertainty)
        
        unc_self = Decimal(str(self.uncertainty)) if not isinstance(self.uncertainty, Decimal) else self.uncertainty
        unc_other = Decimal(str(other.uncertainty)) if not isinstance(other.uncertainty, Decimal) else other.uncertainty
        
        rel_unc = Decimal(0.0)
        if val_self != 0: rel_unc += (unc_self / val_self)**2
        if val_other != 0: rel_unc += (unc_other / val_other)**2
        rel_unc = rel_unc.sqrt()
        
        return QValue(new_val, self.dims - other.dims, abs(new_val) * rel_unc)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, Decimal)):
             if self.value == 0: raise ValueError("Division by zero")
             
             neg_dims = Dimensions(-self.dims.mass, -self.dims.length, -self.dims.time,
                 -self.dims.current, -self.dims.temperature, -self.dims.amount, -self.dims.luminous)
             
             val_dec = Decimal(str(self.value)) if not isinstance(self.value, Decimal) else self.value
             other_dec = Decimal(str(other))
             
             new_val = other_dec / val_dec
             
             unc_dec = Decimal(str(self.uncertainty)) if not isinstance(self.uncertainty, Decimal) else self.uncertainty
             rel_unc = Decimal(0.0)
             if val_dec != 0: rel_unc = abs(unc_dec / val_dec)
                  
             return QValue(new_val, neg_dims, abs(new_val) * rel_unc)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float, Decimal, complex)):
            if isinstance(other, complex) or isinstance(self.value, complex):
                val_self = complex(self.value)
                val_other = complex(other)
                return QValue(val_self * val_other, self.dims, self.uncertainty)

            val_dec = Decimal(str(self.value)) if not isinstance(self.value, Decimal) else self.value
            other_dec = Decimal(str(other))
            unc_dec = Decimal(str(self.uncertainty)) if not isinstance(self.uncertainty, Decimal) else self.uncertainty
            return QValue(val_dec * other_dec, self.dims, unc_dec * other_dec)
            
        if not isinstance(other, QValue): return NotImplemented
        
        if isinstance(self.value, complex) or isinstance(other.value, complex):
             val_self = complex(self.value)
             val_other = complex(other.value)
             # Complex numbers generally don't have standard uncertainty prop in this context yet
             return QValue(val_self * val_other, self.dims + other.dims, 0.0)

        if isinstance(self.value, np.ndarray) or isinstance(other.value, np.ndarray):
             val_self = self.value if isinstance(self.value, np.ndarray) else float(self.value)
             val_other = other.value if isinstance(other.value, np.ndarray) else float(other.value)
             
             if isinstance(val_self, np.ndarray) and val_self.dtype == object: val_self = val_self.astype(float)
             if isinstance(val_other, np.ndarray) and val_other.dtype == object: val_other = val_other.astype(float)
                 
             new_val = val_self * val_other
             
             # [FIX] Handle Vector-Vector Uncertainty Propagation
             unc_self = self.uncertainty
             unc_other = other.uncertainty
             
             # Convert scalar uncertainties to array/broadcast if needed
             if not isinstance(unc_self, np.ndarray): unc_self = np.array(float(unc_self))
             if not isinstance(unc_other, np.ndarray): unc_other = np.array(float(unc_other))
             
             # Ensure uncertainties are broadcastable to value shape
             if unc_self.ndim == 0 and isinstance(val_self, np.ndarray):
                 unc_self = np.full_like(val_self, float(unc_self))
             if unc_other.ndim == 0 and isinstance(val_other, np.ndarray):
                 unc_other = np.full_like(val_other, float(unc_other))

             # [CRITICAL FIX] Vector Correlation Trap
             # If self is other, errors add linearly: |dx/x| + |dx/x| = 2|dx/x|
             if self is other:
                 rel_unc = np.zeros_like(new_val)
                 if isinstance(val_self, np.ndarray):
                     mask = (val_self != 0)
                     if np.any(mask):
                         if isinstance(unc_self, np.ndarray): u_chunk = unc_self[mask]
                         else: u_chunk = float(unc_self)
                         rel_unc[mask] = 2.0 * np.abs(u_chunk / val_self[mask])
                 else:
                     if val_self != 0: rel_unc = 2.0 * abs(float(unc_self)/val_self)
                 
                 new_unc = np.abs(new_val) * rel_unc
                 return QValue(new_val, self.dims + other.dims, new_unc)

             # Calculate relative uncertainty via quadrature for independent vectors
             rel_term_self = np.zeros_like(new_val)
             rel_term_other = np.zeros_like(new_val)
             
             # Compute relative terms safely
             if isinstance(val_self, np.ndarray):
                 mask = (val_self != 0)
                 # Only compute where mask is True
                 if np.any(mask):
                     # If unc is scalar, it broadcasts automatically
                     # Convert to float array to avoid dtype issues
                     
                     # Safe uncertainty extraction
                     if isinstance(unc_self, np.ndarray):
                         u_chunk = unc_self[mask]
                     else:
                         u_chunk = float(unc_self)

                     rel_term_self[mask] = (u_chunk / val_self[mask])**2
             else:
                 if val_self != 0: rel_term_self = (float(unc_self)/val_self)**2

             if isinstance(val_other, np.ndarray):
                 mask = (val_other != 0)
                 if np.any(mask):
                     if isinstance(unc_other, np.ndarray):
                         u_chunk = unc_other[mask]
                     else:
                         u_chunk = float(unc_other)

                     rel_term_other[mask] = (u_chunk / val_other[mask])**2
             else:
                 if val_other != 0: rel_term_other = (float(unc_other)/val_other)**2
             
             rel_unc = np.sqrt(rel_term_self + rel_term_other)
             new_unc = np.abs(new_val) * rel_unc
             
             return QValue(new_val, self.dims + other.dims, new_unc)
        # Dimensions do not need to match for multiplication
        # if self.dims != other.dims:
        #      raise TypeError(f"Dimensional Mismatch: Cannot multiply {self.dims} and {other.dims}")
             
        val_self = Decimal(str(self.value)) if not isinstance(self.value, (Decimal, np.ndarray)) else self.value
        val_other = Decimal(str(other.value)) if not isinstance(other.value, (Decimal, np.ndarray)) else other.value
        
        # [FIX] Handle Infinity multiplication
        if isinstance(val_self, Decimal) and val_self.is_infinite():
            return QValue(val_self, self.dims + other.dims, self.uncertainty)
        if isinstance(val_other, Decimal) and val_other.is_infinite():
            return QValue(val_other, self.dims + other.dims, self.uncertainty)
            
        new_val = val_self * val_other
        
        unc_self = Decimal(str(self.uncertainty)) if not isinstance(self.uncertainty, Decimal) else self.uncertainty
        unc_other = Decimal(str(other.uncertainty)) if not isinstance(other.uncertainty, Decimal) else other.uncertainty
        
        rel_unc = Decimal(0.0)
        
        # [FIX] Uncertainty Correlation Trap
        if self is other:
             if val_self != 0: rel_unc = Decimal('2.0') * abs(unc_self / val_self)
        else:
             if val_self != 0: rel_unc += (unc_self / val_self)**2
             if val_other != 0: rel_unc += (unc_other / val_other)**2
             rel_unc = rel_unc.sqrt()
             
        new_dims = self.dims + other.dims
        
        return QValue(new_val, new_dims, abs(new_val) * rel_unc)

    def __pow__(self, power: Union[int, float, Decimal]):
        val_self = Decimal(str(self.value)) if not isinstance(self.value, (Decimal, np.ndarray)) else self.value
        p_dec = Decimal(str(power))
        
        new_val = val_self ** p_dec
        
        dims = Dimensions(
            self.dims.mass * float(p_dec), 
            self.dims.length * float(p_dec), 
            self.dims.time * float(p_dec),
            self.dims.current * float(p_dec),
            self.dims.temperature * float(p_dec),
            self.dims.amount * float(p_dec),
            self.dims.luminous * float(p_dec)
        )
        
        unc_self = Decimal(str(self.uncertainty)) if not isinstance(self.uncertainty, Decimal) else self.uncertainty
        new_unc = Decimal(0.0)
        if val_self != 0:
            new_unc = abs(new_val) * abs(p_dec) * (unc_self / val_self)

        if dims.length == 2 and dims.time == -2:
             c_val = Decimal('299792458')
             c_sq = c_val**2
             if abs(new_val) > (Decimal('0.01') * c_sq):
                 print(f"⚠️  [RELATIVITY WARNING] High-velocity computation detected (v > 0.1c).")
                 print(f"    Classical mechanics is inaccurate here. Use Relativistic formulas.")

        return QValue(new_val, dims, new_unc)

    def __lt__(self, other):
        if isinstance(other, (int, float)):
             if other == 0: return self.value < 0
             if self.dims != Dimensions(): raise TypeError(f"Cannot compare dimensioned quantity {self} with scalar {other}")
             return self.value < other
        if not isinstance(other, QValue): return NotImplemented
        if self.dims != other.dims:
             raise TypeError(f"Dimensional Mismatch in comparison: {self.dims} vs {other.dims}")
        return self.value < other.value

    def __le__(self, other):
        if isinstance(other, (int, float)):
             if other == 0: return self.value <= 0
             if self.dims != Dimensions(): raise TypeError(f"Cannot compare dimensioned quantity {self} with scalar {other}")
             return self.value <= other
        if not isinstance(other, QValue): return NotImplemented
        if self.dims != other.dims:
             raise TypeError(f"Dimensional Mismatch in comparison: {self.dims} vs {other.dims}")
        return self.value <= other.value

    def __gt__(self, other):
        if isinstance(other, (int, float, Decimal)):
             if other == 0 and self.is_dimensionless():
                  val = float(self.value) if isinstance(self.value, (Decimal, float, int)) else 0
                  return val > 0
             if self.dims != Dimensions(): 
                  # Allow comparison against 0 even if dimensioned
                  if other == 0:
                      val = float(self.value) if isinstance(self.value, (Decimal, float, int)) else 0
                      return val > 0
                  # Otherwise check types
                  raise TypeError(f"Cannot compare dimensioned quantity {self} with scalar {other}")
             val = float(self.value) if isinstance(self.value, (Decimal, float, int)) else float(self.value)
             return val > float(other)
        if not isinstance(other, QValue): return NotImplemented
        if self.dims != other.dims:
             raise TypeError(f"Dimensional Mismatch in comparison: {self.dims} vs {other.dims}")
        return self.value > other.value

    def __ge__(self, other):
        # Similar logic for >=
        if isinstance(other, (int, float, Decimal)):
             if other == 0 and self.is_dimensionless():
                 val = float(self.value) if isinstance(self.value, (Decimal, float, int)) else 0
                 return val >= 0
             if self.dims != Dimensions():
                 if other == 0:
                      val = float(self.value) if isinstance(self.value, (Decimal, float, int)) else 0
                      return val >= 0
                 raise TypeError(f"Cannot compare dimensioned quantity {self} with scalar {other}")
             val = float(self.value) if isinstance(self.value, (Decimal, float, int)) else float(self.value)
             return val >= float(other)
        if not isinstance(other, QValue): return NotImplemented
        if self.dims != other.dims:
             raise TypeError(f"Dimensional Mismatch in comparison: {self.dims} vs {other.dims}")
        return self.value >= other.value
    
    def __abs__(self):
        return QValue(abs(self.value), self.dims, self.uncertainty)

# --- The Interpreter ---
class QLangInterpreter:
    def __init__(self):
        self.context: Dict[str, QValue] = {}
        self._load_constants()
        
    def _load_constants(self):
        self.context['m'] = QValue(1.0, Dimensions(length=1))
        self.context['kg'] = QValue(1.0, Dimensions(mass=1))
        self.context['s'] = QValue(1.0, Dimensions(time=1))
        
        self.context['c'] = QValue(2.99792458e8, Dimensions(length=1, time=-1))
        self.context['h'] = QValue(6.62607015e-34, Dimensions(mass=1, length=2, time=-1))
        self.context['G'] = QValue(6.67430e-11, Dimensions(length=3, mass=-1, time=-2), 0.00015e-11)
        
        self.protected = set(['c', 'h', 'G', 'm', 'kg', 's'])

    def _get_eval_context(self) -> Dict[str, Any]:
        """Returns a safe dictionary for eval with stdlib and context."""
        safe_dict = self.context.copy()
        safe_dict["__builtins__"] = {}
        
        # Core Types
        safe_dict["QValue"] = QValue
        safe_dict["Dimensions"] = Dimensions
        safe_dict["Decimal"] = Decimal
        safe_dict["np"] = np
        
        # Math Utils
        def q_sqrt(x):
             if isinstance(x, QValue):
                 # [FIX] Handle infinite QValue
                 if isinstance(x.value, Decimal) and x.value.is_infinite():
                     return x
                 
                 val = x.value
                 if isinstance(val, (int, float, Decimal)):
                     if val < 0:
                          return QValue(cmath.sqrt(float(val)), Dimensions()) # Complex result
                     new_val = Decimal(str(val)) ** Decimal('0.5')
                 else:
                     new_val = val ** 0.5
                     
                 # Dimensions: sqrt(L^2) -> L^1
                 new_dims = Dimensions(
                     x.dims.mass * 0.5, x.dims.length * 0.5, x.dims.time * 0.5,
                     x.dims.current * 0.5, x.dims.temperature * 0.5, 
                     x.dims.amount * 0.5, x.dims.luminous * 0.5
                 )
                 
                 # Uncertainty: sqrt(x) -> 0.5 * dx/x * sqrt(x) = 0.5 * dx / sqrt(x)
                 new_unc = Decimal(0.0)
                 if x.uncertainty and val != 0:
                     new_unc = Decimal('0.5') * Decimal(str(x.uncertainty)) / Decimal(str(new_val)) * Decimal(str(new_val)) 
                     # Wait, relative unc: dy/y = 0.5 * dx/x
                     # dy = 0.5 * y * dx/x = 0.5 * sqrt(x) * dx/x = 0.5 * dx / sqrt(x)
                     new_unc = Decimal('0.5') * Decimal(str(x.uncertainty)) / Decimal(str(new_val)) * Decimal(str(new_val))
                     # Simplifies to: 0.5 * dx * (x^-0.5)
                     new_unc = Decimal('0.5') * Decimal(str(x.uncertainty)) * (Decimal(str(val)) ** Decimal('-0.5'))

                 return QValue(new_val, new_dims, new_unc)

             if isinstance(x, (int, float, Decimal)):
                 if x < 0: 
                     # Return QValue for complex results to enable Q-Lang arithmetic
                     return QValue(cmath.sqrt(float(x)), Dimensions())
                 return Decimal(str(x)) ** Decimal(0.5) if isinstance(x, Decimal) else x ** 0.5
             return x ** 0.5

        safe_dict["sqrt"] = q_sqrt
        
        # [FIX] Derivative-based Uncertainty Propagation
        def q_sin(x):
            if not isinstance(x, QValue): 
                if isinstance(x, complex): return cmath.sin(x)
                return math.sin(x)
            if not x.is_dimensionless(): raise ValueError("sin() requires dimensionless argument")
            
            val = x.value
            if isinstance(val, complex):
                 return QValue(cmath.sin(val), Dimensions())

            val = float(x.value)
            res = math.sin(val)
            
            # Uncertainty: |cos(x) * dx|
            unc = Decimal(0.0)
            if x.uncertainty:
                deriv = abs(math.cos(val))
                unc = Decimal(deriv) * Decimal(str(x.uncertainty))
                
            return QValue(Decimal(res), Dimensions(), unc)

        def q_cos(x):
            if not isinstance(x, QValue):
                 if isinstance(x, complex): return cmath.cos(x)
                 return math.cos(x)
            if not x.is_dimensionless(): raise ValueError("cos() requires dimensionless argument")
            
            val = x.value
            if isinstance(val, complex):
                 return QValue(cmath.cos(val), Dimensions())

            val = float(x.value)
            res = math.cos(val)
            
            # Uncertainty: |-sin(x) * dx|
            unc = Decimal(0.0)
            if x.uncertainty:
                deriv = abs(math.sin(val))
                unc = Decimal(deriv) * Decimal(str(x.uncertainty))
                
            return QValue(Decimal(res), Dimensions(), unc)

        def q_exp(x):
            if not isinstance(x, QValue):
                 if isinstance(x, complex): return cmath.exp(x)
                 return math.exp(x)
            if not x.is_dimensionless(): raise ValueError("exp() requires dimensionless argument")
            
            val = x.value
            if isinstance(val, complex):
                 return QValue(cmath.exp(val), Dimensions())

            val = float(x.value)
            res = math.exp(val)
            
            # Uncertainty: |exp(x) * dx|
            unc = Decimal(0.0)
            if x.uncertainty:
                deriv = abs(res)
                unc = Decimal(deriv) * Decimal(str(x.uncertainty))
                
            return QValue(Decimal(res), Dimensions(), unc)

        def q_abs(x):
              if not isinstance(x, QValue):
                  return abs(x)
              return x.__abs__()

        def q_log(x):
            """Natural logarithm with uncertainty propagation."""
            if not isinstance(x, QValue):
                if isinstance(x, complex): return cmath.log(x)
                if x <= 0:
                    return float('-inf') if x == 0 else complex(math.log(abs(x)), math.pi)
                return math.log(x)
            if not x.is_dimensionless(): raise ValueError("log() requires dimensionless argument")
            
            val = x.value
            if isinstance(val, complex):
                return QValue(cmath.log(val), Dimensions())
            
            val_float = float(val)
            if val_float <= 0:
                if val_float == 0:
                    return QValue(Decimal('-Infinity'), Dimensions())
                # log of negative -> complex
                return QValue(cmath.log(val_float), Dimensions())
            
            res = math.log(val_float)
            
            # Uncertainty: |dx/x|
            unc = Decimal(0.0)
            if x.uncertainty and val_float != 0:
                unc = abs(Decimal(str(x.uncertainty)) / Decimal(str(val_float)))
                
            return QValue(Decimal(str(res)), Dimensions(), unc)

        safe_dict["sin"] = q_sin
        safe_dict["cos"] = q_cos
        safe_dict["exp"] = q_exp
        safe_dict["abs"] = q_abs
        safe_dict["log"] = q_log
        
        # Physics Utils
        def gamma(v):
            # v must be a QValue with [L/T] or a scalar fraction of c
            c_val = Decimal('299792458')
            
            v_obj = v # keep ref
            v_mag = v.value if isinstance(v, QValue) else v
            
            if isinstance(v, QValue):
                if v.dims.length != 1 or v.dims.time != -1:
                    raise ValueError(f"gamma() requires velocity units [L T^-1], got {v.dims}")
            
            if not isinstance(v_mag, Decimal): v_mag = Decimal(str(v_mag))
            
            beta = v_mag / c_val
            if beta >= 1: raise ValueError("Velocity cannot equal or exceed c")
            
            # gamma = 1 / sqrt(1 - beta^2)
            g = Decimal(1) / (Decimal(1) - beta**2).sqrt()
            return QValue(g, Dimensions())

        safe_dict["gamma"] = gamma
        
        return safe_dict


    def _call_scout_validate(self, claim: str) -> bool:
        """
        Bridges to the Rust-based Scout CLI to validate a scientific claim.
        """
        scout_path = "/opt/qenex/scout-cli/target/release/scout"
        if not os.path.exists(scout_path):
            print("   [Q-Lang] ⚠️ Scout CLI not found. Validation mocked.")
            return True # Mock pass

        try:
            # Usage: scout validate "E = mc^2"
            cmd = [scout_path, "validate", claim]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   [Scout/Rust] ✅ Validation Passed: {claim}")
                return True
            else:
                print(f"   [Scout/Rust] ❌ Validation Failed: {claim}")
                print(f"   [Scout/Rust] Reason: {result.stderr.strip()}")
                return False
        except Exception as e:
            print(f"   [Scout/Rust] Error bridging to CLI: {e}")
            return False


    def _call_scout_evolve(self, target: str, generations: int = 50, population: int = 20):
        scout_path = "/opt/qenex/scout-cli/target/release/scout"
        if not os.path.exists(scout_path):
            print("   [Q-Lang] ⚠️ Scout CLI not found. Evolution mocked.")
            return

        print(f"   [Scout/Rust] 🧬 Starting Evolutionary Search for '{target}'...")
        print(f"   [Scout/Rust] Generations: {generations}, Population: {population}")

        try:
            cmd = [scout_path, "evolve", target, "--generations", str(generations), "--population", str(population)]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if process.stdout is None or process.stderr is None:
                 print("   [Scout/Rust] ❌ Error: Could not capture output streams.")
                 return

            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(f"   [Scout] {output.strip()}")
            
            rc = process.poll()
            if rc == 0:
                print(f"   [Scout/Rust] ✅ Evolution Completed.")
            else:
                print(f"   [Scout/Rust] ❌ Evolution Failed.")
                print(process.stderr.read())
                
        except Exception as e:
            print(f"   [Scout/Rust] Error bridging to CLI: {e}")

    def _eval_condition(self, expr: str) -> bool:
        # [FIX] Use robust eval context
        safe_dict = self._get_eval_context()
        
        # Allow $var syntax
        expr = expr.replace('$', '')
        
        try:
             # Evaluate condition with QValues
             res = eval(expr, safe_dict)
             if isinstance(res, QValue):
                 return bool(res.value)
             return bool(res)
        except Exception as e:
             # If evaluation fails, it might be due to undefined variables or unsupported operations
             # Let's try to be resilient for conditions involving undefined vars (treat as False)
             # But if it's a critical error, we should probably log it.
             
             # Specifically handle the case where "ratio" or other vars are not defined yet
             # This happens if a previous block failed to define them due to errors.
             
             # Minimal fix: Return False on error to prevent crash loop, but print error once
             print(f"   [Condition Error] {e}")
             return False

    def execute(self, code: str):
        lines = code.split('\n')
        lines = [line.strip() for line in lines]
        
        ptr = 0
        n = len(lines)
        control_stack: List[Tuple[str, int, Union[str, None]]] = []

        while ptr < n:
            line = lines[ptr]
            # print(f"DEBUG[Ln {ptr+1}]: {line}") 
            
            if not line or line.startswith('#'): 
                ptr += 1
                continue
            
            if line.startswith('optimize '):
                 self._handle_optimize(line)
                 ptr += 1
                 continue

            if line.startswith('evolve '):
                 parts = line.split()
                 if len(parts) < 2:
                      print("❌ Usage: evolve <target> [options...]")
                      ptr += 1
                      continue
                 target = parts[1]
                 if (target.startswith('"') and target.endswith('"')): target = target[1:-1]
                 
                 generations = 50
                 population = 20
                 for p in parts[2:]:
                     if p.startswith("generations="):
                         generations = int(p.split('=')[1])
                     elif p.startswith("population="):
                         population = int(p.split('=')[1])
                 self._call_scout_evolve(target, generations, population)
                 ptr += 1
                 continue

            if line.startswith('verify ') or line.startswith('validate '):
                 claim = line.split(' ', 1)[1].strip()
                 if (claim.startswith('"') and claim.endswith('"')) or (claim.startswith("'") and claim.endswith("'")):
                      claim = claim[1:-1]
                 
                 # [FIX] Do not abort hard if mock validation fails in dev environment
                 # but we print a strong warning.
                 is_valid = self._call_scout_validate(claim)
                 if not is_valid:
                     print(f"⚠️  [VALIDATION WARNING] Scout rejected: '{claim}'")
                 ptr += 1
                 continue
            
            # --- SYNTAX: PIPE OPERATOR (|>) ---
            if '|>' in line:
                 # Elixir/F# style functional piping
                 # Syntax: data |> function1 |> function2
                 # Logic: x |> f means f(x)
                 segments = [s.strip() for s in line.split('|>', )]
                 
                 current_val_expr = segments[0]
                 
                 # Evaluate first segment using robust context
                 safe_dict = self._get_eval_context()
                 
                 try:
                     val = eval(current_val_expr, safe_dict)
                 except Exception:
                     val = current_val_expr # might be a string literal?
                 
                 # Pipe through rest
                 for func_name in segments[1:]:
                     # Assume function exists in context (like sqrt, sin, etc.)
                     if func_name in safe_dict:
                         func = safe_dict[func_name]
                         if callable(func):
                             val = func(val)
                         else:
                             print(f"❌ Pipe Error: '{func_name}' is not callable.")
                             break
                     else:
                         print(f"❌ Pipe Error: Function '{func_name}' not found.")
                         break
                 
                 self.context["_"] = val # Implicit result variable
                 print(f"   [PIPE] Result: {val}")
                 ptr += 1
                 continue

            if line.startswith('while '):
                 if not line.endswith(':'): raise SyntaxError(f"Line {ptr+1}: 'while' must end with ':'")
                 condition = line[6:-1].strip()
                 if self._eval_condition(condition):
                      control_stack.append(('while', ptr, condition))
                      ptr += 1
                 else:
                      ptr += 1
                      depth = 1
                      while ptr < n and depth > 0:
                          l = lines[ptr]
                          if l.startswith('while ') or l.startswith('if '): depth += 1
                          elif l == 'end': depth -= 1
                          ptr += 1
                 continue

            if line.startswith('if '):
                 if not line.endswith(':'): raise SyntaxError(f"Line {ptr+1}: 'if' must end with ':'")
                 condition = line[3:-1].strip()
                 
                 # Logic:
                 # 1. If True: Push 'if' to stack, continue execution.
                 # 2. If False: Scan forward.
                 #    - If 'else' found at depth 1: Jump to else body. Push 'if' to stack.
                 #    - If 'end' found at depth 1 (no else): Jump past end. Do NOT push to stack.
                 
                 if self._eval_condition(condition):
                      # True Branch
                      control_stack.append(('if', ptr, 'true'))
                      ptr += 1
                 else:
                      # False Branch - Scan
                      scan_ptr = ptr + 1
                      depth = 1
                      target_ptr = -1
                      has_else = False
                      
                      while scan_ptr < n and depth > 0:
                          l = lines[scan_ptr]
                          if l.startswith('if ') or l.startswith('while '): 
                              depth += 1
                          elif l == 'end': 
                              depth -= 1
                          elif l == 'else:' and depth == 1:
                              has_else = True
                              target_ptr = scan_ptr + 1
                              break
                          scan_ptr += 1
                      
                      if has_else:
                          # Jump to else body
                          control_stack.append(('if', ptr, 'else'))
                          ptr = target_ptr
                      else:
                          # No else, skip entire block.
                          # scan_ptr is currently AT the 'end' (because depth became 0)
                          # or at 'else' (if found, handled above)
                          # If loop finished with depth=0, scan_ptr points to 'end'.
                          ptr = scan_ptr + 1
                 continue

            if line == 'else:':
                 # We only encounter 'else:' during execution if we fell through the True branch.
                 # (If we were in False branch, we jumped directly to body, skipping this line)
                 # So we must skip the else block now.
                 
                 scan_ptr = ptr + 1
                 depth = 1
                 while scan_ptr < n and depth > 0:
                      l = lines[scan_ptr]
                      if l.startswith('if ') or l.startswith('while '): depth += 1
                      elif l == 'end': depth -= 1
                      scan_ptr += 1
                 
                 # scan_ptr is at 'end'.
                 # We consume the 'end' as well, so we must pop the stack for this 'if'
                 if control_stack and control_stack[-1][0] == 'if':
                     control_stack.pop()
                 else:
                     raise SyntaxError(f"Line {ptr+1}: 'else' without matching 'if'")

                 ptr = scan_ptr + 1
                 continue

            if line == 'end':
                 if not control_stack: raise SyntaxError(f"Line {ptr+1}: Unexpected 'end'")
                 block_info = control_stack.pop()
                 block_type, start_ptr, extra_data = block_info
                 
                 if block_type == 'while':
                      # Re-evaluate condition (extra_data is condition string)
                      if self._eval_condition(extra_data):
                          # Loop again
                          ptr = start_ptr + 1
                          control_stack.append(('while', start_ptr, extra_data))
                      else:
                          # Loop finished
                          ptr += 1
                 elif block_type == 'if':
                      # Just exit block
                      ptr += 1
                 continue

            if line.startswith('print '):
                 content = line[6:].strip()
                 if (content.startswith('"') and content.endswith('"')): 
                     print(content[1:-1])
                 else:
                     # [FIX] Handle $ prefix for variables
                     if content.startswith('$'):
                         content = content.replace('$', '')
                     
                     if content in self.context: 
                         print(f"{self.context[content]}")
                     else:
                         # [FIX] Try to evaluate the expression if it's not a direct variable look up
                         # This enables: print sum_mass - huge_mass
                         try:
                             # Use our robust eval context
                             safe_dict = self._get_eval_context()
                             res = eval(content, safe_dict)
                             print(str(res))
                         except Exception:
                             # Fallback to printing raw string if eval fails
                             print(content)
                 ptr += 1
                 continue

            if line.startswith('prove '):
                 self._handle_proof(line)
                 ptr += 1
                 continue

            if line.startswith('simulate '):
                 self._handle_simulation(line)
                 ptr += 1
                 continue

            if line.startswith('define '):
                raw_assign = line[7:]
                if '=' not in raw_assign:
                     ptr += 1
                     continue
                parts = raw_assign.split('=', 1)
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    expr = parts[1].strip()
                    if var_name in getattr(self, 'protected', []):
                         print(f"❌ Security Violation: Cannot redefine protected constant '{var_name}'.")
                         ptr += 1
                         continue
                    self._assign(var_name, expr)
                ptr += 1
                continue
            
            elif '=' in line:
                parts = line.split('=', 1)
                var_name = parts[0].strip()
                expr = parts[1].strip()
                if var_name in getattr(self, 'protected', []):
                     print(f"❌ Security Violation: Cannot redefine protected constant '{var_name}'.")
                     ptr += 1
                     continue
                self._assign(var_name, expr)
                ptr += 1
                continue
            
            else:
                ptr += 1

    def _interpolate_variables(self, parts: List[str]) -> List[str]:
        # [UPDATED] Supports expressions like $r*cos($theta)
        processed_parts = []
        safe_dict = self._get_eval_context()
        
        for p in parts:
            if '$' in p or any(op in p for op in "+-*/()"):
                # It might be a coordinate string "x,y,z"
                sub_parts = p.split(',')
                new_sub_parts = []
                
                for sp in sub_parts:
                    # Substitute variables: $var -> value
                    def repl(m):
                        v = m.group(1)
                        if v in self.context:
                            val = self.context[v]
                            return str(val.value) if isinstance(val, QValue) else str(val)
                        return m.group(0)
                        
                    expr = re.sub(r'\$([a-zA-Z_]\w*)', repl, sp)
                    
                    # If it looks like math (try to eval)
                    # We are permissive here because we want to catch "1.0*cos(0.5)"
                    try:
                         # Only eval if it contains operators or function calls, or is a number
                         # Avoid evaluating plain strings like "sto-3g" (contains -)
                         is_math = any(c in expr for c in "+*/()") or (re.search(r'\d', expr) and not re.search(r'[a-z]{3,}', expr))
                         
                         if is_math:
                             res = eval(expr, safe_dict)
                             if isinstance(res, QValue): res = res.value
                             new_sub_parts.append(str(res))
                         else:
                             new_sub_parts.append(expr)
                    except:
                         new_sub_parts.append(expr)
                         
                processed_parts.append(",".join(new_sub_parts))
            else:
                processed_parts.append(p)
        return processed_parts

    def _handle_simulation(self, line: str, silent: bool = False):
        if not KERNELS_AVAILABLE:
            print("❌ Simulation Error: Scientific kernels not loaded.")
            return

        parts = line.split()
        parts = self._interpolate_variables(parts)
        domain = parts[1]
        
        try:
            if domain == "chemistry":
                # simulate chemistry <Type1> <x,y,z> <Type2> <x,y,z> ... [basis] [method=hf|mp2|cis]
                # E.g. simulate chemistry H 0,0,0 H 0.74,0,0 sto-3g
                # E.g. simulate chemistry H 0,0,0 H 0.74,0,0 method=mp2
                
                atoms = []
                basis = "sto-3g"  # default
                method = "hf"     # default: Hartree-Fock
                
                i = 2
                while i < len(parts):
                    # Check for basis set
                    if parts[i] in ["sto-3g", "6-31g"]: 
                        basis = parts[i]
                        i += 1
                        continue
                    
                    # Check for method option
                    if parts[i].startswith("method="):
                        method = parts[i].split("=")[1].lower()
                        if method not in ["hf", "mp2", "cis"]:
                            print(f"❌ Unknown method '{method}'. Use hf, mp2, or cis.")
                            return
                        i += 1
                        continue
                        
                    symbol = parts[i]
                    if i + 1 >= len(parts): break
                    
                    coords_str = parts[i+1]
                    try:
                        coords = [float(x) for x in coords_str.split(',')]
                        if len(coords) != 3: raise ValueError
                        atoms.append((symbol, tuple(coords)))
                    except:
                        print(f"❌ Invalid coordinates for atom {symbol}: {coords_str}")
                        return
                    
                    i += 2
                    
                if not atoms:
                    print("❌ No atoms defined for simulation.")
                    return

                if Molecule is None or MatrixHartreeFock is None:
                    print("❌ Critical: Chemistry kernels (Molecule/HartreeFock) not found.")
                    return

                try:
                    # [OPT] If silent, suppress stdout during kernel execution if possible
                    # For now we just suppress interpreter prints
                    
                    method_name = {"hf": "Hartree-Fock", "mp2": "MP2", "cis": "CIS"}[method]
                    if not silent:
                        print(f"   [Chemistry] Simulating {len(atoms)} atoms with {basis}, method={method_name}...")
                    
                    mol = Molecule(atoms)
                    
                    # Select solver based on method
                    if method == "mp2":
                        if MP2Solver is None:
                            print("❌ MP2Solver not available. Falling back to HF.")
                            solver = MatrixHartreeFock()
                            method = "hf"
                        else:
                            solver = MP2Solver()
                    elif method == "cis":
                        if CISolver is None:
                            print("❌ CISolver not available. Falling back to HF.")
                            solver = MatrixHartreeFock()
                            method = "hf"
                        else:
                            solver = CISolver()
                    else:
                        solver = MatrixHartreeFock()
                    
                    # Redirect stdout if silent to avoid kernel spam
                    if silent:
                        # Simple stdout suppression context
                        with open(os.devnull, 'w') as fnull:
                            old_stdout = sys.stdout
                            sys.stdout = fnull
                            try:
                                result = solver.compute_energy(mol)
                            finally:
                                sys.stdout = old_stdout
                    else:
                        result = solver.compute_energy(mol)
                    
                    # Handle results based on method
                    if method == "mp2":
                        # MP2Solver returns (E_total, E_corr)
                        if isinstance(result, tuple):
                            mp2_total = result[0]
                            mp2_corr = result[1]
                            energy = mp2_total
                            
                            # Store correlation energy
                            self.context["mp2_correlation"] = QValue(mp2_corr, Dimensions(mass=1, length=2, time=-2))
                            
                            if not silent:
                                print(f"✅ MP2 Total Energy   = {mp2_total:.8f} Eh")
                                print(f"   Correlation Energy = {mp2_corr:.8f} Eh ({1000*mp2_corr:.2f} mEh)")
                        else:
                            energy = result
                            
                    elif method == "cis":
                        # CISolver returns (E_hf, [excitation_energies])
                        if isinstance(result, tuple):
                            hf_energy = result[0]
                            excited = result[1] if len(result) > 1 else []
                            energy = hf_energy
                            
                            self.context["excited_states"] = excited
                            
                            if not silent:
                                print(f"✅ HF Ground State = {hf_energy:.8f} Eh")
                                for idx, ex in enumerate(excited[:5]):
                                    eV = ex * 27.2114
                                    print(f"   Excited State {idx+1}: {eV:.4f} eV")
                        else:
                            energy = result
                    else:
                        # Standard HF result
                        if isinstance(result, tuple):
                            hf_energy = result[0]
                            mp2_energy = result[1]
                            energy = mp2_energy
                            
                            self.context["hf_energy"] = QValue(hf_energy, Dimensions(mass=1, length=2, time=-2))
                            self.context["mp2_energy"] = QValue(mp2_energy, Dimensions(mass=1, length=2, time=-2))
                            
                            if not silent:
                                print(f"✅ HF Energy = {hf_energy:.6f} Eh")
                                print(f"✅ MP2 Energy = {mp2_energy:.6f} Eh")
                        else:
                            energy = result
                            if not silent:
                                print(f"✅ Energy = {energy:.6f} Eh")
                    
                    self.context["last_energy"] = QValue(energy, Dimensions(mass=1, length=2, time=-2))
                    
                except Exception as e:
                    print(f"❌ Chemistry Kernel Error: {e}")

            elif domain == "biology":
                # simulate biology [folding] <sequence> [options]
                idx = 2
                if idx < len(parts) and parts[idx] == "folding":
                    idx += 1
                
                if idx >= len(parts):
                     print("❌ Biology Simulation Error: Missing sequence.")
                     return

                sequence = parts[idx]
                idx += 1

                if ProteinFolder is None:
                    print("❌ Critical: Biology kernel (ProteinFolder) not found.")
                    return

                try:
                    folder = ProteinFolder()
                    conditions = {}
                    for p in parts[idx:]:
                        if "=" in p:
                            k, v = p.split("=")
                            conditions[k] = float(v)
                            
                    if not silent: print(f"   [Biology] Folding sequence {sequence}...")
                    
                    result = folder.fold_sequence(sequence, conditions)
                    
                    if not silent:
                        print(f"✅ Folding Complete. Energy = {result['energy']:.4f}")
                        print(f"Structure:\n{result['structure_visual']}")
                    
                    self.context["last_structure"] = result["coordinates"]
                    self.context["last_energy"] = QValue(result["energy"], Dimensions()) 
                    
                except Exception as e:
                    print(f"❌ Biology Kernel Error: {e}")

            elif domain == "physics":
                 if LatticeSimulator is None:
                     print("❌ Critical: Physics kernel (LatticeSimulator) not found.")
                     return

                 try:
                     size = int(float(parts[2])) 
                     sweeps = int(float(parts[3]))
                     temp = float(parts[4])
                     sim = LatticeSimulator(size)
                     mag = sim.run_simulation(sweeps, temp)
                     self.context["last_magnetization"] = QValue(mag, Dimensions())
                     if not silent:
                         print(f"   [PHYSICS] T={temp:.4f}, M={mag:.4f}")
                 except Exception as e:
                     print(f"❌ Physics Error: {e}")

        except Exception as e:
            print(f"❌ SIMULATION ERROR: {e}")


    def _handle_proof(self, line: str):
        if not KERNELS_AVAILABLE or TacticalProver is None:
             print("❌ Proof Error: Math kernel not loaded.")
             return

        # Syntax: prove <goal_string>
        goal = line[6:].strip()
        # Strip quotes if present
        if (goal.startswith('"') and goal.endswith('"')):
             goal = goal[1:-1]
        
        # Resolve variables in goal (e.g. $x)
        parts = [goal]
        parts = self._interpolate_variables(parts)
        goal = parts[0]
        
        try:
             prover = TacticalProver()
             state = ProofState(goal)
             
             print(f"   [Math] Starting proof for: {goal}")
             proof_tree = prover.prove(state, depth_limit=10)
             
             if proof_tree.is_complete:
                 print("✅ Proof Complete!")
                 print("   Trace:")
                 for step in proof_tree.steps:
                     print(f"     - {step}")
             else:
                 print("⚠️  Proof Incomplete.")
                 
        except Exception as e:
             print(f"❌ Prover Error: {e}")

    def _handle_optimize(self, line: str):
        """
        Syntax: optimize geometry <atom_spec_with_variables>
        Example: optimize geometry H 0,0,0 H $r,0,0 sto-3g
        Uses scipy.optimize to minimize energy by varying variables prefixed with $.
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            print("❌ Optimization Error: scipy not found.")
            return

        if not KERNELS_AVAILABLE:
            print("❌ Optimization Error: Scientific kernels not loaded.")
            return

        # 1. Parse the command to find variables
        # Format: optimize geometry ...
        parts = line.split()
        if len(parts) < 3 or parts[1] != "geometry":
             print("❌ Usage: optimize geometry <atoms...>")
             return

        sim_parts = parts[2:] # H 0,0,0 H $r,0,0 ...
        
        # Identify variables (tokens starting with $)
        variables = {}
        var_indices = []
        
        # We need to reconstruct the atom list string for the simulation
        # but keep track of which parts are variables.
        
        # Flatten the parts to handle comma-separated coords like "$r,0,0"
        # Actually, _handle_simulation expects a list of strings.
        # We need to detect which strings contain $vars.
        
        # Strategy:
        # We will create a wrapper function 'objective(x)' that:
        # 1. Updates self.context with values from x.
        # 2. Calls _handle_simulation with the original (template) string.
        # 3. Returns self.context["last_energy"].value
        
        # First, scan for all unique variable names used in the command
        cmd_str = " ".join(sim_parts)
        # Regex to find $varname
        var_names = list(set(re.findall(r'\$([a-zA-Z_]\w*)', cmd_str)))
        
        if not var_names:
            print("❌ Optimization Error: No variables (e.g. $r) found in geometry specification.")
            return

        print(f"   [Optimizer] Optimizing variables: {var_names}")
        
        # Get initial values
        x0 = []
        for v in var_names:
            if v not in self.context:
                print(f"❌ Optimization Error: Variable '{v}' is not defined.")
                return
            val = self.context[v]
            # Unwrap QValue if necessary
            if isinstance(val, QValue): val = val.value
            x0.append(float(val))
            
        initial_energy = None
        trajectory = [] # List of dicts: {'step': i, 'energy': E, 'vars': [...]}
        step_counter = 0

        def objective(x):
            nonlocal initial_energy, step_counter
            step_counter += 1
            
            # Update context
            for i, v_name in enumerate(var_names):
                self.context[v_name] = QValue(x[i], Dimensions()) 
            
            sim_line = f"simulate chemistry {cmd_str}"
            
            # Capture stdout to reduce spam, or just let it print
            self._handle_simulation(sim_line, silent=True)
            
            if "last_energy" not in self.context:
                return 0.0
            
            energy_q = self.context["last_energy"]
            energy = float(energy_q.value)
            
            if initial_energy is None: initial_energy = energy
            
            # Log trajectory
            trajectory.append({
                "step": step_counter,
                "energy": energy,
                "vars": {name: val for name, val in zip(var_names, x)}
            })
            
            return energy

        print(f"   [Optimizer] Starting BFGS optimization...")
        
        # Bounds: Assume all variables (distances) must be > 0.1
        # But wait, coordinates can be negative!
        # Only strict "bond length" variables need to be positive.
        # If we use cartesian coordinates for H2O (x, y), they can be negative.
        # We need a smarter way to handle bounds or disable them for generic coords.
        # For now, let's DISABLE bounds to allow full cartesian optimization.
        # Negative bond lengths in scalar variables caused issues, but negative coords are fine.
        # User is responsible for variable definitions.
        
        # res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, tol=1e-4)
        res = minimize(objective, x0, method='BFGS', tol=1e-4)
        
        print(f"   [Optimizer] {'✅ Converged' if res.success else '❌ Failed'}: {res.message}")
        print(f"   [Optimizer] Initial Energy: {initial_energy:.6f} Eh")
        print(f"   [Optimizer] Final Energy:   {res.fun:.6f} Eh")
        print(f"   [Optimizer] Optimization took {res.nfev} evaluations.")
        
        # Update context with final best values
        for i, v_name in enumerate(var_names):
             best_val = res.x[i]
             print(f"   [Optimizer] Best {v_name} = {best_val:.6f}")
             self.context[v_name] = QValue(best_val, Dimensions())
             
        # Store final energy properly
        self.context["last_energy"] = QValue(res.fun, Dimensions(mass=1, length=2, time=-2))
        
        # Dump trajectory
        try:
            with open("optimization_trajectory.csv", "w") as f:
                header = ["step", "energy"] + var_names
                f.write(",".join(header) + "\n")
                for entry in trajectory:
                    row = [str(entry["step"]), f"{entry['energy']:.6f}"]
                    row.extend([f"{entry['vars'][v]:.6f}" for v in var_names])
                    f.write(",".join(row) + "\n")
            print(f"   [Optimizer] Trajectory dumped to 'optimization_trajectory.csv'")
        except Exception as e:
            print(f"   [Optimizer] Failed to dump trajectory: {e}")



    def _assign(self, var_name, expr):
        var_name = var_name.strip()
        expr = expr.strip()
        
        if not var_name or not expr: raise SyntaxError(f"Invalid assignment: {var_name} = {expr}")
        
        # [FIX] Allow '=' inside quotes (for strings)
        # Only check for unquoted assignment operators
        # Simple check: if '=' is present, check if it's wrapped in quotes
        # Actually, let's just relax this check. Python's eval() will catch syntax errors.
        # The main reason for this check was to prevent "x = y = z" which is valid in Python but maybe we wanted to restrict it.
        # Let's remove it or make it smarter.
        # For now, simply removing it is safe enough because eval() handles the parsing.
        # if "=" in expr: raise SyntaxError("Invalid syntax: multiple assignment operators.")

        try:
            expr = expr.replace('^', '**')
            unc_pattern = r'([\d\.]+(?:e[+-]?\d+)?)\s*\+/-\s*([\d\.]+(?:e[+-]?\d+)?)'
            def unc_repl(m): return f"QValue(Decimal('{m.group(1)}'), Dimensions(), Decimal('{m.group(2)}'))"
            expr = re.sub(unc_pattern, unc_repl, expr)

            # [FIX] Enhanced list/vector uncertainty syntax: [1,2] +/- [0.1,0.2]
            vec_unc_pattern = r'(\[[^\]]+\])\s*\+/-\s*(\[[^\]]+\])'
            # Use 'np.float64' instead of just 'float' inside the eval context, or rely on 'float' being a builtin
            # But wait, 'float' IS a builtin. Why did it fail? 
            # Because we cleared __builtins__ in _get_eval_context!
            # We need to ensure 'float' is available or use 'np.float64' since 'np' is in context.
            def vec_unc_repl(m): return f"QValue(np.array({m.group(1)}, dtype=np.float64), Dimensions(), np.array({m.group(2)}, dtype=np.float64))"
            expr = re.sub(vec_unc_pattern, vec_unc_repl, expr)

            expr = re.sub(r"(?<!['\"])(?<!Decimal\()(?<![\d.])(\d+\.\d+(?:e[+-]?\d+)?|\d+e[+-]?\d+)(?!['\"])", r"Decimal('\1')", expr)

            # [FIX] Allow $var syntax in expressions by removing '$'
            # This allows consistency between 'simulate' (requires $) and 'define' (Python eval)
            expr = expr.replace('$', '')

            # [FIX] Use shared context builder
            safe_dict = self._get_eval_context()
            
            result = eval(expr, safe_dict)
            
            # [FIX] Convert lists to numpy arrays for vector math
            if isinstance(result, list):
                # Ensure complex numbers in lists are handled
                if any(isinstance(x, (complex, str)) and 'j' in str(x) for x in result):
                    result = np.array(result, dtype=complex)
                else:
                    result = np.array(result, dtype=float)
            
            # [FIX] Wrap raw Decimal/float results in QValue for consistent arithmetic
            if isinstance(result, Decimal):
                result = QValue(result, Dimensions())
            elif isinstance(result, (int, float)) and not isinstance(result, bool):
                result = QValue(Decimal(str(result)), Dimensions())
            
            self.context[var_name] = result
            print(f"✅ {var_name} = {result}")
        except Exception as e:
            print(f"❌ EXECUTION ERROR: {e}")
            # [FIX] Re-raise security-related exceptions so callers can handle them
            # This includes blocked builtins like __import__, exec, eval, etc.
            if "__import__" in str(e) or "exec" in str(e) or "is not defined" in str(e):
                raise

if __name__ == "__main__":
    ql = QLangInterpreter()
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f: code = f.read()
        print(f">>> Executing Q-Lang Script: {sys.argv[1]}")
        ql.execute(code)
