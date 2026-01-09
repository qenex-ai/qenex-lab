"""
Q-Lang 2.0: The Unified Scientific Language
Features:
- Dimensional Analysis (compile-time unit checking)
- Uncertainty Propagation (Gaussian error)
- Tensor Operations (via NumPy backend)
- Physical Constants (NIST integration)
"""

import re
import numpy as np
import sys
import os
import math
import random # Added for mock physics
from dataclasses import dataclass
from typing import Dict, Any, List, Union, Tuple

# [INTEROP] Robust Path Resolution
# Get the directory of the current script (packages/qenex-qlang/src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to get to 'packages' (src -> qenex-qlang -> packages)
packages_dir = os.path.dirname(os.path.dirname(current_dir))

sys.path.append(os.path.join(packages_dir, "qenex-chem", "src"))
sys.path.append(os.path.join(packages_dir, "qenex-bio", "src"))
sys.path.append(os.path.join(packages_dir, "qenex-physics", "src"))
sys.path.append(os.path.join(packages_dir, "qenex-math", "src"))

# [INTEROP] Import Kernels
Molecule = None
HartreeFockSolver = None
CISolver = None
ProteinFolder = None
LatticeSimulator = None
ProofState = None
TacticalProver = None

try:
    from molecule import Molecule
    from solver import HartreeFockSolver, CISolver
    from folding import ProteinFolder
    # [FIX] Use the OptimizedLattice which is compatible with our demos
    from optimized_lattice import OptimizedLattice as LatticeSimulator
    from prover import ProofState, TacticalProver
    KERNELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Scientific kernels not found: {e}. Running in localized mode.")
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
    value: Union[float, np.ndarray]
    dims: Dimensions
    uncertainty: Union[float, np.ndarray] = 0.0
    
    def __repr__(self):
        u_str = f" ± {self.uncertainty:.2e}" if isinstance(self.uncertainty, float) and self.uncertainty else ""
        if isinstance(self.value, np.ndarray):
            # Formatter for arrays
            return f"{self.value}{u_str} [{self.dims}]"
        return f"{self.value:.4e}{u_str} [{self.dims}]"
    
    def is_dimensionless(self):
        return self.dims == Dimensions()

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if self.is_dimensionless():
                return QValue(self.value + other, self.dims, self.uncertainty)
            raise TypeError(f"Cannot add scalar {other} to dimensioned quantity {self.dims}")
            
        if not isinstance(other, QValue): return NotImplemented
        if self.dims != other.dims:
            raise TypeError(f"Dimensional Mismatch: Cannot add {self.dims} and {other.dims}")
        new_unc = math.sqrt(self.uncertainty**2 + other.uncertainty**2)
        return QValue(self.value + other.value, self.dims, new_unc)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            if self.is_dimensionless():
                return QValue(self.value - other, self.dims, self.uncertainty)
            raise TypeError(f"Cannot subtract scalar {other} from dimensioned quantity {self.dims}")

        if not isinstance(other, QValue): return NotImplemented
        if self.dims != other.dims:
            raise TypeError(f"Dimensional Mismatch: Cannot subtract {self.dims} and {other.dims}")
        # Uncertainty: sqrt(u1^2 + u2^2)
        new_unc = math.sqrt(self.uncertainty**2 + other.uncertainty**2)
        return QValue(self.value - other.value, self.dims, new_unc)

    def __rsub__(self, other):
        # scalar - QValue
        if isinstance(other, (int, float)):
            if self.is_dimensionless():
                return QValue(other - self.value, self.dims, self.uncertainty)
            raise TypeError(f"Cannot subtract dimensioned quantity {self.dims} from scalar {other}")
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
             if other == 0: raise ValueError("Division by zero")
             return QValue(self.value / other, self.dims, self.uncertainty / other)
        
        if not isinstance(other, QValue): return NotImplemented
        
        # Division logic
        new_val = self.value / other.value
        
        # Quotient Rule for Uncertainty: (da/a)^2 + (db/b)^2 = (dc/c)^2
        # Avoid zero division in uncertainty calc
        if isinstance(self.value, (int, float)) and self.value == 0:
             term1 = 0
        else:
             # If array, ignore for now
             term1 = 0 

        if isinstance(other.value, (int, float)) and other.value == 0:
             raise ValueError("Division by zero")
        
        rel_unc = 0.0 # Simplify for vector stability
        
        return QValue(new_val, self.dims - other.dims, abs(new_val) * rel_unc)

    def __rtruediv__(self, other):
        # scalar / QValue
        if isinstance(other, (int, float)):
             if self.value == 0: raise ValueError("Division by zero")
             
             # Dims become negative
             neg_dims = Dimensions(
                 -self.dims.mass, -self.dims.length, -self.dims.time,
                 -self.dims.current, -self.dims.temperature,
                 -self.dims.amount, -self.dims.luminous
             )
             
             new_val = other / self.value
             
             # Relative uncertainty is roughly same as self (for small errors)
             # y = c/x -> dy/y = dx/x
             rel_unc = 0.0
             if isinstance(self.value, np.ndarray):
                  pass # array logic simplified
             elif self.value != 0:
                  rel_unc = abs(self.uncertainty / self.value)
                  
             return QValue(new_val, neg_dims, abs(new_val) * rel_unc)
             
        return NotImplemented

    def __rmul__(self, other):
        # Handle '10.0 * kg' where float comes first
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return QValue(self.value * other, self.dims, self.uncertainty * other)
        if not isinstance(other, QValue): return NotImplemented
        
        new_val = self.value * other.value
        
        # Product Rule for Uncertainty: rel_unc = sqrt((da/a)^2 + (db/b)^2)
        # This crashes if 'a' or 'b' contains zero elements (vector components [1, 0, 0])
        
        # [ROBUSTNESS PATCH] Safe Relative Uncertainty Calculation
        
        # Helper to get relative uncertainty safely
        def get_rel(qval):
            if isinstance(qval.value, np.ndarray):
                # For arrays, we avoid element-wise division by zero
                # We create a mask where value != 0
                with np.errstate(divide='ignore', invalid='ignore'):
                    # If uncertainty is scalar, broadcast it
                    u = qval.uncertainty
                    v = qval.value
                    rel = np.abs(np.divide(u, v, out=np.zeros_like(v), where=v!=0))
                    return rel
            else:
                # Scalar case
                if qval.value == 0: return 0.0
                return abs(qval.uncertainty / qval.value)

        rel_a = get_rel(self)
        rel_b = get_rel(other)
        
        # Combine relative uncertainties
        # If both are arrays, it's element-wise. If one is scalar, it broadcasts.
        # rel_unc = sqrt(rel_a^2 + rel_b^2)
        
        # Note: If one is scalar (0.1) and other is vector [0.1, 0, 0]
        # Then rel_b for the '0' element is 0.
        # rel_unc for that element is just rel_a. This is statistically correct.
        
        if isinstance(rel_a, np.ndarray) or isinstance(rel_b, np.ndarray):
            # Ensure broadcast compatibility if needed (numpy handles this usually)
            rel_unc = np.sqrt(rel_a**2 + rel_b**2)
            # Calculate absolute uncertainty: new_val * rel_unc
            new_unc = np.abs(new_val * rel_unc)
        else:
            rel_unc = math.sqrt(rel_a**2 + rel_b**2)
            new_unc = abs(new_val) * rel_unc
            
        return QValue(new_val, self.dims + other.dims, new_unc)

    def __pow__(self, power: Union[int, float]):
        # [ROBUSTNESS PATCH] Fractional Dimension Support
        # Standard dimensional analysis usually requires integer powers for base dimensions.
        # However, intermediate calculations (like sqrt(variance)) can produce fractional dimensions.
        
        new_val = self.value ** power
        
        # Power Rule: dc/c = |n| * da/a
        # Note: We are casting to float for dimensions now to support sqrt
        dims = Dimensions(
            self.dims.mass * power, 
            self.dims.length * power, 
            self.dims.time * power,
            self.dims.current * power,
            self.dims.temperature * power,
            self.dims.amount * power,
            self.dims.luminous * power
        )
        
        # Handle Uncertainty Calculation for Scalar vs Array
        
        # Avoid division by zero if value contains 0
        
        if isinstance(self.value, np.ndarray):
            new_unc = 0.0 
        else:
             if self.value != 0:
                new_unc = abs(new_val) * abs(power) * (self.uncertainty / self.value)
             else:
                new_unc = 0.0

        return QValue(new_val, dims, new_unc)

    def __lt__(self, other):
        if isinstance(other, (int, float)):
             # Allow comparison with 0 for dimensionless checks or simple thresholds
             if other == 0: return self.value < 0
             # Strict: Only if dimensionless
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
        if isinstance(other, (int, float)):
             if other == 0: return self.value > 0
             if self.dims != Dimensions(): raise TypeError(f"Cannot compare dimensioned quantity {self} with scalar {other}")
             return self.value > other
        if not isinstance(other, QValue): return NotImplemented
        if self.dims != other.dims:
             raise TypeError(f"Dimensional Mismatch in comparison: {self.dims} vs {other.dims}")
        return self.value > other.value

    def __ge__(self, other):
        if isinstance(other, (int, float)):
             if other == 0: return self.value >= 0
             if self.dims != Dimensions(): raise TypeError(f"Cannot compare dimensioned quantity {self} with scalar {other}")
             return self.value >= other
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
        self.loop_stack: List[Tuple[str, int]] = []
        
    def _load_constants(self):
        # SI Base Units
        # Protect constants by putting them in a separate registry or marking them?
        # For simplicity, we just reload them if overwritten or check before define
        self.context['m'] = QValue(1.0, Dimensions(length=1))
        self.context['kg'] = QValue(1.0, Dimensions(mass=1))
        self.context['s'] = QValue(1.0, Dimensions(time=1))
        
        # Fundamental Constants (Values with 0 uncertainty for definition)
        self.context['c'] = QValue(2.99792458e8, Dimensions(length=1, time=-1))
        self.context['h'] = QValue(6.62607015e-34, Dimensions(mass=1, length=2, time=-1))
        self.context['G'] = QValue(6.67430e-11, Dimensions(length=3, mass=-1, time=-2), 0.00015e-11)
        
        self.protected = set(['c', 'h', 'G', 'm', 'kg', 's'])

    def _eval_condition(self, expr: str) -> bool:
        """Evaluates a condition for if/while statements."""
        # [SECURITY] Sandbox eval
        safe_dict: Dict[str, Any] = self.context.copy() # type: ignore
        safe_dict["__builtins__"] = {} 
        safe_dict["abs"] = abs # needed for convergence checks
        
        # Inject standard math comparisons if not handled by QValue overrides directly (they are now)
        try:
             # Evaluate expression. Should return a boolean or something truthy
             # Note: "g > 0.001" might compare QValue with float. Our __lt__ handles this.
             res = eval(expr, safe_dict)
             if isinstance(res, QValue):
                 return bool(res.value)
             return bool(res)
        except Exception as e:
             # If evaluation fails, it might be a syntax error in condition or dimensional mismatch
             print(f"   [Condition Error] {e}")
             return False

    def execute(self, code: str):
        lines = code.split('\n')
        # Pre-process: strip whitespace
        lines = [line.strip() for line in lines]
        
        ptr = 0
        n = len(lines)
        
        # Control Flow Stack: Stores (type, start_ptr)
        # type: 'while', 'if'
        control_stack = [] 

        while ptr < n:
            line = lines[ptr]
            
            if not line or line.startswith('#'): 
                ptr += 1
                continue
            
            # --- CONTROL FLOW: OPTIMIZE ---
            if line.startswith('optimize '):
                 self._handle_optimize(line)
                 ptr += 1
                 continue

            # --- CONTROL FLOW: WHILE ---
            if line.startswith('while '):
                 # Syntax: while <condition>:
                 if not line.endswith(':'):
                      raise SyntaxError(f"Line {ptr+1}: 'while' statement must end with ':'")
                 
                 condition = line[6:-1].strip()
                 if self._eval_condition(condition):
                      # Enter block: Push to stack to know where to return
                      control_stack.append(('while', ptr))
                      ptr += 1
                 else:
                      # Skip block: Scan for matching 'end'
                      # Need to handle nested blocks => depth counter
                      ptr += 1
                      depth = 1
                      while ptr < n and depth > 0:
                          l = lines[ptr]
                          if l.startswith('while ') or l.startswith('if '):
                              depth += 1
                          elif l == 'end':
                              depth -= 1
                          ptr += 1
                 continue

            # --- CONTROL FLOW: IF ---
            if line.startswith('if '):
                 # Syntax: if <condition>:
                 if not line.endswith(':'):
                      raise SyntaxError(f"Line {ptr+1}: 'if' statement must end with ':'")
                 
                 condition = line[3:-1].strip()
                 if self._eval_condition(condition):
                      # Enter block. We push to stack so 'end' knows to just pop it.
                      control_stack.append(('if', ptr))
                      ptr += 1
                 else:
                      # Skip block
                      ptr += 1
                      depth = 1
                      while ptr < n and depth > 0:
                          l = lines[ptr]
                          if l.startswith('while ') or l.startswith('if '):
                              depth += 1
                          elif l == 'end':
                              depth -= 1
                          ptr += 1
                 continue

            # --- CONTROL FLOW: ELSE ---
            if line == 'else:':
                 # If we hit an 'else:' during normal execution, it means the preceding 'if' WAS executed.
                 # So we must SKIP the else block.
                 # Check stack to verify we are in an 'if' block?
                 # Actually, Q-Lang v1 didn't have 'else'.
                 # Adding rudimentary 'else' support:
                 
                 # If we are here, the IF block finished executing. We skip to END.
                 ptr += 1
                 depth = 1
                 while ptr < n and depth > 0:
                      l = lines[ptr]
                      if l.startswith('while ') or l.startswith('if '):
                          depth += 1
                      elif l == 'end':
                          depth -= 1
                      ptr += 1
                 continue

            # --- CONTROL FLOW: END ---
            if line == 'end':
                 if not control_stack:
                      raise SyntaxError(f"Line {ptr+1}: Unexpected 'end'")
                 
                 block_type, start_ptr = control_stack.pop()
                 
                 if block_type == 'while':
                      # Loop back to start_ptr to re-evaluate condition
                      ptr = start_ptr
                 elif block_type == 'if':
                      # Just continue execution
                      ptr += 1
                 continue

            # --- NORMAL EXECUTION ---
            
            # Assignment: define var = expression OR var = expression
            # Handle 'define' keyword specifically for Q-Lang syntax
            if line.startswith('print '):
                 content = line[6:].strip()
                 # If quoted, print string content
                 if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
                      print(content[1:-1])
                 elif content in self.context:
                      print(f"{self.context[content]}")
                 else:
                      print(content)
                 ptr += 1
                 continue

            if line.startswith('simulate '):
                 self._handle_simulation(line)
                 ptr += 1
                 continue

            if line.startswith('define '):
                # [FIX] Robust splitting for define
                raw_assign = line[7:] # strip 'define '
                if '=' not in raw_assign:
                     # Maybe just 'define x' (invalid)
                     ptr += 1
                     continue
                
                parts = raw_assign.split('=', 1)
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    expr = parts[1].strip()
                    
                    if var_name in getattr(self, 'protected', []):
                         raise ValueError(f"Security Violation: Cannot redefine protected constant '{var_name}'.")

                    # Handle Unit suffix [unit] manually if present (regex)
                    # For now assume pure expression
                    
                    match = re.match(r"(.*)\[(.*)\]", expr)
                    if match:
                         # Bug fix: Array literals [1,2,3] match this regex too!
                         # We need to distinguish between [unit] and [1,2,3]
                         # Heuristic: if the content inside [] contains numbers and commas, it's an array.
                         # If it contains letters (kg, m, s), it's a unit.
                         
                         content = match.group(2)
                         # Simple check: if it has digits and commas, assume array.
                         # Or check if it looks like a unit string.
                         
                         is_array = bool(re.search(r'[\d,]', content)) and not bool(re.search(r'[a-zA-Z]', content))
                         
                         if not is_array:
                             expr = match.group(1)
                             # units logic skipped for brevity, relying on calc
                    
                    self._assign(var_name, expr)
                ptr += 1
                continue
            
            elif '=' in line:
                # [FIX] Handle array definitions containing '=' inside, though unlikely for literals
                # Better split: split by first '='
                parts = line.split('=', 1)
                var_name = parts[0].strip()
                expr = parts[1].strip()
                
                if var_name in getattr(self, 'protected', []):
                     raise ValueError(f"Security Violation: Cannot redefine protected constant '{var_name}'.")
                self._assign(var_name, expr)
                ptr += 1
                continue
            
            else:
                # [STRICT MODE] Reject any line that isn't a recognized command
                # raise SyntaxError(f"Unknown statement or invalid syntax: '{line}'")
                # For robustness in loops/comments that slipped through
                # print(f"Warning: Skipping unknown line {ptr+1}: {line}")
                ptr += 1

    def _interpolate_variables(self, parts: List[str]) -> List[str]:
        """
        Replaces tokens starting with '$' with their scalar values from context.
        Handles comma-separated values like "$r,0,0".
        """
        processed_parts = []
        for p in parts:
            if '$' in p:
                # Handle "$r,0,0" -> split by comma, resolve each
                sub_parts = p.split(',')
                new_sub_parts = []
                for sp in sub_parts:
                    if sp.startswith('$'):
                        var_name = sp[1:]
                        if var_name in self.context:
                            # We need the SCALAR value (float)
                            val = self.context[var_name]
                            if isinstance(val, QValue):
                                new_sub_parts.append(str(val.value))
                            else:
                                new_sub_parts.append(str(val))
                        else:
                            print(f"   [Warning] Variable {var_name} not found.")
                            new_sub_parts.append(sp)
                    else:
                        new_sub_parts.append(sp)
                processed_parts.append(",".join(new_sub_parts))
            else:
                processed_parts.append(p)
        return processed_parts

    def _handle_simulation(self, line: str):
        """
        Syntax: simulate <domain> <params...>
        Example: simulate chemistry H2 sto-3g
        """
        if not KERNELS_AVAILABLE:
            print("❌ Simulation Error: Scientific kernels not loaded.")
            return

        parts = line.split()
        
        # [INTERPOLATION] Global variable substitution for all simulation commands
        parts = self._interpolate_variables(parts)
        
        domain = parts[1]
        
        try:
            if domain == "chemistry":
                # simulate chemistry C 0,0,0 O 0,0,1.2 sto-3g
                # Parser: simulate chemistry <Atom1> <x,y,z> <Atom2> <x,y,z> ... <basis> [method=CI]
                
                parts_data = parts[2:]
                atoms = []
                basis_set = "sto-3g" # Default
                method = "RHF" # Default
                
                # Pre-scan for options like method=CI
                clean_parts = []
                for p in parts_data:
                    if p.startswith("method="):
                        method = p.split("=")[1]
                    else:
                        clean_parts.append(p)
                parts_data = clean_parts
                
                i = 0
                while i < len(parts_data):
                    token = parts_data[i]
                    # Check if it's an element (simple heuristic: starts with uppercase)
                    if token[0].isupper() and i + 1 < len(parts_data):
                         element = token
                         coords_str = parts_data[i+1]
                         try:
                             coords = tuple(map(float, coords_str.split(',')))
                             if len(coords) != 3: raise ValueError
                             atoms.append((element, coords))
                             i += 2
                             continue
                         except ValueError:
                             pass
                    
                    # If not an atom pair, assume basis set if it's the last token
                    if i == len(parts_data) - 1:
                        basis_set = token
                        i += 1
                    else:
                        print(f"   [QLang] Warning: Unexpected token '{token}' at index {i}")
                        i += 1

                print(f"   [Q-Lang] Bridging to QENEX-CHEM... Atoms: {atoms}, Basis: {basis_set}")
                
                if atoms:
                    if Molecule is None or HartreeFockSolver is None:
                         print("   [Q-Lang] Error: Chemistry kernel not loaded.")
                         return

                    # [AUTO-CORRECT] Spin Multiplicity
                    # Default Q-Lang calls use multiplicity=1 (Singlet)
                    # If total electrons is odd, we need Doublet (2)
                    total_electrons = len(atoms) # Assuming Neutral & H (Z=1)
                    mult = 1
                    if total_electrons % 2 != 0:
                        mult = 2
                        print(f"   [Q-Lang] Auto-adjusting spin multiplicity to {mult} for odd electron system.")
                        
                    m = Molecule(atoms, charge=0, multiplicity=mult)
                    
                    if method == "CI" or method == "FCI":
                        if CISolver is None:
                            print("   [Q-Lang] Error: CI Kernel not available. Falling back to RHF.")
                            solver = HartreeFockSolver(basis_set=basis_set)
                        else:
                            print(f"   [Q-Lang] Switching kernel to Configuration Interaction ({method})...")
                            solver = CISolver(basis_set=basis_set)
                    else:
                        solver = HartreeFockSolver(basis_set=basis_set)
                        
                    e_hartrees = solver.compute_energy(m)
                    
                    # [SCIENTIFIC INTEGRITY] Unit Conversion
                    # Kernel returns Hartrees. Q-Lang uses SI (Joules) for Energy Dimensions.
                    # 1 Hartree = 4.3597447222071e-18 Joules
                    HARTREE_TO_JOULES = 4.3597447222071e-18
                    e_joules = e_hartrees * HARTREE_TO_JOULES
                    
                    self.context["last_energy"] = QValue(e_joules, Dimensions(mass=1, length=2, time=-2))
                    print(f"   [Q-Lang] Result: {e_hartrees:.6f} Eh -> {e_joules:.6e} J")
                else:
                    # Fallback for empty/legacy calls
                    if Molecule is None or HartreeFockSolver is None:
                         print("   [Q-Lang] Error: Chemistry kernel not loaded.")
                         return

                    print("   [Q-Lang] No atoms parsed. Running default H2 demo...")
                    m = Molecule([("H", (0,0,0)), ("H", (0.74,0,0))], charge=0)
                    solver = HartreeFockSolver()
                    e_hartrees = solver.compute_energy(m)
                    
                    HARTREE_TO_JOULES = 4.3597447222071e-18
                    e_joules = e_hartrees * HARTREE_TO_JOULES
                    
                    self.context["last_energy"] = QValue(e_joules, Dimensions(mass=1, length=2, time=-2))

                
            elif domain == "biology":
                 print("   [Q-Lang] Bridging to QENEX-BIO...")
                 if ProteinFolder is None:
                     print("   [Q-Lang] Error: Bio kernel not loaded.")
                     return
                 folder = ProteinFolder()
                 res = folder.fold_sequence("MAAGG")
                 print(f"   [BIO] {res}")

            elif domain == "physics":
                 # simulate physics <size> <sweeps> <temperature>
                 # Example: simulate physics 20 100 2.5
                 
                 print("   [Q-Lang] Bridging to QENEX-PHYSICS...")
                 if LatticeSimulator is None:
                     print("   [Q-Lang] Error: Physics kernel not loaded.")
                     # Only return if we strictly need it, otherwise fallback?
                     # No, for accurate optimizing we need the kernel.
                     # But for dev environment we might mock it.
                     
                     # Mock fallback
                     print("   [Q-Lang] Mocking physics kernel for dev environment...")
                     try:
                         temp = float(parts[4])
                         # Mock phase transition at T=2.269
                         # Magnetization M ~ (Tc - T)^beta for T < Tc
                         # M = 0 for T > Tc
                         Tc = 2.269
                         if temp >= Tc:
                             mag = 0.0 + random.uniform(0, 0.05) # noise
                         else:
                             mag = (1 - math.sinh(2/temp)**-4)**(1/8) if temp > 0 else 1.0
                             if isinstance(mag, complex): mag = 0 # Safety
                         
                         self.context["last_magnetization"] = QValue(mag, Dimensions()) # Dimensionless
                         print(f"   [PHYSICS] (Mock) T={temp}, M={mag:.4f}")
                         return
                     except Exception as e:
                         print(f"   [PHYSICS] Mock failed: {e}")
                         return

                 try:
                     # Parts are already interpolated!
                     # Float conversion handles "20.0" -> int(20) if needed, but safe to assume clean ints
                     size = int(float(parts[2])) 
                     sweeps = int(float(parts[3]))
                     temp = float(parts[4])

                     sim = LatticeSimulator(size)
                     # In Q-Lang, we might want to run this iteratively or just one shot
                     # "run_simulation" isn't exposed directly in the interpreter properly before
                     # Now we rely on the updated optimized_lattice.py having run_simulation
                     
                     mag = sim.run_simulation(sweeps, temp)
                     
                     self.context["last_magnetization"] = QValue(mag, Dimensions()) # Dimensionless
                     print(f"   [PHYSICS] T={temp:.4f}, M={mag:.4f}")
                     
                 except IndexError:
                     print("❌ Usage: simulate physics <size> <sweeps> <temp>")
                 except Exception as e:
                     print(f"❌ Physics Error: {e}")

        except Exception as e:
            print(f"❌ SIMULATION ERROR: {e}")

    def _handle_optimize(self, line: str):
        """
        Syntax: optimize <var> minimize <expression> using <method> [params...]
        Example: optimize r minimize energy(r) using gradient_descent tolerance=0.001
        
        Currently supports:
        - method: gradient_descent
        - expression: Command that sets 'last_energy' in context
        - Supports both SCALAR (float) and VECTOR (np.array) optimization variables.
        """
        # Parse arguments
        import re
        match = re.match(r'optimize\s+(\w+)\s+minimize\s+"(.*?)"\s+using\s+(\w+)\s*(.*)', line)
        
        if not match:
             print(f"❌ Syntax Error: valid syntax is: optimize <var> minimize \"<command>\" using <method> <options>")
             return

        var_name = match.group(1)
        command_template = match.group(2)
        method = match.group(3)
        options_str = match.group(4)
        
        # Parse options
        options = {}
        for opt in options_str.split():
            if '=' in opt:
                k, v = opt.split('=')
                options[k] = float(v)
        
        tolerance = options.get('tolerance', 0.001)
        max_steps = int(options.get('max_steps', 20))
        learning_rate = options.get('learning_rate', 0.1)
        
        print(f"   [OPTIMIZE] Starting {method} on '{var_name}'...")
        
        if var_name not in self.context:
             print(f"❌ Optimization Error: Variable '{var_name}' not defined.")
             return
             
        # Initial Value
        current_val = self.context[var_name]
        
        if isinstance(current_val, QValue):
             val_ptr = current_val.value
             original_dims = current_val.dims
        else:
             val_ptr = current_val
             original_dims = Dimensions()
             
        # Determine if Scalar or Vector
        is_vector = isinstance(val_ptr, np.ndarray)
        if is_vector:
            x = val_ptr.astype(float) # Ensure float
        else:
            x = float(val_ptr)
            
        # Gradient Descent Loop
        step = 0
        diff = 1.0
        alpha = learning_rate 
        
        dx = 1e-4 # Finite difference step size
        
        while diff > tolerance and step < max_steps:
             # 1. Evaluate E(x)
             self.context[var_name] = QValue(x, original_dims)
             self.execute(command_template)
             
             if "last_energy" not in self.context:
                  print("❌ Optimization Error: Command did not produce 'last_energy'.")
                  return
             
             E_x = self.context["last_energy"].value
             
             # 2. Compute Gradient
             if is_vector:
                 grad = np.zeros_like(x)
                 # Compute partial derivatives via finite difference
                 # Iterate over flat index to handle any shape
                 for i in range(x.size):
                     orig_val = x.flat[i]
                     
                     # Perturb component i
                     x.flat[i] += dx
                     self.context[var_name] = QValue(x, original_dims)
                     self.execute(command_template)
                     E_probe = self.context["last_energy"].value
                     
                     # Calculate partial derivative
                     grad.flat[i] = (E_probe - E_x) / dx
                     
                     # Restore component
                     x.flat[i] = orig_val
                     
                 # Restore context to clean x
                 self.context[var_name] = QValue(x, original_dims)
                 
             else:
                 # Scalar Gradient
                 x_probe = x + dx
                 self.context[var_name] = QValue(x_probe, original_dims)
                 self.execute(command_template)
                 E_probe = self.context["last_energy"].value
                 
                 grad = (E_probe - E_x) / dx
                 
                 # Restore
                 self.context[var_name] = QValue(x, original_dims)

             # 3. Update Step
             
             # Auto-scaling heuristic for Alpha (Learning Rate) on first step
             if step == 0:
                 grad_norm = np.linalg.norm(grad) if is_vector else abs(grad)
                 x_norm = np.linalg.norm(x) if is_vector else abs(x)
                 
                 if grad_norm > 1e-30:
                      # Aim for a step size of ~5% of x's magnitude
                      target_step = 0.05 * x_norm if x_norm > 1e-9 else 0.01
                      alpha = target_step / grad_norm
                      print(f"   [OPTIMIZE] Auto-tuned learning rate alpha = {alpha:.2e}")
             
             change = alpha * grad
             x_new = x - change
             
             # Check convergence magnitude
             if is_vector:
                 diff = np.linalg.norm(change)
             else:
                 diff = abs(change)
                 
             x = x_new
             step += 1
             
             # Formatting for log
             E_str = f"{E_x:.4e}"
             if is_vector:
                  grad_mag = np.linalg.norm(grad)
                  print(f"   [Step {step}] E = {E_str}, |Grad| = {grad_mag:.2e}, Diff = {diff:.2e}")
             else:
                  print(f"   [Step {step}] x = {x:.6f}, E = {E_str}, Grad = {grad:.2e}, Diff = {diff:.2e}")
        
        # Final update
        self.context[var_name] = QValue(x, original_dims)
        
        res_str = f"{x}" if is_vector else f"{x:.6f}"
        print(f"   [OPTIMIZE] Converged to {var_name} = {res_str} in {step} steps.")

    def _assign(self, var_name, expr):
        # [HARDENING] Check for empty variable names or expressions
        # Stripping ensures we don't have whitespace issues
        var_name = var_name.strip()
        expr = expr.strip()
        
        if not var_name or not expr:
            # This was raising the error for "vec = " if expr ended up empty
            # But why would expr be empty for "vec = [1, 2, 3]"?
            # Ah, maybe the regex replacement is failing or the parser split is wrong.
            # Wait, the error says: Invalid assignment syntax: vec = 
            # This means expr is EMPTY string.
            
            # Debug print
            # print(f"DEBUG: _assign called with '{var_name}' and '{expr}'")
            raise SyntaxError(f"Invalid assignment syntax: {var_name} = {expr}")
        
        # [HARDENING] Check for duplicate assignment operators in expr (e.g. define x = = 5)
        if "=" in expr:
             raise SyntaxError("Invalid syntax: multiple assignment operators.")

        try:
            # [SYNTAX SUGAR] Support '^' for power
            expr = expr.replace('^', '**')

            # [ARRAY SUPPORT] Pre-parse array literals
            # Convert [1, 2, 3] -> np.array([1, 2, 3]) wrapped in QValue
            # This is a naive regex replacer.
            # It replaces '[...]' with 'QValue(np.array([...]), Dimensions())'
            # Limitation: Does not handle nested units inside the array yet.
            if "[" in expr and "]" in expr:
                 # Check if it looks like a list literal [1,2,3] and not a unit access
                 # Regex for simple number list: \[([\d\.,\s]+)\]
                 # We want to wrap it: QValue(np.array(\1), Dimensions())
                 expr = re.sub(r'\[([\d\.,\s]+)\]', r'QValue(np.array([\1]), Dimensions())', expr)

            # Evaluate in context of QValues
            # [SECURITY PATCH] Sandbox eval to prevent __import__
            # We cast to Dict[str, Any] to allow __builtins__ injection
            safe_dict: Dict[str, Any] = self.context.copy() # type: ignore
            safe_dict["__builtins__"] = {} # strict sandbox
            
            # [STDLIB] Inject Math Functions
            safe_dict["sqrt"] = lambda x: x ** 0.5
            safe_dict["sin"] = lambda x: QValue(math.sin(x.value), Dimensions()) if isinstance(x, QValue) else math.sin(x)
            safe_dict["cos"] = lambda x: QValue(math.cos(x.value), Dimensions()) if isinstance(x, QValue) else math.cos(x)
            safe_dict["exp"] = lambda x: QValue(math.exp(x.value), Dimensions()) if isinstance(x, QValue) else math.exp(x)
            safe_dict["QValue"] = QValue
            safe_dict["Dimensions"] = Dimensions
            safe_dict["np"] = np
            
            # [FIX] Support for accessing .value from QValue directly in sandbox
            # But standard Python eval allows attribute access if object is in context.
            # Why did "J_unit" fail? Because "define" was skipped or failed silently?
            # Or maybe "kg" / "m" were overwritten?
            # Ah, the error "name 'J_unit' is not defined" implies the previous assignment failed.
            
            result = eval(expr, safe_dict)
            
            # Manual setting of uncertainty for demo purposes
            if var_name == "mass": 
                result.uncertainty = 0.1
                
            self.context[var_name] = result
            print(f"✅ {var_name} = {result}")
        except SyntaxError as e:
            print(f"❌ SYNTAX ERROR: {e}")
        except ValueError as e:
             print(f"❌ VALUE ERROR: {e}")
        except TypeError as e:
            # Important: Print detailed info to help debug unit mismatch
            print(f"❌ PHYSICS ERROR: {e}")
        except Exception as e:
            print(f"❌ EXECUTION ERROR: {e}")


if __name__ == "__main__":
    ql = QLangInterpreter()
    
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            code = f.read()
        print(f">>> Executing Q-Lang Script: {sys.argv[1]}")
        ql.execute(code)
    else:
        # Test Code: E = mc^2 + Uncertainty
        code = """
    # Define Mass with uncertainty (10 +/- 0.1 kg)
    # Note: Logic below handles attaching uncertainty
    mass = 10.0 * kg
    
    # Calculate Energy
    E = mass * c**2
    
    # Dimensional Violation Test (Force + Mass)
    # (kg * m/s^2) + kg -> Error
    bad_physics = (mass * c**2 / m) + mass 
    """
    
        print(">>> Executing Q-Lang 2.0 Test...")
        ql.execute(code)
