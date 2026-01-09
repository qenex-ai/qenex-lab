
import sys
import os
import numpy as np

# Add all package sources
sys.path.append(os.path.abspath("packages/qenex-physics/src"))
sys.path.append(os.path.abspath("packages/qenex-chem/src"))
sys.path.append(os.path.abspath("packages/qenex-bio/src"))
sys.path.append(os.path.abspath("packages/qenex-math/src"))
sys.path.append(os.path.abspath("packages/qenex-qlang/src"))

from lattice import LatticeSimulator
from molecule import Molecule
from solver import HartreeFockSolver
from genomics import CRISPRAnalyzer
from folding import ProteinFolder
from prover import TacticalProver, ProofState
from interpreter import QLangInterpreter

print(">>> INITIATING QENEX 'RUTHLESS' CHAOS SUITE")
print("-------------------------------------------------------------")

failures = 0

def log_result(round_id, name, success, msg):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"[{round_id}] {name.ljust(30)}: {status} - {msg}")
    return 0 if success else 1

# --- PHYSICS ---

# R1: Time Reversal (Negative Steps)
try:
    print("...R1...")
    sim = LatticeSimulator(dimensions=2, size=10)
    sim.run_simulation(steps=-100, temperature=1.0)
    failures += log_result(1, "Time Reversal", False, "Allowed negative simulation steps")
except ValueError:
    failures += log_result(1, "Time Reversal", True, "Caught negative steps")
except Exception as e:
    failures += log_result(1, "Time Reversal", False, f"Crashed: {e}")

# R2: Trivial Lattice (Size < 2)
try:
    print("...R2...")
    sim = LatticeSimulator(dimensions=2, size=1)
    failures += log_result(2, "Trivial Lattice", False, "Allowed size=1 (No connectivity)")
except ValueError:
    failures += log_result(2, "Trivial Lattice", True, "Caught invalid size")
except Exception as e:
    failures += log_result(2, "Trivial Lattice", False, f"Crashed: {e}")

# --- CHEMISTRY ---

# R3: Invalid Element
try:
    print("...R3...")
    # 'X' is not a real element
    mol = Molecule(atoms=[("X", (0,0,0))])
    # Check if constructor or solver catches it
    solver = HartreeFockSolver()
    solver.compute_energy(mol)
    failures += log_result(3, "Alchemy (Fake Element)", False, "Accepted element 'X'")
except ValueError:
    failures += log_result(3, "Alchemy (Fake Element)", True, "Caught fake element")
except Exception as e:
    failures += log_result(3, "Alchemy (Fake Element)", False, f"Crashed: {e}")

# R4: Spin/Charge Parity Mismatch
try:
    print("...R4...")
    # 2 H atoms (2 electrons). Charge 0. Total Electrons = 2 (Even).
    # Multiplicity 2 (Doublet) requires unpaired electron -> Odd total count.
    # Formula: Multiplicity = 2S + 1. If N_e is Even, S is Integer, M is Odd.
    # If N_e is Odd, S is Half-Integer, M is Even.
    mol = Molecule(atoms=[("H", (0,0,0)), ("H", (0,0,1))], multiplicity=2)
    solver = HartreeFockSolver()
    solver.compute_energy(mol)
    failures += log_result(4, "Spin Parity Mismatch", False, "Accepted Even e- / Even Multiplicity")
except ValueError:
    failures += log_result(4, "Spin Parity Mismatch", True, "Caught spin parity error")
except Exception as e:
    failures += log_result(4, "Spin Parity Mismatch", False, f"Crashed: {e}")

# --- BIOLOGY ---

# R5: Self-Complementarity (Hairpin)
try:
    print("...R5...")
    analyzer = CRISPRAnalyzer()
    # A guide that binds to itself perfectly: GGGGG...CCCCC
    # This renders it useless in vivo.
    hairpin_guide = "GGGGGGGGGCCCCCCCCCC" 
    target = "GGGGGGGGGCCCCCCCCCC"
    # Should warn or fail
    analyzer.calculate_off_target_score(hairpin_guide, target)
    # We are looking for a specific check for low complexity or secondary structure risk
    # Currently it will just run.
    failures += log_result(5, "RNA Secondary Structure", False, "Allowed hairpin guide")
except ValueError:
    failures += log_result(5, "RNA Secondary Structure", True, "Caught hairpin structure")
except Exception as e:
    failures += log_result(5, "RNA Secondary Structure", False, f"Crashed: {e}")

# R6: Stop Codon Injection
try:
    print("...R6...")
    folder = ProteinFolder()
    # '*' represents a stop codon, cannot fold a truncated protein in mid-sequence
    folder.fold_sequence("MET*ALA")
    failures += log_result(6, "Stop Codon Injection", False, "Accepted internal stop codon")
except ValueError:
    failures += log_result(6, "Stop Codon Injection", True, "Caught stop codon")
except Exception as e:
    failures += log_result(6, "Stop Codon Injection", False, f"Crashed: {e}")

# --- MATH ---

# R7: Non-String Goal Injection
try:
    print("...R7...")
    prover = TacticalProver()
    # Passing an integer instead of a proposition string
    state = ProofState(12345)
    prover.prove(state, depth_limit=1)
    failures += log_result(7, "Type Injection (Goal)", False, "Accepted integer as goal")
except TypeError:
    failures += log_result(7, "Type Injection (Goal)", True, "Caught type error")
except ValueError:
    failures += log_result(7, "Type Injection (Goal)", True, "Caught value error")
except Exception as e:
    failures += log_result(7, "Type Injection (Goal)", False, f"Crashed: {e}")

# --- Q-LANG ---

# R8: Base Constant Overwrite (Reality Hacking)
try:
    print("...R8...")
    ql = QLangInterpreter()
    # Try to redefine the speed of light
    ql.execute("define c = 10 [m/s]")
    if ql.context['c'].value == 10.0:
        failures += log_result(8, "Constant Overwrite", False, "Allowed redefinition of 'c'")
    else:
        failures += log_result(8, "Constant Overwrite", True, "Protected fundamental constant")
except Exception as e:
    failures += log_result(8, "Constant Overwrite", True, f"Caught via Exception: {e}")

# R9: Code Injection (Security)
try:
    print("...R9...")
    ql = QLangInterpreter()
    # Try to access python builtins
    ql.execute("define hack = __import__('os').name")
    if 'hack' in ql.context:
        failures += log_result(9, "Code Injection", False, "Executed __import__")
    else:
        failures += log_result(9, "Code Injection", True, "Blocked builtin access")
except Exception as e:
    # If it raised SyntaxError or NameError, that's good
    if "restricted" in str(e) or "name" in str(e):
         failures += log_result(9, "Code Injection", True, f"Blocked: {e}")
    else:
         failures += log_result(9, "Code Injection", False, f"Crashed: {e}")

# R10: Dimensional Power Fractional
try:
    print("...R10...")
    ql = QLangInterpreter()
    # Raising a dimension to a non-integer power (e.g. sqrt(kg)) is valid in physics
    # BUT our Dimensions class uses 'int'. This should fail or be handled safely.
    ql.execute("define root_mass = kg ** 0.5")
    # If it didn't crash, check if dimensions are int
    if 'root_mass' in ql.context:
        d = ql.context['root_mass'].dims
        if isinstance(d.mass, int):
             # It likely rounded 0.5 -> 0 or crash logic
             failures += log_result(10, "Fractional Dimension", False, f"Result dim: {d.mass} (loss of precision?)")
        else:
             failures += log_result(10, "Fractional Dimension", True, "Handled float dimension?")
except TypeError:
     failures += log_result(10, "Fractional Dimension", True, "Caught non-integer dimension")
except Exception as e:
     failures += log_result(10, "Fractional Dimension", False, f"Crashed: {e}")

print("-------------------------------------------------------------")
print(f"TOTAL FAILURES: {failures}/10")
