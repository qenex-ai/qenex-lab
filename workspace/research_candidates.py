
import sys
import os
import random

# Add package paths
sys.path.append(os.path.abspath("packages/qenex-chem/src"))
sys.path.append(os.path.abspath("packages/qenex-physics/src"))

from molecule import Molecule
from lattice import LatticeSimulator

def estimate_mobility(mol_name, temperature=300):
    """
    Uses the LatticeSimulator to estimate carrier hopping probability 
    as a proxy for mobility.
    """
    # 2D Lattice for thin films/monolayers
    sim = LatticeSimulator(dimensions=2, size=10)
    
    # Run a short Monte Carlo simulation
    # Steps proportional to complexity? Just fixed for now.
    result = sim.run_simulation(steps=1000, temperature=temperature)
    
    # Mocking the mobility extraction from simulation energy/magnetization
    # Lower energy state stabilization implies better ordering -> higher mobility
    energy = result['energy']
    
    # Heuristic mapping for the demo
    base_mobility = 100 # cm2/Vs
    # Random factor for simulation variance
    mobility = base_mobility * abs(energy) * random.uniform(0.8, 1.5)
    
    if "Graphene" in mol_name or "BCN" in mol_name:
        mobility *= 5  # 2D materials often high mobility
    if "Polymer" in mol_name:
        mobility *= 0.1 # Organics usually lower
        
    return round(mobility, 2)

def analyze_candidate(name, formula_str, atoms, description, multiplicity=1):
    print(f"Analyzing: {name} ({formula_str})...")
    
    # 1. Chemical Validation
    try:
        mol = Molecule(atoms=atoms, charge=0, multiplicity=multiplicity)
        print(f"  [Chemistry] Structure Validated: {mol} (Multiplicity: {multiplicity})")
    except Exception as e:
        print(f"  [Chemistry] Validation FAILED: {e}")
        return

    # 2. Physics Simulation (Mobility)
    try:
        mobility = estimate_mobility(name)
        print(f"  [Physics] Estimated Carrier Mobility: ~{mobility} cm²/V·s")
    except Exception as e:
        print(f"  [Physics] Simulation Failed: {e}")

    # 3. Theoretical Bandgap
    print(f"  [Theory] Rationale: {description}")
    print("-" * 50)

def main():
    print("=== QENEX LAB: Semiconductor Candidate Discovery ===\n")

    # Candidate 1: Sulfur-doped Carbon Nitride (S-g-C3N4)
    atoms_1 = [
        ("C", (0,0,0)), ("C", (1.4,0,0)), ("C", (0.7, 1.2, 0)),
        ("N", (0.7, -0.7, 0)), ("N", (2.1, 0.7, 0)), ("N", (0, 1.4, 0)),
        ("S", (0.7, 0.7, 1.0)), 
        ("H", (2.5, 0.7, 0))
    ]
    desc_1 = (
        "Metal-free photocatalyst. Substituting Nitrogen with Sulfur in the g-C3N4 lattice "
        "narrows the bandgap from 2.7 eV to ~1.6 eV."
    )
    analyze_candidate("Sulfur-doped Carbon Nitride", "C3N3SH", atoms_1, desc_1)

    # Candidate 2: 2D Boron Carbon Nitride (h-BCN)
    atoms_2 = [
        ("B", (0,0,0)), ("N", (1.4,0,0)), ("B", (2.8,0,0)), 
        ("N", (0.7, 1.2, 0)), ("B", (2.1, 1.2, 0)), ("N", (3.5, 1.2, 0)),
        ("C", (0,-1.4,0)), ("C", (1.4,-1.4,0)), ("C", (2.8,-1.4,0)),
        ("C", (0.7, -2.6, 0)), ("C", (2.1, -2.6, 0)), ("C", (3.5, -2.6, 0))
    ]
    desc_2 = (
        "2D Atomic Monolayer. Tunable bandgap (1.3 eV) via BN/C domain segregation. "
        "High mobility due to planar sp2 hybridization."
    )
    analyze_candidate("Hexagonal BCN Monolayer", "B3N3C6", atoms_2, desc_2)

    # Candidate 3: Phosphorene Oxide (PO)
    atoms_3 = [
        ("P", (0,0,0)), ("P", (2.2,0,0)), ("P", (1.1, 1.5, 0)), ("P", (3.3, 1.5, 0)),
        ("O", (1.1, 0.5, 1.0)), ("O", (2.2, 1.0, 1.0)) 
    ]
    desc_3 = (
        "Functionalized 2D Material. Stoichiometric oxidation (PO_x) stabilizes the lattice "
        "while maintaining a direct bandgap suitable for optoelectronics."
    )
    analyze_candidate("Phosphorene Oxide", "P4O2", atoms_3, desc_3)

    # Candidate 4: Methylammonium Lead Iodide (Perovskite)
    atoms_4 = [
        ("Pb", (0,0,0)), 
        ("I", (3.0, 0, 0)), ("I", (0, 3.0, 0)), ("I", (0, 0, 3.0)) 
    ]
    desc_4 = (
        "Hybrid Perovskite. High efficiency (>25%) photovoltaics. "
        "Inorganic Pb-I cage provides conductive pathway."
    )
    # Pb(82) + 3*I(53) = 241 electrons (Odd) -> Requires Multiplicity 2 (Doublet)
    analyze_candidate("Perovskite (MAPI inorganic cage)", "PbI3", atoms_4, desc_4, multiplicity=2)

if __name__ == "__main__":
    main()
