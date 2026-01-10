import pytest
import os

# Add package paths

from molecule import Molecule
from lattice import LatticeSimulator

def test_molecule_validation():
    # Valid Perovskite
    atoms = [("Pb", (0,0,0)), ("I", (1,0,0))]
    # Pb(82) + I(53) = 135 (Odd) -> Needs Even Multiplicity (e.g., 2)
    m = Molecule(atoms, multiplicity=2)
    assert m.charge == 0
    assert m.multiplicity == 2

    # Invalid Element
    with pytest.raises(ValueError, match="Unknown element"):
        Molecule([("X", (0,0,0))])

    # Spin Parity Violation
    # H(1) -> Odd electrons -> Needs Even multiplicity
    with pytest.raises(ValueError, match="Spin Parity Error"):
        Molecule([("H", (0,0,0))], multiplicity=1)

def test_lattice_thermodynamics():
    sim = LatticeSimulator(dimensions=2, size=10)
    
    # Valid run
    res = sim.run_simulation(steps=100, temperature=300)
    assert "energy" in res
    assert "magnetization" in res
    
    # Thermodynamic violation
    with pytest.raises(ValueError, match="Thermodynamic Violation"):
        sim.run_simulation(steps=10, temperature=-5)

def test_lattice_causality():
    sim = LatticeSimulator(dimensions=2, size=5)
    
    # Causality violation
    with pytest.raises(ValueError, match="Causality Violation"):
        sim.run_simulation(steps=-1, temperature=100)
