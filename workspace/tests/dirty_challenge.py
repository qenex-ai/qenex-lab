
import sys
import os
import time
import numpy as np

# Add package paths
sys.path.append(os.path.abspath("packages/qenex-chem/src"))
sys.path.append(os.path.abspath("packages/qenex-bio/src"))
sys.path.append(os.path.abspath("packages/qenex-physics/src"))
sys.path.append(os.path.abspath("packages/qenex-math/src"))

from molecule import Molecule
from solver import HartreeFockSolver
from folding import ProteinFolder
from lattice import LatticeSimulator
from prover import ProofState, TacticalProver

def section(name):
    print(f"\n{'='*20} {name} {'='*20}")

def try_break(test_name, func):
    print(f"[-] Running Dirty Test: {test_name}...", end=" ")
    try:
        func()
        print("FAILED TO CRASH (System accepted dirty input?!)")
    except Exception as e:
        print(f"CRASHED/CAUGHT: {e}")

def dirty_chem():
    section("DIRTY CHEMISTRY")
    
    # 1. The "Hyper-Ionized" Atom (Stripping more electrons than exist)
    def test_ionization():
        # Hydrogen (1 proton). Charge +2 means -1 electrons.
        m = Molecule([("H", (0,0,0))], charge=2)
        s = HartreeFockSolver()
        s.compute_energy(m)
    try_break("Hyper-Ionization (Negative Electrons)", test_ionization)

    # 2. The "Singularity" (Atoms occupying exact same space)
    def test_singularity():
        m = Molecule([("H", (0,0,0)), ("H", (0,0,0))], charge=0)
        s = HartreeFockSolver()
        s.compute_energy(m)
    try_break("Nuclear Singularity (Overlap)", test_singularity)

    # 3. The "Ghost" Element
    def test_ghost():
        Molecule([("Unobtainium", (0,0,0))])
    try_break("Invalid Element", test_ghost)

def dirty_physics():
    section("DIRTY PHYSICS")
    
    # 1. Zero Kelvin / Negative Temp
    def test_absolute_zero():
        sim = LatticeSimulator(3, 10)
        sim.run_simulation(steps=100, temperature=0.0)
    try_break("Absolute Zero/Negative Temp", test_absolute_zero)
    
    # 2. Causality Violation (Negative Time/Steps)
    def test_neg_steps():
        sim = LatticeSimulator(3, 10)
        sim.run_simulation(steps=-50, temperature=300)
    try_break("Negative Time Steps", test_neg_steps)

    # 3. Dimensional Collapse
    def test_zero_dim():
        LatticeSimulator(dimensions=0, size=10)
    try_break("Zero Dimensions", test_zero_dim)

def dirty_bio():
    section("DIRTY BIOLOGY")
    
    # 1. The "Alien" DNA
    def test_alien_seq():
        folder = ProteinFolder()
        folder.fold_sequence("MAGGZ$$@#")
    try_break("Alien Amino Acids/Injection", test_alien_seq)
    
    # 2. The "Void" Life
    def test_empty_seq():
        folder = ProteinFolder()
        folder.fold_sequence("")
    try_break("Empty Sequence", test_empty_seq)

def dirty_math():
    section("DIRTY MATH")
    
    # 1. Proving Falsehood
    def test_falsehood():
        p = TacticalProver()
        state = ProofState("1 = 0")
        p.prove(state, depth_limit=5)
    try_break("Proving 1=0", test_falsehood)

    # 2. Infinite Recursion
    def test_recursion():
        p = TacticalProver()
        state = ProofState("prime(n)")
        # Giving it a huge depth limit to see if stack overflows or logic breaks
        p.prove(state, depth_limit=-1) 
    try_break("Negative Depth Limit", test_recursion)

if __name__ == "__main__":
    print(">>> QENEX LAB: EXECUTING DIRTY CHALLENGE SUITE <<<")
    dirty_chem()
    dirty_physics()
    dirty_bio()
    dirty_math()
