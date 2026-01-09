# QENEX Chemistry Engine (`qenex_chem`)

## Overview

The **QENEX Chemistry Engine** is a high-performance, autonomous molecular simulation package designed for ab-initio quantum chemistry calculations. It serves as the chemical reasoning module for the QENEX Sovereign Agent.

## Features

### 1. Electronic Structure Methods

- **RHF (Restricted Hartree-Fock):** For closed-shell molecules (e.g., H2, H2O, CH4).
- **UHF (Unrestricted Hartree-Fock):** For open-shell systems, radicals, and bond breaking (e.g., H2 dissociation, O2).
- **Basis Sets:** Native support for **STO-3G** (minimal basis).
- **Integrals:** Fully analytical implementation of:
  - Overlap (S)
  - Kinetic Energy (T)
  - Nuclear Attraction (V_ne)
  - Electron Repulsion Integrals (ERI) via Obara-Saika recurrence.

### 2. Geometry Optimization

- **Analytical Gradients:** Exact derivatives of the Hartree-Fock energy with respect to nuclear coordinates.
  - Much faster and more precise than finite-difference methods.
  - Supports both RHF and UHF gradients.
- **Algorithms:** Steepest Descent optimizer included.

### 3. Convergence Acceleration

- **DIIS (Direct Inversion in the Iterative Subspace):** Accelerates SCF convergence by extrapolating the error vector.
- **Damping:** Static damping for initial iterations to prevent oscillation.

## Usage

### Python API

```python
from qenex_chem.src.molecule import Molecule
from qenex_chem.src.solver import HartreeFockSolver, UHFSolver
from qenex_chem.src.optimizer import GeometryOptimizer

# 1. Define Molecule
mol = Molecule([
    ('O', (0.0, 0.0, 0.0)),
    ('H', (0.0, 0.757, 0.586)),
    ('H', (0.0, -0.757, 0.586))
])

# 2. Choose Solver (RHF default, or UHF)
solver = HartreeFockSolver()

# 3. Optimize Geometry
opt = GeometryOptimizer(solver)
final_mol, energy_history = opt.optimize(mol, method='analytical')

print(f"Final Energy: {energy_history[-1]} Ha")
```

## Structure

- `molecule.py`: Molecule class and atomic data.
- `integrals.py`: Core mathematical engine (JIT-compiled primitives, recursive integrals).
- `solver.py`: RHF and UHF SCF loop implementations, plus Gradient calculation.
- `optimizer.py`: Geometry optimization drivers.

## Performance

- Optimized with `numba` JIT compilation for integral primitives.
- Analytical gradients provide O(1) cost relative to numerical gradients (which are O(3N)).

---

_Maintained by QENEX Sovereign Agent_
