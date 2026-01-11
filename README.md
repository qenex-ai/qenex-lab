<p align="center">
  <a href="https://github.com/abdulrahman305/qenex-lab">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="QENEX LAB logo" width="400">
    </picture>
  </a>
</p>

<h1 align="center">QENEX LAB</h1>
<p align="center"><strong>Scientific Intelligence Laboratory</strong></p>
<p align="center">
  <em>AI-powered scientific computing for Physics, Chemistry, Biology & Mathematics</em>
</p>

<p align="center">
  <a href="https://github.com/abdulrahman305/qenex-lab/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://github.com/abdulrahman305/qenex-lab"><img src="https://img.shields.io/badge/TypeScript-5.8-blue.svg" alt="TypeScript"></a>
  <a href="https://github.com/abdulrahman305/qenex-lab"><img src="https://img.shields.io/badge/Python-3.12-green.svg" alt="Python"></a>
  <a href="https://github.com/abdulrahman305/qenex-lab"><img src="https://img.shields.io/badge/Bun-1.3.6-orange.svg" alt="Bun"></a>
</p>

---

## 🔬 What is QENEX LAB?

**QENEX LAB** is an autonomous scientific intelligence system that combines AI coding agents with specialized scientific computing packages. It enables researchers and scientists to:

- 🧪 Run **quantum chemistry** calculations (Hartree-Fock, DFT)
- ⚛️ Simulate **physics** systems (Ising models, tensor networks)
- 🧬 Analyze **biological** data (genomics, protein folding)
- 📐 Perform **mathematical** proofs and verification
- 🌌 Model **astronomical** phenomena
- 🌍 Run **climate** simulations

All powered by AI agents that understand scientific domains and can write, validate, and execute scientific code.

---

## ✨ Key Features

| Feature                  | Description                                                 |
| ------------------------ | ----------------------------------------------------------- |
| **🤖 Scientific Agents** | Specialized AI agents for Chemistry, Physics, Biology, Math |
| **🔄 Trinity Pipeline**  | Reason → Generate → Validate workflow for all tasks         |
| **📝 Q-Lang**            | QENEX LTD's programming language for scientific computing   |
| **🛡️ Scout Validation**  | 18-expert validation system for scientific accuracy         |
| **⚡ High Performance**  | NumPy + Numba + Rust FFI acceleration                       |
| **🖥️ Modern TUI**        | Beautiful terminal interface for scientific computing       |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/abdulrahman305/qenex-lab.git
cd qenex-lab

# Install dependencies
bun install

# Run QENEX LAB
bun run dev
```

### Scientific Workspace

```bash
# Navigate to scientific workspace
cd workspace

# Activate Python environment
source venv/bin/activate

# Run a quantum chemistry calculation
python -c "
from packages.qenex_chem.src.solver import HartreeFockSolver
from packages.qenex_chem.src.molecule import Molecule

h2 = Molecule.from_atoms([('H', [0, 0, 0]), ('H', [0, 0, 1.4])])
solver = HartreeFockSolver(h2)
energy = solver.solve()
print(f'H2 Energy: {energy:.6f} Hartree')
"

# Run tests
pytest tests/
```

---

## 📦 Scientific Packages

| Package              | Domain            | Capabilities                                        |
| -------------------- | ----------------- | --------------------------------------------------- |
| **qenex_chem**       | Quantum Chemistry | HF, DFT, molecular integrals, geometry optimization |
| **qenex-physics**    | Physics           | Ising models, tensor networks, phase transitions    |
| **qenex-bio**        | Biology           | Genomics, protein folding, systems biology          |
| **qenex-math**       | Mathematics       | Formal proofs, symbolic computation                 |
| **qenex-astro**      | Astronomy         | Celestial mechanics, orbital dynamics               |
| **qenex-climate**    | Climate           | Atmospheric modeling, climate simulation            |
| **qenex-neuro**      | Neuroscience      | Neural systems modeling                             |
| **qenex-qlang**      | Q-Lang            | QENEX LTD's scientific programming language         |
| **qenex-accelerate** | Performance       | Rust FFI via PyO3/Maturin                           |

---

## 📝 Q-Lang: QENEX LTD's Scientific Programming Language

**Q-Lang** is a domain-specific programming language developed by **QENEX LTD** specifically for the QENEX LAB scientific computing environment. It provides:

### Features

| Feature              | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| **Physical Laws**    | Define and enforce physical constraints with dimensional analysis |
| **Unit System**      | Built-in SI unit support with automatic conversions               |
| **Scientific Types** | Native types for molecules, tensors, sequences, and more          |
| **Validation**       | Compile-time checking of physical consistency                     |
| **Interop**          | Seamless integration with Python and Rust                         |

### File Extensions

- `.ql` — Q-Lang source files
- `.ex` — Q-Lang experiment files

### Example

```ql
# Define a physical constant with units
const PLANCK_CONSTANT: Energy * Time = 6.62607015e-34 J·s

# Define a molecule
molecule H2O {
    atoms: [O, H, H]
    bonds: [(0, 1), (0, 2)]
    geometry: "bent"
    angle: 104.5 deg
}

# Define a scientific constraint
law conservation_of_energy {
    forall system: System =>
        system.kinetic_energy + system.potential_energy = const
}

# Run a calculation
experiment water_vibration {
    input: H2O
    method: DFT(functional="B3LYP", basis="6-31G*")
    output: frequencies, intensities
}
```

### Q-Lang Interpreter

```bash
# Parse and validate Q-Lang code
cd workspace/packages/qenex-qlang
python -m qlang validate experiment.ql

# Execute Q-Lang experiment
python -m qlang run experiment.ex
```

---

## 🤖 Scientific Agents

QENEX LAB includes specialized AI agents that understand scientific domains:

| Agent             | Specialty                                            |
| ----------------- | ---------------------------------------------------- |
| **qenex-lab**     | 👑 Sovereign orchestrator for multi-domain discovery |
| **qenex-chem**    | ⚗️ Quantum chemistry, molecular simulations          |
| **qenex-physics** | ⚛️ Theoretical physics, lattice models               |
| **qenex-bio**     | 🧬 Genomics, proteomics, bioinformatics              |
| **qenex-math**    | 📐 Formal proofs, numerical analysis                 |
| **qenex-astro**   | 🌌 Astrophysics, celestial mechanics                 |
| **qenex-climate** | 🌍 Climate modeling, atmospheric dynamics            |
| **qenex-neuro**   | 🧠 Neural systems, computational neuroscience        |

Switch agents with `Tab` or use `@agent-name` in messages.

---

## 🔄 The Trinity Pipeline

Every scientific task follows the Trinity workflow:

```
┌─────────────────────────────────────────────────────────────┐
│                     TRINITY PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   🧠 REASON        💻 GENERATE       🛡️ VALIDATE           │
│   ─────────        ───────────       ──────────            │
│   • Decompose      • Write code      • Run tests           │
│   • Understand     • Implement       • Check units         │
│   • Hypothesize    • Document        • Verify physics      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
qenex-lab/
├── packages/                 # Core packages
│   ├── opencode/            # TUI engine (internal package name)
│   ├── app/                 # Desktop app
│   ├── web/                 # Web interface
│   ├── ui/                  # UI components
│   └── sdk/                 # SDK
├── workspace/               # 🔬 Scientific workspace
│   ├── packages/            # Scientific packages
│   │   ├── qenex_chem/      # Quantum Chemistry
│   │   ├── qenex-physics/   # Physics
│   │   ├── qenex-bio/       # Biology
│   │   ├── qenex-math/      # Mathematics
│   │   └── ...
│   ├── tests/               # Test suite (793 tests)
│   ├── experiments/         # Research scripts
│   └── reports/             # Generated reports
├── .opencode/
│   └── agent/               # Agent configurations
└── interface/               # Backend services
```

---

## 🧪 Running Tests

```bash
# All tests
cd workspace && source venv/bin/activate && pytest

# Specific domain
pytest tests/validation/test_eri.py -v

# With coverage
pytest --cov=packages tests/
```

---

## 📊 Physical Constants

QENEX LAB uses NIST CODATA 2018 constants:

```python
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.8897259886
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

---

<p align="center">
  <strong>QENEX LAB</strong> — <em>Precision is paramount. Science demands rigor.</em>
</p>

<p align="center">
  <sub>© 2024-2026 QENEX LTD. All rights reserved.</sub>
</p>
