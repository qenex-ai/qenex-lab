<p align="center">
  <a href="https://qenex-lab.ai">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="QENEX LAB logo">
    </picture>
  </a>
</p>
<p align="center"><strong>QENEX LAB</strong> - Scientific Intelligence Laboratory</p>
<p align="center">The open source AI coding agent for scientific computing and discovery.</p>

---

## Overview

QENEX LAB is a Scientific Intelligence Laboratory built on the OpenCode TUI framework. It provides specialized tools for scientific computing across Physics, Chemistry, Biology, and Mathematics.

### Key Features

- **Trinity Pipeline**: Reason → Generate → Validate for all scientific tasks
- **Scientific Agents**: Specialized agents for Chemistry, Physics, Biology, and Math
- **Q-Lang Support**: Domain-specific language for scientific constraints and laws
- **Scout CLI Integration**: 18-expert validation system
- **Full OpenCode TUI**: All the power of OpenCode for general development

---

## Installation

```bash
# Clone the repository
git clone https://github.com/qenex-lab/qenex-lab.git
cd qenex-lab

# Install dependencies
bun install

# Run the TUI
bun run dev
```

### Scientific Workspace Setup

```bash
# Navigate to scientific workspace
cd workspace

# Activate Python environment
source venv/bin/activate

# Run tests
pytest tests/
```

---

## Scientific Packages

| Package            | Domain            | Description                                |
| ------------------ | ----------------- | ------------------------------------------ |
| `qenex_chem`       | Quantum Chemistry | HF, DFT, molecular integrals               |
| `qenex-bio`        | Biology           | Genomics, proteomics, systems biology      |
| `qenex-physics`    | Physics           | Lattice models, tensors, quantum mechanics |
| `qenex-math`       | Mathematics       | Formal proofs, numerical analysis          |
| `qenex-qlang`      | Q-Lang            | Scientific DSL interpreter                 |
| `qenex-astro`      | Astronomy         | Celestial mechanics, cosmology             |
| `qenex-neuro`      | Neuroscience      | Neural systems modeling                    |
| `qenex-climate`    | Climate           | Climate and atmospheric modeling           |
| `qenex-accelerate` | Performance       | Rust FFI via PyO3/Maturin                  |

---

## Agents

QENEX LAB includes specialized scientific agents:

- **qenex-lab** - Sovereign Agent for orchestrating scientific discovery
- **qenex-chem** - Quantum chemistry specialist
- **qenex-physics** - Physics and theoretical physics
- **qenex-bio** - Computational biology and bioinformatics
- **qenex-math** - Mathematics and formal verification

### Built-in Agents (from OpenCode)

- **build** - Default, full access agent for development work
- **plan** - Read-only agent for analysis and code exploration

Switch agents with the `Tab` key or use `@agent-name` in messages.

---

## The Trinity Pipeline

For all scientific tasks, follow:

1. **🧠 REASON**: Decompose the problem, understand the physics/math
2. **💻 GENERATE**: Write the implementation (code/proof/model)
3. **🛡️ VALIDATE**: Verify with tests, dimensional analysis, expert validation

---

## Documentation

- [AGENTS.md](workspace/AGENTS.md) - Agent operational guidelines
- [Scientific Packages](workspace/packages/) - Package documentation
- [Q-Lang Guide](workspace/packages/qenex-qlang/) - Q-Lang documentation

---

## Architecture

```
qenex-lab/
├── packages/
│   ├── opencode/          # Core OpenCode TUI
│   ├── ui/                # UI components
│   ├── web/               # Web interface
│   └── ...
├── workspace/             # Scientific workspace
│   ├── packages/          # Scientific packages
│   │   ├── qenex_chem/    # Quantum Chemistry
│   │   ├── qenex-bio/     # Biology
│   │   ├── qenex-physics/ # Physics
│   │   └── ...
│   ├── tests/             # Test suite
│   ├── experiments/       # Research scripts
│   └── reports/           # Generated reports
├── tui/                   # Rust TUI dashboard
├── interface/             # Backend/Frontend
└── .opencode/
    └── agent/             # Agent configurations
```

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## License

MIT License

---

**QENEX LAB** - _Precision is paramount. Science demands rigor._
