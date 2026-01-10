# QENEX LAB - Agent Operational Guidelines

Instructions for AI coding agents operating in the QENEX Scientific Laboratory workspace.

## 1. Repository Structure

```
workspace/
├── packages/
│   ├── qenex_chem/src/     # Quantum Chemistry (underscore naming)
│   ├── qenex-bio/src/      # Biological Systems
│   ├── qenex-physics/src/  # Lattice Models, Tensors
│   ├── qenex-math/src/     # Formal Verification
│   ├── qenex-qlang/src/    # Q-Lang Interpreter
│   └── qenex-accelerate/   # Rust FFI (maturin/PyO3)
├── tests/
│   ├── validation/         # Scientific accuracy tests
│   ├── benchmarks/         # Performance tests
│   └── conftest.py         # Pytest path configuration
├── reports/                # Generated JSON/PDF reports
├── experiments/            # Research scripts
└── venv/                   # Python virtual environment
```

## 2. Environment Setup

**CRITICAL**: Always activate the virtual environment before running Python:

```bash
source venv/bin/activate
```

## 3. Build & Test Commands

```bash
# Install dependencies
source venv/bin/activate && pip install -r requirements.txt

# Run ALL tests
source venv/bin/activate && pytest

# Run specific test file
source venv/bin/activate && pytest tests/validation/test_eri.py

# Run single test function (TDD)
source venv/bin/activate && pytest tests/validation/test_eri.py::test_function_name -v

# Run tests with output
source venv/bin/activate && pytest -v -s tests/test_qlang.py

# Run tests by keyword
source venv/bin/activate && pytest -k "boys" -v

# Code formatting (PEP 8)
source venv/bin/activate && black .

# Type checking
source venv/bin/activate && mypy packages/qenex_chem/src/

# Run Q-Lang script
source venv/bin/activate && python packages/qenex-qlang/src/interpreter.py <script.ql>

# Build Rust accelerator (if modified)
cd packages/qenex-accelerate && maturin develop --release
```

## 4. Code Style Guidelines

### Imports

Order: stdlib → third-party → local. Use dual import pattern:

```python
try:
    from . import integrals as ints
except ImportError:
    import integrals as ints
```

### Type Hints (Required)

```python
def compute_energy(molecule: Molecule, max_iter: int = 100) -> float:
```

### Docstrings (Google-style)

```python
def boys_function(n: int, t: float) -> float:
    """Compute Boys function F_n(t).

    Args:
        n: Order of the Boys function.
        t: Argument (must be >= 0).

    Returns:
        Value of F_n(t).
    """
```

### Data Classes

```python
@dataclass
class Atom:
    symbol: str
    position: np.ndarray
    charge: int = 0
```

### Error Handling

```python
if dist < 1e-10:
    raise ValueError(f"Nuclear singularity: atoms at same position (R={dist:.2e})")
```

### Naming Conventions

- Classes: `PascalCase` (`HartreeFockSolver`)
- Functions/variables: `snake_case` (`compute_eri`)
- Constants: `UPPER_SNAKE` (`BOHR_TO_ANGSTROM`)
- Private: `_leading_underscore` (`_compute_overlap`)

### NumPy - Use vectorized operations

```python
# Good
distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
# Avoid loops when possible
```

## 5. The Trinity Workflow

1. **REASON** - Read existing code, understand physics, plan changes
2. **GENERATE** - Write/edit code (always read file first before editing)
3. **VALIDATE** - Run tests, verify physical plausibility (e.g., E < 0)

## 6. Git Commit Format

```
feat: Add 6-31G* basis set with d-polarization
fix: Correct Boys function for large arguments
refactor: Extract ERI symmetry into separate module
test: Add validation for nuclear repulsion energy
```

## 7. Common Issues

| Issue                        | Fix                                                   |
| ---------------------------- | ----------------------------------------------------- |
| `ModuleNotFoundError`        | Ensure `conftest.py` in `tests/` has path setup       |
| Numba JIT timeout            | First run compiles JIT (30-60s). Subsequent runs fast |
| Rust FFI not found           | `cd qenex-accelerate && maturin develop --release`    |
| `qenex_chem` vs `qenex-chem` | Chemistry uses underscore, others use hyphens         |

## 8. Physical Constants

```python
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.8897259886
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631
```

## 9. Key Files

- `packages/qenex_chem/src/integrals.py` - Core quantum integrals (O(N⁴) ERI)
- `packages/qenex_chem/src/solver.py` - Hartree-Fock SCF implementation
- `packages/qenex_chem/src/molecule.py` - Molecular structure with validation
- `packages/qenex-qlang/src/interpreter.py` - Q-Lang scientific DSL
- `tests/conftest.py` - Pytest path configuration

---

_Maintained by QENEX Sovereign Agent. Last Updated: 2026-01-10_
