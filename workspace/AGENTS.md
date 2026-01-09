# QENEX LAB - Agent Operational Guidelines

This document serves as the **primary instruction set** for autonomous agents (including QENEX Sovereign Agent) operating within the QENEX `workspace`. The repository follows a **Hybrid Monorepo** structure, though the current workspace primarily exposes the **Python Scientific Core**.

## 1. Project Structure & Environment

### Directory Layout

- `packages/qenex-bio`: Biological systems (Folding, Genomics)
- `packages/qenex-chem`: Computational Chemistry (Hartree-Fock, Integrals)
- `packages/qenex-physics`: Lattice models, Tensors
- `packages/qenex-math`: Formal verification (Prover/Verifier)
- `packages/qenex-qlang`: **Q-Lang** Interpreter & Scripts
- `tests/`: Integration and unit tests
- `experiments/`: Research scripts

### Python Environment (Scientific Core)

The workspace uses a pre-configured Virtual Environment.

- **Activation**: `source venv/bin/activate` (Required for ALL python operations)
- **Dependencies**: Listed in `requirements.txt`.
- **Dev Tools**: `black`, `mypy`, `pytest` are pre-installed in `venv`.

## 2. Operational Commands

### Build & Install

```bash
# Install runtime dependencies
source venv/bin/activate && pip install -r requirements.txt

# (Hybrid Mode) If package.json is populated:
# bun install && bun run build
```

### Testing

**Crucial**: Tests are located in `tests/` and often import from `packages/`.
Always verify imports if tests fail.

```bash
# Run ALL tests
source venv/bin/activate && pytest

# Run specific test file
source venv/bin/activate && pytest tests/test_qlang.py

# Run single test case (Best for TDD)
source venv/bin/activate && pytest tests/test_qlang.py::test_integration
```

### Code Quality (Linting & Formatting)

Before committing, ensure code meets standards:

```bash
# Format Code (PEP 8)
source venv/bin/activate && black .

# Type Checking
source venv/bin/activate && mypy packages/
```

### Q-Lang Execution

The proprietary scientific language used for simulations.

```bash
source venv/bin/activate && python3 packages/qenex-qlang/src/interpreter.py packages/qenex-qlang/examples/theory_of_everything.ql
```

## 3. Code Style & Standards

### Python (Strict)

1.  **Type Hints**: Mandatory for function arguments and return types.
    ```python
    def fold_protein(sequence: str, temp: float = 300.0) -> float: ...
    ```
2.  **Imports**: Absolute imports preferred.
    - `from qenex_chem.solver import SCF` (if installed)
    - OR `from packages.qenex_chem.src.solver import SCF` (local dev)
    - _Note_: `interpreter.py` often modifies `sys.path` to enable local imports.
3.  **Docstrings**: Google-style docstrings for all classes and public methods.
4.  **Error Handling**: Use specific exceptions (e.g., `ValueError` for physics violations).
5.  **Data Classes**: Prefer `@dataclass` for structured data (e.g., Particles, Molecules).

### TypeScript (If applicable)

- **Strict**: No `any`. Use `unknown` or generics.
- **Functional**: Prefer pure functions and immutability (`const`).
- **Namespaces**: Use namespaces to group related stateless functions.

## 4. The "Trinity" Agent Workflow

Agents must follow this strict loop to ensure scientific rigor:

1.  **REASON (Plan)**:
    - Analyze the request.
    - Explore files (`ls -R`, `read`).
    - **Draft a plan** before modifying code.
2.  **GENERATE (Act)**:
    - Write/Edit code.
    - **Always read** a file before editing it to avoid overwriting.
    - Use `write` for new files, `edit` for existing.
3.  **VALIDATE (Test)**:
    - Run `pytest` or a specific Q-Lang script.
    - **Never assume success**. If it fails, read the error, fix, and retry.
    - Verify physical plausibility (e.g., Energy < 0).

## 5. Git & Version Control

- **Commit Format**: Conventional Commits
  - `feat: ...` (New science/logic)
  - `fix: ...` (Correction)
  - `refactor: ...` (Cleanup)
  - `test: ...` (Verification)
- **Rules**:
  - **Atomic**: One task per commit.
  - **No Secrets**: Never commit `.env` or keys.
  - **Verify First**: Do not commit broken code.

## 6. Troubleshooting & Common Issues

- **Import Errors**:
  - If `ModuleNotFoundError` occurs in tests, ensure `sys.path` includes `packages/`.
  - Tests in `tests/` usually need:
    ```python
    import sys, os
    # Add packages root to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../packages')))
    ```
- **File Not Found**:
  - Always verify paths with `ls`. `workspace` is the root.
- **Environment**:
  - If `python` command fails, check `source venv/bin/activate` prefix.
- **Integrals Module**:
  - `packages/qenex-chem/src/integrals.py` is computationally intensive. Ensure `scipy` is available.

---

_Maintained by QENEX Sovereign Agent. Last Updated: 2026-01-09_
