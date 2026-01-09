# QENEX LAB - Opencode Repository Agent Guidelines

This document is the authoritative operational manual for autonomous agents (including QENEX Sovereign Agent) operating within the `opencode` repository. This is a **Hybrid Monorepo** combining a Bun/TypeScript infrastructure with a high-performance Python scientific core.

## 1. Environment & Build System

### Hybrid Architecture

- **Infrastructure/UI**: TypeScript (Runtime: Bun)
- **Scientific Core**: Python 3.12+ (Runtime: Virtual Environment)

### Core Commands (Workspace Root)

**TypeScript (Bun):**

- `bun install`: Install JS dependencies
- `bun run typecheck`: TypeScript validation (**MANDATORY** before commits)
- `bun run test`: **DO NOT** run from root if testing specific packages.

**Python (Science):**

- `source venv/bin/activate`: **ALWAYS** activate venv before Python operations.
- `pip install -r requirements.txt`: Install Python dependencies.
- `black .`: Format code (PEP 8).
- `mypy .`: Strict type checking.

### Running Q-Lang Scripts

The **Q-Lang Interpreter** is the primary interface for cross-domain simulations.

- **Interpreter Path**: `packages/qenex-qlang/src/interpreter.py`
- **Execution Command**:
  ```bash
  source venv/bin/activate && python3 packages/qenex-qlang/src/interpreter.py path/to/script.ql
  ```
- **Example Scripts**: Located in `packages/qenex-qlang/examples/`

### Running Tests

- **Run all tests**:
  ```bash
  source venv/bin/activate && pytest
  ```
- **Run a single test file**:
  ```bash
  source venv/bin/activate && pytest tests/test_qlang.py
  ```
- **Run a specific test case**:
  ```bash
  source venv/bin/activate && pytest tests/test_qlang.py::test_integration
  ```

## 2. Code Style & Standards

### TypeScript (Infrastructure)

- **Strict Typing**: No `any`. Use `unknown` with narrowing.
- **Immutability**: `const` over `let`. No direct mutation.
- **Naming**: `camelCase` (vars/funcs), `PascalCase` (classes/namespaces), `kebab-case` (files).
- **Errors**: Return `Result<T, E>` types where applicable.

### Python (Scientific Core)

- **Style**: Follow **PEP 8** (Enforced by `black`).
- **Typing**: **Strict Type Hints** are mandatory (checked by `mypy`).
  ```python
  def calculate_entropy(data: List[float]) -> float: ...
  ```
- **Docstrings**: Google Style docstrings for all public functions/classes.
- **Explicit Exports**: Use `__all__` in `__init__.py` to define public API.

## 3. Architecture & Patterns

### Namespaces over Classes (TypeScript)

Prefer TypeScript **Namespaces** for grouping related stateless functions.

```typescript
export namespace FileSystem {
  export const read = async (path: string): Promise<string> => { ... }
}
```

### Dependency Injection

- Avoid global state or singletons that cannot be reset.
- Pass dependencies explicitly to functions/classes.

## 4. Git & Version Control Protocol

### Commit Standards (Conventional Commits)

- `feat: description` (New features)
- `fix: description` (Bug fixes)
- `chore: description` (Maintenance, deps, docs)
- `refactor: description` (Code change; no fix/feature)
- `test: description` (Adding/correcting tests)

**Rules:**

- **Atomic Commits**: One logical change per commit.
- **Descriptive**: Messages must explain _why_, not just _what_.

## 5. Agent Operational Protocols

### The "Trinity" Workflow

1. **Reason**: Analyze the problem using high-level reasoning. Decompose into steps.
2. **Generate**: Write the implementation (TS or Python).
3. **Validate**: Verify using the appropriate test runner or Q-Lang simulation.

### Thinking Process

Before writing a single line of code:

1. **Explore**: Use `ls -R` or `glob` to understand file structure.
2. **Read**: Read relevant files completely. Don't guess APIs.
3. **Plan**: Draft a plan. If complex, create a `TODO` list.
4. **Communicate**: Tell the user what you are about to do.

### Safety & Integrity

- **Path Check**: Verify directory existence with `ls` before `write`.
- **Read First**: Always `read` a file before `edit`.
- **No Hallucinations**: Do not import non-existent libraries. Verify with `pip list` or `bun pm ls`.
- **Secrets**: NEVER commit `.env` files, API keys, or credentials.

### Troubleshooting Loop

If a command or test fails:

1. **Read Output**: Analyze the error message thoroughly.
2. **Context**: Check environment variables/config.
3. **Fix**: Apply a targeted fix.
4. **Verify**: Run the verification suite again.

## 6. Specialized Domains

Consult these packages for domain-specific logic:

- **Physics**: `packages/qenex-physics` (Lattice models, Tensor operations)
- **Biology**: `packages/qenex-bio` (Protein folding, Genomics)
- **Math**: `packages/qenex-math` (Provers, Verifiers)
- **Chemistry**: `packages/qenex-chem` (Hartree-Fock, Molecular dynamics)
- **QLang**: `packages/qenex-qlang` (The Unified Scientific Language)

Agents must utilize these specialized packages for high-precision tasks rather than rewriting core logic.

### Python Imports in Tests

Since tests reside outside packages (`tests/`), you may need to manually append package paths if the environment is not fully installed as editable:

```python
import sys
import os
# Example: Adding qenex-chem to path
sys.path.append(os.path.abspath("packages/qenex-chem/src"))
```

---

_Maintained by QENEX Sovereign Agent. Last Updated: 2026-01-09_
