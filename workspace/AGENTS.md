# QENEX LAB - Opencode Repository Agent Guidelines

This document is the authoritative operational manual for autonomous agents (including QENEX Sovereign Agent) operating within the `opencode` repository. This is a **Hybrid Monorepo** combining a Bun/TypeScript infrastructure with a high-performance Python scientific core.

## 1. Environment & Build System

### Hybrid Architecture

- **Infrastructure/UI**: TypeScript (Runtime: Bun)
- **Scientific Core**: Python 3.12+ (Runtime: Virtual Environment)

### Core Commands (Workspace Root)

**TypeScript (Bun):**
_Note: Scripts are managed via Bun. Run `bun run` to see all available scripts._

- `bun install` : Install JS dependencies
- `bun run dev` : Start development watchers (runs opencode package)
- `bun run typecheck`: TypeScript validation (**MANDATORY** before commits)
- `bun run test`: **DO NOT** run from root. See testing protocols below.

**Python (Science):**

- `source venv/bin/activate`: **ALWAYS** activate venv before Python operations
- `pip install -r requirements.txt`: Install Python dependencies (numpy, pytest)
- `black .` : Format code (PEP 8)
- `mypy .` : Strict type checking

### Testing Protocols

**TypeScript (Bun Test):**
_Navigate to specific package first (e.g., `packages/opencode` if it exists)_

- **Run All**: `bun test`
- **Single File**: `bun test <path-to-file>`
- **Watch**: `bun test --watch`

**Python (Pytest):**
_Tests are primarily located in the `tests/` directory._

- **Activate Env**: `source venv/bin/activate`
- **Run All**: `pytest tests/`
- **Single File**: `pytest tests/optimization_test.py`
- **Specific Test**: `pytest tests/optimization_test.py::test_function_name`

**Important Note on Python Imports**:
Some tests manually append package paths. If creating new tests, ensure proper path handling:

```python
import sys
import os
sys.path.append(os.path.abspath("packages/qenex-chem/src"))
```

## 2. Code Style & Standards

### TypeScript (Infrastructure)

- **Strict Typing**: No `any`. Use `unknown` with narrowing.
- **Immutability**: `const` over `let`. No direct mutation.
- **Imports**: Use path aliases if available. Group external vs internal.
- **Errors**: Return `Result<T, E>` types where applicable.
- **Logging**: Avoid `console.log` in production code.
- **Naming**:
  - Variables/Functions: `camelCase`
  - Classes/Namespaces: `PascalCase`
  - Files: `kebab-case`
  - Constants: `UPPER_SNAKE_CASE`

### Python (Scientific Core)

- **Style**: Follow **PEP 8**. Enforced by `black`.
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

- Avoid global state or singletons that cannot be reset between tests.
- Pass dependencies explicitly to functions/classes to ensure testability.

### Error Handling

- Handle errors explicitly.
- In TypeScript, consider using result patterns or typed error handling.
- In Python, raise specific exceptions.

## 4. Git & Version Control Protocol

### Commit Standards (Conventional Commits)

- `feat: description` (New features)
- `fix: description` (Bug fixes)
- `chore: description` (Maintenance, deps, docs)
- `refactor: description` (Code change that neither fixes a bug nor adds a feature)
- `test: description` (Adding missing tests or correcting existing tests)

**Rules:**

- **Atomic Commits**: One logical change per commit.
- **Descriptive**: Messages must explain _why_, not just _what_.

## 5. Agent Operational Protocols

### The "Trinity" Workflow

1. **Reason**: Analyze the problem using high-level reasoning. Decompose into steps.
2. **Generate**: Write the implementation (TS or Python).
3. **Validate**: Verify using the appropriate test runner (Bun Test or Pytest).

### Thinking Process

Before writing a single line of code:

1. **Explore**: Use `ls -R` or `glob` to understand the file structure.
2. **Read**: Read relevant files completely. Don't guess APIs.
3. **Plan**: Draft a plan. If complex, create a `TODO` list.
4. **Communicate**: Tell the user what you are about to do.

### Safety & Integrity

- **Path Check**: Verify directory existence with `ls` before `write`.
- **Read First**: Always `read` a file before `edit` to ensure context.
- **No Hallucinations**: Do not import non-existent libraries. Verify with `pip list` or `bun pm ls`.
- **Secrets**: NEVER commit `.env` files, API keys, or credentials.

### Troubleshooting Loop

If a command or test fails:

1. **Read Output**: Analyze the error message thoroughly.
2. **Context**: Check environment variables or configuration files.
3. **Isolate**: Create a minimal reproduction if possible.
4. **Fix**: Apply a targeted fix.
5. **Verify**: Run the verification suite again.

## 6. AI Interaction Rules

- **Context is King**: When editing, read the _entire_ relevant module to understand local conventions.
- **Atomic Edits**: Focus on one logical change at a time.
- **Self-Correction**: If a build command fails, read the _entire_ error log before attempting a fix.

## 7. Specialized Domains

Consult package-specific READMEs for domain constraints.

- **Physics**: `packages/qenex-physics`
- **Biology**: `packages/qenex-bio`
- **Math**: `packages/qenex-math`
- **Chemistry**: `packages/qenex-chem`
- **QLang**: `packages/qenex-qlang`

Agents must utilize these specialized packages for high-precision tasks.

---

_Maintained by QENEX Sovereign Agent. Last Updated: 2026-01-08_
