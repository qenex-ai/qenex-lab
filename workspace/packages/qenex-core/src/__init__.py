"""
QENEX Core - Multi-Language Scientific Discovery Framework

This module provides the foundation for achieving maximum scientific accuracy
across all domains by leveraging the best tools from multiple programming languages:

- Python: Orchestration, ML/AI, data processing
- Julia: High-performance numerical computing, differential equations
- Rust: Memory-safe systems programming, cryptographic proofs
- Zig: Low-level optimization, SIMD operations
- C/Fortran: Legacy scientific libraries (BLAS, LAPACK)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    QENEX Discovery Engine                    │
    ├─────────────────────────────────────────────────────────────┤
    │  Hypothesis → Computation → Validation → Proof → Discovery  │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
    │  │ Python  │  │  Julia  │  │  Rust   │  │   Zig   │       │
    │  │ Scout   │  │ Numeric │  │ Proofs  │  │  SIMD   │       │
    │  │ AI/ML   │  │  DiffEq │  │ Verify  │  │  Fast   │       │
    │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
    │       │            │            │            │             │
    │       └────────────┴────────────┴────────────┘             │
    │                         │                                   │
    │              ┌──────────┴──────────┐                       │
    │              │  Precision Core     │                       │
    │              │  (Arbitrary Prec)   │                       │
    │              └─────────────────────┘                       │
    └─────────────────────────────────────────────────────────────┘

Scientific Domains:
    - Physics: Quantum mechanics, relativity, thermodynamics
    - Chemistry: Molecular dynamics, reaction kinetics, materials
    - Biology: Genomics, proteomics, systems biology
    - Mathematics: Formal proofs, number theory, topology
    - Astronomy: Celestial mechanics, cosmology
    - Medicine: Drug discovery, disease modeling
"""

__version__ = "2.0.0"
__author__ = "QENEX Scientific Intelligence Laboratory"

# Precision module
from .precision import (
    PrecisionEngine, 
    ArbitraryPrecision,
    UncertainValue,
    PrecisionLevel,
    uncertain,
    precise,
)

# Validation module
from .validation import (
    ValidationFramework, 
    EvidenceChain,
    ValidationResult,
    ValidationStatus,
    ValidationLevel,
    CrossValidator,
    validate_numerical,
    validate_conservation,
)

# Proof module
from .proof import (
    FormalProver, 
    TheoremVerifier,
    Proof,
    ProofStatus,
    InferenceRule,
    AxiomSystem,
    create_real_number_axioms,
    prove,
    verify,
)

# Discovery module
from .discovery import (
    DiscoveryPipeline, 
    HypothesisEngine,
    Hypothesis,
    Observation,
    Experiment,
    ExperimentDesigner,
    KnowledgeSynthesizer,
    DiscoveryMode,
    HypothesisStatus,
    discover,
    observe,
    hypothesize,
)

__all__ = [
    # Precision
    'PrecisionEngine',
    'ArbitraryPrecision',
    'UncertainValue',
    'PrecisionLevel',
    'uncertain',
    'precise',
    # Validation
    'ValidationFramework',
    'EvidenceChain',
    'ValidationResult',
    'ValidationStatus',
    'ValidationLevel',
    'CrossValidator',
    'validate_numerical',
    'validate_conservation',
    # Proof
    'FormalProver',
    'TheoremVerifier',
    'Proof',
    'ProofStatus',
    'InferenceRule',
    'AxiomSystem',
    'create_real_number_axioms',
    'prove',
    'verify',
    # Discovery
    'DiscoveryPipeline',
    'HypothesisEngine',
    'Hypothesis',
    'Observation',
    'Experiment',
    'ExperimentDesigner',
    'KnowledgeSynthesizer',
    'DiscoveryMode',
    'HypothesisStatus',
    'discover',
    'observe',
    'hypothesize',
]
