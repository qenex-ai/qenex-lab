"""
QENEX Discovery Engine API
==========================

This module provides programmatic access to the QENEX scientific discovery
capabilities: hypothesis generation, verification, and cross-domain analysis.

Quick Start
-----------

1. Hypothesis Generation::

    from discovery.hypothesis_generator import (
        AutomatedHypothesisGenerator, UniversalPattern
    )
    
    gen = AutomatedHypothesisGenerator()
    
    # Generate from universal patterns
    hypotheses = gen.generate_from_pattern(
        UniversalPattern.SCALING_LAW,
        "neuroscience",
        n_hypotheses=3
    )
    
    # Generate by cross-domain analogy
    hypotheses = gen.generate_by_analogy(
        source_domain="physics",
        source_phenomenon="phase transition",
        target_domain="climate_science"
    )
    
    # Get top hypotheses
    top = gen.get_top_hypotheses(10)

2. Hypothesis Verification::

    from discovery.verification import HypothesisVerifier
    
    verifier = HypothesisVerifier()
    
    report = verifier.verify_hypothesis(
        hypothesis_id="hyp_001",
        statement="Energy scales as mass times velocity squared",
        domain="physics",
        mathematical_form="E = m * v^2"
    )
    
    print(report.summary())
    print(f"Status: {report.overall_status.name}")

3. Visualization::

    from discovery.visualization import DiscoveryReportGenerator
    
    gen = DiscoveryReportGenerator()
    gen.generate_full_dashboard(
        hypotheses=hypotheses,
        verifications=reports,
        save_path="reports/figures/"
    )

Module Reference
----------------

discovery.hypothesis_generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AutomatedHypothesisGenerator
    Main class for generating scientific hypotheses.
    
    Methods:
    - generate_from_pattern(pattern, domain, n_hypotheses=3)
    - generate_by_analogy(source_domain, phenomenon, target_domain)
    - generate_for_open_question(domain, question_index=0)
    - generate_all_cross_domain(n_per_pair=2)
    - get_top_hypotheses(n=10)
    - export_hypotheses(filepath)

UniversalPattern (Enum)
    Universal scientific patterns:
    - SCALING_LAW: Power law relationships
    - PHASE_TRANSITION: Critical phenomena
    - SYMMETRY: Conservation laws
    - FEEDBACK_LOOP: Positive/negative feedback
    - OPTIMIZATION: Extremum principles
    - EMERGENCE: Collective behavior
    - NETWORK_EFFECT: Topology-dependent phenomena
    - DIFFUSION: Transport processes
    - OSCILLATION: Periodic dynamics
    - RENORMALIZATION: Scale-dependent behavior

GeneratedHypothesis
    Data class for generated hypotheses with scoring.
    
    Attributes:
    - id, statement, domain, pattern
    - novelty_score, testability_score, impact_score, plausibility_score
    - composite_score (property)
    - mathematical_form, predictions, falsifiable_tests

discovery.verification
~~~~~~~~~~~~~~~~~~~~~~

HypothesisVerifier
    Main verification engine combining all checkers.
    
    Methods:
    - verify_hypothesis(id, statement, domain, math_form=None, ...)
    - verify_from_generated_hypothesis(hypothesis)
    - batch_verify(hypotheses)
    - export_reports(filepath)

DimensionalAnalyzer
    Performs dimensional analysis on equations.
    
    Methods:
    - get_dimension(quantity_name)
    - check_equation_balance(lhs, rhs, lhs_exp, rhs_exp)
    - parse_mathematical_form(equation)

PhysicalConstraintChecker
    Checks against physical constraints and bounds.
    
    Methods:
    - check_value_bounds(quantity, value)
    - check_conservation(quantity, initial, final)
    - check_for_impossibilities(statement)

VerificationStatus (Enum)
    - PASSED, FAILED, WARNING, SKIPPED, UNCERTAIN

VerificationReport
    Complete verification report.
    
    Attributes:
    - hypothesis_id, checks, overall_status, overall_confidence
    - n_passed, n_failed, n_warnings (properties)
    
    Methods:
    - summary() -> str
    - to_dict() -> Dict

discovery.visualization
~~~~~~~~~~~~~~~~~~~~~~~

HypothesisVisualizer
    Visualization for hypothesis analysis.
    
    Methods:
    - plot_hypothesis_scores(hypotheses)
    - plot_pattern_distribution(hypotheses)
    - plot_domain_network(hypotheses)
    - plot_hypothesis_ranking(hypotheses)

SimulationVisualizer
    Visualization for simulation results.
    
    Methods:
    - plot_parameter_exploration(results)
    - plot_objective_history(history)
    - plot_climate_projection(results)
    - plot_neural_activity(results)

DiscoveryReportGenerator
    Generate comprehensive discovery reports.
    
    Methods:
    - generate_hypothesis_report(hypotheses, save_path)
    - generate_verification_report(reports, save_path)
    - generate_full_dashboard(hypotheses, verifications, save_path)

Q-Lang Commands
---------------

Hypothesis Generation::

    # Generate using pattern
    hypothesize pattern SCALING_LAW domain neuroscience n=3
    
    # Generate by analogy
    hypothesize analogy source=physics phenomenon="phase_transition" target=biology
    
    # Generate for open question
    hypothesize question domain=quantum_chemistry index=0
    
    # Cross-domain discovery
    discover crossdomain n_per_pair=2
    
    # View top hypotheses
    discover top n=5
    
    # Export hypotheses
    discover export path=/path/to/file.json

Verification::

    # Verify a hypothesis
    verify hypothesis statement="Energy equals mass times c squared" domain=physics math="E = m * c^2"
    
    # Verify last generated hypotheses
    verify last
    
    # Check dimensional consistency
    verify dimensional "E = m * v^2"
    
    # Check physical bounds
    verify bounds velocity=1000
    verify bounds temperature=300
    
    # Export verification reports
    verify report path=/path/to/report.json

Examples
--------

Full Discovery Workflow::

    from discovery.hypothesis_generator import AutomatedHypothesisGenerator, UniversalPattern
    from discovery.verification import HypothesisVerifier
    from discovery.visualization import DiscoveryReportGenerator
    
    # 1. Generate hypotheses
    gen = AutomatedHypothesisGenerator()
    
    # Try scaling laws in neuroscience
    hyps = gen.generate_from_pattern(UniversalPattern.SCALING_LAW, "neuroscience")
    
    # Cross-domain analogy
    hyps += gen.generate_by_analogy("physics", "critical point", "biology")
    
    # 2. Verify hypotheses
    verifier = HypothesisVerifier()
    reports = verifier.batch_verify(hyps)
    
    # Filter to valid hypotheses
    valid_hyps = [h for h, r in zip(hyps, reports) if r.overall_status.name == "PASSED"]
    
    # 3. Visualize results
    viz = DiscoveryReportGenerator()
    viz.generate_full_dashboard(hyps, reports, "reports/figures/")
    
    # 4. Export
    gen.export_hypotheses("reports/hypotheses.json")
    verifier.export_reports("reports/verifications.json")

Domain-Specific Analysis::

    from discovery.verification import PhysicalConstraintChecker, DimensionalAnalyzer
    
    # Check physical bounds
    checker = PhysicalConstraintChecker()
    
    result = checker.check_value_bounds("climate_sensitivity", 3.0)
    print(f"Climate sensitivity 3K: {result.status.name}")
    
    result = checker.check_value_bounds("velocity", 4e8)
    print(f"FTL velocity: {result.status.name}")  # FAILED
    
    # Dimensional analysis
    analyzer = DimensionalAnalyzer()
    
    result = analyzer.check_equation_balance(
        ["E"], ["m", "c"],
        [1], [1, 2]
    )
    print(f"E = mc^2: {result.status.name}")  # PASSED

Available Domains
-----------------

The following domains are supported for hypothesis generation:

- quantum_chemistry: Electron correlation, catalysis, battery materials
- climate_science: Tipping points, carbon cycle, radiative forcing
- neuroscience: Neural coding, consciousness, brain-computer interfaces
- astrophysics: Dark matter, exoplanets, gravitational waves
- biology: Protein folding, gene editing, systems biology

Available Physical Quantities (Bounds)
--------------------------------------

- velocity: 0 to c (299792458 m/s)
- temperature: 0 to 1e32 K
- mass: 0 to 1e53 kg
- climate_sensitivity: 1.5 to 6 K
- membrane_potential: -0.1 to 0.1 V
- firing_rate: 0 to 1000 Hz
- stellar_mass: 0.08 to 300 solar masses
- and many more...

Conservation Laws
-----------------

The verification module checks these conservation laws:

- energy: Total energy conserved in isolated systems
- momentum: Conserved in absence of external forces
- angular_momentum: Conserved if no external torque
- charge: Electric charge always conserved
- probability: Total probability equals 1

Author: QENEX Sovereign Agent
Version: 2.0.0
Date: 2026-01-10
"""

__version__ = "2.0.0"
__author__ = "QENEX Sovereign Agent"

# Re-export main classes for convenience
from .hypothesis_generator import (
    AutomatedHypothesisGenerator,
    UniversalPattern,
    GeneratedHypothesis,
    PatternTemplate,
    DomainKnowledge,
    PATTERN_TEMPLATES,
    DOMAIN_KNOWLEDGE,
)

from .verification import (
    HypothesisVerifier,
    DimensionalAnalyzer,
    PhysicalConstraintChecker,
    MathematicalConsistencyChecker,
    CrossDomainConsistencyChecker,
    VerificationStatus,
    VerificationType,
    VerificationResult,
    VerificationReport,
    Dimension,
    DIMENSIONS,
)

try:
    from .visualization import (
        HypothesisVisualizer,
        SimulationVisualizer,
        DiscoveryReportGenerator,
    )
except ImportError:
    # Visualization requires matplotlib
    pass

__all__ = [
    # Hypothesis Generation
    "AutomatedHypothesisGenerator",
    "UniversalPattern",
    "GeneratedHypothesis",
    "PatternTemplate",
    "DomainKnowledge",
    "PATTERN_TEMPLATES",
    "DOMAIN_KNOWLEDGE",
    
    # Verification
    "HypothesisVerifier",
    "DimensionalAnalyzer",
    "PhysicalConstraintChecker",
    "MathematicalConsistencyChecker",
    "CrossDomainConsistencyChecker",
    "VerificationStatus",
    "VerificationType",
    "VerificationResult",
    "VerificationReport",
    "Dimension",
    "DIMENSIONS",
    
    # Visualization
    "HypothesisVisualizer",
    "SimulationVisualizer",
    "DiscoveryReportGenerator",
]
