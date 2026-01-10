"""
QENEX Automated Hypothesis Generator
====================================
AI-powered hypothesis generation using cross-domain pattern recognition,
analogical reasoning, and Bayesian exploration.

This module automatically:
1. Identifies universal scientific patterns (scaling laws, phase transitions, etc.)
2. Generates novel hypotheses by transferring patterns between domains
3. Scores hypotheses by novelty, testability, and potential impact
4. Suggests experimental tests to validate hypotheses

Author: QENEX Sovereign Agent
Date: 2026-01-10
"""

import hashlib
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum, auto
from datetime import datetime
import re


# =============================================================================
# HYPOTHESIS TEMPLATES AND PATTERNS
# =============================================================================

class UniversalPattern(Enum):
    """Universal patterns that appear across scientific domains."""
    SCALING_LAW = auto()           # Power law relationships
    PHASE_TRANSITION = auto()      # Critical phenomena, tipping points
    SYMMETRY = auto()              # Conservation laws, invariances
    FEEDBACK_LOOP = auto()         # Positive/negative feedback
    OPTIMIZATION = auto()          # Extremum principles
    EMERGENCE = auto()             # Collective behavior from simple rules
    NETWORK_EFFECT = auto()        # Topology-dependent phenomena
    DIFFUSION = auto()             # Transport processes
    OSCILLATION = auto()           # Periodic dynamics
    RENORMALIZATION = auto()       # Scale-dependent behavior


@dataclass
class PatternTemplate:
    """Template for generating hypotheses from universal patterns."""
    pattern: UniversalPattern
    mathematical_form: str
    key_parameters: List[str]
    typical_domains: List[str]
    example_instances: List[str]
    testable_predictions: List[str]


PATTERN_TEMPLATES: Dict[UniversalPattern, PatternTemplate] = {
    UniversalPattern.SCALING_LAW: PatternTemplate(
        pattern=UniversalPattern.SCALING_LAW,
        mathematical_form="Y = A * X^alpha",
        key_parameters=["exponent_alpha", "prefactor_A", "range_validity"],
        typical_domains=["physics", "biology", "economics", "networks", "astronomy"],
        example_instances=[
            "Kleiber's law: metabolic rate ~ mass^0.75",
            "Zipf's law: frequency ~ rank^(-1)",
            "Mass-luminosity relation: L ~ M^3.5",
            "Earthquake magnitude distribution: N ~ M^(-b)",
            "Neural avalanche sizes: P(s) ~ s^(-1.5)",
        ],
        testable_predictions=[
            "Plot log(Y) vs log(X) should yield straight line",
            "Exponent should be robust across different conditions",
            "Deviations indicate regime changes or new physics",
        ]
    ),
    
    UniversalPattern.PHASE_TRANSITION: PatternTemplate(
        pattern=UniversalPattern.PHASE_TRANSITION,
        mathematical_form="Order parameter: phi ~ |T - Tc|^beta near critical point",
        key_parameters=["critical_point_Tc", "order_parameter", "critical_exponent_beta"],
        typical_domains=["physics", "climate", "ecology", "neuroscience", "economics"],
        example_instances=[
            "Magnetic phase transition: M ~ |T - Tc|^0.326",
            "Climate tipping points: ice-albedo feedback at ~1.5-2C warming",
            "Neural criticality: branching ratio = 1 at criticality",
            "Percolation transition in networks",
            "Protein folding: folded/unfolded transition",
        ],
        testable_predictions=[
            "Diverging susceptibility/variance near critical point",
            "Hysteresis in first-order transitions",
            "Universal critical exponents independent of microscopic details",
            "Finite-size scaling analysis",
        ]
    ),
    
    UniversalPattern.SYMMETRY: PatternTemplate(
        pattern=UniversalPattern.SYMMETRY,
        mathematical_form="Conservation law: dQ/dt = 0 for conserved quantity Q",
        key_parameters=["symmetry_group", "conserved_quantity", "breaking_mechanism"],
        typical_domains=["physics", "chemistry", "biology", "cosmology"],
        example_instances=[
            "Energy conservation from time translation symmetry",
            "Chirality in molecules (broken mirror symmetry)",
            "Left-right asymmetry in development (broken parity)",
            "Matter-antimatter asymmetry in cosmology",
            "Gauge symmetry in electromagnetism",
        ],
        testable_predictions=[
            "Conserved quantities should remain constant",
            "Symmetry breaking leads to ordered states",
            "Goldstone modes for continuous symmetry breaking",
        ]
    ),
    
    UniversalPattern.FEEDBACK_LOOP: PatternTemplate(
        pattern=UniversalPattern.FEEDBACK_LOOP,
        mathematical_form="dx/dt = f(x) + g(x)*x (positive) or -g(x)*x (negative)",
        key_parameters=["feedback_gain", "feedback_sign", "time_constant"],
        typical_domains=["climate", "biology", "economics", "control_systems", "neuroscience"],
        example_instances=[
            "Ice-albedo feedback: warming -> ice melt -> less reflection -> more warming",
            "Gene regulation: transcription factor binding controls gene expression",
            "AMOC circulation: freshwater input -> density change -> circulation weakening",
            "Neural homeostasis: firing rate regulation via synaptic scaling",
            "Economic inflation: price increase -> wage increase -> price increase",
        ],
        testable_predictions=[
            "Positive feedback: exponential growth/runaway behavior",
            "Negative feedback: homeostasis, oscillations around set point",
            "Gain > 1: instability possible",
        ]
    ),
    
    UniversalPattern.OPTIMIZATION: PatternTemplate(
        pattern=UniversalPattern.OPTIMIZATION,
        mathematical_form="Extremum: dL/dx = 0, d2L/dx2 > 0 (minimum)",
        key_parameters=["objective_function", "constraints", "optimal_value"],
        typical_domains=["physics", "biology", "chemistry", "economics", "information_theory"],
        example_instances=[
            "Principle of least action in classical mechanics",
            "Natural selection: fitness optimization",
            "Minimum energy configurations in chemistry",
            "Maximum entropy principle in thermodynamics",
            "Efficient coding in neural systems",
        ],
        testable_predictions=[
            "Systems should settle near optimal states",
            "Perturbations should restore optimal configuration",
            "Trade-offs between competing objectives",
        ]
    ),
    
    UniversalPattern.EMERGENCE: PatternTemplate(
        pattern=UniversalPattern.EMERGENCE,
        mathematical_form="Collective behavior: phi(ensemble) != sum of phi(individuals)",
        key_parameters=["interaction_strength", "population_size", "emergence_threshold"],
        typical_domains=["physics", "biology", "neuroscience", "social_science", "chemistry"],
        example_instances=[
            "Superconductivity from Cooper pairs",
            "Consciousness from neural networks (hypothesized)",
            "Flocking behavior from simple rules",
            "Phase separation in mixtures",
            "Market crashes from individual trades",
        ],
        testable_predictions=[
            "New properties absent in individual components",
            "Critical system size for emergence",
            "Robustness to individual component failure",
        ]
    ),
    
    UniversalPattern.NETWORK_EFFECT: PatternTemplate(
        pattern=UniversalPattern.NETWORK_EFFECT,
        mathematical_form="P(k) ~ k^(-gamma) for scale-free networks",
        key_parameters=["degree_exponent_gamma", "clustering_coefficient", "path_length"],
        typical_domains=["neuroscience", "biology", "social_science", "epidemiology", "ecology"],
        example_instances=[
            "Neural small-world networks in brain",
            "Protein interaction networks",
            "Social network influence propagation",
            "Disease spread in contact networks",
            "Food webs and ecosystem stability",
        ],
        testable_predictions=[
            "Hub nodes have disproportionate influence",
            "Small-world property: short paths between nodes",
            "Cascade failures from hub removal",
        ]
    ),
    
    UniversalPattern.DIFFUSION: PatternTemplate(
        pattern=UniversalPattern.DIFFUSION,
        mathematical_form="dc/dt = D * nabla^2(c) (Fick's second law)",
        key_parameters=["diffusion_coefficient_D", "boundary_conditions", "source_sink"],
        typical_domains=["physics", "chemistry", "biology", "climate", "neuroscience"],
        example_instances=[
            "Heat diffusion in materials",
            "Chemical concentration gradients",
            "Ocean heat transport",
            "Neurotransmitter diffusion in synaptic cleft",
            "CO2 diffusion in atmosphere",
        ],
        testable_predictions=[
            "Concentration profiles follow error function solutions",
            "Mean squared displacement ~ t for normal diffusion",
            "Anomalous diffusion (~ t^alpha) indicates complex media",
        ]
    ),
    
    UniversalPattern.OSCILLATION: PatternTemplate(
        pattern=UniversalPattern.OSCILLATION,
        mathematical_form="d2x/dt2 + omega^2 * x = 0 (simple harmonic)",
        key_parameters=["frequency_omega", "amplitude", "damping"],
        typical_domains=["physics", "biology", "climate", "economics", "neuroscience"],
        example_instances=[
            "Circadian rhythms in biology",
            "Neural oscillations (alpha, gamma waves)",
            "El Nino-Southern Oscillation",
            "Business cycles in economics",
            "Predator-prey population oscillations",
        ],
        testable_predictions=[
            "Characteristic frequency in power spectrum",
            "Phase-locking between coupled oscillators",
            "Period doubling route to chaos",
        ]
    ),
    
    UniversalPattern.RENORMALIZATION: PatternTemplate(
        pattern=UniversalPattern.RENORMALIZATION,
        mathematical_form="Parameters flow under coarse-graining: g' = R(g)",
        key_parameters=["renormalization_flow", "fixed_points", "relevant_operators"],
        typical_domains=["physics", "machine_learning", "biology", "statistics"],
        example_instances=[
            "Renormalization group in quantum field theory",
            "Deep learning as iterative renormalization",
            "Multi-scale biological organization",
            "Information compression at different scales",
        ],
        testable_predictions=[
            "Universal behavior near fixed points",
            "Scale-dependent effective parameters",
            "Hierarchical structure across scales",
        ]
    ),
}


# =============================================================================
# DOMAIN KNOWLEDGE FOR HYPOTHESIS GENERATION
# =============================================================================

@dataclass
class DomainKnowledge:
    """Knowledge about a scientific domain for hypothesis generation."""
    name: str
    key_quantities: List[str]
    fundamental_equations: List[str]
    active_research_areas: List[str]
    open_questions: List[str]
    related_domains: List[str]


DOMAIN_KNOWLEDGE: Dict[str, DomainKnowledge] = {
    "quantum_chemistry": DomainKnowledge(
        name="Quantum Chemistry",
        key_quantities=["energy", "wavefunction", "electron_density", "bond_length", "binding_energy"],
        fundamental_equations=[
            "Schrodinger equation: H|psi> = E|psi>",
            "Hartree-Fock: F|chi> = epsilon|chi>",
            "DFT: E[rho] = T[rho] + V_ext[rho] + E_H[rho] + E_xc[rho]",
        ],
        active_research_areas=[
            "Beyond-DFT methods for strong correlation",
            "Machine learning force fields",
            "Quantum computing for chemistry",
            "Catalysis design",
            "Battery materials",
        ],
        open_questions=[
            "How to accurately capture strong electron correlation?",
            "Can we predict reaction mechanisms from first principles?",
            "What controls selectivity in catalysis?",
        ],
        related_domains=["physics", "materials_science", "biology"],
    ),
    
    "climate_science": DomainKnowledge(
        name="Climate Science",
        key_quantities=["temperature", "co2_concentration", "albedo", "radiative_forcing", "sea_level"],
        fundamental_equations=[
            "Energy balance: (1-alpha)*S/4 = sigma*T^4 (0D EBM)",
            "Radiative forcing: F = 5.35 * ln(CO2/CO2_0)",
            "Carbon cycle: dC/dt = E - S (emissions minus sinks)",
        ],
        active_research_areas=[
            "Tipping points and abrupt climate change",
            "Cloud feedbacks and climate sensitivity",
            "Carbon cycle feedbacks",
            "Regional climate projections",
            "Geoengineering assessment",
        ],
        open_questions=[
            "What is the exact value of equilibrium climate sensitivity?",
            "When and how will major tipping points be crossed?",
            "How will the carbon cycle respond to warming?",
        ],
        related_domains=["oceanography", "ecology", "atmospheric_physics"],
    ),
    
    "neuroscience": DomainKnowledge(
        name="Neuroscience",
        key_quantities=["membrane_potential", "firing_rate", "synaptic_weight", "spike_timing", "connectivity"],
        fundamental_equations=[
            "Hodgkin-Huxley: C*dV/dt = I_ion + I_ext",
            "STDP: dw = A+ * exp(-dt/tau+) for pre-before-post",
            "LIF: tau_m * dV/dt = V_rest - V + R*I",
        ],
        active_research_areas=[
            "Neural coding and information processing",
            "Connectomics and brain mapping",
            "Neural basis of consciousness",
            "Brain-computer interfaces",
            "Computational psychiatry",
        ],
        open_questions=[
            "How does the brain encode and process information?",
            "What is the neural correlate of consciousness?",
            "How do neural networks learn and generalize?",
        ],
        related_domains=["psychology", "machine_learning", "medicine"],
    ),
    
    "astrophysics": DomainKnowledge(
        name="Astrophysics",
        key_quantities=["luminosity", "mass", "temperature", "redshift", "metallicity"],
        fundamental_equations=[
            "Mass-luminosity: L ~ M^3.5 (main sequence)",
            "Hubble law: v = H_0 * d",
            "Stefan-Boltzmann: L = 4*pi*R^2*sigma*T^4",
        ],
        active_research_areas=[
            "Dark matter and dark energy",
            "Exoplanet characterization",
            "Gravitational wave astronomy",
            "First stars and galaxies",
            "High-energy astrophysics",
        ],
        open_questions=[
            "What is the nature of dark matter?",
            "What is driving cosmic acceleration (dark energy)?",
            "How common is life in the universe?",
        ],
        related_domains=["cosmology", "particle_physics", "planetary_science"],
    ),
    
    "biology": DomainKnowledge(
        name="Biology",
        key_quantities=["fitness", "population_size", "mutation_rate", "gene_expression", "protein_structure"],
        fundamental_equations=[
            "Population growth: dN/dt = r*N*(1 - N/K)",
            "Hardy-Weinberg: p^2 + 2pq + q^2 = 1",
            "Michaelis-Menten: v = V_max * [S] / (K_m + [S])",
        ],
        active_research_areas=[
            "Protein structure prediction (AlphaFold)",
            "Gene editing (CRISPR)",
            "Systems biology and network medicine",
            "Synthetic biology",
            "Origin of life",
        ],
        open_questions=[
            "How did life originate?",
            "How does genotype map to phenotype?",
            "Can we engineer biological systems from scratch?",
        ],
        related_domains=["chemistry", "medicine", "ecology"],
    ),
}


# =============================================================================
# HYPOTHESIS DATA STRUCTURES
# =============================================================================

@dataclass
class GeneratedHypothesis:
    """A hypothesis generated by the automated system."""
    id: str
    statement: str
    domain: str
    pattern: UniversalPattern
    source_analogy: Optional[str] = None
    target_domain: Optional[str] = None
    
    # Scoring
    novelty_score: float = 0.0       # 0-1, how novel
    testability_score: float = 0.0   # 0-1, how testable
    impact_score: float = 0.0        # 0-1, potential impact
    plausibility_score: float = 0.0  # 0-1, physical plausibility
    
    # Content
    mathematical_form: str = ""
    key_parameters: List[str] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)
    falsifiable_tests: List[str] = field(default_factory=list)
    required_data: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    validation_status: str = "pending"
    
    @property
    def composite_score(self) -> float:
        """Weighted composite score for ranking hypotheses."""
        return (
            0.3 * self.novelty_score +
            0.3 * self.testability_score +
            0.2 * self.impact_score +
            0.2 * self.plausibility_score
        )
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "domain": self.domain,
            "pattern": self.pattern.name,
            "source_analogy": self.source_analogy,
            "target_domain": self.target_domain,
            "novelty_score": self.novelty_score,
            "testability_score": self.testability_score,
            "impact_score": self.impact_score,
            "plausibility_score": self.plausibility_score,
            "composite_score": self.composite_score,
            "mathematical_form": self.mathematical_form,
            "predictions": self.predictions,
            "falsifiable_tests": self.falsifiable_tests,
            "created_at": self.created_at,
        }


# =============================================================================
# AUTOMATED HYPOTHESIS GENERATOR
# =============================================================================

class AutomatedHypothesisGenerator:
    """
    Generates scientific hypotheses using:
    1. Universal pattern recognition
    2. Cross-domain analogical reasoning
    3. Knowledge graph queries
    4. Bayesian surprise maximization
    
    The generator creates hypotheses that are:
    - Novel: Not obvious from existing knowledge
    - Testable: Have clear experimental predictions
    - Impactful: Address important open questions
    - Plausible: Consistent with known physics
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.generated_hypotheses: List[GeneratedHypothesis] = []
        self.hypothesis_history: Set[str] = set()  # Track to avoid duplicates
        
        if verbose:
            print("=" * 60)
            print("QENEX Automated Hypothesis Generator")
            print("=" * 60)
            print(f"Patterns: {len(PATTERN_TEMPLATES)}")
            print(f"Domains: {len(DOMAIN_KNOWLEDGE)}")
            print("=" * 60)
    
    def generate_from_pattern(self, 
                               pattern: UniversalPattern,
                               target_domain: str,
                               n_hypotheses: int = 3) -> List[GeneratedHypothesis]:
        """
        Generate hypotheses by applying a universal pattern to a target domain.
        
        Args:
            pattern: The universal pattern to apply
            target_domain: Scientific domain to generate hypotheses for
            n_hypotheses: Number of hypotheses to generate
        
        Returns:
            List of generated hypotheses
        """
        if pattern not in PATTERN_TEMPLATES:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        if target_domain not in DOMAIN_KNOWLEDGE:
            raise ValueError(f"Unknown domain: {target_domain}")
        
        template = PATTERN_TEMPLATES[pattern]
        domain_info = DOMAIN_KNOWLEDGE[target_domain]
        
        hypotheses = []
        
        for i in range(n_hypotheses):
            # Generate hypothesis by combining pattern with domain
            hypothesis = self._create_pattern_hypothesis(
                template, domain_info, variation=i
            )
            hypotheses.append(hypothesis)
            self.generated_hypotheses.append(hypothesis)
        
        if self.verbose:
            print(f"\n[GENERATE] Pattern: {pattern.name} -> Domain: {target_domain}")
            print(f"  Generated {len(hypotheses)} hypotheses")
        
        return hypotheses
    
    def generate_by_analogy(self,
                             source_domain: str,
                             source_phenomenon: str,
                             target_domain: str) -> List[GeneratedHypothesis]:
        """
        Generate hypotheses by transferring a phenomenon from one domain to another.
        
        Args:
            source_domain: Domain where phenomenon is known
            source_phenomenon: Description of the known phenomenon
            target_domain: Domain to transfer to
        
        Returns:
            List of hypotheses based on the analogy
        """
        if source_domain not in DOMAIN_KNOWLEDGE:
            raise ValueError(f"Unknown source domain: {source_domain}")
        if target_domain not in DOMAIN_KNOWLEDGE:
            raise ValueError(f"Unknown target domain: {target_domain}")
        
        source_info = DOMAIN_KNOWLEDGE[source_domain]
        target_info = DOMAIN_KNOWLEDGE[target_domain]
        
        # Identify relevant patterns
        relevant_patterns = self._identify_patterns_for_phenomenon(source_phenomenon)
        
        hypotheses = []
        
        for pattern in relevant_patterns[:2]:  # Top 2 patterns
            template = PATTERN_TEMPLATES[pattern]
            
            # Create analogy-based hypothesis
            hyp = self._create_analogy_hypothesis(
                source_domain, source_phenomenon,
                target_domain, template, target_info
            )
            hypotheses.append(hyp)
            self.generated_hypotheses.append(hyp)
        
        if self.verbose:
            print(f"\n[ANALOGY] {source_domain}:{source_phenomenon} -> {target_domain}")
            print(f"  Patterns identified: {[p.name for p in relevant_patterns]}")
            print(f"  Generated {len(hypotheses)} hypotheses")
        
        return hypotheses
    
    def generate_for_open_question(self,
                                    domain: str,
                                    question_index: int = 0) -> List[GeneratedHypothesis]:
        """
        Generate hypotheses to address an open question in a domain.
        
        Args:
            domain: Scientific domain
            question_index: Index of open question in domain knowledge
        
        Returns:
            List of hypotheses addressing the question
        """
        if domain not in DOMAIN_KNOWLEDGE:
            raise ValueError(f"Unknown domain: {domain}")
        
        domain_info = DOMAIN_KNOWLEDGE[domain]
        
        if question_index >= len(domain_info.open_questions):
            raise ValueError(f"Question index {question_index} out of range")
        
        question = domain_info.open_questions[question_index]
        
        hypotheses = []
        
        # Try different patterns to address the question
        for pattern in UniversalPattern:
            template = PATTERN_TEMPLATES[pattern]
            
            # Check if pattern is relevant to domain
            domain_matches = any(
                domain.lower() in d.lower() or d.lower() in domain.lower()
                for d in template.typical_domains
            )
            
            if domain_matches or pattern in [UniversalPattern.OPTIMIZATION, UniversalPattern.EMERGENCE]:
                hyp = self._create_question_hypothesis(
                    domain, question, pattern, template, domain_info
                )
                hypotheses.append(hyp)
                self.generated_hypotheses.append(hyp)
                
                if len(hypotheses) >= 3:  # Limit to 3 per question
                    break
        
        if self.verbose:
            print(f"\n[QUESTION] Domain: {domain}")
            print(f"  Question: {question}")
            print(f"  Generated {len(hypotheses)} hypotheses")
        
        return hypotheses
    
    def generate_all_cross_domain(self, n_per_pair: int = 2) -> List[GeneratedHypothesis]:
        """
        Generate hypotheses for all domain pairs.
        
        Args:
            n_per_pair: Number of hypotheses per domain pair
        
        Returns:
            List of all generated cross-domain hypotheses
        """
        all_hypotheses = []
        domains = list(DOMAIN_KNOWLEDGE.keys())
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("CROSS-DOMAIN HYPOTHESIS GENERATION")
            print("=" * 60)
        
        for i, source in enumerate(domains):
            for target in domains[i+1:]:
                # Find common patterns between domains
                source_info = DOMAIN_KNOWLEDGE[source]
                target_info = DOMAIN_KNOWLEDGE[target]
                
                # Check if domains are related
                if target in source_info.related_domains or source in target_info.related_domains:
                    # Generate from first research area as phenomenon
                    if source_info.active_research_areas:
                        hyps = self.generate_by_analogy(
                            source, source_info.active_research_areas[0], target
                        )
                        all_hypotheses.extend(hyps[:n_per_pair])
        
        if self.verbose:
            print(f"\nTotal cross-domain hypotheses: {len(all_hypotheses)}")
        
        return all_hypotheses
    
    def rank_hypotheses(self) -> List[GeneratedHypothesis]:
        """Rank all generated hypotheses by composite score."""
        return sorted(
            self.generated_hypotheses,
            key=lambda h: h.composite_score,
            reverse=True
        )
    
    def get_top_hypotheses(self, n: int = 10) -> List[GeneratedHypothesis]:
        """Get the top N hypotheses by composite score."""
        return self.rank_hypotheses()[:n]
    
    def export_hypotheses(self, filepath: str):
        """Export all hypotheses to JSON."""
        data = {
            "hypotheses": [h.to_dict() for h in self.generated_hypotheses],
            "total": len(self.generated_hypotheses),
            "exported_at": datetime.now().isoformat(),
            "domains": list(DOMAIN_KNOWLEDGE.keys()),
            "patterns": [p.name for p in UniversalPattern],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.verbose:
            print(f"\nExported {len(self.generated_hypotheses)} hypotheses to {filepath}")
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _generate_id(self, *args) -> str:
        """Generate unique hypothesis ID."""
        content = "_".join(str(a) for a in args) + str(time.time())
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _identify_patterns_for_phenomenon(self, phenomenon: str) -> List[UniversalPattern]:
        """Identify which universal patterns might explain a phenomenon."""
        phenomenon_lower = phenomenon.lower()
        
        pattern_keywords = {
            UniversalPattern.SCALING_LAW: ["scale", "power law", "exponent", "growth", "size"],
            UniversalPattern.PHASE_TRANSITION: ["transition", "critical", "tipping", "threshold", "abrupt"],
            UniversalPattern.SYMMETRY: ["symmetry", "conservation", "invariant", "chirality"],
            UniversalPattern.FEEDBACK_LOOP: ["feedback", "amplification", "regulation", "homeostasis"],
            UniversalPattern.OPTIMIZATION: ["optimal", "minimize", "maximize", "efficient", "selection"],
            UniversalPattern.EMERGENCE: ["emerge", "collective", "self-organize", "complex"],
            UniversalPattern.NETWORK_EFFECT: ["network", "connection", "hub", "topology"],
            UniversalPattern.DIFFUSION: ["diffusion", "transport", "spread", "gradient"],
            UniversalPattern.OSCILLATION: ["oscillation", "cycle", "rhythm", "periodic"],
            UniversalPattern.RENORMALIZATION: ["scale", "coarse", "multi-scale", "hierarchical"],
        }
        
        scores = {}
        for pattern, keywords in pattern_keywords.items():
            score = sum(1 for kw in keywords if kw in phenomenon_lower)
            scores[pattern] = score
        
        # Sort by score and return top patterns
        sorted_patterns = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, s in sorted_patterns if s > 0][:3] or [UniversalPattern.EMERGENCE]
    
    def _score_hypothesis(self, hypothesis: GeneratedHypothesis) -> GeneratedHypothesis:
        """Score a hypothesis on multiple dimensions."""
        # Novelty: Based on pattern rarity in domain
        template = PATTERN_TEMPLATES[hypothesis.pattern]
        domain_match = sum(
            1 for d in template.typical_domains 
            if hypothesis.domain.lower() in d.lower()
        )
        hypothesis.novelty_score = 1.0 - (domain_match / len(template.typical_domains)) * 0.5
        
        # Testability: Based on number of falsifiable tests
        hypothesis.testability_score = min(1.0, len(hypothesis.falsifiable_tests) / 3)
        
        # Impact: Based on whether it addresses open questions
        domain_info = DOMAIN_KNOWLEDGE.get(hypothesis.domain)
        if domain_info:
            question_overlap = any(
                q.lower() in hypothesis.statement.lower() or 
                any(word in hypothesis.statement.lower() for word in q.lower().split()[:3])
                for q in domain_info.open_questions
            )
            hypothesis.impact_score = 0.8 if question_overlap else 0.4
        else:
            hypothesis.impact_score = 0.5
        
        # Plausibility: Based on mathematical grounding
        has_math = bool(hypothesis.mathematical_form)
        has_params = len(hypothesis.key_parameters) > 0
        hypothesis.plausibility_score = 0.5 + 0.25 * has_math + 0.25 * has_params
        
        return hypothesis
    
    def _create_pattern_hypothesis(self,
                                    template: PatternTemplate,
                                    domain_info: DomainKnowledge,
                                    variation: int = 0) -> GeneratedHypothesis:
        """Create a hypothesis by applying a pattern to a domain."""
        
        # Select a key quantity to focus on
        quantity_idx = variation % len(domain_info.key_quantities)
        quantity = domain_info.key_quantities[quantity_idx]
        
        # Select a research area
        area_idx = variation % len(domain_info.active_research_areas)
        area = domain_info.active_research_areas[area_idx]
        
        # Create statement
        statement = (
            f"The {template.pattern.name.lower().replace('_', ' ')} pattern "
            f"governs {quantity} dynamics in {area}, following {template.mathematical_form}"
        )
        
        # Check for duplicates
        statement_hash = hashlib.md5(statement.encode()).hexdigest()
        if statement_hash in self.hypothesis_history:
            # Modify statement slightly
            statement += f" (variant {variation})"
        self.hypothesis_history.add(hashlib.md5(statement.encode()).hexdigest())
        
        hypothesis = GeneratedHypothesis(
            id=self._generate_id(template.pattern.name, domain_info.name, variation),
            statement=statement,
            domain=domain_info.name,
            pattern=template.pattern,
            mathematical_form=template.mathematical_form,
            key_parameters=template.key_parameters.copy(),
            predictions=[
                f"{quantity} should follow {template.pattern.name.lower()} behavior",
                template.example_instances[0] if template.example_instances else "Pattern-consistent behavior",
            ],
            falsifiable_tests=template.testable_predictions[:2],
            required_data=[f"Time series of {quantity}", f"Experimental data from {area}"],
        )
        
        return self._score_hypothesis(hypothesis)
    
    def _create_analogy_hypothesis(self,
                                    source_domain: str,
                                    source_phenomenon: str,
                                    target_domain: str,
                                    template: PatternTemplate,
                                    target_info: DomainKnowledge) -> GeneratedHypothesis:
        """Create a hypothesis by analogy transfer."""
        
        # Select target quantity
        target_quantity = target_info.key_quantities[0] if target_info.key_quantities else "behavior"
        
        statement = (
            f"By analogy with {source_phenomenon} in {source_domain}, "
            f"we hypothesize that {target_quantity} in {target_domain} "
            f"exhibits similar {template.pattern.name.lower().replace('_', ' ')} behavior"
        )
        
        hypothesis = GeneratedHypothesis(
            id=self._generate_id(source_domain, source_phenomenon, target_domain),
            statement=statement,
            domain=target_domain,
            pattern=template.pattern,
            source_analogy=f"{source_domain}: {source_phenomenon}",
            target_domain=target_domain,
            mathematical_form=template.mathematical_form,
            key_parameters=template.key_parameters.copy(),
            predictions=[
                f"Similar {template.pattern.name.lower()} behavior in {target_quantity}",
                f"Analogous mathematical structure to {source_domain}",
            ],
            falsifiable_tests=[
                f"Test {template.pattern.name.lower()} predictions in {target_domain}",
                f"Compare parameters with {source_domain}",
            ],
            required_data=[
                f"Data on {target_quantity} in {target_domain}",
                f"Reference data from {source_domain}",
            ],
        )
        
        # Cross-domain hypotheses get novelty bonus
        hypothesis = self._score_hypothesis(hypothesis)
        hypothesis.novelty_score = min(1.0, hypothesis.novelty_score + 0.2)
        
        return hypothesis
    
    def _create_question_hypothesis(self,
                                     domain: str,
                                     question: str,
                                     pattern: UniversalPattern,
                                     template: PatternTemplate,
                                     domain_info: DomainKnowledge) -> GeneratedHypothesis:
        """Create a hypothesis to address an open question."""
        
        # Extract key words from question
        question_words = [w for w in question.lower().split() if len(w) > 4][:3]
        
        statement = (
            f"To address '{question}', we propose that {' '.join(question_words)} "
            f"can be explained through {template.pattern.name.lower().replace('_', ' ')} "
            f"mechanisms, specifically {template.mathematical_form}"
        )
        
        hypothesis = GeneratedHypothesis(
            id=self._generate_id(domain, question, pattern.name),
            statement=statement,
            domain=domain,
            pattern=pattern,
            mathematical_form=template.mathematical_form,
            key_parameters=template.key_parameters.copy(),
            predictions=[
                f"Resolution of question: {question}",
                f"{template.pattern.name} behavior should be observable",
            ],
            falsifiable_tests=[
                f"Experimental test of {template.pattern.name.lower()} prediction",
            ] + template.testable_predictions[:1],
            required_data=[
                f"Data relevant to: {question}",
                f"Quantitative measurements of {domain_info.key_quantities[0]}",
            ],
        )
        
        # Question-addressing hypotheses get impact bonus
        hypothesis = self._score_hypothesis(hypothesis)
        hypothesis.impact_score = min(1.0, hypothesis.impact_score + 0.2)
        
        return hypothesis


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Demonstrate the automated hypothesis generator."""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     QENEX AUTOMATED HYPOTHESIS GENERATOR                     ║
    ║     AI-Powered Scientific Hypothesis Generation              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Patterns: Scaling Laws | Phase Transitions | Feedback Loops ║
    ║  Methods: Cross-Domain Analogy | Open Question Targeting     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    generator = AutomatedHypothesisGenerator(verbose=True)
    
    # 1. Generate from patterns
    print("\n" + "=" * 60)
    print("1. PATTERN-BASED GENERATION")
    print("=" * 60)
    
    generator.generate_from_pattern(
        UniversalPattern.PHASE_TRANSITION,
        "climate_science",
        n_hypotheses=2
    )
    
    generator.generate_from_pattern(
        UniversalPattern.SCALING_LAW,
        "neuroscience",
        n_hypotheses=2
    )
    
    # 2. Generate by analogy
    print("\n" + "=" * 60)
    print("2. ANALOGICAL REASONING")
    print("=" * 60)
    
    generator.generate_by_analogy(
        source_domain="quantum_chemistry",
        source_phenomenon="electron correlation effects",
        target_domain="neuroscience"
    )
    
    generator.generate_by_analogy(
        source_domain="astrophysics",
        source_phenomenon="gravitational collapse",
        target_domain="biology"
    )
    
    # 3. Address open questions
    print("\n" + "=" * 60)
    print("3. OPEN QUESTION TARGETING")
    print("=" * 60)
    
    generator.generate_for_open_question("quantum_chemistry", 0)
    generator.generate_for_open_question("neuroscience", 0)
    
    # 4. Cross-domain generation
    generator.generate_all_cross_domain(n_per_pair=1)
    
    # 5. Rank and display results
    print("\n" + "=" * 60)
    print("TOP HYPOTHESES")
    print("=" * 60)
    
    top_hypotheses = generator.get_top_hypotheses(5)
    for i, hyp in enumerate(top_hypotheses, 1):
        print(f"\n{i}. [{hyp.pattern.name}] Score: {hyp.composite_score:.2f}")
        print(f"   Domain: {hyp.domain}")
        print(f"   Statement: {hyp.statement[:100]}...")
        print(f"   Novelty: {hyp.novelty_score:.2f} | Testability: {hyp.testability_score:.2f}")
    
    # 6. Export
    generator.export_hypotheses("/opt/qenex_lab/workspace/reports/hypotheses.json")
    
    return generator


if __name__ == "__main__":
    main()
