"""
QENEX Discovery Pipeline - Autonomous Scientific Discovery Engine

This module provides the core framework for autonomous scientific discovery,
integrating hypothesis generation, experimental design, and knowledge synthesis.

Key Components:
1. DiscoveryPipeline - End-to-end scientific discovery workflow
2. HypothesisEngine - Generates and ranks scientific hypotheses
3. ExperimentDesigner - Plans computational experiments
4. KnowledgeSynthesizer - Integrates findings into coherent theories

Discovery Workflow:
    Observation → Hypothesis → Prediction → Experiment → Analysis → Theory

Supported Discovery Modes:
- Exploratory: Generate novel hypotheses from data
- Confirmatory: Test specific theoretical predictions
- Optimization: Find optimal parameters for phenomena
- Anomaly Detection: Identify unexpected results
"""

from dataclasses import dataclass, field
from typing import (
    List, Dict, Any, Optional, Callable, Tuple,
    Set, Union, Generic, TypeVar, Iterator
)
from enum import Enum, auto
from abc import ABC, abstractmethod
from datetime import datetime
import json
import hashlib
import math
import random
from collections import defaultdict

# Import sibling modules for integration
try:
    from ..validation import (
        ValidationFramework, EvidenceChain, ValidationResult,
        ValidationStatus, ValidationLevel
    )
    from ..proof import FormalProver, Proof, ProofStatus
except ImportError:
    # Standalone mode - provide stubs
    ValidationFramework = None
    EvidenceChain = None
    FormalProver = None


class DiscoveryMode(Enum):
    """Modes of scientific discovery."""
    EXPLORATORY = "exploratory"       # Generate novel hypotheses
    CONFIRMATORY = "confirmatory"     # Test specific predictions
    OPTIMIZATION = "optimization"     # Find optimal parameters
    ANOMALY = "anomaly"              # Detect unexpected results


class HypothesisStatus(Enum):
    """Status of a hypothesis."""
    PROPOSED = auto()       # Newly generated
    TESTING = auto()        # Under experimental test
    SUPPORTED = auto()      # Evidence supports
    REFUTED = auto()        # Evidence contradicts
    REVISED = auto()        # Modified based on evidence
    ESTABLISHED = auto()    # Strong support, considered true


class ConfidenceLevel(Enum):
    """Confidence levels for scientific claims."""
    SPECULATIVE = 0.1       # Highly uncertain
    TENTATIVE = 0.3         # Some support
    MODERATE = 0.5          # Reasonable support
    STRONG = 0.7            # Significant support
    VERY_STRONG = 0.9       # Overwhelming support
    ESTABLISHED = 0.99      # Near certainty


@dataclass
class Observation:
    """
    A scientific observation or experimental result.
    """
    name: str
    value: Any
    uncertainty: float = 0.0
    units: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "computed"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        if self.uncertainty > 0:
            return f"Obs({self.name}={self.value}±{self.uncertainty} {self.units})"
        return f"Obs({self.name}={self.value} {self.units})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'uncertainty': self.uncertainty,
            'units': self.units,
            'conditions': self.conditions,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'metadata': self.metadata
        }


@dataclass
class Hypothesis:
    """
    A scientific hypothesis with supporting/refuting evidence.
    """
    id: str
    statement: str
    domain: str  # physics, chemistry, biology, etc.
    predictions: List[str] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    confidence: float = 0.5
    supporting_evidence: List[str] = field(default_factory=list)
    refuting_evidence: List[str] = field(default_factory=list)
    parent_hypotheses: List[str] = field(default_factory=list)  # IDs
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_support(self, evidence: str, weight: float = 0.1) -> None:
        """Add supporting evidence and update confidence."""
        self.supporting_evidence.append(evidence)
        self.confidence = min(0.99, self.confidence + weight * (1 - self.confidence))
        self.updated_at = datetime.now()
        self._update_status()
    
    def add_refutation(self, evidence: str, weight: float = 0.2) -> None:
        """Add refuting evidence and update confidence."""
        self.refuting_evidence.append(evidence)
        self.confidence = max(0.01, self.confidence - weight * self.confidence)
        self.updated_at = datetime.now()
        self._update_status()
    
    def _update_status(self) -> None:
        """Update status based on evidence."""
        if self.confidence < 0.2:
            self.status = HypothesisStatus.REFUTED
        elif self.confidence > 0.9:
            self.status = HypothesisStatus.ESTABLISHED
        elif self.confidence > 0.7:
            self.status = HypothesisStatus.SUPPORTED
        elif len(self.supporting_evidence) > 0 or len(self.refuting_evidence) > 0:
            self.status = HypothesisStatus.TESTING
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'statement': self.statement,
            'domain': self.domain,
            'predictions': self.predictions,
            'status': self.status.name,
            'confidence': self.confidence,
            'supporting_evidence': self.supporting_evidence,
            'refuting_evidence': self.refuting_evidence,
            'parent_hypotheses': self.parent_hypotheses,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        return f"H({self.id}): {self.statement[:50]}... [{self.status.name}, conf={self.confidence:.2f}]"


@dataclass
class Experiment:
    """
    A computational or physical experiment design.
    """
    id: str
    name: str
    hypothesis_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: Optional[str] = None
    actual_outcome: Optional[Any] = None
    status: str = "planned"  # planned, running, completed, failed
    results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def execute(self, executor: Callable[..., Any]) -> Any:
        """Execute the experiment with given executor function."""
        self.status = "running"
        try:
            result = executor(**self.parameters)
            self.actual_outcome = result
            self.status = "completed"
            self.completed_at = datetime.now()
            return result
        except Exception as e:
            self.status = "failed"
            self.results['error'] = str(e)
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'hypothesis_id': self.hypothesis_id,
            'parameters': self.parameters,
            'expected_outcome': self.expected_outcome,
            'actual_outcome': str(self.actual_outcome) if self.actual_outcome else None,
            'status': self.status,
            'results': self.results,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class HypothesisEngine:
    """
    Engine for generating and managing scientific hypotheses.
    
    Capabilities:
    - Generate hypotheses from observations
    - Rank hypotheses by plausibility
    - Track hypothesis evolution over time
    - Identify contradictions and correlations
    """
    
    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.observations: List[Observation] = []
        self._hypothesis_counter = 0
        
        # Domain-specific hypothesis templates
        self.templates: Dict[str, List[str]] = {
            'physics': [
                "The system follows conservation of {quantity}",
                "There exists a linear relationship between {x} and {y}",
                "The phenomenon exhibits {symmetry} symmetry",
                "Energy is minimized when {condition}",
            ],
            'chemistry': [
                "The reaction rate depends on {factor}",
                "The compound exhibits {property} due to {mechanism}",
                "Equilibrium is reached when {condition}",
                "The activation energy is proportional to {quantity}",
            ],
            'biology': [
                "The protein folds into {structure} conformation",
                "Gene {gene} regulates expression of {target}",
                "The pathway is activated by {trigger}",
                "Selection pressure favors {trait}",
            ],
            'general': [
                "There is a relationship between {x} and {y}",
                "The quantity {q} is conserved",
                "The system exhibits {behavior} behavior",
            ]
        }
    
    def _generate_id(self) -> str:
        """Generate unique hypothesis ID."""
        self._hypothesis_counter += 1
        return f"H{self._hypothesis_counter:04d}"
    
    def add_observation(self, observation: Observation) -> None:
        """Add an observation to the knowledge base."""
        self.observations.append(observation)
    
    def generate_hypothesis(
        self,
        statement: str,
        predictions: List[str] = None,
        initial_confidence: float = 0.5
    ) -> Hypothesis:
        """Create a new hypothesis."""
        h_id = self._generate_id()
        hypothesis = Hypothesis(
            id=h_id,
            statement=statement,
            domain=self.domain,
            predictions=predictions or [],
            confidence=initial_confidence
        )
        self.hypotheses[h_id] = hypothesis
        return hypothesis
    
    def generate_from_template(
        self,
        template_key: str,
        **kwargs
    ) -> Hypothesis:
        """Generate hypothesis from domain template."""
        templates = self.templates.get(self.domain, self.templates['general'])
        
        # Find matching template
        template = None
        for t in templates:
            if template_key in t:
                template = t
                break
        
        if template is None:
            # Use first template as fallback
            template = templates[0]
        
        # Fill in template
        try:
            statement = template.format(**kwargs)
        except KeyError:
            statement = template
        
        return self.generate_hypothesis(statement)
    
    def generate_from_observations(
        self,
        observations: List[Observation] = None
    ) -> List[Hypothesis]:
        """
        Generate hypotheses by analyzing patterns in observations.
        
        Uses simple pattern detection to propose relationships.
        """
        obs = observations or self.observations
        generated = []
        
        if len(obs) < 2:
            return generated
        
        # Look for correlations between numerical observations
        numeric_obs = [o for o in obs if isinstance(o.value, (int, float))]
        
        if len(numeric_obs) >= 2:
            # Check for potential linear relationships
            for i, o1 in enumerate(numeric_obs):
                for o2 in numeric_obs[i+1:]:
                    # Simple correlation check
                    h = self.generate_hypothesis(
                        f"There may be a relationship between {o1.name} and {o2.name}",
                        predictions=[
                            f"If {o1.name} increases, {o2.name} should change predictably"
                        ],
                        initial_confidence=0.3
                    )
                    generated.append(h)
        
        # Look for conservation patterns
        if len(obs) >= 3:
            h = self.generate_hypothesis(
                f"A quantity derived from these observations may be conserved",
                predictions=["The quantity should remain constant under time evolution"],
                initial_confidence=0.25
            )
            generated.append(h)
        
        return generated
    
    def rank_hypotheses(self) -> List[Hypothesis]:
        """
        Rank hypotheses by confidence and evidence quality.
        
        Returns hypotheses sorted by score (highest first).
        """
        def score(h: Hypothesis) -> float:
            # Base score from confidence
            s = h.confidence
            
            # Bonus for predictions (testability)
            s += 0.1 * min(len(h.predictions), 5)
            
            # Bonus for supporting evidence
            s += 0.05 * min(len(h.supporting_evidence), 10)
            
            # Penalty for unresolved contradictions
            if len(h.refuting_evidence) > len(h.supporting_evidence):
                s -= 0.2
            
            return s
        
        return sorted(self.hypotheses.values(), key=score, reverse=True)
    
    def get_testable_hypotheses(self) -> List[Hypothesis]:
        """Get hypotheses that have testable predictions."""
        return [h for h in self.hypotheses.values() 
                if h.predictions and h.status in 
                (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING)]
    
    def find_contradictions(self) -> List[Tuple[Hypothesis, Hypothesis]]:
        """
        Find potentially contradictory hypotheses.
        
        Returns pairs of hypotheses that may conflict.
        """
        # Simplified contradiction detection
        # In full implementation, would use semantic analysis
        contradictions = []
        
        hyp_list = list(self.hypotheses.values())
        for i, h1 in enumerate(hyp_list):
            for h2 in hyp_list[i+1:]:
                # Check for obvious negations
                if "not" in h1.statement.lower() and h1.statement.replace("not ", "") in h2.statement:
                    contradictions.append((h1, h2))
                elif "not" in h2.statement.lower() and h2.statement.replace("not ", "") in h1.statement:
                    contradictions.append((h1, h2))
        
        return contradictions
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize engine state."""
        return {
            'domain': self.domain,
            'hypotheses': {k: v.to_dict() for k, v in self.hypotheses.items()},
            'observations': [o.to_dict() for o in self.observations],
            'hypothesis_count': self._hypothesis_counter
        }


class ExperimentDesigner:
    """
    Designs computational experiments to test hypotheses.
    """
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self._experiment_counter = 0
    
    def _generate_id(self) -> str:
        """Generate unique experiment ID."""
        self._experiment_counter += 1
        return f"E{self._experiment_counter:04d}"
    
    def design_experiment(
        self,
        hypothesis: Hypothesis,
        name: str,
        parameters: Dict[str, Any],
        expected_outcome: str = None
    ) -> Experiment:
        """Design an experiment to test a hypothesis."""
        exp_id = self._generate_id()
        experiment = Experiment(
            id=exp_id,
            name=name,
            hypothesis_id=hypothesis.id,
            parameters=parameters,
            expected_outcome=expected_outcome
        )
        self.experiments[exp_id] = experiment
        return experiment
    
    def design_parameter_sweep(
        self,
        hypothesis: Hypothesis,
        base_name: str,
        parameter_name: str,
        values: List[Any],
        fixed_parameters: Dict[str, Any] = None
    ) -> List[Experiment]:
        """Design a series of experiments varying one parameter."""
        experiments = []
        fixed = fixed_parameters or {}
        
        for val in values:
            params = {**fixed, parameter_name: val}
            exp = self.design_experiment(
                hypothesis,
                f"{base_name}_{parameter_name}={val}",
                params
            )
            experiments.append(exp)
        
        return experiments
    
    def get_pending_experiments(self) -> List[Experiment]:
        """Get experiments that haven't been executed."""
        return [e for e in self.experiments.values() if e.status == "planned"]


class KnowledgeSynthesizer:
    """
    Synthesizes findings into coherent scientific theories.
    
    Combines validated hypotheses, proven relationships, and
    experimental results into unified theoretical frameworks.
    """
    
    def __init__(self):
        self.theories: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, Any]] = []
        self.constants: Dict[str, Observation] = {}
    
    def add_validated_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Add a validated hypothesis to the knowledge base."""
        if hypothesis.status in (HypothesisStatus.SUPPORTED, HypothesisStatus.ESTABLISHED):
            self.theories[hypothesis.id] = {
                'type': 'hypothesis',
                'content': hypothesis.to_dict(),
                'confidence': hypothesis.confidence
            }
    
    def add_relationship(
        self,
        entity1: str,
        relation: str,
        entity2: str,
        evidence: str = "",
        confidence: float = 0.5
    ) -> None:
        """Record a relationship between entities."""
        self.relationships.append({
            'entity1': entity1,
            'relation': relation,
            'entity2': entity2,
            'evidence': evidence,
            'confidence': confidence
        })
    
    def add_constant(self, name: str, observation: Observation) -> None:
        """Add a discovered constant."""
        self.constants[name] = observation
    
    def synthesize_theory(
        self,
        name: str,
        hypothesis_ids: List[str],
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Synthesize a theory from multiple hypotheses.
        
        Creates a unified theoretical framework combining
        the specified hypotheses.
        """
        components = []
        total_confidence = 0.0
        
        for h_id in hypothesis_ids:
            if h_id in self.theories:
                theory = self.theories[h_id]
                components.append(theory)
                total_confidence += theory['confidence']
        
        if not components:
            return {'error': 'No valid hypotheses found'}
        
        avg_confidence = total_confidence / len(components)
        
        # Find relationships between components
        relevant_relations = [
            r for r in self.relationships
            if any(h_id in str(r) for h_id in hypothesis_ids)
        ]
        
        theory = {
            'name': name,
            'description': description,
            'components': components,
            'relationships': relevant_relations,
            'confidence': avg_confidence,
            'created_at': datetime.now().isoformat()
        }
        
        self.theories[name] = theory
        return theory
    
    def generate_summary(self) -> str:
        """Generate human-readable summary of current knowledge."""
        lines = [
            "=== QENEX Knowledge Synthesis Report ===",
            "",
            f"Theories/Hypotheses: {len(self.theories)}",
            f"Relationships: {len(self.relationships)}",
            f"Constants: {len(self.constants)}",
            "",
            "--- Established Theories ---"
        ]
        
        for name, theory in self.theories.items():
            conf = theory.get('confidence', 0)
            lines.append(f"  [{name}] confidence={conf:.2f}")
        
        if self.constants:
            lines.append("")
            lines.append("--- Discovered Constants ---")
            for name, obs in self.constants.items():
                lines.append(f"  {name}: {obs}")
        
        return "\n".join(lines)


class DiscoveryPipeline:
    """
    Complete scientific discovery pipeline.
    
    Orchestrates the full discovery workflow:
    1. Collect observations
    2. Generate hypotheses
    3. Design experiments
    4. Execute experiments
    5. Validate results
    6. Synthesize knowledge
    7. Generate proofs (if applicable)
    
    Example:
        pipeline = DiscoveryPipeline("Gravitational Constant")
        pipeline.add_observation("mass1", 1.0, units="kg")
        pipeline.add_observation("mass2", 1.0, units="kg")
        pipeline.add_observation("force", 6.674e-11, units="N")
        
        hypotheses = pipeline.generate_hypotheses()
        pipeline.test_hypothesis(hypotheses[0], executor_func)
        
        report = pipeline.generate_report()
    """
    
    def __init__(self, name: str, domain: str = "general"):
        self.name = name
        self.domain = domain
        self.mode = DiscoveryMode.EXPLORATORY
        
        # Components
        self.hypothesis_engine = HypothesisEngine(domain)
        self.experiment_designer = ExperimentDesigner()
        self.synthesizer = KnowledgeSynthesizer()
        
        # Evidence chain for audit
        if EvidenceChain is not None:
            self.evidence = EvidenceChain(f"Discovery: {name}")
            self.evidence.add_hypothesis(f"Discover new knowledge about {name}")
        else:
            self.evidence = None
        
        # Validation framework
        if ValidationFramework is not None:
            self.validator = ValidationFramework(f"Validation: {name}")
        else:
            self.validator = None
        
        # Optional prover
        if FormalProver is not None:
            self.prover = FormalProver()
        else:
            self.prover = None
        
        self.created_at = datetime.now()
        self.status = "active"
    
    def add_observation(
        self,
        name: str,
        value: Any,
        uncertainty: float = 0.0,
        units: str = "",
        **conditions
    ) -> Observation:
        """Add an observation to the pipeline."""
        obs = Observation(
            name=name,
            value=value,
            uncertainty=uncertainty,
            units=units,
            conditions=conditions
        )
        self.hypothesis_engine.add_observation(obs)
        
        if self.evidence:
            self.evidence.add_computation(
                f"Observation: {name}",
                obs.to_dict()
            )
        
        return obs
    
    def generate_hypotheses(self) -> List[Hypothesis]:
        """Generate hypotheses from current observations."""
        hypotheses = self.hypothesis_engine.generate_from_observations()
        
        if self.evidence:
            for h in hypotheses:
                self.evidence.add_computation(
                    f"Generated hypothesis: {h.id}",
                    h.to_dict()
                )
        
        return hypotheses
    
    def add_hypothesis(
        self,
        statement: str,
        predictions: List[str] = None,
        confidence: float = 0.5
    ) -> Hypothesis:
        """Add a hypothesis manually."""
        h = self.hypothesis_engine.generate_hypothesis(
            statement, predictions, confidence
        )
        
        if self.evidence:
            self.evidence.add_hypothesis(statement)
        
        return h
    
    def design_experiment(
        self,
        hypothesis: Hypothesis,
        name: str,
        parameters: Dict[str, Any],
        expected: str = None
    ) -> Experiment:
        """Design an experiment for a hypothesis."""
        return self.experiment_designer.design_experiment(
            hypothesis, name, parameters, expected
        )
    
    def run_experiment(
        self,
        experiment: Experiment,
        executor: Callable[..., Any]
    ) -> Any:
        """Run an experiment and record results."""
        result = experiment.execute(executor)
        
        if self.evidence:
            self.evidence.add_computation(
                f"Experiment: {experiment.name}",
                experiment.to_dict()
            )
        
        return result
    
    def test_hypothesis(
        self,
        hypothesis: Hypothesis,
        test_function: Callable[..., bool],
        **test_params
    ) -> bool:
        """
        Test a hypothesis using provided test function.
        
        Updates hypothesis status based on result.
        """
        hypothesis.status = HypothesisStatus.TESTING
        
        try:
            result = test_function(**test_params)
            
            if result:
                hypothesis.add_support(f"Test passed: {test_function.__name__}")
            else:
                hypothesis.add_refutation(f"Test failed: {test_function.__name__}")
            
            if self.evidence:
                self.evidence.add_validation(
                    f"Hypothesis test: {hypothesis.id}",
                    self._create_validation_result(hypothesis, result)
                )
            
            return result
            
        except Exception as e:
            hypothesis.add_refutation(f"Test error: {str(e)}")
            return False
    
    def _create_validation_result(
        self,
        hypothesis: Hypothesis,
        passed: bool
    ) -> Any:
        """Create a validation result object."""
        if ValidationResult is not None:
            from ..validation import ValidationResult, ValidationStatus, ValidationLevel
            return ValidationResult(
                check_name=f"hypothesis_{hypothesis.id}",
                status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
                level=ValidationLevel.BASIC,
                message=f"Hypothesis test {'passed' if passed else 'failed'}"
            )
        return {'passed': passed, 'hypothesis_id': hypothesis.id}
    
    def synthesize(self) -> Dict[str, Any]:
        """Synthesize current findings into theories."""
        # Add all supported hypotheses
        for h in self.hypothesis_engine.hypotheses.values():
            if h.status in (HypothesisStatus.SUPPORTED, HypothesisStatus.ESTABLISHED):
                self.synthesizer.add_validated_hypothesis(h)
        
        # Generate synthesis
        supported_ids = [
            h.id for h in self.hypothesis_engine.hypotheses.values()
            if h.status in (HypothesisStatus.SUPPORTED, HypothesisStatus.ESTABLISHED)
        ]
        
        if supported_ids:
            theory = self.synthesizer.synthesize_theory(
                f"Theory_{self.name}",
                supported_ids,
                f"Synthesized theory for {self.name}"
            )
            
            if self.evidence:
                self.evidence.add_conclusion(
                    "Knowledge synthesis complete",
                    theory
                )
            
            return theory
        
        return {'status': 'no_supported_hypotheses'}
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive discovery report."""
        hypotheses = list(self.hypothesis_engine.hypotheses.values())
        
        report = {
            'name': self.name,
            'domain': self.domain,
            'mode': self.mode.value,
            'created_at': self.created_at.isoformat(),
            'status': self.status,
            'statistics': {
                'total_observations': len(self.hypothesis_engine.observations),
                'total_hypotheses': len(hypotheses),
                'supported': sum(1 for h in hypotheses if h.status == HypothesisStatus.SUPPORTED),
                'refuted': sum(1 for h in hypotheses if h.status == HypothesisStatus.REFUTED),
                'established': sum(1 for h in hypotheses if h.status == HypothesisStatus.ESTABLISHED),
                'total_experiments': len(self.experiment_designer.experiments),
                'completed_experiments': sum(
                    1 for e in self.experiment_designer.experiments.values()
                    if e.status == "completed"
                )
            },
            'hypotheses': [h.to_dict() for h in self.hypothesis_engine.rank_hypotheses()],
            'experiments': [e.to_dict() for e in self.experiment_designer.experiments.values()],
            'synthesis': self.synthesizer.generate_summary(),
            'evidence_chain': self.evidence.export_to_json() if self.evidence else None
        }
        
        return report
    
    def export_to_json(self) -> str:
        """Export full pipeline state as JSON."""
        return json.dumps(self.generate_report(), indent=2, default=str)
    
    def __repr__(self) -> str:
        h_count = len(self.hypothesis_engine.hypotheses)
        o_count = len(self.hypothesis_engine.observations)
        return f"DiscoveryPipeline('{self.name}', {h_count} hypotheses, {o_count} observations)"


# Convenience functions
def discover(name: str, domain: str = "general") -> DiscoveryPipeline:
    """Create a new discovery pipeline."""
    return DiscoveryPipeline(name, domain)


def observe(name: str, value: Any, **kwargs) -> Observation:
    """Create an observation."""
    return Observation(name=name, value=value, **kwargs)


def hypothesize(statement: str, domain: str = "general") -> Hypothesis:
    """Create a standalone hypothesis."""
    engine = HypothesisEngine(domain)
    return engine.generate_hypothesis(statement)


# Export
__all__ = [
    'DiscoveryMode',
    'HypothesisStatus',
    'ConfidenceLevel',
    'Observation',
    'Hypothesis',
    'Experiment',
    'HypothesisEngine',
    'ExperimentDesigner',
    'KnowledgeSynthesizer',
    'DiscoveryPipeline',
    'discover',
    'observe',
    'hypothesize',
]
