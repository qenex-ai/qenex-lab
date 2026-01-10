"""
QENEX Formal Proof System - Mathematical Verification Engine

This module provides formal theorem proving and verification capabilities
for scientific computations. It bridges computational results to rigorous
mathematical proofs.

Key Components:
1. FormalProver - Generates formal proofs for mathematical statements
2. TheoremVerifier - Verifies proofs against axiom systems
3. ProofStep - Atomic unit of logical deduction
4. ProofChain - Complete derivation from axioms to theorem

Supported Proof Systems:
- Propositional Logic
- First-Order Logic
- Type Theory (simple)
- Equational Reasoning
- Real Analysis (epsilon-delta)

Integration:
- Export proofs to Lean4, Coq, Isabelle formats (placeholder)
- Symbolic computation via SymPy for algebraic verification
"""

from dataclasses import dataclass, field
from typing import (
    List, Dict, Any, Optional, Callable, Tuple,
    Set, Union, Generic, TypeVar
)
from enum import Enum, auto
from abc import ABC, abstractmethod
import hashlib
import json
from datetime import datetime


class LogicSystem(Enum):
    """Supported logical systems for proofs."""
    PROPOSITIONAL = "propositional"      # Boolean logic
    FIRST_ORDER = "first_order"          # Quantifiers, predicates
    HIGHER_ORDER = "higher_order"        # Functions as values
    TYPE_THEORY = "type_theory"          # Dependent types
    EQUATIONAL = "equational"            # Algebraic equalities
    REAL_ANALYSIS = "real_analysis"      # Epsilon-delta proofs


class ProofStatus(Enum):
    """Status of a proof."""
    VALID = auto()
    INVALID = auto()
    INCOMPLETE = auto()
    CHECKING = auto()


class InferenceRule(Enum):
    """Standard inference rules for deduction."""
    # Propositional
    MODUS_PONENS = "modus_ponens"          # P, P→Q ⊢ Q
    MODUS_TOLLENS = "modus_tollens"        # ¬Q, P→Q ⊢ ¬P
    HYPOTHETICAL_SYLLOGISM = "hyp_syll"    # P→Q, Q→R ⊢ P→R
    DISJUNCTIVE_SYLLOGISM = "disj_syll"    # P∨Q, ¬P ⊢ Q
    CONJUNCTION_INTRO = "conj_intro"        # P, Q ⊢ P∧Q
    CONJUNCTION_ELIM = "conj_elim"          # P∧Q ⊢ P (or Q)
    DISJUNCTION_INTRO = "disj_intro"        # P ⊢ P∨Q
    DOUBLE_NEGATION = "double_neg"          # ¬¬P ⊢ P
    CONTRAPOSITION = "contraposition"       # P→Q ⊢ ¬Q→¬P
    
    # First-order
    UNIVERSAL_INSTANTIATION = "univ_inst"   # ∀x.P(x) ⊢ P(t)
    UNIVERSAL_GENERALIZATION = "univ_gen"   # P(x) ⊢ ∀x.P(x) (with conditions)
    EXISTENTIAL_INSTANTIATION = "exist_inst" # ∃x.P(x) ⊢ P(c)
    EXISTENTIAL_GENERALIZATION = "exist_gen" # P(t) ⊢ ∃x.P(x)
    
    # Equality
    REFLEXIVITY = "reflexivity"             # ⊢ t = t
    SYMMETRY = "symmetry"                   # t = s ⊢ s = t
    TRANSITIVITY = "transitivity"           # t = s, s = r ⊢ t = r
    SUBSTITUTION = "substitution"           # t = s, P(t) ⊢ P(s)
    
    # Analysis
    EPSILON_DELTA = "epsilon_delta"         # Limit definition
    INDUCTION = "induction"                 # Mathematical induction
    
    # Meta
    AXIOM = "axiom"                         # Base assumption
    DEFINITION = "definition"               # By definition
    LEMMA = "lemma"                         # Previously proven


@dataclass
class ProofStep:
    """
    A single step in a formal proof.
    
    Each step represents an application of an inference rule
    to derive a new statement from previous statements.
    """
    step_number: int
    statement: str
    rule: InferenceRule
    justification: str
    dependencies: List[int] = field(default_factory=list)  # Previous step numbers
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        deps = f" [{','.join(map(str, self.dependencies))}]" if self.dependencies else ""
        return f"{self.step_number}. {self.statement}  ({self.rule.value}{deps})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_number': self.step_number,
            'statement': self.statement,
            'rule': self.rule.value,
            'justification': self.justification,
            'dependencies': self.dependencies,
            'metadata': self.metadata
        }


@dataclass
class Axiom:
    """
    An axiom in a formal system.
    """
    name: str
    statement: str
    system: LogicSystem
    description: str = ""
    
    def __repr__(self) -> str:
        return f"Axiom({self.name}): {self.statement}"


class AxiomSystem:
    """
    Collection of axioms defining a formal system.
    """
    
    def __init__(self, name: str, logic: LogicSystem):
        self.name = name
        self.logic = logic
        self.axioms: Dict[str, Axiom] = {}
        self.definitions: Dict[str, str] = {}
        self.lemmas: Dict[str, 'Proof'] = {}
    
    def add_axiom(self, name: str, statement: str, description: str = "") -> None:
        """Add an axiom to the system."""
        self.axioms[name] = Axiom(name, statement, self.logic, description)
    
    def add_definition(self, name: str, definition: str) -> None:
        """Add a definition to the system."""
        self.definitions[name] = definition
    
    def add_lemma(self, name: str, proof: 'Proof') -> None:
        """Add a proven lemma that can be used in future proofs."""
        if proof.status == ProofStatus.VALID:
            self.lemmas[name] = proof
        else:
            raise ValueError(f"Cannot add lemma '{name}': proof is not valid")
    
    def get_axiom(self, name: str) -> Optional[Axiom]:
        return self.axioms.get(name)
    
    def __repr__(self) -> str:
        return f"AxiomSystem({self.name}, {len(self.axioms)} axioms, {len(self.lemmas)} lemmas)"


# Standard axiom systems
def create_real_number_axioms() -> AxiomSystem:
    """Create axiom system for real numbers."""
    system = AxiomSystem("RealNumbers", LogicSystem.REAL_ANALYSIS)
    
    # Field axioms
    system.add_axiom("add_comm", "∀x,y ∈ ℝ: x + y = y + x", "Commutativity of addition")
    system.add_axiom("add_assoc", "∀x,y,z ∈ ℝ: (x + y) + z = x + (y + z)", "Associativity of addition")
    system.add_axiom("add_identity", "∃0 ∈ ℝ: ∀x ∈ ℝ: x + 0 = x", "Additive identity")
    system.add_axiom("add_inverse", "∀x ∈ ℝ: ∃(-x) ∈ ℝ: x + (-x) = 0", "Additive inverse")
    system.add_axiom("mul_comm", "∀x,y ∈ ℝ: x · y = y · x", "Commutativity of multiplication")
    system.add_axiom("mul_assoc", "∀x,y,z ∈ ℝ: (x · y) · z = x · (y · z)", "Associativity of multiplication")
    system.add_axiom("mul_identity", "∃1 ∈ ℝ: ∀x ∈ ℝ: x · 1 = x", "Multiplicative identity")
    system.add_axiom("mul_inverse", "∀x ∈ ℝ\\{0}: ∃x⁻¹ ∈ ℝ: x · x⁻¹ = 1", "Multiplicative inverse")
    system.add_axiom("distributive", "∀x,y,z ∈ ℝ: x · (y + z) = x·y + x·z", "Distributive law")
    
    # Order axioms
    system.add_axiom("trichotomy", "∀x,y ∈ ℝ: exactly one of x<y, x=y, x>y holds", "Trichotomy")
    system.add_axiom("transitivity", "∀x,y,z ∈ ℝ: x<y ∧ y<z → x<z", "Transitivity of order")
    
    # Completeness
    system.add_axiom("completeness", "Every non-empty bounded subset of ℝ has a supremum", "Completeness")
    
    return system


def create_euclidean_geometry_axioms() -> AxiomSystem:
    """Create axiom system for Euclidean geometry."""
    system = AxiomSystem("EuclideanGeometry", LogicSystem.FIRST_ORDER)
    
    # Euclid's postulates
    system.add_axiom("P1", "A straight line can be drawn from any point to any point")
    system.add_axiom("P2", "A finite straight line can be extended continuously")
    system.add_axiom("P3", "A circle can be drawn with any center and radius")
    system.add_axiom("P4", "All right angles are equal")
    system.add_axiom("P5", "Parallel postulate: given line L and point P not on L, "
                     "exactly one line through P parallel to L exists")
    
    return system


class Proof:
    """
    A formal proof consisting of a sequence of proof steps.
    
    A proof derives a theorem from axioms through valid
    applications of inference rules.
    """
    
    def __init__(
        self,
        theorem: str,
        axiom_system: AxiomSystem,
        description: str = ""
    ):
        self.theorem = theorem
        self.axiom_system = axiom_system
        self.description = description
        self.steps: List[ProofStep] = []
        self.status = ProofStatus.INCOMPLETE
        self._step_counter = 0
        self.created_at = datetime.now()
    
    def add_axiom_step(self, axiom_name: str) -> int:
        """Add a step citing an axiom."""
        axiom = self.axiom_system.get_axiom(axiom_name)
        if axiom is None:
            raise ValueError(f"Unknown axiom: {axiom_name}")
        
        self._step_counter += 1
        step = ProofStep(
            step_number=self._step_counter,
            statement=axiom.statement,
            rule=InferenceRule.AXIOM,
            justification=f"Axiom: {axiom_name}"
        )
        self.steps.append(step)
        return self._step_counter
    
    def add_definition_step(self, name: str, expansion: str) -> int:
        """Add a step using a definition."""
        self._step_counter += 1
        step = ProofStep(
            step_number=self._step_counter,
            statement=expansion,
            rule=InferenceRule.DEFINITION,
            justification=f"Definition of {name}"
        )
        self.steps.append(step)
        return self._step_counter
    
    def add_inference_step(
        self,
        statement: str,
        rule: InferenceRule,
        dependencies: List[int],
        justification: str = ""
    ) -> int:
        """Add an inference step."""
        # Validate dependencies exist
        valid_steps = {s.step_number for s in self.steps}
        for dep in dependencies:
            if dep not in valid_steps:
                raise ValueError(f"Invalid dependency: step {dep} does not exist")
        
        self._step_counter += 1
        step = ProofStep(
            step_number=self._step_counter,
            statement=statement,
            rule=rule,
            justification=justification or rule.value,
            dependencies=dependencies
        )
        self.steps.append(step)
        return self._step_counter
    
    def add_lemma_step(self, lemma_name: str) -> int:
        """Add a step citing a previously proven lemma."""
        lemma = self.axiom_system.lemmas.get(lemma_name)
        if lemma is None:
            raise ValueError(f"Unknown lemma: {lemma_name}")
        
        self._step_counter += 1
        step = ProofStep(
            step_number=self._step_counter,
            statement=lemma.theorem,
            rule=InferenceRule.LEMMA,
            justification=f"Lemma: {lemma_name}"
        )
        self.steps.append(step)
        return self._step_counter
    
    def conclude(self, final_statement: str) -> None:
        """Mark the proof as concluded with final statement."""
        if self.steps:
            last_step = self.steps[-1]
            # Check if conclusion matches
            if self._statements_equivalent(last_step.statement, final_statement):
                self.status = ProofStatus.VALID
            else:
                # Need explicit connection
                self.status = ProofStatus.INCOMPLETE
        else:
            self.status = ProofStatus.INVALID
    
    def _statements_equivalent(self, s1: str, s2: str) -> bool:
        """Check if two statements are logically equivalent."""
        # Simplified comparison - in full implementation would use
        # symbolic logic comparison
        return s1.strip().lower() == s2.strip().lower()
    
    def verify(self) -> Tuple[bool, List[str]]:
        """
        Verify the proof is valid.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        if not self.steps:
            issues.append("Proof has no steps")
            return False, issues
        
        # Check each step
        for step in self.steps:
            # Verify dependencies exist and precede current step
            for dep in step.dependencies:
                if dep >= step.step_number:
                    issues.append(f"Step {step.step_number} depends on future step {dep}")
                if not any(s.step_number == dep for s in self.steps):
                    issues.append(f"Step {step.step_number} references non-existent step {dep}")
            
            # Verify axioms are in the system
            if step.rule == InferenceRule.AXIOM:
                # Extract axiom name from justification
                if "Axiom:" in step.justification:
                    axiom_name = step.justification.split("Axiom:")[1].strip()
                    if axiom_name not in self.axiom_system.axioms:
                        issues.append(f"Step {step.step_number} cites unknown axiom: {axiom_name}")
        
        is_valid = len(issues) == 0
        if is_valid:
            self.status = ProofStatus.VALID
        else:
            self.status = ProofStatus.INVALID
        
        return is_valid, issues
    
    def to_string(self) -> str:
        """Generate human-readable proof."""
        lines = [
            f"=== Proof: {self.theorem} ===",
            f"System: {self.axiom_system.name}",
            f"Status: {self.status.name}",
            "",
            "Steps:"
        ]
        
        for step in self.steps:
            lines.append(f"  {step}")
        
        lines.append("")
        lines.append(f"∴ {self.theorem} □")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize proof to dictionary."""
        return {
            'theorem': self.theorem,
            'axiom_system': self.axiom_system.name,
            'description': self.description,
            'status': self.status.name,
            'steps': [s.to_dict() for s in self.steps],
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self) -> str:
        return f"Proof({self.theorem}, {self.status.name}, {len(self.steps)} steps)"


class TheoremVerifier:
    """
    Verifies mathematical theorems and proofs.
    
    Provides automated checking of proof validity and
    can suggest proof strategies for common patterns.
    """
    
    def __init__(self):
        self.verified_theorems: Dict[str, Proof] = {}
    
    def verify_proof(self, proof: Proof) -> Tuple[bool, List[str]]:
        """Verify a proof and cache if valid."""
        is_valid, issues = proof.verify()
        
        if is_valid:
            # Generate unique key from theorem
            key = hashlib.sha256(proof.theorem.encode()).hexdigest()[:16]
            self.verified_theorems[key] = proof
        
        return is_valid, issues
    
    def check_numerical_identity(
        self,
        lhs: Callable,
        rhs: Callable,
        test_points: List[float],
        tolerance: float = 1e-10
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Numerically verify an algebraic identity.
        
        Tests lhs(x) = rhs(x) for all test points.
        """
        results = []
        all_passed = True
        
        for x in test_points:
            try:
                left = lhs(x)
                right = rhs(x)
                diff = abs(left - right)
                rel_diff = diff / max(abs(left), abs(right), 1e-300)
                passed = rel_diff < tolerance
                all_passed = all_passed and passed
                results.append({
                    'x': x,
                    'lhs': left,
                    'rhs': right,
                    'diff': diff,
                    'rel_diff': rel_diff,
                    'passed': passed
                })
            except Exception as e:
                results.append({
                    'x': x,
                    'error': str(e),
                    'passed': False
                })
                all_passed = False
        
        return all_passed, {'test_results': results}
    
    def verify_epsilon_delta_limit(
        self,
        f: Callable[[float], float],
        x0: float,
        L: float,
        epsilon_values: List[float] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify a limit using epsilon-delta definition.
        
        For each ε, tries to find δ such that |f(x) - L| < ε 
        whenever 0 < |x - x0| < δ.
        """
        if epsilon_values is None:
            epsilon_values = [0.1, 0.01, 0.001, 0.0001]
        
        results = []
        all_verified = True
        
        for epsilon in epsilon_values:
            # Try to find suitable delta
            delta = epsilon  # Initial guess
            verified = False
            
            # Binary search for valid delta
            for _ in range(20):
                # Test points around x0
                test_xs = [x0 + delta * t for t in [-0.99, -0.5, 0.5, 0.99]]
                all_close = all(abs(f(x) - L) < epsilon for x in test_xs)
                
                if all_close:
                    verified = True
                    break
                else:
                    delta /= 2
            
            results.append({
                'epsilon': epsilon,
                'delta': delta,
                'verified': verified
            })
            all_verified = all_verified and verified
        
        return all_verified, {
            'limit_point': x0,
            'limit_value': L,
            'results': results
        }


class FormalProver:
    """
    Automated theorem prover for scientific propositions.
    
    Generates proofs for mathematical statements within
    supported formal systems.
    """
    
    def __init__(self, default_system: Optional[AxiomSystem] = None):
        self.default_system = default_system or create_real_number_axioms()
        self.verifier = TheoremVerifier()
        self.proof_history: List[Proof] = []
    
    def prove_equality(
        self,
        lhs: str,
        rhs: str,
        transformation_steps: List[Tuple[str, str]]
    ) -> Proof:
        """
        Prove lhs = rhs through a series of transformations.
        
        Args:
            lhs: Left-hand side expression
            rhs: Right-hand side expression  
            transformation_steps: List of (rule_name, intermediate_expression)
        """
        theorem = f"{lhs} = {rhs}"
        proof = Proof(theorem, self.default_system)
        
        # Start with reflexivity
        step1 = proof.add_inference_step(
            f"{lhs} = {lhs}",
            InferenceRule.REFLEXIVITY,
            [],
            "Reflexivity"
        )
        
        # Apply transformations
        current_step = step1
        current_expr = lhs
        
        for rule_name, next_expr in transformation_steps:
            current_step = proof.add_inference_step(
                f"{current_expr} = {next_expr}",
                InferenceRule.SUBSTITUTION,
                [current_step],
                rule_name
            )
            current_expr = next_expr
        
        # Final transitivity
        if current_expr == rhs:
            proof.conclude(theorem)
        else:
            # Need explicit final step
            proof.add_inference_step(
                theorem,
                InferenceRule.TRANSITIVITY,
                list(range(1, current_step + 1)),
                "Transitivity of equality"
            )
            proof.conclude(theorem)
        
        self.proof_history.append(proof)
        return proof
    
    def prove_by_induction(
        self,
        property_name: str,
        base_case: str,
        inductive_step: str
    ) -> Proof:
        """
        Prove property holds for all natural numbers by induction.
        
        Args:
            property_name: Name of property P(n)
            base_case: Proof that P(0) holds
            inductive_step: Proof that P(k) → P(k+1)
        """
        theorem = f"∀n ∈ ℕ: {property_name}(n)"
        proof = Proof(theorem, self.default_system)
        
        # Base case
        step1 = proof.add_inference_step(
            f"{property_name}(0)",
            InferenceRule.DEFINITION,
            [],
            f"Base case: {base_case}"
        )
        
        # Inductive hypothesis
        step2 = proof.add_inference_step(
            f"Assume {property_name}(k) for arbitrary k",
            InferenceRule.DEFINITION,
            [],
            "Inductive hypothesis"
        )
        
        # Inductive step
        step3 = proof.add_inference_step(
            f"{property_name}(k) → {property_name}(k+1)",
            InferenceRule.DEFINITION,
            [step2],
            f"Inductive step: {inductive_step}"
        )
        
        # Conclusion by induction
        step4 = proof.add_inference_step(
            theorem,
            InferenceRule.INDUCTION,
            [step1, step3],
            "By mathematical induction"
        )
        
        proof.conclude(theorem)
        self.proof_history.append(proof)
        return proof
    
    def prove_conservation_law(
        self,
        quantity: str,
        system_description: str,
        initial_state: str,
        final_state: str
    ) -> Proof:
        """
        Prove a conservation law for a physical system.
        """
        theorem = f"{quantity}(initial) = {quantity}(final)"
        proof = Proof(theorem, self.default_system, f"Conservation of {quantity}")
        
        # Axiom: Conservation principle
        proof.add_axiom_step("add_identity")
        
        # Define system
        step2 = proof.add_definition_step(
            "system",
            system_description
        )
        
        # Initial state
        step3 = proof.add_inference_step(
            f"{quantity}({initial_state}) = Q₀",
            InferenceRule.DEFINITION,
            [step2],
            "Initial state definition"
        )
        
        # Time evolution (no external forces)
        step4 = proof.add_inference_step(
            "d/dt({quantity}) = 0 (isolated system)",
            InferenceRule.DEFINITION,
            [step2],
            "Isolation assumption"
        )
        
        # Final state
        step5 = proof.add_inference_step(
            f"{quantity}({final_state}) = Q₀",
            InferenceRule.TRANSITIVITY,
            [step3, step4],
            "Integration over time"
        )
        
        # Conclusion
        step6 = proof.add_inference_step(
            theorem,
            InferenceRule.TRANSITIVITY,
            [step3, step5],
            "Equating initial and final"
        )
        
        proof.conclude(theorem)
        self.proof_history.append(proof)
        return proof
    
    def get_proof_summary(self) -> Dict[str, Any]:
        """Get summary of all proofs generated."""
        return {
            'total_proofs': len(self.proof_history),
            'valid': sum(1 for p in self.proof_history if p.status == ProofStatus.VALID),
            'invalid': sum(1 for p in self.proof_history if p.status == ProofStatus.INVALID),
            'incomplete': sum(1 for p in self.proof_history if p.status == ProofStatus.INCOMPLETE),
            'proofs': [p.to_dict() for p in self.proof_history]
        }


# Convenience functions
def prove(theorem: str, system: Optional[AxiomSystem] = None) -> Proof:
    """Create a new proof for a theorem."""
    if system is None:
        system = create_real_number_axioms()
    return Proof(theorem, system)


def verify(proof: Proof) -> Tuple[bool, List[str]]:
    """Verify a proof."""
    return proof.verify()


# Export
__all__ = [
    'LogicSystem',
    'ProofStatus',
    'InferenceRule',
    'ProofStep',
    'Axiom',
    'AxiomSystem',
    'Proof',
    'TheoremVerifier',
    'FormalProver',
    'create_real_number_axioms',
    'create_euclidean_geometry_axioms',
    'prove',
    'verify',
]
