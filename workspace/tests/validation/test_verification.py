"""
Tests for the QENEX Formal Verification Module.

Tests dimensional analysis, physical constraint checking, 
mathematical consistency, and cross-domain validation.
"""

import pytest
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'packages', 'qenex-core', 'src'))

from discovery.verification import (
    HypothesisVerifier,
    DimensionalAnalyzer,
    PhysicalConstraintChecker,
    MathematicalConsistencyChecker,
    CrossDomainConsistencyChecker,
    VerificationStatus,
    VerificationType,
    Dimension,
    DIMENSIONS
)


class TestDimension:
    """Test the Dimension class."""
    
    def test_dimensionless(self):
        """Test dimensionless quantity."""
        d = Dimension()
        assert d.is_dimensionless()
        assert str(d) == "dimensionless"
    
    def test_mass_dimension(self):
        """Test mass dimension."""
        d = Dimension({'M': 1})
        assert not d.is_dimensionless()
        assert 'M' in str(d)
    
    def test_dimension_multiplication(self):
        """Test multiplying dimensions."""
        mass = Dimension({'M': 1})
        velocity = Dimension({'L': 1, 'T': -1})
        momentum = mass * velocity
        
        assert momentum.exponents['M'] == 1
        assert momentum.exponents['L'] == 1
        assert momentum.exponents['T'] == -1
    
    def test_dimension_division(self):
        """Test dividing dimensions."""
        length = Dimension({'L': 1})
        time = Dimension({'T': 1})
        velocity = length / time
        
        assert velocity.exponents['L'] == 1
        assert velocity.exponents['T'] == -1
    
    def test_dimension_power(self):
        """Test raising dimension to power."""
        velocity = Dimension({'L': 1, 'T': -1})
        velocity_squared = velocity ** 2
        
        assert velocity_squared.exponents['L'] == 2
        assert velocity_squared.exponents['T'] == -2
    
    def test_dimension_equality(self):
        """Test dimension equality."""
        d1 = Dimension({'M': 1, 'L': 2, 'T': -2})
        d2 = Dimension({'M': 1, 'L': 2, 'T': -2})
        d3 = Dimension({'M': 1, 'L': 2, 'T': -1})
        
        assert d1 == d2
        assert d1 != d3


class TestDimensionalAnalyzer:
    """Test the DimensionalAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        return DimensionalAnalyzer()
    
    def test_get_known_dimension(self, analyzer):
        """Test getting dimension of known quantities."""
        energy_dim = analyzer.get_dimension("energy")
        assert energy_dim is not None
        assert energy_dim.exponents['M'] == 1
        assert energy_dim.exponents['L'] == 2
        assert energy_dim.exponents['T'] == -2
    
    def test_get_unknown_dimension(self, analyzer):
        """Test getting dimension of unknown quantity."""
        dim = analyzer.get_dimension("unknown_quantity")
        assert dim is None
    
    def test_balanced_equation(self, analyzer):
        """Test checking a balanced equation: E = m * c^2."""
        result = analyzer.check_equation_balance(
            lhs_quantities=["E"],
            rhs_quantities=["m", "c"],
            lhs_exponents=[1],
            rhs_exponents=[1, 2]
        )
        
        assert result.status == VerificationStatus.PASSED
        assert result.confidence > 0.5
    
    def test_unbalanced_equation(self, analyzer):
        """Test checking an unbalanced equation: E = m * c."""
        result = analyzer.check_equation_balance(
            lhs_quantities=["E"],
            rhs_quantities=["m", "c"],
            lhs_exponents=[1],
            rhs_exponents=[1, 1]  # Wrong: should be c^2
        )
        
        assert result.status == VerificationStatus.FAILED
    
    def test_parse_equation(self, analyzer):
        """Test parsing a mathematical equation."""
        parsed = analyzer.parse_mathematical_form("E = m * c^2")
        
        assert "E" in parsed["lhs"]
        assert "m" in parsed["rhs"]
        assert "c" in parsed["rhs"]
    
    def test_unknown_quantities(self, analyzer):
        """Test handling unknown quantities."""
        result = analyzer.check_equation_balance(
            lhs_quantities=["foo"],
            rhs_quantities=["bar", "baz"],
        )
        
        assert result.status == VerificationStatus.UNCERTAIN


class TestPhysicalConstraintChecker:
    """Test the PhysicalConstraintChecker class."""
    
    @pytest.fixture
    def checker(self):
        return PhysicalConstraintChecker()
    
    def test_valid_velocity(self, checker):
        """Test velocity within bounds."""
        result = checker.check_value_bounds("velocity", 1000)  # 1000 m/s
        assert result.status == VerificationStatus.PASSED
    
    def test_invalid_velocity(self, checker):
        """Test velocity exceeding speed of light."""
        result = checker.check_value_bounds("velocity", 4e8)  # > c
        assert result.status == VerificationStatus.FAILED
    
    def test_valid_temperature(self, checker):
        """Test valid temperature."""
        result = checker.check_value_bounds("temperature", 300)  # 300 K
        assert result.status == VerificationStatus.PASSED
    
    def test_invalid_negative_temperature(self, checker):
        """Test negative temperature (invalid in classical sense)."""
        result = checker.check_value_bounds("temperature", -10)  # Below absolute zero
        assert result.status == VerificationStatus.FAILED
    
    def test_unknown_quantity(self, checker):
        """Test unknown quantity."""
        result = checker.check_value_bounds("unknown_qty", 100)
        assert result.status == VerificationStatus.UNCERTAIN
    
    def test_conservation_satisfied(self, checker):
        """Test conservation law satisfied."""
        result = checker.check_conservation("energy", 100.0, 100.0)
        assert result.status == VerificationStatus.PASSED
    
    def test_conservation_violated(self, checker):
        """Test conservation law violated."""
        result = checker.check_conservation("energy", 100.0, 150.0)
        assert result.status == VerificationStatus.FAILED
    
    def test_check_impossibilities_found(self, checker):
        """Test detecting physical impossibilities."""
        result = checker.check_for_impossibilities(
            "This perpetual motion machine generates faster than light particles"
        )
        assert result.status == VerificationStatus.FAILED
    
    def test_check_impossibilities_clean(self, checker):
        """Test valid statement without impossibilities."""
        result = checker.check_for_impossibilities(
            "Energy is conserved in this isolated quantum system"
        )
        assert result.status == VerificationStatus.PASSED


class TestMathematicalConsistencyChecker:
    """Test the MathematicalConsistencyChecker class."""
    
    @pytest.fixture
    def checker(self):
        return MathematicalConsistencyChecker()
    
    def test_valid_power_law(self, checker):
        """Test valid power law form."""
        result = checker.check_pattern_validity("Y ~ X^alpha")
        assert result.status in [VerificationStatus.PASSED, VerificationStatus.WARNING]
    
    def test_valid_exponential(self, checker):
        """Test valid exponential form."""
        result = checker.check_pattern_validity("exp(-E/kT)")
        assert result.status == VerificationStatus.PASSED
    
    def test_unbalanced_parentheses(self, checker):
        """Test unbalanced parentheses."""
        result = checker.check_pattern_validity("f(x = x^2")
        assert result.status == VerificationStatus.FAILED
    
    def test_empty_form(self, checker):
        """Test empty mathematical form."""
        result = checker.check_pattern_validity("")
        assert result.status == VerificationStatus.SKIPPED


class TestCrossDomainConsistencyChecker:
    """Test the CrossDomainConsistencyChecker class."""
    
    @pytest.fixture
    def checker(self):
        return CrossDomainConsistencyChecker()
    
    def test_known_cross_domain(self, checker):
        """Test known cross-domain relationship."""
        result = checker.check_cross_domain_consistency(
            "physics", "chemistry",
            "Quantum mechanics governs molecular bonding"
        )
        assert result.status in [VerificationStatus.PASSED, VerificationStatus.WARNING]
    
    def test_universal_principle_violation(self, checker):
        """Test violation of universal principle."""
        result = checker.check_cross_domain_consistency(
            "physics", "biology",
            "This perpetual motion machine powers cellular metabolism"
        )
        assert result.status == VerificationStatus.FAILED


class TestHypothesisVerifier:
    """Test the main HypothesisVerifier class."""
    
    @pytest.fixture
    def verifier(self):
        return HypothesisVerifier(verbose=False)
    
    def test_verify_valid_hypothesis(self, verifier):
        """Test verifying a valid physics hypothesis."""
        report = verifier.verify_hypothesis(
            hypothesis_id="test_001",
            statement="Kinetic energy follows E = 0.5 * m * v^2",
            domain="physics",
            mathematical_form="E = 0.5 * m * v^2"
        )
        
        assert report.overall_status in [VerificationStatus.PASSED, VerificationStatus.WARNING]
        assert report.overall_confidence > 0.5
        assert report.n_failed == 0
    
    def test_verify_invalid_hypothesis(self, verifier):
        """Test verifying an invalid hypothesis."""
        report = verifier.verify_hypothesis(
            hypothesis_id="test_002",
            statement="A perpetual motion machine can achieve faster than light travel",
            domain="physics"
        )
        
        assert report.overall_status == VerificationStatus.FAILED
        assert report.n_failed > 0
    
    def test_verify_cross_domain_hypothesis(self, verifier):
        """Test verifying a cross-domain hypothesis."""
        report = verifier.verify_hypothesis(
            hypothesis_id="test_003",
            statement="Neural networks exhibit phase transition behavior",
            domain="neuroscience",
            source_domain="physics"
        )
        
        assert report.hypothesis_id == "test_003"
        assert len(report.checks) > 0
    
    def test_verification_report_summary(self, verifier):
        """Test that verification report generates summary."""
        report = verifier.verify_hypothesis(
            hypothesis_id="test_004",
            statement="Test hypothesis",
            domain="physics"
        )
        
        summary = report.summary()
        assert "VERIFICATION REPORT" in summary
        assert "test_004" in summary
    
    def test_verification_report_to_dict(self, verifier):
        """Test converting verification report to dict."""
        report = verifier.verify_hypothesis(
            hypothesis_id="test_005",
            statement="Test hypothesis",
            domain="physics"
        )
        
        d = report.to_dict()
        assert "hypothesis_id" in d
        assert "checks" in d
        assert "overall_status" in d
    
    def test_batch_verify(self, verifier):
        """Test batch verification of hypotheses."""
        hypotheses = [
            {"id": "h1", "statement": "Valid physics claim", "domain": "physics"},
            {"id": "h2", "statement": "Another claim", "domain": "chemistry"},
        ]
        
        reports = verifier.batch_verify(hypotheses)
        assert len(reports) == 2
    
    def test_export_reports(self, verifier, tmp_path):
        """Test exporting verification reports."""
        # Generate a report first
        verifier.verify_hypothesis(
            hypothesis_id="export_test",
            statement="Test for export",
            domain="physics"
        )
        
        filepath = tmp_path / "test_reports.json"
        verifier.export_reports(str(filepath))
        
        assert filepath.exists()
        
        import json
        with open(filepath) as f:
            data = json.load(f)
        
        assert "reports" in data
        assert "total" in data
        assert data["total"] >= 1


class TestPredefinedDimensions:
    """Test predefined dimensions in the module."""
    
    def test_energy_dimension(self):
        """Test energy dimension is correct: M L^2 T^-2."""
        energy = DIMENSIONS["energy"]
        assert energy.exponents['M'] == 1
        assert energy.exponents['L'] == 2
        assert energy.exponents['T'] == -2
    
    def test_force_dimension(self):
        """Test force dimension is correct: M L T^-2."""
        force = DIMENSIONS["force"]
        assert force.exponents['M'] == 1
        assert force.exponents['L'] == 1
        assert force.exponents['T'] == -2
    
    def test_momentum_dimension(self):
        """Test momentum dimension is correct: M L T^-1."""
        momentum = DIMENSIONS["momentum"]
        assert momentum.exponents['M'] == 1
        assert momentum.exponents['L'] == 1
        assert momentum.exponents['T'] == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
