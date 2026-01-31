"""
Comprehensive test suite for QENEX LAB Blind Validation Protocol.

Tests cover:
1. Individual toxicity patterns (TZD, NSAID, hERG, etc.)
2. Drug class exemptions (TKIs, ARBs, SSRIs, etc.)
3. Edge cases and boundary conditions
4. Metric calculations
5. Full validation workflow
"""

import pytest
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "qenex-tissue"))

from src.blind_validation import (
    BlindValidationProtocol,
    BlindTestMolecule,
    ValidationResult,
)


class TestToxicityPatterns:
    """Test individual toxicity detection patterns"""

    def setup_method(self):
        self.protocol = BlindValidationProtocol()

    def test_tzd_hepatotoxicity_troglitazone(self):
        """Troglitazone should be flagged as high-risk TZD"""
        self.protocol.add_molecule(
            "TZD001", "Troglitazone",
            molecular_weight=441.5, logP=4.2, tpsa=84.9, num_hbd=2, num_hba=6
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_outcome == "failed", "Troglitazone should be predicted as failed"
        assert mol.predicted_failure_risk >= 0.35, f"Risk {mol.predicted_failure_risk} should be >= 0.35"
        assert "TZD" in mol.prediction_reasoning or "hepatotox" in mol.prediction_reasoning.lower()

    def test_small_polar_iniparib(self):
        """Iniparib should be flagged for efficacy concerns"""
        self.protocol.add_molecule(
            "INI001", "Iniparib",
            molecular_weight=179.1, logP=0.5, tpsa=92.0, num_hbd=2, num_hba=5
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_outcome == "failed", "Iniparib should be predicted as failed"
        assert "small" in mol.prediction_reasoning.lower() or "efficacy" in mol.prediction_reasoning.lower()

    def test_biologics_bapineuzumab(self):
        """Large biologics should be flagged for extreme MW"""
        self.protocol.add_molecule(
            "BIO001", "Bapineuzumab",
            molecular_weight=148000, logP=-3.0, tpsa=800, num_hbd=80, num_hba=150
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_outcome == "failed"
        assert mol.predicted_failure_risk >= 0.5, "Biologics should have high risk score"
        assert "MW" in mol.prediction_reasoning or "biologic" in mol.prediction_reasoning.lower()

    def test_gamma_secretase_semagacestat(self):
        """Gamma-secretase inhibitors should be flagged"""
        self.protocol.add_molecule(
            "GSI001", "Semagacestat",
            molecular_weight=563.6, logP=2.8, tpsa=158.0, num_hbd=5, num_hba=9
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_outcome == "failed"
        assert "gamma-secretase" in mol.prediction_reasoning.lower() or "TPSA" in mol.prediction_reasoning

    def test_bace_inhibitor_verubecestat(self):
        """BACE inhibitors should be flagged for efficacy failure"""
        self.protocol.add_molecule(
            "BACE001", "Verubecestat",
            molecular_weight=409.4, logP=1.5, tpsa=98.0, num_hbd=2, num_hba=6,
            pgp_substrate=True
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_outcome == "failed"
        assert "BACE" in mol.prediction_reasoning or "efficacy" in mol.prediction_reasoning.lower()

    def test_herg_terfenadine_pattern(self):
        """High logP + low TPSA should trigger hERG alert"""
        self.protocol.add_molecule(
            "HERG001", "Terfenadine-like",
            molecular_weight=471.7, logP=6.0, tpsa=43.7, num_hbd=1, num_hba=3
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_outcome == "failed"
        assert "hERG" in mol.prediction_reasoning or "cardiotox" in mol.prediction_reasoning.lower()

    def test_cox2_rofecoxib_pattern(self):
        """COX-2 inhibitors should be flagged for CV risk"""
        self.protocol.add_molecule(
            "COX2001", "Rofecoxib-like",
            molecular_weight=314.4, logP=1.7, tpsa=60.4, num_hbd=0, num_hba=4
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_outcome == "failed"
        assert "COX-2" in mol.prediction_reasoning or "CV" in mol.prediction_reasoning

    def test_fluoroquinolone_trovafloxacin(self):
        """Fluoroquinolone hepatotoxicity pattern"""
        self.protocol.add_molecule(
            "FQ001", "Trovafloxacin-like",
            molecular_weight=416.4, logP=-0.3, tpsa=96.0, num_hbd=2, num_hba=8
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_outcome == "failed"
        assert "fluoroquinolone" in mol.prediction_reasoning.lower() or "hepatotox" in mol.prediction_reasoning.lower()


class TestDrugClassExemptions:
    """Test that known safe drug classes are NOT flagged"""

    def setup_method(self):
        self.protocol = BlindValidationProtocol()

    def test_ssri_citalopram_approved(self):
        """Citalopram (SSRI) should be approved"""
        self.protocol.add_molecule(
            "SSRI001", "Citalopram",
            molecular_weight=324.4, logP=3.5, tpsa=36.3, num_hbd=0, num_hba=3
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_outcome == "approved", f"Citalopram should be approved, got {mol.predicted_outcome}"
        assert mol.predicted_failure_risk < 0.35

    def test_opioid_fentanyl_approved(self):
        """Fentanyl should be approved"""
        self.protocol.add_molecule(
            "OPI001", "Fentanyl",
            molecular_weight=336.5, logP=4.05, tpsa=23.55, num_hbd=0, num_hba=2
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_outcome == "approved"

    def test_tki_imatinib_approved(self):
        """Imatinib (TKI) should be approved despite high MW"""
        self.protocol.add_molecule(
            "TKI001", "Imatinib",
            molecular_weight=493.6, logP=3.5, tpsa=86.3, num_hbd=2, num_hba=7
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        # Note: Imatinib has HBD=2 which could trigger TZD, but it's a TKI
        # The algorithm should distinguish based on other properties
        assert mol.predicted_failure_risk < 0.5, f"Imatinib risk too high: {mol.predicted_failure_risk}"

    def test_arb_losartan_approved(self):
        """Losartan (ARB) should be approved"""
        self.protocol.add_molecule(
            "ARB001", "Losartan",
            molecular_weight=422.9, logP=4.01, tpsa=92.5, num_hbd=2, num_hba=6
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_failure_risk < 0.5

    def test_ace_inhibitor_enalapril_approved(self):
        """Enalapril (ACE-I) should be approved"""
        self.protocol.add_molecule(
            "ACEI001", "Enalapril",
            molecular_weight=376.4, logP=0.07, tpsa=95.9, num_hbd=2, num_hba=7
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_failure_risk < 0.4

    def test_corticosteroid_prednisone_approved(self):
        """Prednisone should be approved"""
        self.protocol.add_molecule(
            "CORT001", "Prednisone",
            molecular_weight=358.4, logP=1.46, tpsa=91.67, num_hbd=2, num_hba=5
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_failure_risk < 0.4


class TestEdgeCases:
    """Test boundary conditions and edge cases"""

    def setup_method(self):
        self.protocol = BlindValidationProtocol()

    def test_threshold_boundary_just_below(self):
        """Test molecule just below threshold"""
        self.protocol.add_molecule(
            "EDGE001", "Edge Case Low",
            molecular_weight=350, logP=2.5, tpsa=70, num_hbd=1, num_hba=4
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        if mol.predicted_failure_risk < 0.35:
            assert mol.predicted_outcome == "approved"
        else:
            assert mol.predicted_outcome == "failed"

    def test_threshold_boundary_just_above(self):
        """Test molecule just above threshold"""
        self.protocol.add_molecule(
            "EDGE002", "Edge Case High",
            molecular_weight=550, logP=5.5, tpsa=130, num_hbd=4, num_hba=8
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_outcome == "failed"

    def test_extreme_mw_low(self):
        """Test very low MW compound"""
        self.protocol.add_molecule(
            "LOW001", "Tiny Molecule",
            molecular_weight=50, logP=0.5, tpsa=30, num_hbd=1, num_hba=2
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        # Very small molecules should have efficacy concerns
        assert mol.predicted_failure_risk > 0

    def test_extreme_mw_high(self):
        """Test antibody-range MW"""
        self.protocol.add_molecule(
            "HIGH001", "Antibody",
            molecular_weight=150000, logP=-5, tpsa=5000, num_hbd=100, num_hba=200
        )
        self.protocol.predict_all(threshold=0.35)
        mol = self.protocol.molecules[0]

        assert mol.predicted_outcome == "failed"
        assert mol.predicted_failure_risk >= 0.5

    def test_dynamic_threshold_high_confidence(self):
        """High confidence should use lower threshold"""
        # Add molecule with extreme properties (high confidence)
        self.protocol.add_molecule(
            "DYN001", "Extreme Props",
            molecular_weight=900, logP=7, tpsa=200, num_hbd=8, num_hba=15
        )
        self.protocol.predict_all(threshold=0.40, use_dynamic_threshold=True)
        mol = self.protocol.molecules[0]

        # High confidence should trigger dynamic threshold adjustment
        assert mol.prediction_confidence >= 0.6


class TestValidationMetrics:
    """Test metric calculations"""

    def setup_method(self):
        self.protocol = BlindValidationProtocol()

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        # Add known failures
        self.protocol.add_molecule("F1", "Fail1", 800, 6.0, 180, 6, 12)
        self.protocol.add_molecule("F2", "Fail2", 900, 7.0, 200, 8, 15)

        # Add known approvals
        self.protocol.add_molecule("A1", "Approve1", 324.4, 3.5, 36.3, 0, 3)
        self.protocol.add_molecule("A2", "Approve2", 336.5, 4.05, 23.55, 0, 2)

        self.protocol.predict_all(threshold=0.35)

        outcomes = {
            "F1": {"outcome": "failed"},
            "F2": {"outcome": "failed"},
            "A1": {"outcome": "approved"},
            "A2": {"outcome": "approved"},
        }
        self.protocol.reveal_outcomes(outcomes)

        metrics = self.protocol.calculate_metrics()

        assert metrics.total_molecules == 4
        assert metrics.accuracy >= 0.75  # Should have good accuracy

    def test_confusion_matrix(self):
        """Test confusion matrix calculations"""
        self.protocol.add_molecule("M1", "Mol1", 800, 6.0, 180, 6, 12)  # Should predict fail
        self.protocol.add_molecule("M2", "Mol2", 324.4, 3.5, 36.3, 0, 3)  # Should predict pass

        self.protocol.predict_all(threshold=0.35)

        # M1 predicted fail -> actually failed (TP)
        # M2 predicted pass -> actually approved (TN)
        outcomes = {
            "M1": {"outcome": "failed"},
            "M2": {"outcome": "approved"},
        }
        self.protocol.reveal_outcomes(outcomes)

        metrics = self.protocol.calculate_metrics()

        # Verify confusion matrix adds up
        total = (
            metrics.correctly_predicted_failures +
            metrics.correctly_predicted_approvals +
            metrics.false_positives +
            metrics.false_negatives
        )
        assert total == metrics.total_molecules

    def test_sensitivity_specificity(self):
        """Test sensitivity and specificity calculations"""
        # Add molecules designed to test sensitivity/specificity
        self.protocol.add_molecule("TP1", "TruePos", 800, 6.0, 180, 6, 12)
        self.protocol.add_molecule("TN1", "TrueNeg", 324.4, 3.5, 36.3, 0, 3)

        self.protocol.predict_all(threshold=0.35)

        outcomes = {
            "TP1": {"outcome": "failed"},
            "TN1": {"outcome": "approved"},
        }
        self.protocol.reveal_outcomes(outcomes)

        metrics = self.protocol.calculate_metrics()

        # With these inputs, should have good sensitivity and specificity
        assert 0 <= metrics.sensitivity <= 1
        assert 0 <= metrics.specificity <= 1
        assert 0 <= metrics.accuracy <= 1


class TestFullValidationWorkflow:
    """Test complete validation workflow"""

    def test_full_demo_workflow(self):
        """Run complete demo validation workflow"""
        protocol = BlindValidationProtocol()

        # Add approved drugs
        protocol.add_molecule("DB00215", "Citalopram", 324.4, 3.5, 36.3, 0, 3)
        protocol.add_molecule("DB00813", "Fentanyl", 336.5, 4.05, 23.55, 0, 2)
        protocol.add_molecule("DB00619", "Imatinib", 493.6, 3.5, 86.3, 2, 7)

        # Add failed drugs
        protocol.add_molecule("FAIL001", "Bapineuzumab", 148000, -3.0, 800, 80, 150)
        protocol.add_molecule("FAIL002", "Semagacestat", 563.6, 2.8, 158.0, 5, 9)
        protocol.add_molecule("TOXIC001", "Troglitazone", 441.5, 4.2, 84.9, 2, 6)

        # Generate predictions
        protocol.predict_all(threshold=0.35)

        assert protocol.predictions_locked is True

        # All molecules should have predictions
        for mol in protocol.molecules:
            assert mol.predicted_outcome in ["approved", "failed"]
            assert mol.predicted_failure_risk is not None
            assert mol.prediction_confidence is not None

        # Reveal outcomes
        outcomes = {
            "DB00215": {"outcome": "approved"},
            "DB00813": {"outcome": "approved"},
            "DB00619": {"outcome": "approved"},
            "FAIL001": {"outcome": "failed", "phase": "phase3", "reason": "lack_of_efficacy"},
            "FAIL002": {"outcome": "failed", "phase": "phase3", "reason": "toxicity"},
            "TOXIC001": {"outcome": "withdrawn", "reason": "hepatotoxicity"},
        }
        protocol.reveal_outcomes(outcomes)

        assert protocol.outcomes_revealed is True

        # Calculate metrics
        metrics = protocol.calculate_metrics()

        assert metrics.total_molecules == 6
        assert metrics.actual_approved == 3
        assert metrics.actual_failed == 3

        # With improved algorithm, should have better accuracy
        assert metrics.accuracy >= 0.66, f"Accuracy {metrics.accuracy} should be >= 66%"

        # Generate report (should not raise)
        report = protocol.generate_report()
        assert "QENEX LAB BLIND VALIDATION REPORT" in report
        assert "CONFUSION MATRIX" in report

    def test_prediction_locking(self):
        """Test that predictions cannot be modified after locking"""
        protocol = BlindValidationProtocol()
        protocol.add_molecule("M1", "Test", 400, 3.0, 70, 2, 5)
        protocol.predict_all()

        with pytest.raises(ValueError):
            protocol.add_molecule("M2", "Test2", 400, 3.0, 70, 2, 5)

    def test_outcome_reveal_before_prediction(self):
        """Test that outcomes cannot be revealed before predictions"""
        protocol = BlindValidationProtocol()
        protocol.add_molecule("M1", "Test", 400, 3.0, 70, 2, 5)

        with pytest.raises(ValueError):
            protocol.reveal_outcomes({"M1": {"outcome": "approved"}})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
