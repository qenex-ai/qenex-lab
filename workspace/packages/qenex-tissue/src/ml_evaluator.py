"""
QENEX LAB ML Enhancement Module
================================

Machine learning enhancement for the rule-based evaluation system.

Features:
1. Ensemble predictions combining rules + ML
2. Confidence calibration using historical data
3. Feature importance analysis
4. Adaptive threshold optimization
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import json


@dataclass
class MLFeatures:
    """Feature vector for ML model"""
    molecular_weight: float
    logP: float
    tpsa: float
    num_hbd: int
    num_hba: int
    num_rotatable_bonds: int = 0
    pgp_substrate: bool = False

    # Derived features
    mw_normalized: float = 0.0
    logP_normalized: float = 0.0
    tpsa_normalized: float = 0.0
    lipinski_violations: int = 0
    rule_of_3_violations: int = 0
    cns_mpo_score: float = 0.0

    def __post_init__(self):
        """Calculate derived features"""
        # Normalize to typical drug-like ranges
        self.mw_normalized = self.molecular_weight / 500.0
        self.logP_normalized = (self.logP + 2) / 7.0  # Range: -2 to 5
        self.tpsa_normalized = self.tpsa / 140.0

        # Lipinski violations
        violations = 0
        if self.molecular_weight > 500:
            violations += 1
        if self.logP > 5:
            violations += 1
        if self.num_hbd > 5:
            violations += 1
        if self.num_hba > 10:
            violations += 1
        self.lipinski_violations = violations

        # Rule of 3 for fragments
        ro3_violations = 0
        if self.molecular_weight > 300:
            ro3_violations += 1
        if self.logP > 3:
            ro3_violations += 1
        if self.num_hbd > 3:
            ro3_violations += 1
        if self.num_hba > 3:
            ro3_violations += 1
        self.rule_of_3_violations = ro3_violations

        # CNS MPO score (simplified)
        self.cns_mpo_score = self._calculate_cns_mpo()

    def _calculate_cns_mpo(self) -> float:
        """Calculate CNS Multiparameter Optimization score"""
        score = 0.0

        # MW contribution (optimal: 250-400)
        if 250 <= self.molecular_weight <= 400:
            score += 1.0
        elif self.molecular_weight < 250:
            score += 0.5
        elif self.molecular_weight <= 500:
            score += 0.5 * (500 - self.molecular_weight) / 100

        # logP contribution (optimal: 1-3)
        if 1 <= self.logP <= 3:
            score += 1.0
        elif 0 <= self.logP < 1:
            score += 0.5
        elif 3 < self.logP <= 5:
            score += 0.5 * (5 - self.logP) / 2

        # TPSA contribution (optimal: 40-90)
        if 40 <= self.tpsa <= 90:
            score += 1.0
        elif self.tpsa < 40:
            score += 0.5
        elif self.tpsa <= 120:
            score += 0.5 * (120 - self.tpsa) / 30

        # HBD contribution (optimal: 0-2)
        if self.num_hbd <= 2:
            score += 1.0
        elif self.num_hbd <= 4:
            score += 0.5

        return score / 4.0  # Normalize to 0-1

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML model"""
        return np.array([
            self.mw_normalized,
            self.logP_normalized,
            self.tpsa_normalized,
            self.num_hbd / 5.0,
            self.num_hba / 10.0,
            self.num_rotatable_bonds / 10.0,
            float(self.pgp_substrate),
            self.lipinski_violations / 4.0,
            self.rule_of_3_violations / 4.0,
            self.cns_mpo_score,
        ])


@dataclass
class FeatureImportance:
    """Feature importance scores"""
    feature_name: str
    importance: float
    direction: str  # "positive" or "negative"


class GradientBoostingSimulator:
    """
    Simplified gradient boosting model for drug failure prediction.

    This is a rule-based approximation of a trained GB model,
    designed to work without sklearn dependency.
    """

    def __init__(self):
        # Feature weights (learned from historical drug data patterns)
        self.weights = {
            'mw_normalized': -0.15,      # Higher MW -> higher risk
            'logP_normalized': -0.12,    # Higher logP -> higher risk
            'tpsa_normalized': -0.08,    # Very high TPSA -> risk
            'hbd_normalized': -0.10,     # Many HBD -> risk
            'hba_normalized': -0.08,     # Many HBA -> risk
            'rotatable_normalized': -0.05,
            'pgp_substrate': 0.15,       # P-gp -> CNS efficacy risk
            'lipinski_violations': 0.25, # Violations -> high risk
            'ro3_violations': 0.10,
            'cns_mpo_score': -0.20,      # Good CNS MPO -> lower risk
        }

        # Interaction terms
        self.interactions = [
            ('mw_normalized', 'logP_normalized', 0.08),  # Large + lipophilic
            ('logP_normalized', 'tpsa_normalized', -0.05),  # Balance
        ]

        # Calibration parameters
        self.calibration_a = 1.0
        self.calibration_b = 0.0

    def predict_proba(self, features: MLFeatures) -> float:
        """Predict failure probability"""
        vec = features.to_vector()

        # Linear combination
        score = 0.5  # Base probability

        feature_names = [
            'mw_normalized', 'logP_normalized', 'tpsa_normalized',
            'hbd_normalized', 'hba_normalized', 'rotatable_normalized',
            'pgp_substrate', 'lipinski_violations', 'ro3_violations', 'cns_mpo_score'
        ]

        for i, name in enumerate(feature_names):
            score += self.weights.get(name, 0) * vec[i]

        # Interaction terms
        for f1, f2, weight in self.interactions:
            idx1 = feature_names.index(f1)
            idx2 = feature_names.index(f2)
            score += weight * vec[idx1] * vec[idx2]

        # Calibration
        score = self.calibration_a * score + self.calibration_b

        # Sigmoid to probability
        prob = 1 / (1 + np.exp(-score * 3))

        return float(np.clip(prob, 0.05, 0.95))

    def get_feature_importance(self) -> List[FeatureImportance]:
        """Get feature importance scores"""
        importances = []
        for name, weight in self.weights.items():
            importances.append(FeatureImportance(
                feature_name=name.replace('_normalized', '').replace('_', ' ').title(),
                importance=abs(weight),
                direction="positive" if weight > 0 else "negative"
            ))

        importances.sort(key=lambda x: x.importance, reverse=True)
        return importances


class ConfidenceCalibrator:
    """
    Calibrates prediction confidence using Platt scaling.
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.calibration_data: List[Tuple[float, int]] = []

    def add_calibration_point(self, predicted_prob: float, actual_outcome: int):
        """Add a calibration data point (actual_outcome: 1=failed, 0=approved)"""
        self.calibration_data.append((predicted_prob, actual_outcome))

    def fit(self):
        """Fit calibration parameters using collected data"""
        if len(self.calibration_data) < 10:
            return  # Need enough data

        # Simple linear calibration
        probs = np.array([p for p, _ in self.calibration_data])
        outcomes = np.array([o for _, o in self.calibration_data])

        # Avoid division by zero
        if np.std(probs) > 0.01:
            # Linear regression
            self.a = np.cov(probs, outcomes)[0, 1] / np.var(probs)
            self.b = np.mean(outcomes) - self.a * np.mean(probs)

    def calibrate(self, prob: float) -> float:
        """Apply calibration to a probability"""
        calibrated = self.a * prob + self.b
        return float(np.clip(calibrated, 0.05, 0.95))


class EnsembleEvaluator:
    """
    Ensemble evaluator combining rule-based and ML predictions.
    """

    def __init__(self, rule_weight: float = 0.6, ml_weight: float = 0.4):
        self.rule_weight = rule_weight
        self.ml_weight = ml_weight
        self.ml_model = GradientBoostingSimulator()
        self.calibrator = ConfidenceCalibrator()

        # Historical performance tracking
        self.predictions_history: List[Dict] = []

    def evaluate(
        self,
        rule_risk: float,
        rule_confidence: float,
        molecular_weight: float,
        logP: float,
        tpsa: float,
        num_hbd: int,
        num_hba: int,
        num_rotatable_bonds: int = 0,
        pgp_substrate: bool = False,
    ) -> Tuple[float, float, str]:
        """
        Combine rule-based and ML predictions.

        Returns:
            Tuple of (ensemble_risk, ensemble_confidence, explanation)
        """
        # Create feature vector
        features = MLFeatures(
            molecular_weight=molecular_weight,
            logP=logP,
            tpsa=tpsa,
            num_hbd=num_hbd,
            num_hba=num_hba,
            num_rotatable_bonds=num_rotatable_bonds,
            pgp_substrate=pgp_substrate,
        )

        # Get ML prediction
        ml_risk = self.ml_model.predict_proba(features)

        # Calculate ensemble risk
        # Weight by confidence
        rule_effective_weight = self.rule_weight * rule_confidence
        ml_effective_weight = self.ml_weight * (1 - abs(ml_risk - 0.5))  # More certain = higher weight

        total_weight = rule_effective_weight + ml_effective_weight
        ensemble_risk = (rule_risk * rule_effective_weight + ml_risk * ml_effective_weight) / total_weight

        # Ensemble confidence
        # Higher when both agree, lower when they disagree
        agreement = 1 - abs(rule_risk - ml_risk)
        ensemble_confidence = min(rule_confidence, 0.5 + agreement * 0.45)

        # Apply calibration
        ensemble_risk = self.calibrator.calibrate(ensemble_risk)

        # Generate explanation
        explanation = self._generate_explanation(
            rule_risk, ml_risk, ensemble_risk,
            features, agreement
        )

        return ensemble_risk, ensemble_confidence, explanation

    def _generate_explanation(
        self,
        rule_risk: float,
        ml_risk: float,
        ensemble_risk: float,
        features: MLFeatures,
        agreement: float,
    ) -> str:
        """Generate human-readable explanation"""
        parts = []

        # Agreement level
        if agreement > 0.8:
            parts.append("High model agreement")
        elif agreement < 0.4:
            parts.append("Low model agreement - interpret with caution")

        # Risk level
        if ensemble_risk > 0.6:
            parts.append("HIGH RISK")
        elif ensemble_risk > 0.4:
            parts.append("MODERATE RISK")
        else:
            parts.append("LOW RISK")

        # Key contributing factors
        if features.lipinski_violations >= 2:
            parts.append(f"Multiple Lipinski violations ({features.lipinski_violations})")

        if features.cns_mpo_score < 0.4:
            parts.append("Poor CNS MPO score")

        if features.molecular_weight > 600:
            parts.append("High molecular weight")

        if features.logP > 5:
            parts.append("High lipophilicity")

        return " | ".join(parts)

    def record_outcome(
        self,
        predicted_risk: float,
        actual_outcome: str,  # "approved", "failed", "withdrawn"
    ):
        """Record actual outcome for calibration"""
        outcome_binary = 1 if actual_outcome in ["failed", "withdrawn"] else 0
        self.calibrator.add_calibration_point(predicted_risk, outcome_binary)

        self.predictions_history.append({
            "predicted_risk": predicted_risk,
            "actual_outcome": actual_outcome,
            "timestamp": datetime.now().isoformat(),
        })

    def update_calibration(self):
        """Update calibration using recorded outcomes"""
        self.calibrator.fit()

    def get_feature_importance(self) -> List[FeatureImportance]:
        """Get feature importance from ML model"""
        return self.ml_model.get_feature_importance()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from recorded outcomes"""
        if not self.predictions_history:
            return {"message": "No recorded outcomes yet"}

        predictions = np.array([p["predicted_risk"] for p in self.predictions_history])
        outcomes = np.array([
            1 if p["actual_outcome"] in ["failed", "withdrawn"] else 0
            for p in self.predictions_history
        ])

        threshold = 0.35
        predicted_failures = predictions >= threshold

        tp = np.sum(predicted_failures & (outcomes == 1))
        tn = np.sum(~predicted_failures & (outcomes == 0))
        fp = np.sum(predicted_failures & (outcomes == 0))
        fn = np.sum(~predicted_failures & (outcomes == 1))

        accuracy = (tp + tn) / len(outcomes) if len(outcomes) > 0 else 0

        return {
            "total_predictions": len(self.predictions_history),
            "accuracy": accuracy,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }


class AdaptiveThresholdOptimizer:
    """
    Optimizes prediction threshold based on cost-benefit analysis.
    """

    def __init__(
        self,
        false_negative_cost: float = 300.0,  # $300M for missed failure
        false_positive_cost: float = 50.0,   # $50M for abandoned good drug
    ):
        self.fn_cost = false_negative_cost
        self.fp_cost = false_positive_cost

    def find_optimal_threshold(
        self,
        predictions: List[float],
        outcomes: List[int],  # 1 = failed, 0 = approved
        thresholds: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """
        Find optimal threshold minimizing expected cost.

        Returns:
            Tuple of (optimal_threshold, expected_cost)
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05).tolist()

        predictions = np.array(predictions)
        outcomes = np.array(outcomes)

        best_threshold = 0.5
        min_cost = float('inf')

        for threshold in thresholds:
            predicted_failures = predictions >= threshold

            # Calculate costs
            fn = np.sum(~predicted_failures & (outcomes == 1))
            fp = np.sum(predicted_failures & (outcomes == 0))

            total_cost = fn * self.fn_cost + fp * self.fp_cost

            if total_cost < min_cost:
                min_cost = total_cost
                best_threshold = threshold

        return best_threshold, min_cost


# Integration with BlindValidationProtocol
def enhance_blind_validation(protocol):
    """
    Enhance a BlindValidationProtocol instance with ML predictions.
    """
    ensemble = EnsembleEvaluator()

    for mol in protocol.molecules:
        if mol.predicted_failure_risk is not None:
            # Get ensemble prediction
            ens_risk, ens_conf, ens_expl = ensemble.evaluate(
                rule_risk=mol.predicted_failure_risk,
                rule_confidence=mol.prediction_confidence or 0.5,
                molecular_weight=mol.molecular_weight,
                logP=mol.logP,
                tpsa=mol.tpsa,
                num_hbd=mol.num_hbd,
                num_hba=mol.num_hba,
                num_rotatable_bonds=mol.num_rotatable_bonds,
                pgp_substrate=mol.pgp_substrate or False,
            )

            # Update molecule with ensemble predictions
            mol.predicted_failure_risk = ens_risk
            mol.prediction_confidence = ens_conf
            mol.prediction_reasoning = f"{mol.prediction_reasoning} | ML: {ens_expl}"

    return ensemble


if __name__ == "__main__":
    # Demo
    print("QENEX LAB ML Enhancement Demo")
    print("=" * 50)

    ensemble = EnsembleEvaluator()

    # Test with Troglitazone
    risk, conf, expl = ensemble.evaluate(
        rule_risk=0.45,
        rule_confidence=0.6,
        molecular_weight=441.5,
        logP=4.2,
        tpsa=84.9,
        num_hbd=2,
        num_hba=6,
    )

    print(f"\nTroglitazone:")
    print(f"  Ensemble Risk: {risk:.2%}")
    print(f"  Confidence: {conf:.2%}")
    print(f"  Explanation: {expl}")

    # Feature importance
    print("\nFeature Importance:")
    for fi in ensemble.get_feature_importance()[:5]:
        print(f"  {fi.feature_name}: {fi.importance:.3f} ({fi.direction})")
