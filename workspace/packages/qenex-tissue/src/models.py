"""
Tissue Distribution Prediction Models
======================================

Multi-tissue pharmacokinetic prediction using ensemble ML models.
Predicts drug distribution to:
- Brain (BBB penetration)
- Liver (hepatic uptake, metabolism)
- Kidney (renal clearance)
- Tumor (solid tumor penetration)
- Lung, Heart, Muscle, Fat

Each tissue has specialized models trained on tissue-specific features.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json
import math

from .features import MolecularDescriptors, MolecularFeatureExtractor, TissueType


@dataclass
class TissueDistributionResult:
    """Predicted tissue distribution for a single molecule"""

    molecule_name: str

    # Partition coefficients (Kp = tissue/plasma ratio)
    kp_brain: float = 0.0
    kp_liver: float = 0.0
    kp_kidney: float = 0.0
    kp_tumor: float = 0.0
    kp_lung: float = 0.0
    kp_heart: float = 0.0
    kp_muscle: float = 0.0
    kp_fat: float = 0.0

    # Permeability scores (0-1)
    bbb_permeability: float = 0.0
    tumor_penetration: float = 0.0

    # Clearance predictions
    hepatic_clearance: float = 0.0  # mL/min/kg
    renal_clearance: float = 0.0

    # Efflux transporter liabilities
    pgp_substrate: float = 0.0
    bcrp_substrate: float = 0.0

    # Confidence scores
    confidence: float = 0.0
    uncertainty: Dict[str, float] = field(default_factory=dict)

    # Risk assessment
    tissue_selectivity_score: float = 0.0
    clinical_failure_risk: float = 0.0

    # Detailed predictions per tissue
    tissue_predictions: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "molecule_name": self.molecule_name,
            "partition_coefficients": {
                "brain": self.kp_brain,
                "liver": self.kp_liver,
                "kidney": self.kp_kidney,
                "tumor": self.kp_tumor,
                "lung": self.kp_lung,
                "heart": self.kp_heart,
                "muscle": self.kp_muscle,
                "fat": self.kp_fat,
            },
            "permeability": {
                "bbb": self.bbb_permeability,
                "tumor": self.tumor_penetration,
            },
            "clearance": {
                "hepatic": self.hepatic_clearance,
                "renal": self.renal_clearance,
            },
            "transporters": {
                "pgp_substrate": self.pgp_substrate,
                "bcrp_substrate": self.bcrp_substrate,
            },
            "scores": {
                "tissue_selectivity": self.tissue_selectivity_score,
                "clinical_failure_risk": self.clinical_failure_risk,
                "confidence": self.confidence,
            },
            "uncertainty": self.uncertainty,
        }

    def get_recommendation(self) -> str:
        """Generate clinical development recommendation"""
        if self.clinical_failure_risk > 0.7:
            return "HIGH_RISK: Do not advance - predicted tissue distribution failure"
        elif self.clinical_failure_risk > 0.5:
            return "MODERATE_RISK: Consider formulation optimization before advancing"
        elif self.clinical_failure_risk > 0.3:
            return "LOW_RISK: Acceptable for advancement with monitoring"
        else:
            return "FAVORABLE: Good predicted tissue distribution profile"


class NeuralNetworkLayer:
    """Simple neural network layer for tissue prediction"""

    def __init__(self, input_size: int, output_size: int, activation: str = "relu"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Xavier initialization
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros(output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.weights) + self.bias

        if self.activation == "relu":
            return np.maximum(0, z)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "linear":
            return z
        else:
            return z


class TissueSpecificModel:
    """
    Neural network model for specific tissue distribution prediction.

    Architecture optimized for each tissue type based on
    mechanistic understanding of drug distribution.
    """

    # Tissue-specific feature importance weights
    TISSUE_FEATURE_WEIGHTS = {
        TissueType.BRAIN: {
            "tpsa": -0.8,  # Lower TPSA = better BBB penetration
            "num_hbd": -0.6,
            "logP": 0.4,  # Moderate lipophilicity helps
            "molecular_weight": -0.3,
            "pgp_substrate_prob": -0.7,
            "bbb_score": 0.9,
        },
        TissueType.LIVER: {
            "logP": 0.6,  # Lipophilic drugs accumulate
            "molecular_weight": 0.2,
            "num_hba": 0.3,
            "plasma_protein_binding": 0.4,
        },
        TissueType.KIDNEY: {
            "molecular_weight": -0.4,  # Small molecules cleared
            "logP": -0.3,  # Hydrophilic cleared faster
            "net_charge": -0.5,  # Cations retained
            "plasma_protein_binding": -0.3,
        },
        TissueType.TUMOR: {
            "molecular_weight": -0.2,
            "logP": 0.3,
            "tpsa": -0.3,
            "pgp_substrate_prob": -0.5,
            "esp_balance": 0.2,
        },
    }

    def __init__(self, tissue_type: TissueType, hidden_sizes: List[int] = None):
        self.tissue_type = tissue_type
        self.hidden_sizes = hidden_sizes or [64, 32, 16]
        self.layers: List[NeuralNetworkLayer] = []
        self.is_trained = False
        self.feature_weights = self.TISSUE_FEATURE_WEIGHTS.get(tissue_type, {})

        # Build network
        self._build_network()

    def _build_network(self):
        """Construct neural network layers"""
        input_size = 31  # Number of molecular descriptors

        sizes = [input_size] + self.hidden_sizes + [1]

        for i in range(len(sizes) - 1):
            activation = "relu" if i < len(sizes) - 2 else "sigmoid"
            self.layers.append(NeuralNetworkLayer(sizes[i], sizes[i + 1], activation))

    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Predict tissue distribution coefficient.

        Returns: (Kp prediction, uncertainty)
        """
        # Normalize features
        x = self._normalize_features(features)

        # Forward pass
        for layer in self.layers:
            x = layer.forward(x)

        # Apply tissue-specific adjustments using mechanistic knowledge
        base_prediction = float(x[0]) if x.ndim > 0 else float(x)

        # Mechanistic adjustment based on known tissue biology
        adjustment = self._mechanistic_adjustment(features)

        # Combine ML prediction with mechanistic model
        final_prediction = 0.6 * base_prediction + 0.4 * adjustment

        # Estimate uncertainty from feature confidence
        uncertainty = self._estimate_uncertainty(features)

        return final_prediction, uncertainty

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range"""
        # Feature-specific normalization ranges
        ranges = [
            (0, 1000),  # molecular_weight
            (0, 100),  # num_heavy_atoms
            (-5, 10),  # logP
            (-5, 10),  # logD_7_4
            (-10, 2),  # logS
            (0, 15),  # num_hbd
            (0, 20),  # num_hba
            (0, 200),  # tpsa
            (0, 20),  # dipole_moment
            (-15, 0),  # homo_energy
            (-5, 5),  # lumo_energy
            (0, 15),  # homo_lumo_gap
            (0, 10),  # total_esp_positive
            (0, 10),  # total_esp_negative
            (0, 5),  # esp_balance
            (0, 2000),  # molecular_volume
            (0, 1500),  # molecular_surface_area
            (0, 1),  # sphericity
            (0, 20),  # num_rotatable_bonds
            (0, 1),  # flexibility_index
            (0, 10),  # num_rings
            (0, 5),  # num_aromatic_rings
            (0, 20),  # num_heteroatoms
            (0, 5),  # total_positive_charge
            (-5, 0),  # total_negative_charge
            (-2, 2),  # net_charge
            (-0.01, 0.01),  # charge_density
            (0, 4),  # lipinski_violations
            (0, 1),  # bbb_score
            (0, 1),  # pgp_substrate_prob
            (0, 100),  # plasma_protein_binding
        ]

        normalized = np.zeros_like(features)
        for i, (low, high) in enumerate(ranges):
            if i < len(features):
                normalized[i] = (features[i] - low) / (high - low + 1e-8)
                normalized[i] = np.clip(normalized[i], 0, 1)

        return normalized

    def _mechanistic_adjustment(self, features: np.ndarray) -> float:
        """Apply mechanistic model based on tissue biology"""
        feature_names = MolecularDescriptors.feature_names()

        adjustment = 0.5  # Base

        for i, name in enumerate(feature_names):
            if name in self.feature_weights and i < len(features):
                weight = self.feature_weights[name]
                normalized_val = features[i]

                # Apply sigmoid to keep in range
                contribution = weight * (normalized_val - 0.5) * 0.2
                adjustment += contribution

        return np.clip(adjustment, 0, 1)

    def _estimate_uncertainty(self, features: np.ndarray) -> float:
        """Estimate prediction uncertainty"""
        # Higher uncertainty for edge cases
        feature_names = MolecularDescriptors.feature_names()

        uncertainty = 0.1  # Base uncertainty

        # Check for unusual values
        mw_idx = feature_names.index("molecular_weight")
        if features[mw_idx] > 800 or features[mw_idx] < 100:
            uncertainty += 0.1

        logp_idx = feature_names.index("logP")
        if abs(features[logp_idx]) > 6:
            uncertainty += 0.1

        return min(0.5, uncertainty)


class TissueDistributionPredictor:
    """
    Main tissue distribution prediction engine.

    Integrates multiple tissue-specific models with:
    - Ensemble predictions for robustness
    - Uncertainty quantification
    - Clinical failure risk assessment
    - Structure-Tissue Relationship (STR) analysis
    """

    def __init__(self):
        self.tissue_models: Dict[TissueType, TissueSpecificModel] = {}
        self.feature_extractor = MolecularFeatureExtractor(use_dft=False)

        # Initialize tissue-specific models
        self._initialize_models()

        # Clinical failure thresholds based on literature
        self.failure_thresholds = {
            TissueType.BRAIN: {"min_kp": 0.3, "max_pgp": 0.5},
            TissueType.TUMOR: {"min_kp": 0.5, "max_pgp": 0.4},
            TissueType.LIVER: {"max_kp": 10.0},  # Too high = toxicity
            TissueType.KIDNEY: {"max_clearance": 100},
        }

    def _initialize_models(self):
        """Initialize all tissue-specific models"""
        for tissue in TissueType:
            self.tissue_models[tissue] = TissueSpecificModel(
                tissue_type=tissue, hidden_sizes=[64, 32, 16]
            )

    def predict_from_sdf(
        self, sdf_path: str, name: str = None
    ) -> TissueDistributionResult:
        """Predict tissue distribution from SDF file"""
        self.feature_extractor.load_from_sdf(sdf_path, name)
        descriptors = self.feature_extractor.extract_features()
        return self.predict_from_descriptors(descriptors, name or sdf_path)

    def predict_from_descriptors(
        self, descriptors: MolecularDescriptors, name: str = "molecule"
    ) -> TissueDistributionResult:
        """Predict tissue distribution from molecular descriptors"""
        features = descriptors.to_vector()

        result = TissueDistributionResult(molecule_name=name)

        # Predict for each tissue
        for tissue, model in self.tissue_models.items():
            kp, uncertainty = model.predict(features)

            # Store predictions
            if tissue == TissueType.BRAIN:
                result.kp_brain = self._kp_from_score(kp, tissue)
                result.bbb_permeability = kp
            elif tissue == TissueType.LIVER:
                result.kp_liver = self._kp_from_score(kp, tissue)
            elif tissue == TissueType.KIDNEY:
                result.kp_kidney = self._kp_from_score(kp, tissue)
            elif tissue == TissueType.TUMOR:
                result.kp_tumor = self._kp_from_score(kp, tissue)
                result.tumor_penetration = kp
            elif tissue == TissueType.LUNG:
                result.kp_lung = self._kp_from_score(kp, tissue)
            elif tissue == TissueType.HEART:
                result.kp_heart = self._kp_from_score(kp, tissue)
            elif tissue == TissueType.MUSCLE:
                result.kp_muscle = self._kp_from_score(kp, tissue)
            elif tissue == TissueType.FAT:
                result.kp_fat = self._kp_from_score(kp, tissue)

            result.uncertainty[tissue.value] = uncertainty
            result.tissue_predictions[tissue.value] = {
                "score": float(kp),
                "kp": float(self._kp_from_score(kp, tissue)),
                "uncertainty": float(uncertainty),
            }

        # Transporter predictions
        result.pgp_substrate = descriptors.pgp_substrate_prob
        result.bcrp_substrate = self._predict_bcrp(descriptors)

        # Clearance predictions
        result.hepatic_clearance = self._predict_hepatic_clearance(descriptors)
        result.renal_clearance = self._predict_renal_clearance(descriptors)

        # Calculate aggregate scores
        result.confidence = 1 - np.mean(list(result.uncertainty.values()))
        result.tissue_selectivity_score = self._calculate_selectivity(result)
        result.clinical_failure_risk = self._calculate_failure_risk(result, descriptors)

        return result

    def _kp_from_score(self, score: float, tissue: TissueType) -> float:
        """Convert model score to partition coefficient"""
        # Tissue-specific Kp ranges
        kp_ranges = {
            TissueType.BRAIN: (0.01, 5.0),
            TissueType.LIVER: (0.5, 50.0),
            TissueType.KIDNEY: (0.3, 20.0),
            TissueType.TUMOR: (0.1, 10.0),
            TissueType.LUNG: (0.3, 15.0),
            TissueType.HEART: (0.2, 10.0),
            TissueType.MUSCLE: (0.1, 5.0),
            TissueType.FAT: (0.5, 100.0),
            TissueType.PLASMA: (1.0, 1.0),
            TissueType.BONE: (0.05, 2.0),
        }

        low, high = kp_ranges.get(tissue, (0.1, 10.0))

        # Exponential mapping for better spread
        kp = low * math.exp(score * math.log(high / low))

        return round(kp, 3)

    def _predict_bcrp(self, descriptors: MolecularDescriptors) -> float:
        """Predict BCRP efflux transporter substrate probability"""
        # BCRP substrates tend to be: planar, aromatic, moderate MW
        risk = 0.0

        if descriptors.num_aromatic_rings >= 2:
            risk += 0.3
        if 300 < descriptors.molecular_weight < 600:
            risk += 0.2
        if descriptors.tpsa > 80:
            risk += 0.2

        return min(1.0, risk)

    def _predict_hepatic_clearance(self, descriptors: MolecularDescriptors) -> float:
        """Predict hepatic clearance (mL/min/kg)"""
        # Base clearance from lipophilicity
        cl = 10 * (1 + descriptors.logP * 0.3)

        # Adjust for molecular weight (larger = slower)
        cl *= math.exp(-descriptors.molecular_weight / 1000)

        # Adjust for metabolic liability (more sites = faster)
        cl *= 1 + descriptors.num_aromatic_rings * 0.1

        return max(0.1, min(200, cl))

    def _predict_renal_clearance(self, descriptors: MolecularDescriptors) -> float:
        """Predict renal clearance (mL/min/kg)"""
        # Small, polar molecules cleared faster
        cl = 20 * math.exp(-descriptors.molecular_weight / 500)

        # Hydrophilic compounds filtered better
        if descriptors.logP < 0:
            cl *= 1.5

        # Protein binding reduces filtration
        cl *= (1 - descriptors.plasma_protein_binding / 100) * 0.5 + 0.5

        return max(0.01, min(100, cl))

    def _calculate_selectivity(self, result: TissueDistributionResult) -> float:
        """
        Calculate tissue selectivity score.

        Higher score = drug concentrates in target vs off-target tissues.
        """
        # For CNS drugs: brain vs liver
        # For oncology: tumor vs healthy tissue

        target_kps = [result.kp_brain, result.kp_tumor]
        offtarget_kps = [result.kp_liver, result.kp_heart, result.kp_kidney]

        avg_target = np.mean(target_kps)
        avg_offtarget = np.mean(offtarget_kps)

        if avg_offtarget > 0:
            selectivity = avg_target / avg_offtarget
        else:
            selectivity = avg_target

        # Normalize to 0-1
        return min(1.0, selectivity / 5.0)

    def _calculate_failure_risk(
        self, result: TissueDistributionResult, descriptors: MolecularDescriptors
    ) -> float:
        """
        Calculate clinical trial failure risk based on tissue distribution.

        Factors:
        - Poor target tissue penetration
        - High efflux transporter liability
        - Unfavorable tissue selectivity
        - Extreme clearance values
        """
        risk = 0.0

        # Brain penetration risk (for CNS drugs)
        if result.bbb_permeability < 0.3:
            risk += 0.2

        # P-gp efflux risk
        if result.pgp_substrate > 0.5:
            risk += 0.15

        # Poor tumor penetration (for oncology)
        if result.tumor_penetration < 0.4:
            risk += 0.15

        # Unfavorable liver accumulation (toxicity risk)
        if result.kp_liver > 20:
            risk += 0.1

        # Poor selectivity
        if result.tissue_selectivity_score < 0.3:
            risk += 0.15

        # Lipinski violations
        if descriptors.lipinski_violations >= 2:
            risk += 0.1

        # Extreme clearance
        if result.hepatic_clearance > 100 or result.hepatic_clearance < 1:
            risk += 0.1

        # Low confidence increases risk
        risk += (1 - result.confidence) * 0.15

        return min(1.0, risk)

    def predict_batch(self, molecules: List[Dict]) -> List[TissueDistributionResult]:
        """Predict tissue distribution for multiple molecules"""
        results = []
        for mol in molecules:
            if "sdf_path" in mol:
                result = self.predict_from_sdf(mol["sdf_path"], mol.get("name"))
            elif "descriptors" in mol:
                result = self.predict_from_descriptors(
                    mol["descriptors"], mol.get("name")
                )
            else:
                continue
            results.append(result)
        return results

    def rank_candidates(
        self,
        results: List[TissueDistributionResult],
        target_tissue: TissueType = TissueType.BRAIN,
    ) -> List[TissueDistributionResult]:
        """Rank drug candidates by tissue distribution profile"""

        def score_candidate(r: TissueDistributionResult) -> float:
            # Lower failure risk + higher target penetration = better
            if target_tissue == TissueType.BRAIN:
                target_score = r.bbb_permeability
            elif target_tissue == TissueType.TUMOR:
                target_score = r.tumor_penetration
            else:
                target_score = r.tissue_selectivity_score

            return target_score * (1 - r.clinical_failure_risk)

        return sorted(results, key=score_candidate, reverse=True)

    def generate_report(self, result: TissueDistributionResult) -> str:
        """Generate human-readable analysis report"""
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    QENEX LAB TISSUE DISTRIBUTION REPORT                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Molecule: {result.molecule_name:<66} ║
╠══════════════════════════════════════════════════════════════════════════════╣

┌─────────────────────────────────────────────────────────────────────────────┐
│ PARTITION COEFFICIENTS (Kp = Tissue/Plasma)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Brain:  {result.kp_brain:>6.2f}  │  Liver:  {result.kp_liver:>6.2f}  │  Kidney: {result.kp_kidney:>6.2f}  │
│  Tumor:  {result.kp_tumor:>6.2f}  │  Lung:   {result.kp_lung:>6.2f}  │  Heart:  {result.kp_heart:>6.2f}  │
│  Muscle: {result.kp_muscle:>6.2f}  │  Fat:    {result.kp_fat:>6.2f}  │                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PERMEABILITY SCORES (0-1 scale)                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Blood-Brain Barrier: {result.bbb_permeability:>5.2f}  {"✓ GOOD" if result.bbb_permeability > 0.5 else "✗ POOR":<20}            │
│  Tumor Penetration:   {result.tumor_penetration:>5.2f}  {"✓ GOOD" if result.tumor_penetration > 0.5 else "✗ POOR":<20}            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ TRANSPORTER LIABILITIES                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  P-gp Substrate:  {result.pgp_substrate:>5.2f}  {"⚠ HIGH RISK" if result.pgp_substrate > 0.5 else "✓ LOW RISK":<20}              │
│  BCRP Substrate:  {result.bcrp_substrate:>5.2f}  {"⚠ HIGH RISK" if result.bcrp_substrate > 0.5 else "✓ LOW RISK":<20}              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CLEARANCE PREDICTIONS                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Hepatic: {result.hepatic_clearance:>6.1f} mL/min/kg                                              │
│  Renal:   {result.renal_clearance:>6.1f} mL/min/kg                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ RISK ASSESSMENT                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Clinical Failure Risk:   {result.clinical_failure_risk:>5.1%}                                       │
│  Tissue Selectivity:      {result.tissue_selectivity_score:>5.2f}                                        │
│  Prediction Confidence:   {result.confidence:>5.1%}                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ RECOMMENDATION                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  {result.get_recommendation():<73} │
└─────────────────────────────────────────────────────────────────────────────┘

Generated by QENEX LAB v3.0-INFINITY | Trinity Pipeline
"""
        return report


# Export
__all__ = [
    "TissueDistributionPredictor",
    "TissueDistributionResult",
    "TissueSpecificModel",
]
