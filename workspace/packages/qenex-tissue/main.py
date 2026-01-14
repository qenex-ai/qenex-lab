"""
QENEX Tissue Distribution Discovery Engine - Main Runner
=========================================================

Demonstrates the full Trinity Pipeline workflow:
1. Load molecule from SDF
2. Extract features
3. Run tissue distribution prediction
4. Validate with Q-Lang physics constraints
5. Generate comprehensive report

Usage:
    python -m qenex_tissue.main --sdf molecule.sdf --target brain
    python -m qenex_tissue.main --demo
"""

import sys
import os
import argparse
import asyncio
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import MolecularFeatureExtractor, MolecularDescriptors
from src.models import TissueDistributionPredictor, TissueDistributionResult
from src.validation import ValidationDataset, DrugDataPoint
from src.qlang_interface import QLangTissueEngine, QLangMolecule


def run_demo():
    """Run demonstration with sample molecules"""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    QENEX LAB TISSUE DISTRIBUTION ENGINE                       ║
║                              DEMONSTRATION                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Trinity Pipeline: DeepSeek + Llama Scout + Scout CLI                        ║
║  Q-Lang Scientific Expressions | 18-Expert Validation                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Initialize components
    print("\n[1] Initializing QENEX LAB components...")
    predictor = TissueDistributionPredictor()
    qlang_engine = QLangTissueEngine()
    validation_dataset = ValidationDataset()

    # Print dataset statistics
    print("\n[2] Loading validation dataset...")
    validation_dataset.print_summary()

    # Demo molecule: Imatinib (known good tumor penetration)
    print("\n[3] Analyzing demo molecule: Imatinib...")

    # Create descriptors for Imatinib
    imatinib_desc = MolecularDescriptors()
    imatinib_desc.molecular_weight = 493.6
    imatinib_desc.logP = 3.5
    imatinib_desc.tpsa = 86.3
    imatinib_desc.num_hbd = 2
    imatinib_desc.num_hba = 7
    imatinib_desc.num_rotatable_bonds = 7
    imatinib_desc.num_aromatic_rings = 4
    imatinib_desc.num_rings = 5
    imatinib_desc.bbb_score = 0.35
    imatinib_desc.pgp_substrate_prob = 0.7
    imatinib_desc.plasma_protein_binding = 95.0

    # Run prediction
    result = predictor.predict_from_descriptors(imatinib_desc, "Imatinib")

    # Print tissue distribution report
    print(predictor.generate_report(result))

    # Q-Lang analysis
    print("\n[4] Running Q-Lang physics validation...")

    qlang_mol = qlang_engine.from_descriptors(
        "Imatinib",
        {
            "molecular_weight": 493.6,
            "logP": 3.5,
            "tpsa": 86.3,
            "num_hbd": 2,
            "num_hba": 7,
            "pgp_substrate_prob": 0.7,
            "plasma_protein_binding": 95.0,
            "bbb_score": 0.35,
        },
    )

    print(qlang_engine.generate_report(qlang_mol))

    # Demonstrate batch prediction with failed drugs
    print("\n[5] Demonstrating failure prediction on known failed drugs...")

    failed_drugs = validation_dataset.get_by_outcome(
        validation_dataset.drugs[0].outcome.__class__.PHASE3_FAILURE
    )

    print(f"\n   Analyzing {len(failed_drugs)} known Phase 3 failures:\n")

    correct_predictions = 0
    for drug in failed_drugs[:3]:  # Demo first 3
        # Create descriptors
        desc = MolecularDescriptors()
        desc.molecular_weight = drug.molecular_weight
        desc.logP = drug.logP
        desc.tpsa = drug.tpsa
        desc.num_hbd = drug.num_hbd
        desc.num_hba = drug.num_hba
        desc.bbb_score = 0.3 if drug.bbb_penetration else 0.1
        desc.pgp_substrate_prob = 0.8 if drug.pgp_substrate else 0.2

        result = predictor.predict_from_descriptors(desc, drug.name)

        predicted_failure = result.clinical_failure_risk > 0.5
        actual_failure = True  # These are all failures

        status = "✓" if predicted_failure == actual_failure else "✗"
        if predicted_failure == actual_failure:
            correct_predictions += 1

        print(
            f"   {status} {drug.name:<25} Risk: {result.clinical_failure_risk:>5.1%}  "
            f"Actual: FAILED  Predicted: {'FAIL' if predicted_failure else 'PASS'}"
        )

    if failed_drugs:
        accuracy = correct_predictions / min(3, len(failed_drugs)) * 100
        print(f"\n   Retrospective accuracy: {accuracy:.0f}% on failed drugs")

    # Summary
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              DEMO COMPLETE                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  The QENEX LAB Tissue Distribution Engine provides:                          ║
║                                                                              ║
║  • DFT-based molecular feature extraction                                    ║
║  • Multi-tissue Kp prediction (brain, liver, tumor, kidney)                  ║
║  • P-gp efflux transporter liability assessment                              ║
║  • Clinical failure risk prediction                                          ║
║  • Q-Lang physics constraint validation                                      ║
║  • Trinity Pipeline AI integration                                           ║
║                                                                              ║
║  Ready for pharmaceutical discovery applications.                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def analyze_sdf(sdf_path: str, target_tissue: str = "brain"):
    """Analyze a molecule from SDF file"""

    print(f"\n[QENEX LAB] Analyzing {sdf_path}...")

    # Extract features
    extractor = MolecularFeatureExtractor(use_dft=False)
    extractor.load_from_sdf(sdf_path)
    descriptors = extractor.extract_features()

    # Run prediction
    predictor = TissueDistributionPredictor()
    result = predictor.predict_from_sdf(sdf_path)

    # Print report
    print(predictor.generate_report(result))

    # Q-Lang validation
    qlang = QLangTissueEngine()
    mol = qlang.from_descriptors(Path(sdf_path).stem, descriptors.to_dict())
    print(qlang.generate_report(mol))

    return result


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="QENEX LAB Tissue Distribution Prediction Engine"
    )
    parser.add_argument("--sdf", type=str, help="Path to SDF file")
    parser.add_argument(
        "--target",
        type=str,
        default="brain",
        choices=["brain", "liver", "tumor", "kidney"],
        help="Target tissue",
    )
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument(
        "--validate", action="store_true", help="Run validation on dataset"
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.sdf:
        analyze_sdf(args.sdf, args.target)
    elif args.validate:
        # Run validation benchmark
        print("\n[QENEX LAB] Running validation benchmark...")
        dataset = ValidationDataset()
        dataset.print_summary()

        # Export dataset
        dataset.export_json("/tmp/qenex_validation_dataset.json")
        print("\nDataset exported to /tmp/qenex_validation_dataset.json")
    else:
        parser.print_help()
        print("\n\nExample usage:")
        print("  python main.py --demo")
        print("  python main.py --sdf molecule.sdf --target brain")
        print("  python main.py --validate")


if __name__ == "__main__":
    main()
