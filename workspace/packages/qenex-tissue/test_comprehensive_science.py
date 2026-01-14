from src.qlang_interface import QLangTissueEngine, QLangValue, QLangUnit


def test_comprehensive_science():
    print("============================================================")
    print("   QENEX LAB COMPREHENSIVE SCIENCE DISCOVERY TEST")
    print("============================================================")

    engine = QLangTissueEngine()

    # Test Case 1: hERG Blockade (Terfenadine-like)
    # High logP, basic amine features
    terfenadine = engine.from_descriptors(
        "Terfenadine_Proxy",
        {
            "molecular_weight": 471.7,
            "logP": 6.5,  # Very lipophilic
            "tpsa": 43.7,  # Low polar surface area
            "num_hbd": 1,
            "num_hba": 3,
        },
    )

    # Test Case 2: CYP3A4 Substrate (Ritonavir-like properties)
    ritonavir = engine.from_descriptors(
        "Ritonavir_Proxy",
        {
            "molecular_weight": 720.9,
            "logP": 5.9,
            "tpsa": 145.8,
            "num_hbd": 4,
            "num_hba": 9,
        },
    )

    # Test Case 3: Synthetic Complexity (Large natural product)
    complex_mol = engine.from_descriptors(
        "Complex_Macrolide",
        {
            "molecular_weight": 850.0,
            "logP": 2.5,
            "tpsa": 180.0,
            "num_hbd": 8,
            "num_hba": 14,
        },
    )

    # Test Case 4: Genotoxicity (Small reactive electrophile)
    genotoxin = engine.from_descriptors(
        "Small_Electrophile",
        {
            "molecular_weight": 140.0,
            "logP": 0.8,
            "tpsa": 25.0,
            "num_hbd": 0,
            "num_hba": 2,
        },
    )

    test_molecules = [terfenadine, ritonavir, complex_mol, genotoxin]

    for mol in test_molecules:
        print(f"\nAnalyzing: {mol.name}")
        report = engine.generate_report(mol)

        # Extract relevant sections for brevity
        lines = report.split("\n")
        in_laws = False
        for line in lines:
            if "PHYSICS LAWS EVALUATION" in line:
                in_laws = True
            elif "PREDICTED TISSUE DISTRIBUTION" in line:
                in_laws = False

            if in_laws and ("✓ APPLICABLE" in line or "PHYSICS" in line):
                print(line)


if __name__ == "__main__":
    test_comprehensive_science()
