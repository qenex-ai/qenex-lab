from src.universal import UniversalDiscoveryEngine, ScientificConcept, DiscoveryGoal


def run_civilization_test():
    print("=================================================================")
    print("   QENEX LAB: CIVILIZATION ACCELERATION PROTOCOL v1.0")
    print("=================================================================")

    engine = UniversalDiscoveryEngine()

    # 1. CLEAN ENERGY: Solid State Battery Electrolyte
    # Goal: Enable 1000-mile EV range
    lgps_super = ScientificConcept(
        name="LGPS-Super-X",
        domain="energy",
        properties={
            "ionic_conductivity": 1.2e-2,  # Very high (12 mS/cm)
            "stability_window": 5.0,  # 5V stability
            "density": 2.5,
        },
    )

    energy_goal = DiscoveryGoal("1000-Mile EV Battery", ["conductivity > 10mS/cm"])

    # 2. MATERIALS: Room Temp Superconductor
    # Goal: Lossless power grids / Levitation trains
    lk99_opt = ScientificConcept(
        name="Cu-Apatite-Quantum",
        domain="material",
        properties={
            "critical_temp": 285.0,  # 285K = 12°C (Room Temp!)
            "critical_current": 1000.0,
        },
    )

    mat_goal = DiscoveryGoal("Lossless Power Grid", ["Tc > 273K"])

    # RUN EVALUATION
    results = []
    results.append(engine.evaluate(lgps_super, energy_goal))
    results.append(engine.evaluate(lk99_opt, mat_goal))

    print("\n>>> DISCOVERY REPORT <<<\n")

    for res in results:
        print(f"Concept: {res['concept']}")
        print(f"Goal:    {res['goal']}")
        print(f"Score:   {res['score'] * 100:.0f}%")
        print("Proofs:")
        for proof in res["proofs"]:
            print(f"  {proof}")
        print("-" * 50)


if __name__ == "__main__":
    run_civilization_test()
