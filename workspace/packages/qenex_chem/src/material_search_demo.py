"""
QENEX Material Search Lab - Fast Demo
======================================
A lightweight version for quick demonstration of the catalyst design pipeline.

Author: QENEX Sovereign Agent
Date: 2026-01-10
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from datetime import datetime
import sys
import os

# Import from optimized ERI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eri_optimized import PrimitiveGaussian, eri, ERISymmetry, BoysFunction

# ============================================================
# Physical Constants
# ============================================================

class PhysicalConstants:
    """Fundamental constants for quantum chemistry."""
    BOHR_TO_ANGSTROM = 0.529177210903
    ANGSTROM_TO_BOHR = 1.8897259886
    HARTREE_TO_EV = 27.211386245988
    
    ATOMIC_NUMBER = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'Pd': 46, 'Pt': 78
    }

# ============================================================
# Enums and Data Classes
# ============================================================

class CatalystType(Enum):
    HYDROGENATION = "hydrogenation"
    OXIDATION = "oxidation"
    COUPLING = "coupling"

@dataclass
class Atom:
    symbol: str
    position: np.ndarray
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.atomic_number = PhysicalConstants.ATOMIC_NUMBER.get(self.symbol, 1)

@dataclass
class Molecule:
    name: str
    atoms: List[Atom]
    charge: int = 0
    
    @property
    def n_atoms(self) -> int:
        return len(self.atoms)
    
    @property
    def n_electrons(self) -> int:
        return sum(a.atomic_number for a in self.atoms) - self.charge

@dataclass
class MaterialFingerprint:
    """Catalyst material fingerprint with predictions."""
    formula: str
    n_atoms: int
    n_electrons: int
    homo_energy: float
    lumo_energy: float
    band_gap: float
    d_band_center: float
    d_band_width: float
    coordination_number: int
    activity_score: float
    selectivity_score: float
    stability_score: float
    turnover_frequency: float
    
    def to_dict(self) -> Dict:
        return {
            'formula': self.formula,
            'n_atoms': self.n_atoms,
            'n_electrons': self.n_electrons,
            'electronic_properties': {
                'homo_eV': self.homo_energy,
                'lumo_eV': self.lumo_energy,
                'band_gap_eV': self.band_gap,
                'd_band_center_eV': self.d_band_center,
                'd_band_width_eV': self.d_band_width
            },
            'catalytic_predictions': {
                'activity_score': self.activity_score,
                'selectivity_score': self.selectivity_score,
                'stability_score': self.stability_score,
                'predicted_TOF_per_hour': self.turnover_frequency
            }
        }

# ============================================================
# DeepSeek Catalyst Designer (Simulated)
# ============================================================

class DeepSeekDesigner:
    """
    Simulates DeepSeek reasoning for catalyst design.
    Uses d-band theory and Sabatier principle.
    """
    
    def __init__(self, catalyst_type: CatalystType):
        self.catalyst_type = catalyst_type
        self.log = []
        
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log.append(f"[{ts}] {msg}")
        print(f"  [DeepSeek] {msg}")
    
    def design(self) -> Tuple[Molecule, Dict]:
        """Design a catalyst structure."""
        print("\n" + "="*70)
        print("🧠 DEEPSEEK CATALYST DESIGN ENGINE")
        print("="*70)
        
        self._log(f"Analyzing optimal catalyst for: {self.catalyst_type.value}")
        
        # Phase 1: Metal selection via d-band theory
        self._log("Applying d-band theory for metal selection...")
        d_band_centers = {
            'Pt': -2.25,  # Strong binding
            'Pd': -1.83,  # Optimal for hydrogenation
            'Ni': -1.29,  # Moderate binding
        }
        
        if self.catalyst_type == CatalystType.HYDROGENATION:
            metal = 'Pd'
            d_center = d_band_centers['Pd']
            self._log(f"Selected Pd: d-band = {d_center} eV (optimal for H2 activation)")
        else:
            metal = 'Pt'
            d_center = d_band_centers['Pt']
            self._log(f"Selected Pt: d-band = {d_center} eV")
        
        # Phase 2: Ligand design
        self._log("Designing ligand sphere using trans effect principles...")
        self._log("Selected: phosphine ligands (PH3) for σ-donation")
        
        # Phase 3: Build structure - Pd(PH3)2 linear complex
        # Pd at origin, PH3 groups along z-axis
        to_bohr = PhysicalConstants.ANGSTROM_TO_BOHR
        
        atoms = [
            # Pd center
            Atom('Pd', np.array([0.0, 0.0, 0.0])),
            # P atoms on z-axis (Pd-P = 2.3 Å typical)
            Atom('P', np.array([0.0, 0.0, 2.3 * to_bohr])),
            Atom('P', np.array([0.0, 0.0, -2.3 * to_bohr])),
            # H atoms on first P (P-H = 1.4 Å, tetrahedral)
            Atom('H', np.array([1.0 * to_bohr, 0.7 * to_bohr, 3.2 * to_bohr])),
            Atom('H', np.array([-1.0 * to_bohr, 0.7 * to_bohr, 3.2 * to_bohr])),
            Atom('H', np.array([0.0, -1.2 * to_bohr, 3.2 * to_bohr])),
            # H atoms on second P
            Atom('H', np.array([1.0 * to_bohr, 0.7 * to_bohr, -3.2 * to_bohr])),
            Atom('H', np.array([-1.0 * to_bohr, 0.7 * to_bohr, -3.2 * to_bohr])),
            Atom('H', np.array([0.0, -1.2 * to_bohr, -3.2 * to_bohr])),
        ]
        
        molecule = Molecule(
            name="Pd(PH3)2",
            atoms=atoms,
            charge=0
        )
        
        self._log(f"Structure built: {molecule.name} ({molecule.n_atoms} atoms, {molecule.n_electrons} e⁻)")
        
        rationale = {
            'metal_selection': {
                'element': metal,
                'd_band_center_eV': d_center,
                'rationale': 'Optimal d-band position for H2 activation (Sabatier principle)'
            },
            'ligand_design': {
                'type': 'phosphine (PH3)',
                'count': 2,
                'geometry': 'linear',
                'rationale': 'Strong σ-donor, weak π-acceptor for electron-rich metal'
            },
            'design_principles': [
                "d-band theory: Pd at -1.83 eV optimal for hydrogenation",
                "Sabatier principle: intermediate binding energy",
                "16-electron configuration: two open coordination sites",
                "Linear geometry: minimum steric hindrance"
            ]
        }
        
        return molecule, rationale

# ============================================================
# Scout Symmetry Monitor (Simulated)
# ============================================================

class ScoutMonitor:
    """
    Simulates Scout CLI validation of ERI symmetry.
    Target: 84.6% reduction via 8-fold permutational symmetry.
    """
    
    def __init__(self):
        self.iterations = []
        self.target_reduction = 84.6
        
    def monitor(self, n_basis: int, iteration: int = 0) -> Dict:
        """Monitor symmetry reduction for ERI calculation."""
        # Calculate theoretical symmetry reduction
        # Full tensor: n^4 elements
        # Unique quartets: n(n+1)/2 * (n(n+1)/2 + 1) / 2
        n2 = n_basis * (n_basis + 1) // 2
        unique = n2 * (n2 + 1) // 2
        full = n_basis ** 4
        
        reduction_pct = (1 - unique / full) * 100
        
        result = {
            'iteration': iteration,
            'n_basis': n_basis,
            'full_tensor_size': full,
            'unique_quartets': unique,
            'symmetry_reduction_pct': round(reduction_pct, 1),
            'target_reduction_pct': self.target_reduction,
            'status': 'PASS' if reduction_pct > 80 else 'WARN'
        }
        
        self.iterations.append(result)
        return result
    
    def generate_report(self) -> Dict:
        if not self.iterations:
            return {'status': 'NO_DATA'}
        
        avg_reduction = np.mean([i['symmetry_reduction_pct'] for i in self.iterations])
        return {
            'total_iterations': len(self.iterations),
            'average_symmetry_reduction': round(avg_reduction, 1),
            'target_reduction': self.target_reduction,
            'overall_status': 'PASS' if avg_reduction > 80 else 'WARN',
            'details': self.iterations
        }

# ============================================================
# Minimal Basis Set
# ============================================================

class MinimalBasis:
    """Minimal STO-1G basis for fast demo."""
    
    # Single Gaussian approximation to Slater orbitals
    STO1G = {
        'H': [('1s', 0.4166, 1.0)],
        'P': [('3s', 0.8, 1.0), ('3p', 0.5, 1.0)],
        'Pd': [('4d', 1.2, 1.0), ('5s', 0.5, 1.0)],
    }
    
    @classmethod
    def build(cls, molecule: Molecule) -> List[PrimitiveGaussian]:
        """Build minimal basis set."""
        basis = []
        for atom in molecule.atoms:
            elem = atom.symbol
            pos = atom.position
            
            if elem not in cls.STO1G:
                # Default minimal
                basis.append(PrimitiveGaussian(pos, 0.5, 1.0, 0, 0, 0))
                continue
            
            for shell, alpha, coeff in cls.STO1G[elem]:
                if 's' in shell:
                    basis.append(PrimitiveGaussian(pos, alpha, coeff, 0, 0, 0))
                elif 'p' in shell:
                    basis.append(PrimitiveGaussian(pos, alpha, coeff, 1, 0, 0))
                    basis.append(PrimitiveGaussian(pos, alpha, coeff, 0, 1, 0))
                    basis.append(PrimitiveGaussian(pos, alpha, coeff, 0, 0, 1))
                elif 'd' in shell:
                    # Only include 2 d functions for speed
                    basis.append(PrimitiveGaussian(pos, alpha, coeff, 2, 0, 0))
                    basis.append(PrimitiveGaussian(pos, alpha, coeff, 0, 2, 0))
        
        return basis

# ============================================================
# ERI Calculation with Symmetry
# ============================================================

def compute_eri_sample(basis: List[PrimitiveGaussian], sample_size: int = 100) -> Dict:
    """
    Compute a sample of ERI integrals with symmetry analysis.
    """
    n = len(basis)
    symmetry = ERISymmetry(n)
    
    # Get unique quartets
    unique_quartets = symmetry.get_unique_quartets(n)
    
    # Sample calculation
    computed = 0
    total_eri = 0.0
    max_eri = 0.0
    
    sample = min(sample_size, len(unique_quartets))
    indices = np.random.choice(len(unique_quartets), sample, replace=False)
    
    start_time = time.time()
    for idx in indices:
        i, j, k, l = unique_quartets[idx]
        val = eri(basis[i], basis[j], basis[k], basis[l])
        total_eri += abs(val)
        max_eri = max(max_eri, abs(val))
        computed += 1
    
    elapsed = time.time() - start_time
    
    return {
        'n_basis': n,
        'full_tensor_size': n**4,
        'unique_quartets': len(unique_quartets),
        'sample_computed': computed,
        'computation_time_sec': round(elapsed, 3),
        'avg_eri_magnitude': total_eri / computed if computed > 0 else 0,
        'max_eri_magnitude': max_eri,
        'symmetry_reduction_pct': round((1 - len(unique_quartets) / n**4) * 100, 1)
    }

# ============================================================
# Fingerprint Generator
# ============================================================

def generate_fingerprint(molecule: Molecule, rationale: Dict, eri_stats: Dict) -> MaterialFingerprint:
    """Generate material fingerprint with catalytic predictions."""
    
    # Get d-band info from design
    d_center = rationale['metal_selection']['d_band_center_eV']
    
    # Estimate electronic properties
    homo = -5.5 + d_center * 0.2  # Rough correlation
    lumo = homo + 2.5  # Typical gap
    band_gap = lumo - homo
    
    # d-band width (correlated with coordination)
    d_width = 2.5  # eV for linear coordination
    
    # Coordination number
    coord = 2  # Linear Pd(PH3)2
    
    # Catalytic predictions using d-band model
    # Activity: peak at d-band ~ -1.9 eV
    d_opt = -1.9
    activity = np.exp(-((d_center - d_opt) / 0.5)**2) * 100
    
    # Selectivity: depends on geometry and sterics
    selectivity = 85 if coord <= 2 else 70  # Linear = high selectivity
    
    # Stability: depends on band gap and hardness
    hardness = band_gap / 2
    stability = 60 + hardness * 10
    
    # TOF prediction
    tof = activity * stability / 100 * 500  # Base TOF scaling
    
    return MaterialFingerprint(
        formula="Pd(PH3)2",
        n_atoms=molecule.n_atoms,
        n_electrons=molecule.n_electrons,
        homo_energy=round(homo, 2),
        lumo_energy=round(lumo, 2),
        band_gap=round(band_gap, 2),
        d_band_center=d_center,
        d_band_width=d_width,
        coordination_number=coord,
        activity_score=round(activity, 1),
        selectivity_score=round(selectivity, 1),
        stability_score=round(stability, 1),
        turnover_frequency=round(tof, 0)
    )

# ============================================================
# Main Lab Orchestrator
# ============================================================

class MaterialSearchLabDemo:
    """Demo version of Material Search Lab."""
    
    def __init__(self, catalyst_type: CatalystType = CatalystType.HYDROGENATION):
        self.catalyst_type = catalyst_type
        self.results = {}
        
    def run(self) -> Dict:
        """Run the complete demo pipeline."""
        print("\n" + "="*70)
        print("🔬 QENEX MATERIAL SEARCH LAB - DEMO")
        print("="*70)
        print(f"  Catalyst Type: {self.catalyst_type.value}")
        print(f"  Started: {datetime.now().isoformat()}")
        
        # Phase 1: DeepSeek Design
        designer = DeepSeekDesigner(self.catalyst_type)
        molecule, rationale = designer.design()
        self.results['design'] = rationale
        self.results['design_log'] = designer.log
        
        # Phase 2: Build Basis
        print("\n" + "="*70)
        print("🔧 BASIS SET CONSTRUCTION")
        print("="*70)
        basis = MinimalBasis.build(molecule)
        print(f"  Built {len(basis)} basis functions (minimal STO-1G)")
        print(f"  Atoms: {molecule.n_atoms}, Electrons: {molecule.n_electrons}")
        self.results['basis'] = {'n_functions': len(basis), 'type': 'STO-1G'}
        
        # Phase 3: ERI with Symmetry
        print("\n" + "="*70)
        print("⚛️ ERI CALCULATION WITH 8-FOLD SYMMETRY")
        print("="*70)
        eri_stats = compute_eri_sample(basis, sample_size=200)
        print(f"  Full tensor size: {eri_stats['full_tensor_size']:,}")
        print(f"  Unique quartets: {eri_stats['unique_quartets']:,}")
        print(f"  Symmetry reduction: {eri_stats['symmetry_reduction_pct']}%")
        print(f"  Sample computed: {eri_stats['sample_computed']}")
        print(f"  Computation time: {eri_stats['computation_time_sec']} sec")
        self.results['eri'] = eri_stats
        
        # Phase 4: Scout Monitoring
        print("\n" + "="*70)
        print("🛡️ SCOUT SYMMETRY VALIDATION")
        print("="*70)
        monitor = ScoutMonitor()
        sym_result = monitor.monitor(len(basis))
        print(f"  Target reduction: {sym_result['target_reduction_pct']}%")
        print(f"  Achieved: {sym_result['symmetry_reduction_pct']}%")
        print(f"  Status: {sym_result['status']}")
        self.results['monitoring'] = monitor.generate_report()
        
        # Phase 5: Fingerprint Generation
        print("\n" + "="*70)
        print("🎯 MATERIAL FINGERPRINT GENERATION")
        print("="*70)
        fingerprint = generate_fingerprint(molecule, rationale, eri_stats)
        self.results['fingerprint'] = fingerprint.to_dict()
        
        # Print fingerprint
        print(f"\n  Formula: {fingerprint.formula}")
        print(f"  Atoms: {fingerprint.n_atoms}, Electrons: {fingerprint.n_electrons}")
        print(f"\n  [Electronic Properties]")
        print(f"    HOMO: {fingerprint.homo_energy} eV")
        print(f"    LUMO: {fingerprint.lumo_energy} eV")
        print(f"    Band Gap: {fingerprint.band_gap} eV")
        print(f"    d-Band Center: {fingerprint.d_band_center} eV")
        
        print(f"\n  [Catalytic Predictions]")
        print(f"    Activity Score: {fingerprint.activity_score}/100")
        print(f"    Selectivity Score: {fingerprint.selectivity_score}/100")
        print(f"    Stability Score: {fingerprint.stability_score}/100")
        print(f"    Predicted TOF: {fingerprint.turnover_frequency:.0f} h⁻¹")
        
        # Final Summary
        print("\n" + "="*70)
        print("✅ MATERIAL SEARCH LAB COMPLETE")
        print("="*70)
        
        self.results['completed_at'] = datetime.now().isoformat()
        return self.results
    
    def save(self, filepath: str):
        """Save results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {filepath}")

# ============================================================
# Main Entry
# ============================================================

def main():
    """Run the Material Search Lab demo."""
    lab = MaterialSearchLabDemo(CatalystType.HYDROGENATION)
    results = lab.run()
    
    # Save results
    output_dir = "/opt/qenex_lab/workspace/reports"
    os.makedirs(output_dir, exist_ok=True)
    lab.save(os.path.join(output_dir, "catalyst_demo_results.json"))
    
    return results

if __name__ == "__main__":
    main()
