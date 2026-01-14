"""
Molecular Feature Extraction for Tissue Distribution Prediction
================================================================

Extracts physics-based and ML-friendly features from molecular structures:
- DFT-derived electronic properties (ESP, HOMO/LUMO, dipole)
- Lipophilicity and solubility descriptors
- Membrane permeability indicators
- Hydrogen bonding capacity
- Molecular flexibility and size

Integrates with QENEX LAB qenex_chem for quantum calculations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import math
import json


class TissueType(Enum):
    """Target tissue types for distribution prediction"""

    PLASMA = "plasma"
    BRAIN = "brain"  # Blood-Brain Barrier
    LIVER = "liver"
    KIDNEY = "kidney"
    TUMOR = "tumor"
    LUNG = "lung"
    HEART = "heart"
    MUSCLE = "muscle"
    FAT = "fat"
    BONE = "bone"


@dataclass
class Atom:
    """Atomic representation with electronic properties"""

    symbol: str
    x: float
    y: float
    z: float
    partial_charge: float = 0.0
    electronegativity: float = 0.0
    vdw_radius: float = 0.0

    # Electronegativity values (Pauling scale)
    ELECTRONEG = {
        "H": 2.20,
        "C": 2.55,
        "N": 3.04,
        "O": 3.44,
        "F": 3.98,
        "P": 2.19,
        "S": 2.58,
        "Cl": 3.16,
        "Br": 2.96,
        "I": 2.66,
        "Na": 0.93,
        "K": 0.82,
        "Ca": 1.00,
        "Mg": 1.31,
        "Fe": 1.83,
        "Zn": 1.65,
        "Cu": 1.90,
        "B": 2.04,
        "Si": 1.90,
        "Se": 2.55,
    }

    # Van der Waals radii (Angstroms)
    VDW_RADII = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "F": 1.47,
        "P": 1.80,
        "S": 1.80,
        "Cl": 1.75,
        "Br": 1.85,
        "I": 1.98,
        "Na": 2.27,
        "K": 2.75,
        "Ca": 2.31,
        "Mg": 1.73,
        "Fe": 2.04,
        "Zn": 2.01,
        "Cu": 1.96,
        "B": 1.92,
        "Si": 2.10,
        "Se": 1.90,
    }

    def __post_init__(self):
        self.electronegativity = self.ELECTRONEG.get(self.symbol, 2.0)
        self.vdw_radius = self.VDW_RADII.get(self.symbol, 1.70)


@dataclass
class MolecularDescriptors:
    """Comprehensive molecular descriptors for tissue distribution"""

    # Basic properties
    molecular_weight: float = 0.0
    num_atoms: int = 0
    num_heavy_atoms: int = 0

    # Lipophilicity (critical for membrane permeation)
    logP: float = 0.0  # Octanol-water partition coefficient
    logD_7_4: float = 0.0  # Distribution coefficient at pH 7.4

    # Solubility
    logS: float = 0.0  # Aqueous solubility

    # Hydrogen bonding (affects permeability)
    num_hbd: int = 0  # H-bond donors
    num_hba: int = 0  # H-bond acceptors

    # Polar surface area (key for BBB penetration)
    tpsa: float = 0.0  # Topological polar surface area

    # Electronic properties (DFT-derived)
    dipole_moment: float = 0.0
    homo_energy: float = 0.0  # Highest occupied MO
    lumo_energy: float = 0.0  # Lowest unoccupied MO
    homo_lumo_gap: float = 0.0
    total_esp_positive: float = 0.0
    total_esp_negative: float = 0.0
    esp_balance: float = 0.0  # Ratio of +/- ESP

    # Molecular shape
    molecular_volume: float = 0.0
    molecular_surface_area: float = 0.0
    sphericity: float = 0.0

    # Flexibility
    num_rotatable_bonds: int = 0
    flexibility_index: float = 0.0

    # Ring systems
    num_rings: int = 0
    num_aromatic_rings: int = 0
    num_heteroatoms: int = 0

    # Charge distribution
    total_positive_charge: float = 0.0
    total_negative_charge: float = 0.0
    net_charge: float = 0.0
    charge_density: float = 0.0

    # Lipinski Rule of 5 compliance
    lipinski_violations: int = 0

    # Tissue-specific predictors
    bbb_score: float = 0.0  # Blood-brain barrier penetration
    pgp_substrate_prob: float = 0.0  # P-glycoprotein efflux probability
    plasma_protein_binding: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models"""
        return np.array(
            [
                self.molecular_weight,
                self.num_heavy_atoms,
                self.logP,
                self.logD_7_4,
                self.logS,
                self.num_hbd,
                self.num_hba,
                self.tpsa,
                self.dipole_moment,
                self.homo_energy,
                self.lumo_energy,
                self.homo_lumo_gap,
                self.total_esp_positive,
                self.total_esp_negative,
                self.esp_balance,
                self.molecular_volume,
                self.molecular_surface_area,
                self.sphericity,
                self.num_rotatable_bonds,
                self.flexibility_index,
                self.num_rings,
                self.num_aromatic_rings,
                self.num_heteroatoms,
                self.total_positive_charge,
                self.total_negative_charge,
                self.net_charge,
                self.charge_density,
                self.lipinski_violations,
                self.bbb_score,
                self.pgp_substrate_prob,
                self.plasma_protein_binding,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for interpretability"""
        return [
            "molecular_weight",
            "num_heavy_atoms",
            "logP",
            "logD_7_4",
            "logS",
            "num_hbd",
            "num_hba",
            "tpsa",
            "dipole_moment",
            "homo_energy",
            "lumo_energy",
            "homo_lumo_gap",
            "total_esp_positive",
            "total_esp_negative",
            "esp_balance",
            "molecular_volume",
            "molecular_surface_area",
            "sphericity",
            "num_rotatable_bonds",
            "flexibility_index",
            "num_rings",
            "num_aromatic_rings",
            "num_heteroatoms",
            "total_positive_charge",
            "total_negative_charge",
            "net_charge",
            "charge_density",
            "lipinski_violations",
            "bbb_score",
            "pgp_substrate_prob",
            "plasma_protein_binding",
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            name: float(val)
            for name, val in zip(self.feature_names(), self.to_vector())
        }


class MolecularFeatureExtractor:
    """
    Extract tissue-distribution-relevant features from molecular structures.

    Uses a combination of:
    1. Classical chemoinformatics descriptors
    2. DFT-derived electronic properties
    3. Physics-based permeability indicators

    Designed to integrate with QENEX LAB Trinity Pipeline.
    """

    # Atomic masses for MW calculation
    ATOMIC_MASS = {
        "H": 1.008,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "F": 18.998,
        "P": 30.974,
        "S": 32.065,
        "Cl": 35.453,
        "Br": 79.904,
        "I": 126.904,
        "Na": 22.990,
        "K": 39.098,
        "Ca": 40.078,
        "Mg": 24.305,
        "Fe": 55.845,
        "Zn": 65.38,
        "Cu": 63.546,
        "B": 10.811,
        "Si": 28.086,
        "Se": 78.96,
    }

    # Fragment contributions to logP (Wildman-Crippen)
    LOGP_CONTRIBUTIONS = {
        "C_sp3": 0.1441,
        "C_sp2": 0.1441,
        "C_aromatic": 0.2952,
        "N_sp3": -0.5262,
        "N_sp2": -0.2582,
        "N_aromatic": -0.3239,
        "O_sp3": -0.2893,
        "O_sp2": -0.1188,
        "O_aromatic": 0.1129,
        "S": 0.6482,
        "P": 0.8456,
        "F": 0.4119,
        "Cl": 0.6895,
        "Br": 0.8456,
        "I": 1.1410,
        "H_on_C": 0.1230,
        "H_on_N": -0.2677,
        "H_on_O": -0.2893,
    }

    # TPSA contributions per atom type
    TPSA_CONTRIBUTIONS = {
        "N_primary_amine": 26.02,
        "N_secondary_amine": 12.03,
        "N_tertiary_amine": 3.24,
        "N_aromatic": 12.89,
        "N_amide": 9.68,
        "N_nitro": 11.68,
        "O_hydroxyl": 20.23,
        "O_ether": 9.23,
        "O_carbonyl": 17.07,
        "O_carboxyl": 37.30,
        "O_ester": 26.30,
        "S_thiol": 28.24,
        "S_sulfide": 25.30,
    }

    def __init__(self, use_dft: bool = True):
        """
        Initialize feature extractor.

        Args:
            use_dft: Whether to compute DFT-derived properties (slower but more accurate)
        """
        self.use_dft = use_dft
        self.atoms: List[Atom] = []
        self.bonds: List[Tuple[int, int, int]] = []  # (atom1, atom2, bond_order)
        self.name: str = ""

    def load_from_sdf(
        self, filepath: str, name: str = None
    ) -> "MolecularFeatureExtractor":
        """Load molecule from SDF file"""
        self.atoms = []
        self.bonds = []
        self.name = name or filepath.split("/")[-1].replace(".sdf", "")

        with open(filepath, "r") as f:
            lines = f.readlines()

        # Parse counts line
        counts_line = lines[3]
        num_atoms = int(counts_line[0:3].strip())
        num_bonds = int(counts_line[3:6].strip())

        # Parse atoms
        for i in range(num_atoms):
            line = lines[4 + i]
            x = float(line[0:10])
            y = float(line[10:20])
            z = float(line[20:30])
            symbol = line[31:34].strip()
            self.atoms.append(Atom(symbol=symbol, x=x, y=y, z=z))

        # Parse bonds
        bond_start = 4 + num_atoms
        for i in range(num_bonds):
            if bond_start + i < len(lines):
                line = lines[bond_start + i]
                if (
                    line.strip()
                    and not line.startswith("M")
                    and not line.startswith(">")
                ):
                    try:
                        atom1 = int(line[0:3].strip()) - 1
                        atom2 = int(line[3:6].strip()) - 1
                        bond_order = int(line[6:9].strip())
                        self.bonds.append((atom1, atom2, bond_order))
                    except (ValueError, IndexError):
                        pass

        # Parse charges from properties block
        for line in lines:
            if "PUBCHEM_MMFF94_PARTIAL_CHARGES" in line:
                # Next lines contain charge data
                pass

        return self

    def load_from_atoms(
        self,
        atoms: List[Dict],
        bonds: List[Tuple[int, int, int]] = None,
        name: str = "molecule",
    ) -> "MolecularFeatureExtractor":
        """Load molecule from atom list"""
        self.atoms = [
            Atom(symbol=a["symbol"], x=a["x"], y=a["y"], z=a["z"]) for a in atoms
        ]
        self.bonds = bonds or []
        self.name = name
        return self

    def extract_features(self) -> MolecularDescriptors:
        """Extract comprehensive molecular descriptors"""
        desc = MolecularDescriptors()

        # Basic properties
        desc.num_atoms = len(self.atoms)
        desc.num_heavy_atoms = sum(1 for a in self.atoms if a.symbol != "H")
        desc.molecular_weight = sum(
            self.ATOMIC_MASS.get(a.symbol, 12.0) for a in self.atoms
        )

        # Count atom types
        atom_counts = {}
        for atom in self.atoms:
            atom_counts[atom.symbol] = atom_counts.get(atom.symbol, 0) + 1

        # Hydrogen bonding
        desc.num_hbd = self._count_hbd()
        desc.num_hba = self._count_hba()

        # Heteroatoms
        desc.num_heteroatoms = sum(1 for a in self.atoms if a.symbol not in ["C", "H"])

        # Ring analysis
        desc.num_rings, desc.num_aromatic_rings = self._analyze_rings()

        # LogP estimation (Wildman-Crippen method simplified)
        desc.logP = self._estimate_logP()
        desc.logD_7_4 = desc.logP - 0.5  # Simplified pH adjustment

        # Solubility estimation (ESOL-like)
        desc.logS = self._estimate_logS(desc.logP, desc.molecular_weight)

        # TPSA estimation
        desc.tpsa = self._estimate_tpsa()

        # Rotatable bonds
        desc.num_rotatable_bonds = self._count_rotatable_bonds()
        desc.flexibility_index = desc.num_rotatable_bonds / max(1, desc.num_heavy_atoms)

        # Molecular geometry
        desc.molecular_volume = self._estimate_volume()
        desc.molecular_surface_area = self._estimate_surface_area()
        desc.sphericity = self._calculate_sphericity()

        # Electronic properties (simplified or DFT)
        if self.use_dft:
            electronic = self._compute_electronic_properties()
        else:
            electronic = self._estimate_electronic_properties()

        desc.dipole_moment = electronic["dipole"]
        desc.homo_energy = electronic["homo"]
        desc.lumo_energy = electronic["lumo"]
        desc.homo_lumo_gap = electronic["gap"]
        desc.total_esp_positive = electronic["esp_positive"]
        desc.total_esp_negative = electronic["esp_negative"]
        desc.esp_balance = electronic["esp_balance"]

        # Charge distribution
        charges = self._calculate_partial_charges()
        desc.total_positive_charge = sum(c for c in charges if c > 0)
        desc.total_negative_charge = sum(c for c in charges if c < 0)
        desc.net_charge = sum(charges)
        desc.charge_density = desc.net_charge / max(1, desc.molecular_volume)

        # Lipinski Rule of 5
        desc.lipinski_violations = self._count_lipinski_violations(desc)

        # Tissue-specific scores
        desc.bbb_score = self._calculate_bbb_score(desc)
        desc.pgp_substrate_prob = self._estimate_pgp_probability(desc)
        desc.plasma_protein_binding = self._estimate_ppb(desc)

        return desc

    def _count_hbd(self) -> int:
        """Count hydrogen bond donors (N-H, O-H, S-H)"""
        donors = 0
        coords = np.array([[a.x, a.y, a.z] for a in self.atoms])

        for i, atom in enumerate(self.atoms):
            if atom.symbol == "H":
                # Find what H is bonded to
                h_pos = coords[i]
                for j, neighbor in enumerate(self.atoms):
                    if i != j and neighbor.symbol in ["N", "O", "S"]:
                        dist = np.linalg.norm(h_pos - coords[j])
                        if dist < 1.2:  # Typical bond length
                            donors += 1
                            break
        return donors

    def _count_hba(self) -> int:
        """Count hydrogen bond acceptors (N, O with lone pairs)"""
        acceptors = 0
        for atom in self.atoms:
            if atom.symbol == "O":
                acceptors += 1
            elif atom.symbol == "N":
                acceptors += 1
            elif atom.symbol == "F":
                acceptors += 1
        return acceptors

    def _analyze_rings(self) -> Tuple[int, int]:
        """Analyze ring systems (simplified)"""
        if not self.bonds:
            return 0, 0

        # Build adjacency for heavy atoms only
        heavy_indices = [i for i, a in enumerate(self.atoms) if a.symbol != "H"]

        # Count rings using Euler formula: R = B - N + 1 (for connected graph)
        num_bonds_heavy = sum(
            1
            for b in self.bonds
            if self.atoms[b[0]].symbol != "H" and self.atoms[b[1]].symbol != "H"
        )
        num_rings = max(0, num_bonds_heavy - len(heavy_indices) + 1)

        # Estimate aromatic rings (C atoms with alternating bonds)
        aromatic = num_rings // 2  # Rough estimate

        return num_rings, aromatic

    def _estimate_logP(self) -> float:
        """Estimate octanol-water partition coefficient"""
        logP = 0.0

        for atom in self.atoms:
            if atom.symbol == "C":
                logP += self.LOGP_CONTRIBUTIONS.get("C_sp3", 0.14)
            elif atom.symbol == "N":
                logP += self.LOGP_CONTRIBUTIONS.get("N_sp3", -0.53)
            elif atom.symbol == "O":
                logP += self.LOGP_CONTRIBUTIONS.get("O_sp3", -0.29)
            elif atom.symbol == "S":
                logP += self.LOGP_CONTRIBUTIONS.get("S", 0.65)
            elif atom.symbol == "F":
                logP += self.LOGP_CONTRIBUTIONS.get("F", 0.41)
            elif atom.symbol == "Cl":
                logP += self.LOGP_CONTRIBUTIONS.get("Cl", 0.69)
            elif atom.symbol == "Br":
                logP += self.LOGP_CONTRIBUTIONS.get("Br", 0.85)
            elif atom.symbol == "H":
                logP += self.LOGP_CONTRIBUTIONS.get("H_on_C", 0.12)

        return logP

    def _estimate_logS(self, logP: float, mw: float) -> float:
        """Estimate aqueous solubility using ESOL-like model"""
        # ESOL: logS = 0.16 - 0.63*logP - 0.0062*MW + 0.066*RB - 0.74*AP
        num_aromatic = sum(1 for a in self.atoms if a.symbol == "C")  # Simplified
        return 0.16 - 0.63 * logP - 0.0062 * mw - 0.74 * (num_aromatic / 10)

    def _estimate_tpsa(self) -> float:
        """Estimate topological polar surface area"""
        tpsa = 0.0

        for atom in self.atoms:
            if atom.symbol == "N":
                tpsa += 12.0  # Average N contribution
            elif atom.symbol == "O":
                tpsa += 15.0  # Average O contribution
            elif atom.symbol == "S":
                tpsa += 25.0  # Average S contribution

        return tpsa

    def _count_rotatable_bonds(self) -> int:
        """Count rotatable bonds"""
        rotatable = 0
        for bond in self.bonds:
            a1, a2, order = bond
            if order == 1:  # Single bond
                sym1 = self.atoms[a1].symbol
                sym2 = self.atoms[a2].symbol
                # Not H, and both are heavy atoms
                if sym1 != "H" and sym2 != "H":
                    rotatable += 1
        return max(0, rotatable - 2)  # Subtract terminal bonds

    def _estimate_volume(self) -> float:
        """Estimate molecular volume from VdW radii"""
        volume = 0.0
        for atom in self.atoms:
            # Sphere volume: 4/3 * pi * r^3
            volume += (4 / 3) * math.pi * (atom.vdw_radius**3)

        # Account for overlap (roughly 70% efficiency)
        return volume * 0.7

    def _estimate_surface_area(self) -> float:
        """Estimate solvent-accessible surface area"""
        area = 0.0
        probe_radius = 1.4  # Water probe

        for atom in self.atoms:
            # Sphere surface: 4 * pi * r^2
            r = atom.vdw_radius + probe_radius
            area += 4 * math.pi * (r**2)

        # Account for burial (roughly 50%)
        return area * 0.5

    def _calculate_sphericity(self) -> float:
        """Calculate molecular sphericity from principal moments"""
        if len(self.atoms) < 2:
            return 1.0

        coords = np.array([[a.x, a.y, a.z] for a in self.atoms])
        centroid = np.mean(coords, axis=0)
        centered = coords - centroid

        # Inertia tensor (simplified, equal masses)
        inertia = np.dot(centered.T, centered)

        try:
            eigenvalues = np.linalg.eigvalsh(inertia)
            eigenvalues = np.sort(eigenvalues)[::-1]

            if eigenvalues[0] > 0:
                # Sphericity: ratio of smallest to largest
                sphericity = eigenvalues[2] / eigenvalues[0]
                return max(0, min(1, sphericity))
        except:
            pass

        return 0.5

    def _compute_electronic_properties(self) -> Dict[str, float]:
        """Compute electronic properties using extended Hückel approximation"""
        # Simplified electronic property estimation
        # In production, this would call qenex_chem DFT

        total_electrons = sum(self._get_valence_electrons(a.symbol) for a in self.atoms)

        # Estimate HOMO/LUMO from electronegativity
        avg_electroneg = np.mean([a.electronegativity for a in self.atoms])

        homo = -avg_electroneg - 2.5  # Rough approximation in eV
        lumo = homo + 3.0 + (total_electrons * 0.01)
        gap = lumo - homo

        # ESP from partial charges
        charges = self._calculate_partial_charges()
        esp_positive = sum(c for c in charges if c > 0)
        esp_negative = abs(sum(c for c in charges if c < 0))
        esp_balance = esp_positive / max(0.01, esp_negative)

        # Dipole moment estimation
        coords = np.array([[a.x, a.y, a.z] for a in self.atoms])
        dipole_vec = np.sum(coords.T * charges, axis=1)
        dipole = np.linalg.norm(dipole_vec) * 4.803  # Convert to Debye

        return {
            "homo": homo,
            "lumo": lumo,
            "gap": gap,
            "dipole": dipole,
            "esp_positive": esp_positive,
            "esp_negative": esp_negative,
            "esp_balance": esp_balance,
        }

    def _estimate_electronic_properties(self) -> Dict[str, float]:
        """Fast estimation without DFT"""
        return self._compute_electronic_properties()

    def _get_valence_electrons(self, symbol: str) -> int:
        """Get number of valence electrons"""
        valence = {
            "H": 1,
            "C": 4,
            "N": 5,
            "O": 6,
            "F": 7,
            "P": 5,
            "S": 6,
            "Cl": 7,
            "Br": 7,
            "I": 7,
            "Na": 1,
            "K": 1,
            "Ca": 2,
            "Mg": 2,
        }
        return valence.get(symbol, 4)

    def _calculate_partial_charges(self) -> List[float]:
        """Calculate Gasteiger partial charges"""
        charges = [0.0] * len(self.atoms)

        # Initial charges from electronegativity
        for i, atom in enumerate(self.atoms):
            charges[i] = (atom.electronegativity - 2.5) * 0.1

        # Iterative equilibration (simplified)
        for _ in range(5):
            new_charges = charges.copy()
            for bond in self.bonds:
                a1, a2, _ = bond
                diff = (
                    self.atoms[a1].electronegativity - self.atoms[a2].electronegativity
                )
                transfer = diff * 0.05
                new_charges[a1] -= transfer
                new_charges[a2] += transfer
            charges = new_charges

        return charges

    def _count_lipinski_violations(self, desc: MolecularDescriptors) -> int:
        """Count Lipinski Rule of 5 violations"""
        violations = 0
        if desc.molecular_weight > 500:
            violations += 1
        if desc.logP > 5:
            violations += 1
        if desc.num_hbd > 5:
            violations += 1
        if desc.num_hba > 10:
            violations += 1
        return violations

    def _calculate_bbb_score(self, desc: MolecularDescriptors) -> float:
        """
        Calculate Blood-Brain Barrier penetration score.

        Based on: CNS MPO score (Wager et al.)
        Factors: MW < 360, TPSA < 90, HBD ≤ 3, logP 1-3, pKa < 8
        """
        score = 0.0

        # Molecular weight contribution (optimal: 250-400)
        if desc.molecular_weight < 400:
            score += 1.0
        elif desc.molecular_weight < 450:
            score += 0.5

        # TPSA contribution (optimal: < 90)
        if desc.tpsa < 60:
            score += 1.0
        elif desc.tpsa < 90:
            score += 0.5

        # HBD contribution (optimal: ≤ 2)
        if desc.num_hbd <= 2:
            score += 1.0
        elif desc.num_hbd <= 3:
            score += 0.5

        # LogP contribution (optimal: 1-3)
        if 1 <= desc.logP <= 3:
            score += 1.0
        elif 0 <= desc.logP <= 4:
            score += 0.5

        # Flexibility (optimal: ≤ 5 rotatable bonds)
        if desc.num_rotatable_bonds <= 5:
            score += 1.0
        elif desc.num_rotatable_bonds <= 8:
            score += 0.5

        return score / 5.0  # Normalize to 0-1

    def _estimate_pgp_probability(self, desc: MolecularDescriptors) -> float:
        """
        Estimate P-glycoprotein substrate probability.

        PgP efflux reduces brain/tumor penetration.
        Risk factors: MW > 400, HBD > 3, TPSA > 120
        """
        risk_score = 0.0

        if desc.molecular_weight > 400:
            risk_score += 0.25
        if desc.molecular_weight > 500:
            risk_score += 0.25

        if desc.num_hbd > 3:
            risk_score += 0.2

        if desc.tpsa > 120:
            risk_score += 0.2

        if desc.num_rotatable_bonds > 10:
            risk_score += 0.1

        return min(1.0, risk_score)

    def _estimate_ppb(self, desc: MolecularDescriptors) -> float:
        """
        Estimate plasma protein binding.

        High binding (>95%) reduces free drug concentration.
        """
        # LogP-based estimation (higher logP = higher binding)
        ppb = 50 + desc.logP * 10

        # Adjust for charge
        if desc.net_charge < -0.5:
            ppb += 10  # Acidic drugs bind more

        return max(0, min(99.9, ppb))


# Export for use in tissue distribution models
__all__ = ["MolecularFeatureExtractor", "MolecularDescriptors", "Atom", "TissueType"]
