import sys
import math
import random
import numpy as np

# ==========================================
# SDF Parser
# ==========================================


class MoleculeStructure:
    def __init__(self, name):
        self.name = name
        self.atoms = []  # List of (symbol, x, y, z)
        self.charges = {}  # Map index (1-based) to charge
        self.donors = []  # List of atom indices
        self.acceptors = []  # List of atom indices

    def add_atom(self, symbol, x, y, z):
        self.atoms.append({"symbol": symbol, "x": x, "y": y, "z": z})

    def get_coords(self):
        return np.array([[a["x"], a["y"], a["z"]] for a in self.atoms])

    def center_mass(self):
        coords = self.get_coords()
        return np.mean(coords, axis=0)

    def translate(self, vec):
        for atom in self.atoms:
            atom["x"] += vec[0]
            atom["y"] += vec[1]
            atom["z"] += vec[2]

    def rotate(self, axis, theta):
        """Rotate atoms around an axis by theta radians"""
        axis = axis / np.linalg.norm(axis)
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

        rot_matrix = np.array(
            [
                [aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)],
                [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab)],
                [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc],
            ]
        )

        for atom in self.atoms:
            v = np.array([atom["x"], atom["y"], atom["z"]])
            v_new = np.dot(rot_matrix, v)
            atom["x"], atom["y"], atom["z"] = v_new


def parse_sdf(filepath, name):
    mol = MoleculeStructure(name)
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Simple parser assuming standard SDF format from PubChem
    # Skip header
    try:
        counts_line = lines[3]
        num_atoms = int(counts_line[0:3].strip())

        # Parse Atoms
        for i in range(num_atoms):
            line = lines[4 + i]
            x = float(line[0:10])
            y = float(line[10:20])
            z = float(line[20:30])
            symbol = line[31:34].strip()
            mol.add_atom(symbol, x, y, z)

            # Heuristic for Donors/Acceptors
            idx = i + 1
            if symbol == "O" or symbol == "N":
                mol.acceptors.append(idx)
                # If it has H attached (in 3D, distance check needed, but let's assume heavy atoms for now)
                # Actually PubChem 3D includes H atoms.

        # Find H attachments for Donors
        coords = mol.get_coords()
        for i, atom in enumerate(mol.atoms):
            if atom["symbol"] == "H":
                # Find neighbor
                h_pos = coords[i]
                for j, neighbor in enumerate(mol.atoms):
                    if i == j:
                        continue
                    if neighbor["symbol"] in ["O", "N"]:
                        dist = np.linalg.norm(h_pos - coords[j])
                        if dist < 1.2:  # Bonded
                            mol.donors.append(j + 1)  # The heavy atom is the donor root

        # Parse Charges
        # Look for > <PUBCHEM_MMFF94_PARTIAL_CHARGES>
        charge_start = -1
        for idx, line in enumerate(lines):
            if "PUBCHEM_MMFF94_PARTIAL_CHARGES" in line:
                charge_start = idx + 1
                break

        if charge_start != -1:
            count = int(lines[charge_start])
            for i in range(count):
                line = lines[charge_start + 1 + i]
                parts = line.split()
                atom_idx = int(parts[0])
                charge = float(parts[1])
                mol.charges[atom_idx] = charge

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

    return mol


# ==========================================
# Energy Function
# ==========================================


def calculate_interaction_energy(mol1, mol2):
    """
    Calculate non-bonded interaction energy (Electrostatic + VdW)
    """
    E_elec = 0.0
    E_vdw = 0.0

    # Constants
    COULOMB_CONST = 332.06  # kcal/mol * Angstrom / e^2

    # VdW params (approximate AMBER-like)
    # Element: (Radius, Epsilon)
    vdw_params = {
        "H": (1.4870, 0.0157),
        "C": (1.9080, 0.1094),
        "N": (1.8240, 0.1700),
        "O": (1.6612, 0.2100),
        "P": (2.1000, 0.2000),
        "S": (2.0000, 0.2500),
    }

    coords1 = mol1.get_coords()
    coords2 = mol2.get_coords()

    for i, at1 in enumerate(mol1.atoms):
        q1 = mol1.charges.get(i + 1, 0.0)
        p1 = vdw_params.get(at1["symbol"], (1.7, 0.15))
        pos1 = coords1[i]

        for j, at2 in enumerate(mol2.atoms):
            q2 = mol2.charges.get(j + 1, 0.0)
            p2 = vdw_params.get(at2["symbol"], (1.7, 0.15))
            pos2 = coords2[j]

            dist = np.linalg.norm(pos1 - pos2)
            if dist < 0.1:
                dist = 0.1  # Avoid singularity

            # Electrostatics
            E_elec += (q1 * q2) / dist

            # VdW (Lennard-Jones 6-12)
            # Rmin_ij = R_i + R_j
            # Eps_ij = sqrt(eps_i * eps_j)
            rmin = p1[0] + p2[0]
            eps = math.sqrt(p1[1] * p2[1])

            term6 = (rmin / dist) ** 6
            term12 = term6 * term6
            E_vdw += eps * (term12 - 2 * term6)

    E_elec *= COULOMB_CONST
    return E_elec + E_vdw


# ==========================================
# Screening Logic
# ==========================================


def screen_interactions():
    print("Parsing Molecules...")
    curcumin = parse_sdf("curcumin.sdf", "Curcumin")
    coformers = [
        parse_sdf("nicotinamide.sdf", "Nicotinamide"),
        parse_sdf("piperine.sdf", "Piperine"),
        parse_sdf("ascorbic_acid.sdf", "Ascorbic Acid"),
        parse_sdf("citric_acid.sdf", "Citric Acid"),
    ]

    # Filter failed parses
    coformers = [c for c in coformers if c is not None]

    if curcumin is None:
        print("Failed to parse Curcumin.")
        return

    # Identify Curcumin Binding Sites (Phenolic OH)
    # Find OH groups on the rings
    # In Curcumin, the phenolic oxygens are usually heavy atom indices:
    # We can find them by looking for O bonded to C_aromatic and H.
    # For this simplified script, we'll try to dock to *every* Donor/Acceptor on Curcumin
    # and pick the best score.

    results = []

    print("\nStarting Interaction Screening (Classical Forcefield)...")
    print(
        f"{'Co-former':<20} | {'Best Energy (kcal/mol)':<25} | {'Interaction Type':<30}"
    )
    print("-" * 80)

    for cof in coformers:
        best_energy = 9999.0
        best_geom = "None"

        # Center coformer at origin first
        cof_center = cof.center_mass()
        cof.translate(-cof_center)

        # Docking Search
        # Try to match every donor of Cof with every acceptor of Cur and vice-versa

        # 1. Coformer Donor -> Curcumin Acceptor
        for cof_idx in cof.donors:
            for cur_idx in curcumin.acceptors:
                # Target Position: Place coformer such that donor H is near acceptor O
                # Vector math: A_pos + (H-Bond Dist) * direction

                # Simplified: Just place the centroids at a distance and rotate randomly
                # This is a Monte Carlo rigid docking

                cur_pos = np.array(
                    [
                        curcumin.atoms[cur_idx - 1]["x"],
                        curcumin.atoms[cur_idx - 1]["y"],
                        curcumin.atoms[cur_idx - 1]["z"],
                    ]
                )

                # Run 50 random rotations around this site
                for _ in range(50):
                    # Random rotation
                    axis = np.random.rand(3)
                    angle = random.uniform(0, 2 * math.pi)
                    cof.rotate(axis, angle)

                    # Place coformer 2.8 Angstroms away from target atom
                    # We need to find the atom in cof that corresponds to cof_idx and place IT near cur_pos
                    cof_atom_pos = np.array(
                        [
                            cof.atoms[cof_idx - 1]["x"],
                            cof.atoms[cof_idx - 1]["y"],
                            cof.atoms[cof_idx - 1]["z"],
                        ]
                    )

                    # Translation vector
                    shift = (
                        cur_pos - cof_atom_pos + np.random.rand(3) * 0.5
                    )  # Add jitter
                    shift = (
                        shift / np.linalg.norm(shift) * 2.8
                    )  # 2.8 A H-bond distance (approx)

                    # Wait, we want the ATOMS to be close.
                    # Vector to move Cof_Atom to Cur_Atom
                    vec = cur_pos - cof_atom_pos
                    # Normalize direction
                    dist = np.linalg.norm(vec)
                    direction = vec / dist
                    # Move to 2.8 A distance
                    final_pos = cur_pos - direction * 2.8
                    trans_vec = final_pos - cof_atom_pos

                    # Apply temporary translation
                    cof.translate(trans_vec)

                    # Calc Energy
                    E = calculate_interaction_energy(curcumin, cof)

                    if E < best_energy:
                        best_energy = E
                        cur_sym = curcumin.atoms[cur_idx - 1]["symbol"]
                        cof_sym = cof.atoms[cof_idx - 1]["symbol"]
                        best_geom = (
                            f"Cof({cof_sym}{cof_idx}) -> Cur({cur_sym}{cur_idx})"
                        )

                    # Reset (translate back)
                    cof.translate(-trans_vec)

        # 2. Curcumin Donor -> Coformer Acceptor
        for cur_idx in curcumin.donors:
            for cof_idx in cof.acceptors:
                cur_pos = np.array(
                    [
                        curcumin.atoms[cur_idx - 1]["x"],
                        curcumin.atoms[cur_idx - 1]["y"],
                        curcumin.atoms[cur_idx - 1]["z"],
                    ]
                )

                for _ in range(50):
                    axis = np.random.rand(3)
                    angle = random.uniform(0, 2 * math.pi)
                    cof.rotate(axis, angle)

                    cof_atom_pos = np.array(
                        [
                            cof.atoms[cof_idx - 1]["x"],
                            cof.atoms[cof_idx - 1]["y"],
                            cof.atoms[cof_idx - 1]["z"],
                        ]
                    )
                    vec = cur_pos - cof_atom_pos
                    dist = np.linalg.norm(vec)
                    if dist < 0.1:
                        direction = np.array([1, 0, 0])
                    else:
                        direction = vec / dist

                    final_pos = cur_pos - direction * 2.8
                    trans_vec = final_pos - cof_atom_pos

                    cof.translate(trans_vec)
                    E = calculate_interaction_energy(curcumin, cof)

                    if E < best_energy:
                        best_energy = E
                        cur_sym = curcumin.atoms[cur_idx - 1]["symbol"]
                        cof_sym = cof.atoms[cof_idx - 1]["symbol"]
                        best_geom = (
                            f"Cur({cur_sym}{cur_idx}) -> Cof({cof_sym}{cof_idx})"
                        )

                    cof.translate(-trans_vec)

        results.append((cof.name, best_energy, best_geom))
        print(f"{cof.name:<20} | {best_energy:.2f}{' kcal/mol':<15} | {best_geom}")

    # Identify Winner
    results.sort(key=lambda x: x[1])
    winner = results[0]
    print("-" * 80)
    print(f"Winner: {winner[0]} with Binding Energy {winner[1]:.2f} kcal/mol")

    # Save the winner info for QM step
    with open("winner_info.txt", "w") as f:
        f.write(f"{winner[0]},{winner[1]},{winner[2]}")


if __name__ == "__main__":
    screen_interactions()
