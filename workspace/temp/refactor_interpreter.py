import sys
import os

path = "packages/qenex-qlang/src/interpreter.py"
with open(path, "r") as f:
    content = f.read()

# Chemistry
chem_old = """                if Molecule is None or MatrixHartreeFock is None:
                    print("   [Q-Lang] Mocking chemistry kernel...")
                    print(f"   [CHEM] System: {atoms}")
                    print(f"   [CHEM] Basis: {basis}")
                    print(f"   [CHEM] Energy: -1.117 Eh (Mock)")
                    self.context["last_energy"] = QValue(-1.117, Dimensions(mass=1, length=2, time=-2))
                    return

                try:
                    mol = Molecule(atoms)"""
chem_new = """                if Molecule is None or MatrixHartreeFock is None:
                    print("❌ Critical: Chemistry kernels (Molecule/HartreeFock) not found.")
                    return

                try:
                    mol = Molecule(atoms)"""

# Biology
bio_old = """                if ProteinFolder is None:
                    print("   [Q-Lang] Mocking biology kernel...")
                    print(f"   [BIO] Folding sequence: {sequence}")
                    print(f"   [BIO] Energy: -10.0 (Mock)")
                    return

                try:
                    folder = ProteinFolder()"""
bio_new = """                if ProteinFolder is None:
                    print("❌ Critical: Biology kernel (ProteinFolder) not found.")
                    return

                try:
                    folder = ProteinFolder()"""

# Physics
phys_old = """                 if LatticeSimulator is None:
                     print("   [Q-Lang] Mocking physics kernel...")
                     try:
                         temp = float(parts[4])
                         Tc = 2.269
                         if temp >= Tc: mag = 0.0 + random.uniform(0, 0.05)
                         else: mag = (1 - math.sinh(2/temp)**-4)**(1/8) if temp > 0 else 1.0
                         if isinstance(mag, complex): mag = 0
                         self.context["last_magnetization"] = QValue(mag, Dimensions())
                         print(f"   [PHYSICS] (Mock) T={temp}, M={mag:.4f}")
                     except Exception as e:
                         print(f"   [PHYSICS] Mock failed: {e}")
                     return

                 try:
                     size = int(float(parts[2]))"""
phys_new = """                 if LatticeSimulator is None:
                     print("❌ Critical: Physics kernel (LatticeSimulator) not found.")
                     return

                 try:
                     size = int(float(parts[2]))"""

# Perform replacements
if chem_old in content:
    content = content.replace(chem_old, chem_new)
    print("Replaced Chemistry mock.")
else:
    print("Chemistry mock not found.")

if bio_old in content:
    content = content.replace(bio_old, bio_new)
    print("Replaced Biology mock.")
else:
    print("Biology mock not found.")

if phys_old in content:
    content = content.replace(phys_old, phys_new)
    print("Replaced Physics mock.")
else:
    print("Physics mock not found.")

with open(path, "w") as f:
    f.write(content)
