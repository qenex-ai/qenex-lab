"""
Molecule Module
Defines chemical structures for quantum simulation.
"""

from typing import List, Tuple

class Molecule:
    """
    Represents a molecular structure.
    """
    
    def __init__(self, atoms: List[Tuple[str, Tuple[float, float, float]]], charge: int = 0, multiplicity: int = 1):
        # [SECURITY PATCH] Element Validation
        valid_elements = set(["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "P", "S", "Pb", "I"]) # Expanded for Bio-Chem (DNA/Proteins) + Perovskites
        
        atomic_numbers = {
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, 
            "N": 7, "O": 8, "F": 9, "Ne": 10, "P": 15, "S": 16,
            "Pb": 82, "I": 53
        }

        for element, _ in atoms:
            if element not in valid_elements:
                raise ValueError(f"Alchemy Error: Unknown element '{element}'.")
        
        # [SECURITY PATCH] Spin Multiplicity Check (2S+1 rule)
        # Total electrons = Sum(Z) - charge
        total_protons = sum(atomic_numbers[element] for element, _ in atoms)
        total_electrons = total_protons - charge
        
        # If Ne is Even, Multiplicity must be Odd (1, 3, 5...)
        # If Ne is Odd, Multiplicity must be Even (2, 4, 6...)
        if total_electrons % 2 == 0:
             if multiplicity % 2 == 0:
                 raise ValueError(f"Spin Parity Error: Even electrons ({total_electrons}) require Odd multiplicity (Got {multiplicity}).")
        else:
             if multiplicity % 2 != 0:
                 raise ValueError(f"Spin Parity Error: Odd electrons ({total_electrons}) require Even multiplicity (Got {multiplicity}).")

        self.atoms = atoms
        self.charge = charge
        self.multiplicity = multiplicity
        
    def __repr__(self):
        formula = "".join([a[0] for a in self.atoms])
        return f"Molecule({formula}, charge={self.charge})"
