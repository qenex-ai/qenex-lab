"""
Solver Module
Implements Hartree-Fock Self-Consistent Field (SCF) method.
Supports s and p orbitals (up to L=1) using updated integrals module.
"""

import numpy as np
import integrals as ints
from molecule import Molecule

# ==========================================
# Basis Set Construction
# ==========================================

class ContractedGaussian:
    """
    Represents a contracted Gaussian function (Basis Function in the SCF calculation).
    Phi(r) = Sum_k d_k * g_k(alpha_k, r)
    """
    def __init__(self, primitives, label=""):
        self.primitives = primitives
        self.label = label
        
    def normalize(self):
        """
        Ensures the contracted function is normalized: (Phi|Phi) = 1
        """
        overlap = 0.0
        for p1 in self.primitives:
            for p2 in self.primitives:
                overlap += ints.overlap(p1, p2)
        
        if overlap < 1e-14:
            raise ValueError(f"Normalization overlap too small: {overlap}")
            
        norm_factor = 1.0 / np.sqrt(overlap)
        for p in self.primitives:
            p.coeff *= norm_factor
            p.N *= norm_factor # Update precomputed normalization

class CISolver:
    """
    Placeholder CISolver to satisfy imports.
    """
    def __init__(self, basis_set="sto-3g"):
        pass
    def compute_energy(self, molecule, method="FCI"):
        return 0.0

class HartreeFockSolver:
    """
    Restricted Hartree-Fock (RHF) Solver for closed-shell molecules.
    Supports STO-3G and 6-31G basis sets.
    """
    def __init__(self, basis="sto-3g"):
        self.basis_name = basis.lower()

    def build_basis(self, molecule: Molecule):
        """
        Constructs the basis set (list of ContractedGaussians) for the molecule.
        """
        basis_set = []
        
        # 6-31G Implementation
        if self.basis_name == '6-31g':
            for atom_idx, (element, coord) in enumerate(molecule.atoms):
                coord = np.array(coord)
                
                if element == 'H':
                    # Inner 1s (3 prims)
                    prim_in = [ints.BasisFunction(coord, a, c, (0,0,0)) for a, c in ints.BasisSet631G.H_inner]
                    cgf_in = ContractedGaussian(prim_in, label=f"H_{atom_idx}_1s_in")
                    cgf_in.normalize()
                    basis_set.append(cgf_in)
                    
                    # Outer 1s (1 prim)
                    prim_out = [ints.BasisFunction(coord, a, c, (0,0,0)) for a, c in ints.BasisSet631G.H_outer]
                    cgf_out = ContractedGaussian(prim_out, label=f"H_{atom_idx}_1s_out")
                    cgf_out.normalize()
                    basis_set.append(cgf_out)
                
                elif element == 'He':
                    # Inner 1s
                    prim_in = [ints.BasisFunction(coord, a, c, (0,0,0)) for a, c in ints.BasisSet631G.He_inner]
                    cgf_in = ContractedGaussian(prim_in, label=f"He_{atom_idx}_1s_in")
                    cgf_in.normalize()
                    basis_set.append(cgf_in)
                    
                    # Outer 1s
                    prim_out = [ints.BasisFunction(coord, a, c, (0,0,0)) for a, c in ints.BasisSet631G.He_outer]
                    cgf_out = ContractedGaussian(prim_out, label=f"He_{atom_idx}_1s_out")
                    cgf_out.normalize()
                    basis_set.append(cgf_out)
                    
                elif element in ['C', 'N', 'O']:
                    # Core 1s
                    data_core = getattr(ints.BasisSet631G, f"{element}_core_1s")
                    prim_core = [ints.BasisFunction(coord, a, c, (0,0,0)) for a, c in data_core]
                    cgf_core = ContractedGaussian(prim_core, label=f"{element}_{atom_idx}_1s")
                    cgf_core.normalize()
                    basis_set.append(cgf_core)
                    
                    # Valence Inner (2s, 2p)
                    data_val_in = getattr(ints.BasisSet631G, f"{element}_val_inner")
                    exps = data_val_in['exponents']
                    s_cs = data_val_in['s_coeffs']
                    p_cs = data_val_in['p_coeffs']
                    
                    # Inner 2s
                    prim_2s_in = [ints.BasisFunction(coord, a, c, (0,0,0)) for a, c in zip(exps, s_cs)]
                    cgf_2s_in = ContractedGaussian(prim_2s_in, label=f"{element}_{atom_idx}_2s_in")
                    cgf_2s_in.normalize()
                    basis_set.append(cgf_2s_in)
                    
                    # Inner 2p (x, y, z)
                    for d_lbl, lmn in [('px', (1,0,0)), ('py', (0,1,0)), ('pz', (0,0,1))]:
                        prim_2p_in = [ints.BasisFunction(coord, a, c, lmn) for a, c in zip(exps, p_cs)]
                        cgf_2p_in = ContractedGaussian(prim_2p_in, label=f"{element}_{atom_idx}_{d_lbl}_in")
                        cgf_2p_in.normalize()
                        basis_set.append(cgf_2p_in)
                        
                    # Valence Outer (2s, 2p)
                    data_val_out = getattr(ints.BasisSet631G, f"{element}_val_outer")
                    exps = data_val_out['exponents']
                    s_cs = data_val_out['s_coeffs']
                    p_cs = data_val_out['p_coeffs']
                    
                    # Outer 2s
                    prim_2s_out = [ints.BasisFunction(coord, a, c, (0,0,0)) for a, c in zip(exps, s_cs)]
                    cgf_2s_out = ContractedGaussian(prim_2s_out, label=f"{element}_{atom_idx}_2s_out")
                    cgf_2s_out.normalize()
                    basis_set.append(cgf_2s_out)
                    
                    # Outer 2p
                    for d_lbl, lmn in [('px', (1,0,0)), ('py', (0,1,0)), ('pz', (0,0,1))]:
                        prim_2p_out = [ints.BasisFunction(coord, a, c, lmn) for a, c in zip(exps, p_cs)]
                        cgf_2p_out = ContractedGaussian(prim_2p_out, label=f"{element}_{atom_idx}_{d_lbl}_out")
                        cgf_2p_out.normalize()
                        basis_set.append(cgf_2p_out)

                elif element in ['P', 'S']:
                    # Core 1s
                    data_core_1s = getattr(ints.BasisSet631G, f"{element}_core_1s")
                    prim_core_1s = [ints.BasisFunction(coord, a, c, (0,0,0)) for a, c in data_core_1s]
                    cgf_core_1s = ContractedGaussian(prim_core_1s, label=f"{element}_{atom_idx}_1s")
                    cgf_core_1s.normalize()
                    basis_set.append(cgf_core_1s)
                    
                    # Core 2sp (2s, 2p)
                    data_core_2sp = getattr(ints.BasisSet631G, f"{element}_core_2sp")
                    exps = data_core_2sp['exponents']
                    s_cs = data_core_2sp['s_coeffs']
                    p_cs = data_core_2sp['p_coeffs']
                    
                    # Core 2s
                    prim_2s = [ints.BasisFunction(coord, a, c, (0,0,0)) for a, c in zip(exps, s_cs)]
                    cgf_2s = ContractedGaussian(prim_2s, label=f"{element}_{atom_idx}_2s")
                    cgf_2s.normalize()
                    basis_set.append(cgf_2s)
                    
                    # Core 2p
                    for d_lbl, lmn in [('px', (1,0,0)), ('py', (0,1,0)), ('pz', (0,0,1))]:
                        prim_2p = [ints.BasisFunction(coord, a, c, lmn) for a, c in zip(exps, p_cs)]
                        cgf_2p = ContractedGaussian(prim_2p, label=f"{element}_{atom_idx}_{d_lbl}_core")
                        cgf_2p.normalize()
                        basis_set.append(cgf_2p)

                    # Valence Inner 3sp
                    data_val_in = getattr(ints.BasisSet631G, f"{element}_val_inner")
                    exps = data_val_in['exponents']
                    s_cs = data_val_in['s_coeffs']
                    p_cs = data_val_in['p_coeffs']
                    
                    prim_3s_in = [ints.BasisFunction(coord, a, c, (0,0,0)) for a, c in zip(exps, s_cs)]
                    cgf_3s_in = ContractedGaussian(prim_3s_in, label=f"{element}_{atom_idx}_3s_in")
                    cgf_3s_in.normalize()
                    basis_set.append(cgf_3s_in)
                    
                    for d_lbl, lmn in [('px', (1,0,0)), ('py', (0,1,0)), ('pz', (0,0,1))]:
                        prim_3p_in = [ints.BasisFunction(coord, a, c, lmn) for a, c in zip(exps, p_cs)]
                        cgf_3p_in = ContractedGaussian(prim_3p_in, label=f"{element}_{atom_idx}_{d_lbl}_3p_in")
                        cgf_3p_in.normalize()
                        basis_set.append(cgf_3p_in)
                        
                    # Valence Outer 3sp
                    data_val_out = getattr(ints.BasisSet631G, f"{element}_val_outer")
                    exps = data_val_out['exponents']
                    s_cs = data_val_out['s_coeffs']
                    p_cs = data_val_out['p_coeffs']
                    
                    prim_3s_out = [ints.BasisFunction(coord, a, c, (0,0,0)) for a, c in zip(exps, s_cs)]
                    cgf_3s_out = ContractedGaussian(prim_3s_out, label=f"{element}_{atom_idx}_3s_out")
                    cgf_3s_out.normalize()
                    basis_set.append(cgf_3s_out)
                    
                    for d_lbl, lmn in [('px', (1,0,0)), ('py', (0,1,0)), ('pz', (0,0,1))]:
                        prim_3p_out = [ints.BasisFunction(coord, a, c, lmn) for a, c in zip(exps, p_cs)]
                        cgf_3p_out = ContractedGaussian(prim_3p_out, label=f"{element}_{atom_idx}_{d_lbl}_3p_out")
                        cgf_3p_out.normalize()
                        basis_set.append(cgf_3p_out)

                else:
                    print(f"Warning: Element {element} not fully supported in 6-31G build_basis. Skipping.")

            return basis_set

        # Default STO-3G (existing code)
        # Z mapping for element properties
        # In STO-3G, we check if 1s is core or valence based on row.
        # H, He: 1s is valence.
        # Li - F: 1s is core, 2s/2p are valence.
        
        for atom_idx, (element, coord) in enumerate(molecule.atoms):
            coord = np.array(coord)
            
            if element in ['H', 'He']:
                # H, He: Only 1s orbital
                zeta = ints.STO3G.zeta[element]
                primitives = []
                for alpha_prime, d in ints.STO3G.basis_1s:
                    alpha = alpha_prime * (zeta**2)
                    primitives.append(ints.BasisFunction(coord, alpha, d, (0,0,0)))
                
                cgf = ContractedGaussian(primitives, label=f"{element}_{atom_idx}_1s")
                cgf.normalize()
                basis_set.append(cgf)
                
            elif element in ['Li', 'Be', 'B', 'C', 'N', 'O', 'F']:
                # Row 2: 1s (core) + 2s, 2px, 2py, 2pz (valence)
                
                # --- 1s Core ---
                zeta_1s = ints.STO3G.zeta.get(f"{element}_1s", 1.0) # Fallback should not happen if defined
                prim_1s = []
                for alpha_prime, d in ints.STO3G.basis_1s:
                    alpha = alpha_prime * (zeta_1s**2)
                    prim_1s.append(ints.BasisFunction(coord, alpha, d, (0,0,0)))
                
                cgf_1s = ContractedGaussian(prim_1s, label=f"{element}_{atom_idx}_1s")
                cgf_1s.normalize()
                basis_set.append(cgf_1s)
                
                # --- 2s Valence ---
                zeta_val = ints.STO3G.zeta[element]
                prim_2s = []
                for alpha_prime, d in ints.STO3G.basis_2s:
                    alpha = alpha_prime * (zeta_val**2)
                    prim_2s.append(ints.BasisFunction(coord, alpha, d, (0,0,0)))
                
                cgf_2s = ContractedGaussian(prim_2s, label=f"{element}_{atom_idx}_2s")
                cgf_2s.normalize()
                basis_set.append(cgf_2s)
                
                # --- 2p Valence (px, py, pz) ---
                # STO-3G 2p shares exponents with 2s but different contraction coeffs (basis_2p)
                # We iterate over directions
                for direction, lmn in [('px', (1,0,0)), ('py', (0,1,0)), ('pz', (0,0,1))]:
                    prim_2p = []
                    for alpha_prime, d in ints.STO3G.basis_2p:
                        alpha = alpha_prime * (zeta_val**2)
                        prim_2p.append(ints.BasisFunction(coord, alpha, d, lmn))
                    
                    cgf_2p = ContractedGaussian(prim_2p, label=f"{element}_{atom_idx}_{direction}")
                    cgf_2p.normalize()
                    basis_set.append(cgf_2p)
            
            else:
                # Fallback for undefined elements, assuming simple 1s model or skipping
                print(f"Warning: Element {element} not fully supported in STO-3G build_basis. Skipping.")
            
        return basis_set

    def compute_nuclear_repulsion(self, molecule: Molecule):
        """
        Calculates classic nuclear repulsion energy.
        """
        energy = 0.0
        atoms = molecule.atoms
        Z_map = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                el_i, pos_i = atoms[i]
                el_j, pos_j = atoms[j]
                
                Zi = Z_map.get(el_i, 0)
                Zj = Z_map.get(el_j, 0)
                
                dist = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                if dist > 1e-12:
                    energy += Zi * Zj / dist
        return energy

    def compute_energy(self, molecule: Molecule, max_iter=50, tolerance=1e-8):
        """
        Main SCF Loop.
        """
        atoms = molecule.atoms
        basis = self.build_basis(molecule)
        N = len(basis)
        
        if N == 0:
            return 0.0
            
        print(f"Basis functions: {N}")
        for b in basis:
            print(f"  {b.label}")

        # Precompute Integrals
        # Matrices
        S = np.zeros((N, N))
        T = np.zeros((N, N))
        V = np.zeros((N, N))
        ERI = np.zeros((N, N, N, N))
        
        Z_map = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        atom_qs = []
        for el, pos in atoms:
            atom_qs.append( (Z_map.get(el, 0), np.array(pos)) )

        # 1-Electron Integrals
        print("Computing 1-electron integrals...")
        for i in range(N):
            for j in range(N):
                s_val = 0.0
                t_val = 0.0
                v_val = 0.0
                
                for pi in basis[i].primitives:
                    for pj in basis[j].primitives:
                        s_val += ints.overlap(pi, pj)
                        t_val += ints.kinetic(pi, pj)
                        
                        for Z, pos in atom_qs:
                            v_val += ints.nuclear_attraction(pi, pj, pos, Z)
                            
                S[i, j] = s_val
                T[i, j] = t_val
                V[i, j] = v_val
        
        H_core = T + V
        
        # 2-Electron Integrals (ERI)
        print("Computing 2-electron integrals...")
        # (mu nu | lambda sigma)
        # Naive O(N^4) loop. 
        # For small systems (Water: N=7), 7^4 = 2401 integrals. Very fast.
        for mu in range(N):
            for nu in range(N):
                for lam in range(N):
                    for sig in range(N):
                        val = 0.0
                        for p_mu in basis[mu].primitives:
                            for p_nu in basis[nu].primitives:
                                for p_lam in basis[lam].primitives:
                                    for p_sig in basis[sig].primitives:
                                        val += ints.eri(p_mu, p_nu, p_lam, p_sig)
                        ERI[mu, nu, lam, sig] = val

        # Orthogonalization Matrix X = S^(-1/2)
        evals, evecs = np.linalg.eigh(S)
        # Check for linear dependence
        threshold = 1e-6
        keep_indices = [i for i, val in enumerate(evals) if val > threshold]
        
        if len(keep_indices) < N:
            print(f"Warning: Linear dependence detected. Keeping {len(keep_indices)}/{N} basis functions.")
            # For simplicity, we just invert relevant subspace or error out.
            # But let's construct X properly.
            
        inv_sqrt_evals = np.array([1.0/np.sqrt(e) if e > threshold else 0.0 for e in evals])
        X = evecs @ np.diag(inv_sqrt_evals) @ evecs.T
        
        # Initial Guess P = 0 (Core Hamiltonian Guess)
        # Solve F = H_core first to get initial C
        F_0 = H_core
        F_0_prime = X.T @ F_0 @ X
        eps_0, C_0_prime = np.linalg.eigh(F_0_prime)
        C_0 = X @ C_0_prime
        
        # Initialize loop variables
        C = C_0
        eps = eps_0
        
        # Determine Occupancy
        n_elec = sum(Z_map.get(at[0], 0) for at in atoms) - molecule.charge
        n_occ = int(n_elec // 2)
        print(f"Electrons: {n_elec}, Occupied Orbitals: {n_occ}")
        
        if n_elec % 2 != 0:
            print("Warning: Open shell system (odd electrons). RHF results may be incorrect.")
            
        P = np.zeros((N, N))
        for mu in range(N):
            for nu in range(N):
                for a in range(n_occ):
                    P[mu, nu] += 2.0 * C_0[mu, a] * C_0[nu, a]

        electronic_energy = 0.0
        old_energy = 0.0
        
        print("Starting SCF Loop...")
        for iteration in range(max_iter):
            # Build G matrix
            # G_mu_nu = Sum_lam_sig P_lam_sig * [ (mu nu | lam sig) - 0.5 (mu lam | nu sig) ]
            
            # Vectorized contraction:
            J = np.einsum('ls,mnls->mn', P, ERI)
            K = np.einsum('ls,mlns->mn', P, ERI)
            G = J - 0.5 * K
            
            # Fock Matrix
            F = H_core + G
            
            # Transform to orthogonal basis
            F_prime = X.T @ F @ X
            
            # Diagonalize
            eps, C_prime = np.linalg.eigh(F_prime)
            
            # Transform coefficients back
            C = X @ C_prime
            
            # Update Density
            P_new = np.zeros((N, N))
            for mu in range(N):
                for nu in range(N):
                    for a in range(n_occ):
                        P_new[mu, nu] += 2.0 * C[mu, a] * C[nu, a]
            
            # Damping
            if iteration > 0:
                alpha = 0.5
                P = alpha * P_new + (1.0 - alpha) * P
            else:
                P = P_new
            
            # Calculate Energy
            current_E = 0.5 * np.sum(P * (H_core + F))
            
            diff = np.abs(current_E - old_energy)
            # print(f"Iter {iteration}: E = {current_E:.8f} (diff {diff:.2e})")
            
            if diff < tolerance and iteration > 1:
                electronic_energy = current_E
                print(f"Converged in {iteration} iterations.")
                break
                
            old_energy = current_E
            electronic_energy = current_E 

        nuclear_repulsion = self.compute_nuclear_repulsion(molecule)
        total_energy = electronic_energy + nuclear_repulsion
        
        print(f"Nuclear Repulsion: {nuclear_repulsion:.6f}")
        print(f"Electronic Energy: {electronic_energy:.6f}")
        print(f"HF Energy:         {total_energy:.6f}")
        
        # ==========================================
        # MP2 Correlation
        # ==========================================
        
        print("Starting MP2 Calculation...")
        
        # 1. Transform ERIs from AO to MO basis
        # C matrix columns are MOs. C_mu_p
        # (pq|rs) = Sum_mu_nu_lam_sig C_mu_p C_nu_q C_lam_r C_sig_s (mu nu | lam sig)
        
        # We need to transform ERI[mu, nu, lam, sig] -> MO_ERI[p, q, r, s]
        # Optimization: Partial transformations
        # Step 1: (p nu | lam sig) = Sum_mu C_mu_p (mu nu | lam sig)
        # Tensor contraction: (N,N,N,N) * (N,N) -> (N,N,N,N)
        
        # C shape: (N, N) where columns are orbitals
        # ERI shape: (N, N, N, N)
        
        # Step 1: Transform index 1 (mu -> p)
        # ERI_1[p, nu, lam, sig] = Sum_mu C[mu, p] * ERI[mu, nu, lam, sig]
        ERI_1 = np.einsum('mp,mnls->pnls', C, ERI)
        
        # Step 2: Transform index 2 (nu -> q)
        # ERI_2[p, q, lam, sig] = Sum_nu C[nu, q] * ERI_1[p, nu, lam, sig]
        ERI_2 = np.einsum('nq,pnls->pqls', C, ERI_1)
        
        # Step 3: Transform index 3 (lam -> r)
        # ERI_3[p, q, r, sig] = Sum_lam C[lam, r] * ERI_2[p, q, lam, sig]
        ERI_3 = np.einsum('lr,pqls->pqrs', C, ERI_2)
        
        # Step 4: Transform index 4 (sig -> s)
        # MO_ERI[p, q, r, s] = Sum_sig C[sig, s] * ERI_3[p, q, r, sig]
        MO_ERI = np.einsum('ks,pqrk->pqrs', C, ERI_3)
        
        # MO Integrals (ia|jb) in Chemist notation <ij|ab> ?
        # Physics notation: <pq|rs> = Integral phi_p*(1) phi_q*(2) 1/r12 phi_r(1) phi_s(2)
        # Chemist notation: (pq|rs) = Integral phi_p*(1) phi_q(1) 1/r12 phi_r*(2) phi_s(2)
        # Our ERI code computes (mu nu | lam sig) = (1 1 | 2 2)
        # So MO_ERI is (pq|rs) = [p(1) q(1) | r(2) s(2)]
        
        # MP2 Formula:
        # E_MP2 = Sum_i<j Sum_a<b ...
        # Standard Spin-Orbital Formula: E = 1/4 Sum_ijab |<ij||ab>|^2 / (e_i + e_j - e_a - e_b)
        # For Restricted Closed Shell (RHF-MP2):
        # E_MP2 = Sum_i_occ Sum_j_occ Sum_a_virt Sum_b_virt [ t_ij^ab ( 2 t_ij^ab - t_ij^ba ) ] / D_ijab
        # where t_ij^ab = (ia|jb)
        # D_ijab = eps_i + eps_j - eps_a - eps_b
        
        # Indices:
        # i, j: Occupied (0 to n_occ-1)
        # a, b: Virtual (n_occ to N-1)
        
        mp2_energy = 0.0
        
        occ_range = range(n_occ)
        virt_range = range(n_occ, N)
        
        for i in occ_range:
            for j in occ_range:
                for a in virt_range:
                    for b in virt_range:
                        # (ia|jb)
                        # MO_ERI indices are [p, q, r, s] corresponding to [1, 1, 2, 2]
                        # So (ia|jb) = MO_ERI[i, a, j, b]
                        t_iajb = MO_ERI[i, a, j, b]
                        t_ibja = MO_ERI[i, b, j, a] # (ib|ja)
                        
                        num = t_iajb * (2.0 * t_iajb - t_ibja)
                        denom = eps[i] + eps[j] - eps[a] - eps[b]
                        
                        if abs(denom) > 1e-12:
                            mp2_energy += num / denom
        
        print(f"MP2 Correlation:   {mp2_energy:.6f}")
        print(f"Total MP2 Energy:  {total_energy + mp2_energy:.6f}")

        return total_energy, total_energy + mp2_energy
