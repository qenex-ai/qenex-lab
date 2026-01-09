import numpy as np
from molecule import Molecule
import integrals as ints

class DIIS:
    """
    Direct Inversion in the Iterative Subspace (DIIS) accelerator.
    """
    def __init__(self, max_history=6):
        self.error_vectors = []
        self.fock_matrices = []
        self.max_history = max_history

    def update(self, F, D, S):
        # Error vector e = FDS - SDF
        error = F @ D @ S - S @ D @ F
        
        # Orthogonalize error vector (optional but good practice, here we just use raw)
        # Flatten for storage
        self.error_vectors.append(error)
        self.fock_matrices.append(F)
        
        if len(self.error_vectors) > self.max_history:
            self.error_vectors.pop(0)
            self.fock_matrices.pop(0)

    def extrapolate(self):
        n = len(self.error_vectors)
        if n < 2:
            return None
            
        # Build B matrix
        B = np.zeros((n + 1, n + 1))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0
        
        for i in range(n):
            for j in range(n):
                # Dot product of error vectors
                # e_i . e_j
                val = np.sum(self.error_vectors[i] * self.error_vectors[j])
                B[i, j] = val
                
        # RHS vector
        rhs = np.zeros(n + 1)
        rhs[-1] = -1
        
        try:
            coeffs = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            return None
            
        # Linear combination of Fock matrices
        F_new = np.zeros_like(self.fock_matrices[0])
        for i in range(n):
            F_new += coeffs[i] * self.fock_matrices[i]
            
        return F_new

class HartreeFockSolver:
    def build_basis(self, molecule: Molecule):
        # Delegate to integrals module to build basis
        return ints.build_basis(molecule)

    def compute_nuclear_repulsion(self, molecule: Molecule):
        energy = 0.0
        atoms = molecule.atoms
        Z_map = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10}
        
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                el_i, pos_i = atoms[i]
                el_j, pos_j = atoms[j]
                
                Zi = Z_map.get(el_i, 0)
                Zj = Z_map.get(el_j, 0)
                
                dist = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                if dist > 1e-12:
                    energy += (Zi * Zj) / dist
        return energy

    def compute_energy(self, molecule: Molecule, max_iter=100, tolerance=1e-8, verbose=True):
        """
        Restricted Hartree-Fock (RHF) for closed-shell systems.
        """
        atoms = molecule.atoms
        basis = self.build_basis(molecule)
        N = len(basis)
        self.basis = basis # Store for gradient usage if needed
        
        if N == 0: return 0.0, 0.0
        
        Z_map = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10}
        total_electrons = sum(Z_map.get(at[0], 0) for at in atoms) - molecule.charge
        
        if total_electrons % 2 != 0:
            raise ValueError(f"RHF requires even number of electrons (found {total_electrons}). Use UHF.")

        # Integrals
        S = np.zeros((N, N))
        T = np.zeros((N, N))
        V = np.zeros((N, N))
        ERI = np.zeros((N, N, N, N))
        
        atom_qs = [(Z_map.get(el, 0), np.array(pos)) for el, pos in atoms]
        
        if verbose: print("Computing integrals...")
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
        
        # ERI
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
                        
        # Orthogonalization matrix X = S^-1/2
        evals, evecs = np.linalg.eigh(S)
        inv_sqrt_evals = np.array([1.0/np.sqrt(e) if e > 1e-6 else 0.0 for e in evals])
        X = evecs @ np.diag(inv_sqrt_evals) @ evecs.T
        
        # Initial Guess
        F_0_prime = X.T @ H_core @ X
        eps_0, C_0_prime = np.linalg.eigh(F_0_prime)
        C = X @ C_0_prime
        
        P = np.zeros((N, N))
        n_occ = total_electrons // 2
        
        for mu in range(N):
            for nu in range(N):
                for a in range(n_occ):
                    P[mu, nu] += 2.0 * C[mu, a] * C[nu, a]
                    
        old_energy = 0.0
        diis = DIIS()
        
        # Initialize variables
        curr_E = 0.0
        eps = np.zeros(N)
        
        # Initialize variables to avoid unbound errors if loop doesn't run or logic fails
        curr_E = 0.0
        eps = np.zeros(N)
        
        if verbose: print("Starting SCF Loop...")
        
        for iteration in range(max_iter):
            # G = J - 0.5 K
            # J_uv = Sum_ls P_ls (uv|ls)
            # K_uv = Sum_ls P_ls (ul|vs)
            
            G = np.zeros((N, N))
            
            # Vectorized construction of G
            # J part: (mn|ls) * P_ls -> contract on last two indices
            J = np.einsum('ls,mnls->mn', P, ERI)
            
            # K part: (ml|ns) * P_ls -> we need to transpose ERI to align
            # ERI is (mu, nu, lam, sig). We want (mu, lam, nu, sig)
            K = np.einsum('ls,mlns->mn', P, ERI)
            
            G = J - 0.5 * K
            
            F = H_core + G
            
            # DIIS
            diis.update(F, P, S)
            if iteration >= 2:
                F_diis = diis.extrapolate()
                if F_diis is not None:
                    F = F_diis
            
            # Diagonalize
            F_prime = X.T @ F @ X
            eps, C_prime = np.linalg.eigh(F_prime)
            C = X @ C_prime
            
            # New Density
            P_new = np.zeros((N, N))
            for mu in range(N):
                for nu in range(N):
                    for a in range(n_occ):
                        P_new[mu, nu] += 2.0 * C[mu, a] * C[nu, a]
                        
            # Damping
            if iteration < 5:
                P = 0.5 * P + 0.5 * P_new
            else:
                P = P_new
                
            # Energy
            e_elec = 0.5 * np.sum(P * (H_core + F))
            curr_E = e_elec
            
            diff = abs(curr_E - old_energy)
            if verbose: print(f"Iter {iteration}: E = {curr_E:.8f} (diff {diff:.2e})")
            
            if diff < tolerance:
                if verbose: print("Converged.")
                break
                
            old_energy = curr_E
            
        nuc_rep = self.compute_nuclear_repulsion(molecule)
        total_E = curr_E + nuc_rep
        
        # Store state
        self.P = P
        self.C = C
        self.eps = eps
        self.n_occ = n_occ
        self.ERI = ERI # Store for gradient usage
        self.H_core = H_core # Store for gradient
        
        return total_E, total_E

    def compute_gradient(self, molecule: Molecule):
        """
        RHF Analytical Gradient.
        """
        if not hasattr(self, 'P'):
            self.compute_energy(molecule, verbose=False)
            
        P = self.P
        basis = self.basis
        N = len(basis)
        
        # Energy Weighted Density W for RHF
        # W_uv = 2 * Sum_i^occ eps_i * C_ui * C_vi
        W = np.zeros((N, N))
        for mu in range(N):
            for nu in range(N):
                for i in range(self.n_occ):
                    W[mu, nu] += 2.0 * self.eps[i] * self.C[mu, i] * self.C[nu, i]
                    
        gradients = []
        Z_map = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10}
        
        for atom_idx, (element, coord) in enumerate(molecule.atoms):
            grad_E = np.zeros(3)
            
            # 1. Nuclear Repulsion Gradient
            for j, (el_j, pos_j) in enumerate(molecule.atoms):
                if atom_idx == j: continue
                Zi = Z_map.get(element, 0)
                Zj = Z_map.get(el_j, 0)
                diff = np.array(coord) - np.array(pos_j)
                dist = np.linalg.norm(diff)
                if dist > 1e-12:
                    grad_E += - (Zi * Zj * diff) / (dist**3)
            
            grad_1e = np.zeros(3)
            grad_2e = np.zeros(3)
            grad_S = np.zeros(3)
            
            # Pre-fetch derivatives to save loops? No, memory expensive. Loop and compute.
            for mu in range(N):
                for nu in range(N):
                    # Overlap Derivative contribution: -Tr(W dS)
                    dS = np.zeros(3)
                    for p_mu in basis[mu].primitives:
                        for p_nu in basis[nu].primitives:
                            dS += ints.overlap_deriv(p_mu, p_nu, atom_idx, molecule.atoms)
                    grad_S += W[mu, nu] * dS
                    
                    # 1-electron Derivative: Tr(P dH) = Tr(P (dT + dV))
                    dT = np.zeros(3)
                    dV = np.zeros(3)
                    for p_mu in basis[mu].primitives:
                        for p_nu in basis[nu].primitives:
                            dT += ints.kinetic_deriv(p_mu, p_nu, atom_idx, molecule.atoms)
                            dV += ints.nuclear_attraction_deriv(p_mu, p_nu, atom_idx, molecule.atoms)
                    
                    grad_1e += P[mu, nu] * (dT + dV)
                    
                    # 2-electron Derivative
                    # 0.5 * Sum_lam_sig P_uv P_ls * [ (uv|ls)' - 0.5 (ul|vs)' ]
                    for lam in range(N):
                        for sig in range(N):
                            d_eri = np.zeros(3)
                            d_eri_ex = np.zeros(3)
                            
                            for p_mu in basis[mu].primitives:
                                for p_nu in basis[nu].primitives:
                                    for p_lam in basis[lam].primitives:
                                        for p_sig in basis[sig].primitives:
                                            # (mu nu | lam sig)
                                            d_eri += ints.eri_deriv(p_mu, p_nu, p_lam, p_sig, atom_idx, molecule.atoms)
                                            # (mu lam | nu sig)
                                            d_eri_ex += ints.eri_deriv(p_mu, p_lam, p_nu, p_sig, atom_idx, molecule.atoms)
                            
                            # Coulomb: P_uv * P_ls * (uv|ls)'
                            # Factor 0.5 comes from 0.5 * Tr(P G). 
                            # G = J - 0.5 K.
                            # dE = 0.5 * Sum(P_uv * dJ_uv) - 0.25 * Sum(P_uv * dK_uv) ... slightly complex.
                            # Standard RHF deriv:
                            # 0.5 * Sum_uvls P_uv P_ls [ (uv|ls)' - 0.5 (ul|vs)' ]
                            
                            term = P[mu, nu] * P[lam, sig] * (d_eri - 0.5 * d_eri_ex)
                            grad_2e += 0.5 * term

            total_grad = grad_E + grad_1e + grad_2e - grad_S
            gradients.append(total_grad)
            
        return gradients

class UHFSolver(HartreeFockSolver):
    """
    Unrestricted Hartree-Fock (UHF) Solver for open-shell systems.
    """
    def compute_energy(self, molecule: Molecule, max_iter=100, tolerance=1e-8, verbose=True):
        atoms = molecule.atoms
        basis = self.build_basis(molecule)
        N = len(basis)
        self.basis = basis # Store for gradient
        
        if N == 0: return 0.0, 0.0

        if verbose:
            print(f"UHF Basis functions: {N}")

        # Integrals (Same as RHF)
        # We can reuse the parent class method if refactored, but here we just recompute or copy logic.
        # To avoid code duplication in a real refactor we'd split integrals out. 
        # For now, inline computation for speed of implementation.
        
        S = np.zeros((N, N))
        T = np.zeros((N, N))
        V = np.zeros((N, N))
        ERI = np.zeros((N, N, N, N))
        
        Z_map = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10}
        atom_qs = [(Z_map.get(el, 0), np.array(pos)) for el, pos in atoms]

        if verbose: print("UHF: Computing integrals...")
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
        
        # ERI
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

        # Orthogonalization
        evals, evecs = np.linalg.eigh(S)
        inv_sqrt_evals = np.array([1.0/np.sqrt(e) if e > 1e-6 else 0.0 for e in evals])
        X = evecs @ np.diag(inv_sqrt_evals) @ evecs.T
        
        # Initial Guess (Core Hamiltonian)
        F_0_prime = X.T @ H_core @ X
        eps_0, C_0_prime = np.linalg.eigh(F_0_prime)
        C_0 = X @ C_0_prime
        
        # Determine N_alpha, N_beta
        total_electrons = sum(Z_map.get(at[0], 0) for at in atoms) - molecule.charge
        multiplicity = molecule.multiplicity
        
        # N_alpha - N_beta = multiplicity - 1
        # N_alpha + N_beta = total
        n_alpha = (total_electrons + multiplicity - 1) // 2
        n_beta = total_electrons - n_alpha
        
        if verbose:
            print(f"UHF: Electrons={total_electrons}, Multiplicity={multiplicity}")
            print(f"     N_alpha={n_alpha}, N_beta={n_beta}")
        
        # Break symmetry for initial guess
        C_alpha = C_0.copy()
        C_beta = C_0.copy()
        
        # Mix HOMO/LUMO for beta to ensure different spatial orbitals if possible
        if n_beta > 0 and n_beta < N:
             homo = n_beta - 1
             lumo = n_beta
             angle = np.pi / 4.0
             cb_homo = np.cos(angle)*C_beta[:, homo] + np.sin(angle)*C_beta[:, lumo]
             cb_lumo = -np.sin(angle)*C_beta[:, homo] + np.cos(angle)*C_beta[:, lumo]
             C_beta[:, homo] = cb_homo
             C_beta[:, lumo] = cb_lumo
        
        # Build initial densities
        P_alpha = np.zeros((N, N))
        P_beta = np.zeros((N, N))
        
        for mu in range(N):
            for nu in range(N):
                for a in range(n_alpha):
                    P_alpha[mu, nu] += C_alpha[mu, a] * C_alpha[nu, a]
                for a in range(n_beta):
                    P_beta[mu, nu] += C_beta[mu, a] * C_beta[nu, a]
                    
        P_total = P_alpha + P_beta
        old_energy = 0.0
        
        # Initialize variables to avoid unbound errors
        current_E = 0.0
        eps_a = np.zeros(N)
        eps_b = np.zeros(N)
        
        # DIIS
        diis_alpha = DIIS()
        diis_beta = DIIS()
        
        if verbose: print("Starting UHF SCF Loop...")
        
        for iteration in range(max_iter):
            # J_total
            J = np.einsum('ls,mnls->mn', P_total, ERI)
            
            # K matrices
            K_alpha = np.einsum('ls,mlns->mn', P_alpha, ERI)
            K_beta = np.einsum('ls,mlns->mn', P_beta, ERI)
            
            F_alpha = H_core + J - K_alpha
            F_beta = H_core + J - K_beta
            
            # DIIS
            diis_alpha.update(F_alpha, P_alpha, S)
            diis_beta.update(F_beta, P_beta, S)
            
            F_a_use = F_alpha
            F_b_use = F_beta
            
            # Apply DIIS after a few iterations
            if iteration >= 4:
                F_a_diis = diis_alpha.extrapolate()
                F_b_diis = diis_beta.extrapolate()
                if F_a_diis is not None: F_a_use = F_a_diis
                if F_b_diis is not None: F_b_use = F_b_diis
            
            # Diagonalize
            Fa_prime = X.T @ F_a_use @ X
            eps_a, Ca_prime = np.linalg.eigh(Fa_prime)
            C_alpha = X @ Ca_prime
            
            Fb_prime = X.T @ F_b_use @ X
            eps_b, Cb_prime = np.linalg.eigh(Fb_prime)
            C_beta = X @ Cb_prime
            
            # New Densities
            P_alpha_new = np.zeros((N, N))
            P_beta_new = np.zeros((N, N))
            
            for mu in range(N):
                for nu in range(N):
                    for a in range(n_alpha):
                        P_alpha_new[mu, nu] += C_alpha[mu, a] * C_alpha[nu, a]
                    for a in range(n_beta):
                        P_beta_new[mu, nu] += C_beta[mu, a] * C_beta[nu, a]
                        
            # Damping
            if iteration < 8:
                damp = 0.5
                P_alpha = damp*P_alpha + (1-damp)*P_alpha_new
                P_beta = damp*P_beta + (1-damp)*P_beta_new
            else:
                P_alpha = P_alpha_new
                P_beta = P_beta_new
            
            P_total = P_alpha + P_beta
            
            # Energy Calculation
            # E = 0.5 * Tr[ P_t H + P_a F_a + P_b F_b ]
            e_core = 0.5 * np.sum(P_total * H_core)
            e_a = 0.5 * np.sum(P_alpha * F_alpha) # Use variational F, not DIIS
            e_b = 0.5 * np.sum(P_beta * F_beta)
            current_E = e_core + e_a + e_b
            
            diff = abs(current_E - old_energy)
            if verbose: print(f"Iter {iteration}: E = {current_E:.8f} (diff {diff:.2e})")
            
            if diff < tolerance and iteration > 1:
                if verbose: print("UHF Converged.")
                break
                
            old_energy = current_E
            
        # Store State
        self.P = P_total 
        self.P_alpha = P_alpha
        self.P_beta = P_beta
        self.C_alpha = C_alpha
        self.C_beta = C_beta
        self.eps_alpha = eps_a
        self.eps_beta = eps_b
        
        # Spin Contamination <S^2>
        # <S^2> = s(s+1) + N_beta - Sum_ij |<psi_i^alpha | psi_j^beta>|^2
        s_z = (n_alpha - n_beta) / 2.0
        exact_s2 = s_z * (s_z + 1.0)
        
        S_ab_MO = C_alpha.T @ S @ C_beta
        overlap_sum = 0.0
        for i in range(n_alpha):
            for j in range(n_beta):
                overlap_sum += S_ab_MO[i, j]**2
                
        calc_s2 = exact_s2 + n_beta - overlap_sum
        if verbose: print(f"<S^2>: {calc_s2:.4f} (Exact: {exact_s2:.4f})")
        
        nuc_rep = self.compute_nuclear_repulsion(molecule)
        total_E = current_E + nuc_rep
        if verbose: print(f"Total UHF Energy: {total_E:.6f}")
        
        return total_E, total_E

    def compute_gradient(self, molecule: Molecule):
        """
        UHF Analytical Gradient.
        """
        if not hasattr(self, 'P_alpha'):
            self.compute_energy(molecule, verbose=False)
            
        P_t = self.P_alpha + self.P_beta
        P_a = self.P_alpha
        P_b = self.P_beta
        basis = self.basis
        
        # Energy Weighted Density W
        # W_total = W_alpha + W_beta
        Z_map = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10}
        total_elec = sum(Z_map.get(at[0], 0) for at in molecule.atoms) - molecule.charge
        mult = molecule.multiplicity
        n_a = (total_elec + mult - 1) // 2
        n_b = total_elec - n_a
        
        W = np.zeros_like(P_t)
        
        for mu in range(len(basis)):
            for nu in range(len(basis)):
                for i in range(n_a):
                    W[mu, nu] += 2.0 * self.eps_alpha[i] * self.C_alpha[mu, i] * self.C_alpha[nu, i]
                # Wait, factor of 2? No. UHF occupations are 1.0.
                # RHF: 2 * eps * C * C. UHF: 1 * eps * C * C.
                # Let's correct RHF W matrix too if I copied it wrong... 
                # RHF code had: W += 2.0 * eps * C * C. Correct for closed shell.
                
        # Reset W for UHF (Occupancy is 1)
        W = np.zeros_like(P_t)
        for mu in range(len(basis)):
            for nu in range(len(basis)):
                for i in range(n_a):
                    W[mu, nu] += self.eps_alpha[i] * self.C_alpha[mu, i] * self.C_alpha[nu, i]
                for i in range(n_b):
                    W[mu, nu] += self.eps_beta[i] * self.C_beta[mu, i] * self.C_beta[nu, i]
                    
        gradients = []
        N = len(basis)
        
        for atom_idx, (element, coord) in enumerate(molecule.atoms):
            grad_E = np.zeros(3)
            
            # 1. Nuclear
            for j, (el_j, pos_j) in enumerate(molecule.atoms):
                if atom_idx == j: continue
                Zi = Z_map.get(element, 0)
                Zj = Z_map.get(el_j, 0)
                diff = np.array(coord) - np.array(pos_j)
                dist = np.linalg.norm(diff)
                if dist > 1e-12:
                    grad_E += - (Zi * Zj * diff) / (dist**3)
            
            grad_1e = np.zeros(3)
            grad_2e = np.zeros(3)
            grad_S = np.zeros(3)
            
            for mu in range(N):
                for nu in range(N):
                    # Overlap
                    dS = np.zeros(3)
                    for p_mu in basis[mu].primitives:
                        for p_nu in basis[nu].primitives:
                            dS += ints.overlap_deriv(p_mu, p_nu, atom_idx, molecule.atoms)
                    grad_S += W[mu, nu] * dS
                    
                    # 1e
                    dT = np.zeros(3)
                    dV = np.zeros(3)
                    for p_mu in basis[mu].primitives:
                        for p_nu in basis[nu].primitives:
                            dT += ints.kinetic_deriv(p_mu, p_nu, atom_idx, molecule.atoms)
                            dV += ints.nuclear_attraction_deriv(p_mu, p_nu, atom_idx, molecule.atoms)
                    
                    grad_1e += P_t[mu, nu] * (dT + dV)
                    
                    # 2e
                    for lam in range(N):
                        for sig in range(N):
                            d_eri = np.zeros(3)
                            d_eri_ex = np.zeros(3)
                            
                            for p_mu in basis[mu].primitives:
                                for p_nu in basis[nu].primitives:
                                    for p_lam in basis[lam].primitives:
                                        for p_sig in basis[sig].primitives:
                                            d_eri += ints.eri_deriv(p_mu, p_nu, p_lam, p_sig, atom_idx, molecule.atoms)
                                            d_eri_ex += ints.eri_deriv(p_mu, p_lam, p_nu, p_sig, atom_idx, molecule.atoms)
                            
                            # Coulomb: 0.5 * P_t * P_t * d(ii|jj)
                            grad_2e += 0.5 * P_t[mu, nu] * P_t[lam, sig] * d_eri
                            
                            # Exchange
                            grad_2e -= 0.5 * P_a[mu, nu] * P_a[lam, sig] * d_eri_ex
                            grad_2e -= 0.5 * P_b[mu, nu] * P_b[lam, sig] * d_eri_ex
                            
            total_grad = grad_E + grad_1e + grad_2e - grad_S
            gradients.append(total_grad)
            
        return gradients
