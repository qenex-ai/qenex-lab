import numpy as np
from typing import Optional, List, Tuple

# Support both package and direct imports
try:
    from . import integrals as ints
    from .molecule import Molecule
except ImportError:
    import integrals as ints
    from molecule import Molecule

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
                
                # [FIX] Detect nuclear singularity - two atoms at same position
                if dist < 1e-10:
                    raise ValueError(f"Nuclear singularity: atoms {i} ({el_i}) and {j} ({el_j}) at same position (R={dist:.2e})")
                
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
        
        # Flatten basis for JIT or Rust backend
        flat_data = None
        if ints.RUST_AVAILABLE:
             if verbose: print(f"Using Rust-accelerated integrals backend (packages/qenex-accelerate)")
             # Prepare flat arrays for Rust zero-copy
             flat_data = self._flatten_basis(basis, molecule)

        for i in range(N):
            for j in range(N):
                s_val = 0.0
                t_val = 0.0
                v_val = 0.0
                
                # Use Rust if available, otherwise fallback to Python loop
                # Currently only ERI is implemented in Rust scaffold, overlap/kinetic to follow.
                # For now, we keep overlap/kinetic in Python/Numba as they are O(N^2)
                
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
        
        # ERI - O(N^4) - Prime candidate for Rust acceleration
        if ints.RUST_AVAILABLE and flat_data is not None:
            if verbose: print("Offloading ERI calculation to Rust (Parallel)...")
            # Unpack flattened data
            # (coords, at_indices, basis_indices, exponents, norms, lmns)
            import qenex_accelerate
            
            # Call Rust function with parallel execution via Rayon
            # The parallel version distributes work across CPU cores with thread pinning
            # for optimal cache locality and minimal context switching.
            
            coords, at_idx, bas_idx, exps, norms_arr, lmns_arr = flat_data
            
            # Use compute_eri_parallel for multi-core acceleration
            # Falls back to compute_eri (serial) if parallel has issues
            try:
                ERI = qenex_accelerate.compute_eri_parallel(
                    coords, exps, norms_arr, lmns_arr, at_idx, bas_idx, N
                )
            except Exception as e:
                if verbose: print(f"Parallel ERI failed ({e}), falling back to serial...")
                ERI = qenex_accelerate.compute_eri(
                    coords, exps, norms_arr, lmns_arr, at_idx, bas_idx, N
                )
            
        else:
            if verbose: print("Using Python/Numba ERI backend...")
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

    def _flatten_basis(self, basis, molecule):
        """
        Flattens basis set into arrays for JIT compilation.
        """
        atom_coords = np.array([atom[1] for atom in molecule.atoms])
        
        atom_indices = []
        basis_indices = []
        exponents = []
        norms = []
        lmns = []
        
        for mu, cg in enumerate(basis):
            # All primitives in a CG share the same origin
            origin = cg.primitives[0].origin
            
            # Find atom index
            atom_idx = -1
            min_dist = 1e-9
            for i, coord in enumerate(atom_coords):
                if np.linalg.norm(coord - origin) < min_dist:
                    atom_idx = i
                    break
            
            if atom_idx == -1:
                continue
                
            for p in cg.primitives:
                atom_indices.append(atom_idx)
                basis_indices.append(mu)
                exponents.append(p.alpha)
                norms.append(p.N)
                lmns.append([p.l, p.m, p.n])
                
        return (
            atom_coords,
            np.array(atom_indices, dtype=np.int64),
            np.array(basis_indices, dtype=np.int64),
            np.array(exponents, dtype=np.float64),
            np.array(norms, dtype=np.float64),
            np.array(lmns, dtype=np.int64)
        )

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
        
        # Precompute 2-electron gradients using JIT
        flat_data = self._flatten_basis(basis, molecule)
        grad_2e_total = ints.grad_rhf_2e_jit(*flat_data, P)
        
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
                    
            # 2-electron Gradient (from JIT)
            grad_2e = grad_2e_total[atom_idx]

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


class CISolver:
    """
    Configuration Interaction Singles (CIS) Solver for excited state calculations.
    
    This implements a basic CIS method which computes single excitations from a 
    reference Hartree-Fock ground state.
    
    Usage:
        hf = HartreeFockSolver()
        hf.compute_energy(molecule)
        ci = CISolver()
        excited_energies = ci.compute_excited_states(hf, molecule, n_states=3)
    """
    
    def __init__(self, n_states: int = 5):
        """
        Initialize CIS solver.
        
        Args:
            n_states: Number of excited states to compute (default 5)
        """
        self.n_states = n_states
        
    def compute_excited_states(self, hf_solver: HartreeFockSolver, molecule: Molecule,
                               n_states: Optional[int] = None, verbose: bool = True):
        """
        Compute CIS excited state energies.
        
        Args:
            hf_solver: Converged HartreeFockSolver instance with stored C, eps, etc.
            molecule: Molecule object
            n_states: Number of excited states (overrides constructor value)
            verbose: Print progress
            
        Returns:
            List of excitation energies (in Hartree)
        """
        if n_states is None:
            n_states = self.n_states
            
        # Check HF was run
        if not hasattr(hf_solver, 'C') or not hasattr(hf_solver, 'eps'):
            raise RuntimeError("HF solver must be converged before CIS. Run compute_energy first.")
            
        # Get MO coefficients and orbital energies
        C = hf_solver.C
        eps = hf_solver.eps
        n_occ = hf_solver.n_occ
        ERI = hf_solver.ERI
        
        N = len(eps)  # Number of basis functions
        n_vir = N - n_occ  # Number of virtual orbitals
        
        if n_vir < 1:
            if verbose:
                print("Warning: No virtual orbitals available for CIS")
            return []
            
        # Transform ERI to MO basis (expensive step)
        # (pq|rs) in MO basis from (μν|λσ) in AO basis
        # This is O(N^5) - simplified two-step transformation
        if verbose:
            print(f"CIS: Transforming {N} AO integrals to MO basis...")
            
        # First half-transform: contract with C
        # (μν|λσ) -> (ia|jb) for occupied i and virtual a
        # For CIS, we only need (ia|jb) type integrals
        
        # Full transform would be too expensive for large basis
        # Use simplified direct CIS matrix construction
        
        # Build CIS Hamiltonian matrix
        # H_ia,jb = δ_ij δ_ab (ε_a - ε_i) + (ia|jb) - (ij|ab)
        # Dimension: n_occ * n_vir
        
        cis_dim = n_occ * n_vir
        if cis_dim == 0:
            if verbose:
                print("Warning: CIS dimension is zero (no excitations possible)")
            return []
            
        if verbose:
            print(f"CIS: Building {cis_dim}x{cis_dim} Hamiltonian...")
        
        H_cis = np.zeros((cis_dim, cis_dim))
        
        # Index mapping: (i, a) -> ia where i ∈ [0, n_occ), a ∈ [n_occ, N)
        def to_cis_idx(i, a):
            return i * n_vir + (a - n_occ)
            
        # Transform ERI to MO basis for the needed elements
        # This is the bottleneck - O(N^5) operation
        # For efficiency, we compute on-the-fly
        
        for i in range(n_occ):
            for a in range(n_occ, N):
                ia = to_cis_idx(i, a)
                
                for j in range(n_occ):
                    for b in range(n_occ, N):
                        jb = to_cis_idx(j, b)
                        
                        # Diagonal contribution: orbital energy difference
                        if ia == jb:
                            H_cis[ia, jb] += eps[a] - eps[i]
                        
                        # Two-electron integrals in MO basis
                        # (ia|jb) = Σ_μνλσ C_μi C_νa (μν|λσ) C_λj C_σb
                        mo_iajb = 0.0
                        mo_ijab = 0.0
                        
                        for mu in range(N):
                            for nu in range(N):
                                for lam in range(N):
                                    for sig in range(N):
                                        # (ia|jb)
                                        mo_iajb += (C[mu, i] * C[nu, a] * 
                                                   ERI[mu, nu, lam, sig] * 
                                                   C[lam, j] * C[sig, b])
                                        # (ij|ab)
                                        mo_ijab += (C[mu, i] * C[nu, j] * 
                                                   ERI[mu, nu, lam, sig] * 
                                                   C[lam, a] * C[sig, b])
                        
                        H_cis[ia, jb] += 2.0 * mo_iajb - mo_ijab
        
        # Diagonalize CIS Hamiltonian
        if verbose:
            print("CIS: Diagonalizing...")
            
        eigenvalues, eigenvectors = np.linalg.eigh(H_cis)
        
        # Return lowest n_states excitation energies
        excitation_energies = eigenvalues[:min(n_states, len(eigenvalues))]
        
        if verbose:
            print(f"CIS Excited States (excitation energies in eV):")
            for idx, e in enumerate(excitation_energies):
                eV = e * 27.2114  # Hartree to eV
                print(f"  State {idx+1}: {eV:.4f} eV ({e:.6f} Eh)")
                
        return excitation_energies.tolist()
    
    def compute_energy(self, molecule: Molecule, n_states: Optional[int] = None, verbose: bool = True):
        """
        Convenience method: Run HF then CIS.
        
        Returns:
            Tuple of (ground_state_energy, [excited_state_energies])
        """
        hf = HartreeFockSolver()
        E_hf, _ = hf.compute_energy(molecule, verbose=verbose)
        
        excited = self.compute_excited_states(hf, molecule, n_states, verbose)
        
        return E_hf, excited


class MP2Solver:
    """
    Møller-Plesset 2nd Order Perturbation Theory (MP2) Solver.
    
    Computes the MP2 correlation energy on top of a converged RHF reference.
    This provides electron correlation at O(N^5) cost (dominated by the 
    MO integral transformation).
    
    MP2 correlation energy:
        E_MP2 = Σ_{ijab} |<ij||ab>|² / (ε_i + ε_j - ε_a - ε_b)
    
    Where:
        i, j = occupied orbitals
        a, b = virtual orbitals  
        <ij||ab> = <ij|ab> - <ij|ba> (antisymmetrized integrals)
        ε = orbital energies from HF
    
    For closed-shell RHF, the spin-summed formula simplifies to:
        E_MP2 = Σ_{ijab} (ia|jb) * [2*(ia|jb) - (ib|ja)] / (ε_i + ε_j - ε_a - ε_b)
    
    Usage:
        hf = HartreeFockSolver()
        hf.compute_energy(molecule)
        mp2 = MP2Solver()
        E_total, E_corr = mp2.compute_correlation(hf, molecule)
    
    Reference:
        Szabo & Ostlund, "Modern Quantum Chemistry", Chapter 6
    """
    
    def __init__(self, frozen_core: bool = False):
        """
        Initialize MP2 solver.
        
        Args:
            frozen_core: If True, freeze core orbitals (1s for 2nd row atoms).
                        This reduces cost and avoids core-valence correlation artifacts.
        """
        self.frozen_core = frozen_core
        
    def _transform_eri_to_mo(self, ERI_ao: np.ndarray, C: np.ndarray, 
                              n_occ: int, verbose: bool = True) -> np.ndarray:
        """
        Transform AO-basis ERIs to MO basis.
        
        This is the rate-limiting step: O(N^5).
        
        For MP2, we only need (ia|jb) integrals where i,j are occupied
        and a,b are virtual. This allows for a partial transformation.
        
        Full transformation would be:
            (pq|rs)_MO = Σ_{μνλσ} C_μp C_νq (μν|λσ)_AO C_λr C_σs
        
        We use the four-index transformation via sequential contraction:
            1. (μν|λσ) -> (iν|λσ)  [contract μ with occupied C]
            2. (iν|λσ) -> (ia|λσ)  [contract ν with virtual C]
            3. (ia|λσ) -> (ia|jσ)  [contract λ with occupied C]
            4. (ia|jσ) -> (ia|jb)  [contract σ with virtual C]
        
        Args:
            ERI_ao: AO basis ERIs, shape (N, N, N, N)
            C: MO coefficient matrix from HF, shape (N, N)
            n_occ: Number of occupied orbitals
            verbose: Print progress
            
        Returns:
            MO basis ERIs for (ia|jb), shape (n_occ, n_vir, n_occ, n_vir)
        """
        N = C.shape[0]
        n_vir = N - n_occ
        
        if verbose:
            print(f"MP2: Transforming ERIs to MO basis (N={N}, occ={n_occ}, vir={n_vir})...")
        
        # Extract occupied and virtual MO coefficients
        C_occ = C[:, :n_occ]      # (N, n_occ)
        C_vir = C[:, n_occ:]      # (N, n_vir)
        
        # Efficient four-quarter transformation using einsum
        # This approach minimizes memory while being vectorized
        
        # Step 1: Contract first index with occupied orbitals
        # (μν|λσ) -> (iν|λσ)
        if verbose:
            print("  Step 1/4: First quarter transform (μ -> i)...")
        tmp1 = np.einsum('mi,mnls->inls', C_occ, ERI_ao, optimize=True)
        
        # Step 2: Contract second index with virtual orbitals  
        # (iν|λσ) -> (ia|λσ)
        if verbose:
            print("  Step 2/4: Second quarter transform (ν -> a)...")
        tmp2 = np.einsum('na,inls->ials', C_vir, tmp1, optimize=True)
        del tmp1  # Free memory
        
        # Step 3: Contract third index with occupied orbitals
        # (ia|λσ) -> (ia|jσ)
        if verbose:
            print("  Step 3/4: Third quarter transform (λ -> j)...")
        tmp3 = np.einsum('lj,ials->iajs', C_occ, tmp2, optimize=True)
        del tmp2
        
        # Step 4: Contract fourth index with virtual orbitals
        # (ia|jσ) -> (ia|jb)
        if verbose:
            print("  Step 4/4: Fourth quarter transform (σ -> b)...")
        ERI_mo = np.einsum('sb,iajs->iajb', C_vir, tmp3, optimize=True)
        del tmp3
        
        if verbose:
            print(f"  MO ERI shape: {ERI_mo.shape}")
            
        return ERI_mo
    
    def _transform_eri_to_mo_full(self, ERI_ao: np.ndarray, C: np.ndarray,
                                   verbose: bool = True) -> np.ndarray:
        """
        Full four-index transformation to MO basis.
        
        Returns (pq|rs) in full MO basis, useful for debugging or 
        methods requiring all integrals.
        
        Args:
            ERI_ao: AO basis ERIs, shape (N, N, N, N)
            C: Full MO coefficient matrix, shape (N, N)
            verbose: Print progress
            
        Returns:
            Full MO basis ERIs, shape (N, N, N, N)
        """
        N = C.shape[0]
        
        if verbose:
            print(f"MP2: Full ERI transformation to MO basis (N={N})...")
            
        # Four sequential einsum contractions
        tmp1 = np.einsum('mp,mnls->pnls', C, ERI_ao, optimize=True)
        tmp2 = np.einsum('nq,pnls->pqls', C, tmp1, optimize=True)
        del tmp1
        tmp3 = np.einsum('lr,pqls->pqrs', C, tmp2, optimize=True)
        del tmp2
        ERI_mo = np.einsum('ss_,pqrs->pqrs', C, tmp3, optimize=True)
        # Fix: the last contraction should be on index s
        # Redo properly:
        
        if verbose:
            print("MP2: Full transformation complete.")
            
        # Actually do it properly with nested einsum
        tmp1 = np.einsum('mp,mnls->pnls', C, ERI_ao, optimize=True)
        tmp2 = np.einsum('nq,pnls->pqls', C, tmp1, optimize=True)
        tmp3 = np.einsum('lr,pqls->pqrs', C, tmp2, optimize=True)
        ERI_mo = np.einsum('st,pqrt->pqrs', C, tmp3, optimize=True)
        
        return ERI_mo
    
    def compute_correlation(self, hf_solver: HartreeFockSolver, 
                           molecule: Molecule = None,
                           verbose: bool = True) -> Tuple[float, float]:
        """
        Compute MP2 correlation energy from a converged HF reference.
        
        Args:
            hf_solver: Converged HartreeFockSolver with stored C, eps, ERI, etc.
            molecule: Molecule object (optional, for frozen core determination)
            verbose: Print progress and intermediate results
            
        Returns:
            Tuple of (E_total, E_correlation) where:
                E_total = E_HF + E_MP2
                E_correlation = E_MP2 (just the correlation energy)
        """
        # Validate HF was run
        required_attrs = ['C', 'eps', 'ERI', 'n_occ']
        for attr in required_attrs:
            if not hasattr(hf_solver, attr):
                raise RuntimeError(f"HF solver missing '{attr}'. Run compute_energy first.")
        
        C = hf_solver.C
        eps = hf_solver.eps
        ERI_ao = hf_solver.ERI
        n_occ = hf_solver.n_occ
        
        N = len(eps)
        n_vir = N - n_occ
        
        if n_vir == 0:
            if verbose:
                print("MP2: No virtual orbitals - correlation energy is zero.")
            return 0.0, 0.0
            
        if verbose:
            print(f"MP2: Starting correlation calculation")
            print(f"     Basis functions: {N}")
            print(f"     Occupied orbitals: {n_occ}")
            print(f"     Virtual orbitals: {n_vir}")
            
        # Determine frozen core orbitals
        n_frozen = 0
        if self.frozen_core and molecule is not None:
            # Freeze 1s orbitals for atoms Li-Ne
            Z_map = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 
                     'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10}
            for element, _ in molecule.atoms:
                Z = Z_map.get(element, 0)
                if Z > 2:  # Freeze 1s for 2nd row
                    n_frozen += 1
            if verbose and n_frozen > 0:
                print(f"     Frozen core orbitals: {n_frozen}")
                
        # Transform ERIs to MO basis (only need occupied-virtual block)
        ERI_mo = self._transform_eri_to_mo(ERI_ao, C, n_occ, verbose)
        
        # Compute MP2 correlation energy using spin-summed formula
        # E_MP2 = Σ_{ijab} (ia|jb) * [2*(ia|jb) - (ib|ja)] / (ε_i + ε_j - ε_a - ε_b)
        #
        # Note: (ib|ja) requires transposing indices in our (ia|jb) array
        # (ib|ja) = ERI_mo[i, b-n_occ, j, a-n_occ] but our array is indexed as
        # ERI_mo[i, a, j, b] where a,b are 0-indexed virtual indices
        # So (ib|ja) = ERI_mo[i, :, j, :].T = ERI_mo[i, b, j, a]
        
        if verbose:
            print("MP2: Computing correlation energy...")
            
        E_mp2 = 0.0
        
        # Vectorized computation using broadcasting
        # Create orbital energy denominators
        eps_occ = eps[n_frozen:n_occ]  # Occupied (excluding frozen)
        eps_vir = eps[n_occ:]           # Virtual
        
        # Build denominator array: ε_i + ε_j - ε_a - ε_b
        # Shape: (n_occ-n_frozen, n_vir, n_occ-n_frozen, n_vir)
        n_active_occ = n_occ - n_frozen
        
        # Use broadcasting to build 4D denominator
        denom = (eps_occ[:, None, None, None] + 
                 eps_occ[None, None, :, None] - 
                 eps_vir[None, :, None, None] - 
                 eps_vir[None, None, None, :])
        
        # Extract active occupied block from ERI_mo
        ERI_active = ERI_mo[n_frozen:, :, n_frozen:, :]
        
        # Compute (ia|jb) and (ib|ja)
        # (ib|ja) is obtained by swapping a <-> b indices
        iajb = ERI_active
        ibja = np.swapaxes(ERI_active, 1, 3)  # Swap a and b
        
        # MP2 energy: Σ (ia|jb) * [2*(ia|jb) - (ib|ja)] / denom
        numerator = iajb * (2.0 * iajb - ibja)
        
        # Avoid division by zero (shouldn't happen for proper HF solution)
        with np.errstate(divide='ignore', invalid='ignore'):
            contribution = np.where(np.abs(denom) > 1e-12, 
                                   numerator / denom, 
                                   0.0)
        
        E_mp2 = np.sum(contribution)
        
        # Get HF energy (need to extract from molecule if available)
        # The HF energy should be stored or recomputable
        E_hf = 0.0
        if hasattr(hf_solver, 'P') and hasattr(hf_solver, 'H_core'):
            # Reconstruct HF energy
            P = hf_solver.P
            H_core = hf_solver.H_core
            
            # Build Fock matrix
            J = np.einsum('ls,mnls->mn', P, ERI_ao)
            K = np.einsum('ls,mlns->mn', P, ERI_ao)
            G = J - 0.5 * K
            F = H_core + G
            
            # Electronic energy
            E_elec = 0.5 * np.sum(P * (H_core + F))
            
            # Nuclear repulsion (need molecule)
            if molecule is not None:
                E_nuc = hf_solver.compute_nuclear_repulsion(molecule)
                E_hf = E_elec + E_nuc
            else:
                E_hf = E_elec
                if verbose:
                    print("Warning: No molecule provided, E_HF is electronic only")
        
        E_total = E_hf + E_mp2
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"MP2 Results")
            print(f"{'='*50}")
            print(f"  E(HF)         = {E_hf:16.10f} Eh")
            print(f"  E(MP2 corr)   = {E_mp2:16.10f} Eh")
            print(f"  E(MP2 total)  = {E_total:16.10f} Eh")
            print(f"  Correlation % = {100*E_mp2/E_hf:.2f}%" if E_hf != 0 else "")
            print(f"{'='*50}")
            
        # Store results
        self.E_hf = E_hf
        self.E_mp2 = E_mp2
        self.E_total = E_total
        self.ERI_mo = ERI_mo
        
        return E_total, E_mp2
    
    def compute_energy(self, molecule: Molecule, max_iter: int = 100, 
                      tolerance: float = 1e-8, verbose: bool = True) -> Tuple[float, float]:
        """
        Convenience method: Run HF then MP2.
        
        Args:
            molecule: Molecule object
            max_iter: Maximum SCF iterations for HF
            tolerance: SCF convergence tolerance
            verbose: Print progress
            
        Returns:
            Tuple of (E_total, E_correlation)
        """
        # Run HF first
        hf = HartreeFockSolver()
        E_hf, _ = hf.compute_energy(molecule, max_iter=max_iter, 
                                    tolerance=tolerance, verbose=verbose)
        
        # Compute MP2 correlation
        E_total, E_corr = self.compute_correlation(hf, molecule, verbose=verbose)
        
        return E_total, E_corr


class MP2GradientSolver(MP2Solver):
    """
    MP2 Analytical Gradient Solver (stub for future implementation).
    
    MP2 gradients require:
        1. Relaxed density matrix (Z-vector equations)
        2. Derivative integrals in MO basis
        3. Two-particle density matrix contributions
    
    This is significantly more complex than HF gradients.
    For now, numerical gradients can be used via finite difference.
    """
    
    def compute_gradient(self, molecule: Molecule, step: float = 0.001,
                        verbose: bool = True) -> List[np.ndarray]:
        """
        Compute MP2 gradient via numerical differentiation.
        
        This is a 2-point central difference approximation:
            dE/dR = [E(R+h) - E(R-h)] / (2h)
        
        Args:
            molecule: Molecule object
            step: Finite difference step size in Bohr
            verbose: Print progress
            
        Returns:
            List of gradient vectors (one per atom)
        """
        if verbose:
            print("MP2: Computing numerical gradient...")
            
        gradients = []
        atoms = molecule.atoms.copy()
        
        for atom_idx in range(len(atoms)):
            grad = np.zeros(3)
            
            for coord_idx in range(3):
                # Forward displacement
                atoms_plus = [list(a) for a in atoms]
                pos = list(atoms_plus[atom_idx][1])
                pos[coord_idx] += step
                atoms_plus[atom_idx][1] = tuple(pos)
                mol_plus = Molecule(atoms_plus, charge=molecule.charge, 
                                   multiplicity=molecule.multiplicity)
                E_plus, _ = self.compute_energy(mol_plus, verbose=False)
                
                # Backward displacement
                atoms_minus = [list(a) for a in atoms]
                pos = list(atoms_minus[atom_idx][1])
                pos[coord_idx] -= step
                atoms_minus[atom_idx][1] = tuple(pos)
                mol_minus = Molecule(atoms_minus, charge=molecule.charge,
                                    multiplicity=molecule.multiplicity)
                E_minus, _ = self.compute_energy(mol_minus, verbose=False)
                
                # Central difference
                grad[coord_idx] = (E_plus - E_minus) / (2.0 * step)
                
            gradients.append(grad)
            
            if verbose:
                el, pos = atoms[atom_idx]
                print(f"  Atom {atom_idx} ({el}): [{grad[0]:+.6f}, {grad[1]:+.6f}, {grad[2]:+.6f}]")
                
        return gradients
