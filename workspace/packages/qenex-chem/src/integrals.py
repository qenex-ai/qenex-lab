"""
Integrals Module
Provides analytical integrals for Gaussian primitives and basis set definitions (STO-3G).
Supports s and p orbitals via Obara-Saika recurrence relations.
"""

import numpy as np
from scipy.special import erf

# ==========================================
# STO-3G Basis Set Definitions
# ==========================================

class STO3G:
    """
    Standard STO-3G parameters.
    Exponents (alpha) and Contraction Coefficients (d).
    """
    # Normalized 1s shell (alpha, d)
    basis_1s = [
        (0.109818, 0.444635),
        (0.405771, 0.535328),
        (2.227660, 0.154329)
    ]
    
    # Normalized 2s shell (alpha, d)
    basis_2s = [
        (0.0751386, 0.700115),
        (0.231031, 0.399513),
        (0.994203, -0.0999672)
    ]

    # Normalized 2p shell (alpha, d)
    basis_2p = [
        (0.0751386, 0.391957),
        (0.231031, 0.607684),
        (0.994203, 0.155916)
    ]

    # Standard molecular exponents (zeta)
    # Values roughly correspond to Slater rules or standard STO-3G sets
    zeta = {
        'H': 1.24,
        'He': 1.69,
        'Li': 1.03,  # Li 1s=2.69? Usually 2s is valence. Treating 1s as core with different zeta.
                     # For simplicity in this solver, we might just focus on valence or use standard molecular sets.
                     # Below are standard STO-3G scale factors:
        'Be': 1.325, 
        'B': 1.45,
        'C': 1.72,
        'N': 1.95,
        'O': 2.25,
        'F': 2.55,
        
        # Core 1s zetas for 2nd row (approximate)
        'Li_1s': 2.69,
        'Be_1s': 3.68,
        'B_1s': 4.68,
        'C_1s': 5.67,
        'N_1s': 6.67,
        'O_1s': 7.66,
        'F_1s': 8.65
    }

# ==========================================
# 6-31G Basis Set Definitions
# ==========================================

class BasisSet631G:
    """
    Standard 6-31G parameters.
    Source: Standard literature values (e.g. from EMSL Basis Set Exchange).
    Structure:
      H: 4s -> [2s] (Split Valence: 3 inner, 1 outer)
      C-F: 6-31G (Core 1s: 6 prims; Valence 2sp: 3 inner, 1 outer)
      P-S: 66-31G approx (Core 1s,2s,2p; Valence 3sp)
    """
    
    # Hydrogen (4s contracted to 2s)
    # Inner 1s (3 prims)
    H_inner = [
        (18.7311370, 0.03349460),
        (2.8253937, 0.23472695),
        (0.6401217, 0.81375733)
    ]
    # Outer 1s (1 prim)
    H_outer = [
        (0.1612778, 1.0000000)
    ]

    # Helium (He) - Standard 6-31G
    He_inner = [
        (38.4216340, 0.03349460),
        (5.7760130, 0.23472695),
        (1.2931780, 0.81375733)
    ]
    He_outer = [
        (0.3258400, 1.0000000)
    ]

    # Carbon (C)
    # Core 1s (6 prims)
    C_core_1s = [
        (3047.5249000, 0.0018347),
        (457.3695100, 0.0139948),
        (103.9486900, 0.0685866),
        (29.2101550, 0.2322409),
        (9.2866630, 0.4690699),
        (3.1639270, 0.3604552)
    ]
    # Valence Inner 2sp (3 prims) - Shared exponents
    C_val_inner = {
        'exponents': [7.8682724, 1.8812885, 0.5442493],
        's_coeffs': [-0.1193324, -0.1608542, 1.1434564],
        'p_coeffs': [0.0689991, 0.3164240, 0.7443083]
    }
    # Valence Outer 2sp (1 prim)
    C_val_outer = {
        'exponents': [0.1687144],
        's_coeffs': [1.0000000],
        'p_coeffs': [1.0000000]
    }

    # Nitrogen (N)
    N_core_1s = [
        (4173.5110000, 0.0018347),
        (627.4579000, 0.0139948),
        (142.9021000, 0.0685866),
        (40.2343300, 0.2322409),
        (12.8202100, 0.4690699),
        (4.3904370, 0.3604552)
    ]
    N_val_inner = {
        'exponents': [11.6263580, 2.7162800, 0.7722180],
        's_coeffs': [-0.1149612, -0.1593168, 1.1354180],
        'p_coeffs': [0.0675797, 0.3239072, 0.7408951]
    }
    N_val_outer = {
        'exponents': [0.2120313],
        's_coeffs': [1.0000000],
        'p_coeffs': [1.0000000]
    }

    # Oxygen (O)
    O_core_1s = [
        (5484.6717000, 0.0018311),
        (825.2349500, 0.0139501),
        (188.0469600, 0.0684451),
        (52.9645000, 0.2327143),
        (16.8975700, 0.4701930),
        (5.7996353, 0.3585209)
    ]
    O_val_inner = {
        'exponents': [15.5396160, 3.5999336, 1.0137618],
        's_coeffs': [-0.1107775, -0.1480263, 1.1307670],
        'p_coeffs': [0.0708743, 0.3397528, 0.7271586]
    }
    O_val_outer = {
        'exponents': [0.2700058],
        's_coeffs': [1.0000000],
        'p_coeffs': [1.0000000]
    }
    
    # Phosphorus (P)
    # Core 1s (6 prims)
    P_core_1s = [
        (19410.0000, 0.001851),
        (2921.0000, 0.014166),
        (665.4000, 0.069772),
        (187.6000, 0.240706),
        (60.1500, 0.485141),
        (20.9100, 0.331456)
    ]
    # Core 2sp (L-shell) - Often treated as 2s and 2p separately or sp in some sets.
    # Standard 6-31G for 3rd row often uses separate s and p for core or 2sp.
    # For P, we use 6 primitives for 2s and 2p (separate or shared). 
    # Here we use shared exponents for 2s/2p core block (common in Pople sets).
    P_core_2sp = {
        'exponents': [423.5000, 95.8400, 30.5600, 11.0800, 4.2980, 1.6950],
        's_coeffs': [-0.007626, -0.075421, -0.235697, 0.098319, 0.697666, 0.422501],
        'p_coeffs': [0.011883, 0.078440, 0.250550, 0.490807, 0.375258, 0.026417]
    }
    # Valence 3sp (Split 3, 1)
    P_val_inner = {
        'exponents': [2.8640, 1.0880, 0.4354],
        's_coeffs': [-0.276495, 0.230728, 0.890696],
        'p_coeffs': [-0.092176, 0.400537, 0.672670]
    }
    P_val_outer = {
        'exponents': [0.1364],
        's_coeffs': [1.0000],
        'p_coeffs': [1.0000]
    }

    # Sulfur (S)
    S_core_1s = [
        (22660.0000, 0.001835),
        (3404.0000, 0.014023),
        (774.2000, 0.069002),
        (218.0000, 0.238125),
        (69.8900, 0.482062),
        (24.3100, 0.334066)
    ]
    S_core_2sp = {
        'exponents': [507.0000, 115.6000, 37.1600, 13.5600, 5.2820, 2.0830],
        's_coeffs': [-0.007505, -0.073420, -0.228518, 0.088192, 0.686522, 0.433431],
        'p_coeffs': [0.011707, 0.076840, 0.244365, 0.485122, 0.385472, 0.035985]
    }
    S_val_inner = {
        'exponents': [3.6190, 1.3530, 0.5332],
        's_coeffs': [-0.301323, 0.270966, 0.865487],
        'p_coeffs': [-0.088737, 0.408078, 0.669830]
    }
    S_val_outer = {
        'exponents': [0.1654],
        's_coeffs': [1.0000],
        'p_coeffs': [1.0000]
    }

# ==========================================
# Math Helpers
# ==========================================

def factorial2(n):
    """Double factorial (n!!)"""
    if n <= 0: return 1
    result = 1
    for i in range(n, 0, -2):
        result *= i
    return result

def normalize_primitive(alpha, lmn):
    """
    Normalization constant for a primitive gaussian x^l y^m z^n e^(-alpha*r^2)
    """
    l, m, n = lmn
    # Formula: (2*alpha/pi)^0.75 * (4*alpha)^(L/2) / sqrt((2l-1)!! (2m-1)!! (2n-1)!!)
    L = l + m + n
    pre = (2.0 * alpha / np.pi)**0.75
    ang = (4.0 * alpha)**(L / 2.0)
    denom = np.sqrt(factorial2(2*l-1) * factorial2(2*m-1) * factorial2(2*n-1))
    return pre * ang / denom

def boys(n, t):
    """
    Boys function F_n(t) = integral_0^1 u^(2n) exp(-t u^2) du
    """
    if t < 1e-8:
        # Taylor expansion for small t: 1/(2n+1) - t/(2n+3)
        return 1.0 / (2*n + 1) - t / (2*n + 3)
    else:
        # Recursive approximation or use gamma functions.
        # For simplicity and robustness, we use the error function relation for F0
        # and downward recursion for higher orders?
        # Actually, for small N (up to 4-5), we can use a direct implementation 
        # based on incomplete gamma function if available, but scipy.special.erf is all we have.
        
        # F0(t) = 0.5 * sqrt(pi/t) * erf(sqrt(t))
        val_0 = 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))
        if n == 0:
            return val_0
            
        # Upward recursion is unstable?
        # Standard approach: Compute F_max and recurse down.
        # F_n(t) = (2t F_{n+1}(t) + exp(-t)) / (2n+1) -> F_{n+1} = ...
        # Downward: F_{n}(t) = ( (2n+3)F_{n+1}(t) + exp(-t) ) / (2t) ? No.
        # F_m(t) = (2t F_{m+1} + exp(-t))/(2m+1)
        # So F_{m+1} is needed for F_m? That's upward relation for F_m from F_{m+1}.
        # Correct relation: F_m(t) = [exp(-t) + 2t F_{m+1}(t)] / (2m+1) is NOT correct.
        # Correct is: (2m+1)F_m(t) - 2t F_{m+1}(t) = exp(-t)
        # So F_{m+1} = [ (2m+1)F_m - exp(-t) ] / (2t) -> This is UNSTABLE for small t.
        
        # We need downward recursion. Compute F_max then go down.
        # How to compute F_max directly? Incomplete Gamma.
        # For this simplified solver, we will use a naive precise implementation for specific N 
        # or just simple numerical integration (slow but safe) or math trick.
        
        # Since we only need up to L=4 (pp|pp), let's implement specific cases or numerical integration.
        # Using numerical integration (Gauss-Chebyshev or similar) is robust.
        # Or simple:
        from scipy.special import gammainc, gamma
        # F_n(t) = 0.5 * t^(-n-0.5) * gamma(n+0.5) * gammainc(n+0.5, t) ??
        # gammainc in scipy is normalized by gamma(a). So it returns P(a, x).
        # Gamma_inc_unnormalized(a, x) = gammainc(a,x) * gamma(a)
        # F_n(t) = 0.5 * t^(-n - 0.5) * gamma(n + 0.5) * gammainc(n + 0.5, t)
        
        return 0.5 * (t ** (-n - 0.5)) * gamma(n + 0.5) * gammainc(n + 0.5, t)

# ==========================================
# Primitive Gaussian Class
# ==========================================

class BasisFunction:
    def __init__(self, origin, alpha, coeff, lmn=(0,0,0)):
        self.origin = np.array(origin)
        self.alpha = alpha
        self.coeff = coeff
        self.l, self.m, self.n = lmn
        self.L = self.l + self.m + self.n
        self.norm = normalize_primitive(alpha, lmn)
        self.N = self.coeff * self.norm

# ==========================================
# Obara-Saika Recurrence Engine
# ==========================================

def gaussian_product_center(alpha1, A, alpha2, B):
    return (alpha1 * A + alpha2 * B) / (alpha1 + alpha2)

def overlap_1d(l1, l2, xA, xB, alpha, beta):
    """
    1D Overlap Integral between (x-xA)^l1 and (x-xB)^l2
    """
    p = alpha + beta
    rp = (alpha * xA + beta * xB) / p
    dist = xA - xB
    
    # Base case: S_00
    # The prefactor exp(-alpha*beta/p * dist^2) is handled in the 3D wrapper usually, 
    # but for 1D recurrence we keep the structure simple.
    # Standard Obara Saika for Overlap:
    # S_ij = (Pi - Ai) S_{i-1,j} + 1/2p ( (i-1) S_{i-2,j} + j S_{i-1,j-1} )
    # Let's compute a table of S integrals up to l1, l2.
    
    # We need to compute up to S[l1][l2]
    # S[i][j]
    S = np.zeros((l1 + 1, l2 + 1))
    
    # S_00 for 1D (without the exponential part, which we multiply at the end)
    # 1D integral of exp(-alpha(x-A)^2 -beta(x-B)^2) = sqrt(pi/p) * exp(...)
    # We include sqrt(pi/p) here.
    S[0, 0] = np.sqrt(np.pi / p)
    
    # Fill S table
    # We use recurrence:
    # (x-A) G_a G_b = (P-A) G_a G_b + 1/2p d/dPA ... 
    # S_{i+1, j} = (XP - XA) S_{i,j} + 1/(2p) * ( i*S_{i-1,j} + j*S_{i,j-1} )
    
    XP = rp
    XA = xA
    XB = xB
    
    for i in range(l1 + 1):
        for j in range(l2 + 1):
            if i == 0 and j == 0:
                continue
                
            # Compute S[i,j]
            # Prefer incrementing i (recurrence on a)
            if i > 0:
                term1 = (XP - XA) * S[i-1, j]
                term2 = (i-1) * S[i-2, j] if i > 1 else 0.0
                term3 = j * S[i-1, j-1] if j > 0 else 0.0
                S[i, j] = term1 + (1.0 / (2*p)) * (term2 + term3)
            else:
                # Increment j (recurrence on b)
                # S_{i, j+1} = (XP - XB) S_{i, j} + 1/2p ( i S_{i-1, j} + j S_{i, j-1} )
                term1 = (XP - XB) * S[i, j-1]
                term2 = i * S[i-1, j-1] if i > 0 else 0.0
                term3 = (j-1) * S[i, j-2] if j > 1 else 0.0
                S[i, j] = term1 + (1.0 / (2*p)) * (term2 + term3)
                
    return S[l1, l2]

def overlap(a, b):
    """
    3D Overlap Integral (a|b)
    """
    alpha = a.alpha
    beta = b.alpha
    p = alpha + beta
    
    # Exponential prefactor
    diff = a.origin - b.origin
    dist2 = np.dot(diff, diff)
    pre = np.exp(- (alpha * beta / p) * dist2)
    
    sx = overlap_1d(a.l, b.l, a.origin[0], b.origin[0], alpha, beta)
    sy = overlap_1d(a.m, b.m, a.origin[1], b.origin[1], alpha, beta)
    sz = overlap_1d(a.n, b.n, a.origin[2], b.origin[2], alpha, beta)
    
    return a.N * b.N * pre * sx * sy * sz

def kinetic(a, b):
    """
    Kinetic Integral (a|T|b)
    Using the identity T = -0.5 * Laplacian
    (a | -0.5 nabla^2 | b)
    We iterate over x,y,z components.
    T_x = (a | -0.5 d^2/dx^2 | b)
    
    Identity: -0.5 d^2/dx^2 g_b = -0.5 [ 4b^2 x_b^2 - 2b(2l+1) + ... complex recurrence ]
    Easier: Use relation between T and Overlap derivatives.
    T = beta * (2(l+m+n) + 3) * S(a,b) - 2*beta^2 * ( S(a, b+2x) + S(a, b+2y) + S(a, b+2z) )
    Wait, that's for s-orbitals.
    
    General formula for kinetic term on component x with l2:
    T_1d = -0.5 * [ -2*beta*(2*l2+1)*S_{l1,l2} + 4*beta^2*S_{l1, l2+2} + l2(l2-1)*S_{l1, l2-2} ] 
    Wait, derivative of Gaussian x^l exp(-bx^2):
    d/dx = l x^{l-1} exp - 2b x^{l+1} exp
    d^2/dx^2 = l(l-1)x^{l-2} - 2b(2l+1)x^l + 4b^2 x^{l+2}
    
    So <a| -0.5 d2/dx2 |b> = -0.5 * [ l2(l2-1)<a|b_{l2-2}> - 2beta(2l2+1)<a|b_{l2}> + 4beta^2 <a|b_{l2+2}> ]
    (Need to handle l2 < 2 cases where term is 0)
    """
    alpha = a.alpha
    beta = b.alpha
    p = alpha + beta
    
    # Precompute 1D overlaps needed
    # We need overlaps for b's angular momentum varying: l2, l2-2, l2+2
    
    def get_term_1d(l1, l2, xA, xB):
        # Base overlap <l1|l2>
        s0 = overlap_1d(l1, l2, xA, xB, alpha, beta)
        
        # Term 1: l2(l2-1) <l1|l2-2>
        t1 = 0.0
        if l2 >= 2:
            s_minus = overlap_1d(l1, l2-2, xA, xB, alpha, beta)
            t1 = l2 * (l2 - 1) * s_minus
            
        # Term 2: -2*beta*(2*l2+1) <l1|l2>
        t2 = -2.0 * beta * (2.0 * l2 + 1.0) * s0
        
        # Term 3: 4*beta^2 <l1|l2+2>
        s_plus = overlap_1d(l1, l2+2, xA, xB, alpha, beta)
        t3 = 4.0 * beta**2 * s_plus
        
        return -0.5 * (t1 + t2 + t3)

    # Calculate prefactor
    diff = a.origin - b.origin
    dist2 = np.dot(diff, diff)
    pre = np.exp(- (alpha * beta / p) * dist2)
    
    # 1D Overlaps for mixing
    Sx = overlap_1d(a.l, b.l, a.origin[0], b.origin[0], alpha, beta)
    Sy = overlap_1d(a.m, b.m, a.origin[1], b.origin[1], alpha, beta)
    Sz = overlap_1d(a.n, b.n, a.origin[2], b.origin[2], alpha, beta)
    
    Tx = get_term_1d(a.l, b.l, a.origin[0], b.origin[0])
    Ty = get_term_1d(a.m, b.m, a.origin[1], b.origin[1])
    Tz = get_term_1d(a.n, b.n, a.origin[2], b.origin[2])
    
    # Total T = Tx*Sy*Sz + Sx*Ty*Sz + Sx*Sy*Tz
    val = Tx*Sy*Sz + Sx*Ty*Sz + Sx*Sy*Tz
    
    return a.N * b.N * pre * val

def nuclear_attraction(a, b, C, Z):
    """
    Nuclear Attraction Integral (a | -Z/r_C | b)
    Using Obara-Saika recurrence for V
    """
    alpha = a.alpha
    beta = b.alpha
    p = alpha + beta
    
    Rp = gaussian_product_center(alpha, a.origin, beta, b.origin)
    diff = a.origin - b.origin
    dist2 = np.dot(diff, diff)
    kab = np.exp(- (alpha * beta / p) * dist2)
    
    # Target center relative coords
    RPC = Rp - C
    
    # We need auxiliary integrals Theta_N = F_N(p * R_PC^2)
    # Total angular momentum L = la + lb + ma + mb + na + nb
    L_total = a.L + b.L
    
    # Compute Boys functions up to L_total
    T_val = p * np.dot(RPC, RPC)
    boys_vals = [boys(n, T_val) for n in range(L_total + 1)]
    
    # OS Recurrence for V: V[i, j, k] corresponds to integral with momentum indices
    # We implement a generalized recursion.
    # V_{n_x, n_y, n_z}^{(m)}
    # Start with s-s integral V_0^(m) = 2*pi/p * K_ab * F_m(T)
    
    # We need to build up to (la, ma, na) and (lb, mb, nb)
    # It's easier to handle each cartesian direction independently? No, they are coupled by F_m.
    # Actually, they are not coupled if C is the expansion center? No.
    # OS V relation couples axes? No, standard OS decouples if we use auxiliary index m.
    # (a+1_i | V | b )^m = (Pi - Ai) (a|b)^m - (Pi - Ci) (a|b)^{m+1} + ...
    
    # We will compute 3D integrals iteratively.
    # Since we only need one target integral, we can memoize or use dynamic programming.
    # Key: (la, lb, ma, mb, na, nb, m)
    
    memo = {}

    def get_v(la, lb, ma, mb, na, nb, m):
        key = (la, lb, ma, mb, na, nb, m)
        if key in memo:
            return memo[key]
        
        # Base case
        if la == 0 and lb == 0 and ma == 0 and mb == 0 and na == 0 and nb == 0:
            return (2.0 * np.pi / p) * kab * boys_vals[m]
        
        # Recurrence
        # Reduce highest angular momentum
        if la > 0:
            # (a+1_x | ... )
            # Decrement la -> la-1
            # Using i=0 (x)
            term1 = (Rp[0] - a.origin[0]) * get_v(la-1, lb, ma, mb, na, nb, m)
            term2 = (Rp[0] - C[0]) * get_v(la-1, lb, ma, mb, na, nb, m+1)
            term3 = (la-1)/(2*p) * ( get_v(la-2, lb, ma, mb, na, nb, m) - get_v(la-2, lb, ma, mb, na, nb, m+1) ) if la > 1 else 0.0
            term4 = lb/(2*p) * ( get_v(la-1, lb-1, ma, mb, na, nb, m) - get_v(la-1, lb-1, ma, mb, na, nb, m+1) ) if lb > 0 else 0.0
            
            res = term1 - term2 + term3 + term4
            memo[key] = res
            return res
            
        if lb > 0:
            # ( ... | b+1_x )
            # Decrement lb -> lb-1
            term1 = (Rp[0] - b.origin[0]) * get_v(la, lb-1, ma, mb, na, nb, m)
            term2 = (Rp[0] - C[0]) * get_v(la, lb-1, ma, mb, na, nb, m+1)
            term3 = la/(2*p) * ( get_v(la-1, lb-1, ma, mb, na, nb, m) - get_v(la-1, lb-1, ma, mb, na, nb, m+1) ) if la > 0 else 0.0
            term4 = (lb-1)/(2*p) * ( get_v(la, lb-2, ma, mb, na, nb, m) - get_v(la, lb-2, ma, mb, na, nb, m+1) ) if lb > 1 else 0.0
            
            res = term1 - term2 + term3 + term4
            memo[key] = res
            return res

        # If x is done, move to y (ma, mb)
        if ma > 0:
            term1 = (Rp[1] - a.origin[1]) * get_v(la, lb, ma-1, mb, na, nb, m)
            term2 = (Rp[1] - C[1]) * get_v(la, lb, ma-1, mb, na, nb, m+1)
            term3 = (ma-1)/(2*p) * ( get_v(la, lb, ma-2, mb, na, nb, m) - get_v(la, lb, ma-2, mb, na, nb, m+1) ) if ma > 1 else 0.0
            term4 = mb/(2*p) * ( get_v(la, lb, ma-1, mb-1, na, nb, m) - get_v(la, lb, ma-1, mb-1, na, nb, m+1) ) if mb > 0 else 0.0
            res = term1 - term2 + term3 + term4
            memo[key] = res
            return res

        if mb > 0:
            term1 = (Rp[1] - b.origin[1]) * get_v(la, lb, ma, mb-1, na, nb, m)
            term2 = (Rp[1] - C[1]) * get_v(la, lb, ma, mb-1, na, nb, m+1)
            term3 = ma/(2*p) * ( get_v(la, lb, ma-1, mb-1, na, nb, m) - get_v(la, lb, ma-1, mb-1, na, nb, m+1) ) if ma > 0 else 0.0
            term4 = (mb-1)/(2*p) * ( get_v(la, lb, ma, mb-2, na, nb, m) - get_v(la, lb, ma, mb-2, na, nb, m+1) ) if mb > 1 else 0.0
            res = term1 - term2 + term3 + term4
            memo[key] = res
            return res
            
        # If y is done, move to z (na, nb)
        if na > 0:
            term1 = (Rp[2] - a.origin[2]) * get_v(la, lb, ma, mb, na-1, nb, m)
            term2 = (Rp[2] - C[2]) * get_v(la, lb, ma, mb, na-1, nb, m+1)
            term3 = (na-1)/(2*p) * ( get_v(la, lb, ma, mb, na-2, nb, m) - get_v(la, lb, ma, mb, na-2, nb, m+1) ) if na > 1 else 0.0
            term4 = nb/(2*p) * ( get_v(la, lb, ma, mb, na-1, nb-1, m) - get_v(la, lb, ma, mb, na-1, nb-1, m+1) ) if nb > 0 else 0.0
            res = term1 - term2 + term3 + term4
            memo[key] = res
            return res

        if nb > 0:
            term1 = (Rp[2] - b.origin[2]) * get_v(la, lb, ma, mb, na, nb-1, m)
            term2 = (Rp[2] - C[2]) * get_v(la, lb, ma, mb, na, nb-1, m+1)
            term3 = na/(2*p) * ( get_v(la, lb, ma, mb, na-1, nb-1, m) - get_v(la, lb, ma, mb, na-1, nb-1, m+1) ) if na > 0 else 0.0
            term4 = (nb-1)/(2*p) * ( get_v(la, lb, ma, mb, na, nb-2, m) - get_v(la, lb, ma, mb, na, nb-2, m+1) ) if nb > 1 else 0.0
            res = term1 - term2 + term3 + term4
            memo[key] = res
            return res
            
        return 0.0 # Should not reach

    val = get_v(a.l, b.l, a.m, b.m, a.n, b.n, 0)
    return -Z * a.N * b.N * val

def eri(a, b, c, d):
    """
    Two-Electron Repulsion Integral (ab|cd)
    Using Obara-Saika Recurrence
    """
    alpha = a.alpha
    beta = b.alpha
    gamma = c.alpha
    delta = d.alpha
    
    p = alpha + beta
    q = gamma + delta
    alpha_Q = p * q / (p + q)
    
    Rp = gaussian_product_center(alpha, a.origin, beta, b.origin)
    Rq = gaussian_product_center(gamma, c.origin, delta, d.origin)
    Rpq = Rp - Rq
    
    diff_ab = a.origin - b.origin
    diff_cd = c.origin - d.origin
    
    kab = np.exp(- (alpha * beta / p) * np.dot(diff_ab, diff_ab))
    kcd = np.exp(- (gamma * delta / q) * np.dot(diff_cd, diff_cd))
    prefactor = (2 * np.pi**2.5) / (p * q * np.sqrt(p + q)) * kab * kcd
    
    # Auxiliaries
    L_total = a.L + b.L + c.L + d.L
    T_val = alpha_Q * np.dot(Rpq, Rpq)
    boys_vals = [boys(n, T_val) for n in range(L_total + 1)]
    
    memo = {}

    def get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m):
        # 13 indices key
        key = (la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m)
        if key in memo: return memo[key]
        
        rho = alpha_Q

        # Base Case
        if sum(key[:-1]) == 0:
            return prefactor * boys_vals[m]
            
        # Recurrence
        # Handle la > 0
        if la > 0:
            # (a+1_i | ... )
            # Terms:
            # (Pi - Ai) (a|...)^m
            # (Wi - Pi) (a|...)^{m+1}
            # 1/2p (a-1 | ...)^m - alpha_q/p * 1/2p (a-1 | ...)^{m+1} ?? Check coefficients
            # Standard OS:
            # (a+1,b|cd)^m = (Pi - Ai)(ab|cd)^m + (Wi - Pi)(ab|cd)^{m+1}
            #              + 1/2p * [ ai( (a-1,b|cd)^m - rho/p (a-1,b|cd)^{m+1} )
            #                       + bi( (a,b-1|cd)^m - rho/p (a,b-1|cd)^{m+1} )
            #                       + 0.5/q * ( ci ( ... ) + di (...) )? No. 
            #                       + rho/q * ( ci(ab|c-1 d)^{m+1} + di(ab|c d-1)^{m+1} ) ]
            # rho = p*q/(p+q)
            
            rho = alpha_Q
            # P is Rp, W is Rpq weighted? No W is (pRp + qRq)/(p+q).
            # W - P = (pRp + qRq)/(p+q) - Rp = q(Rq - Rp)/(p+q) = - (q / (p+q)) * Rpq
            
            Wi_minus_Pi_x = - (q / (p + q)) * (Rp[0] - Rq[0])
            Pi_minus_Ai_x = (Rp[0] - a.origin[0])
            
            term1 = Pi_minus_Ai_x * get_eri(la-1, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m)
            term2 = Wi_minus_Pi_x * get_eri(la-1, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
            
            term3 = 0.0
            if la > 1:
                val_minus = get_eri(la-2, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m)
                val_minus_m1 = get_eri(la-2, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                term3 = (la-1)/(2*p) * (val_minus - (rho/p)*val_minus_m1)
                
            term4 = 0.0
            if lb > 0:
                val_minus = get_eri(la-1, lb-1, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m)
                val_minus_m1 = get_eri(la-1, lb-1, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                term4 = lb/(2*p) * (val_minus - (rho/p)*val_minus_m1)
            
            term5 = 0.0
            if lc > 0:
                # 0.5/ (p+q) * c ... = rho / (p*q) * ...
                val_c = get_eri(la-1, lb, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                term5 = lc / (2*(p+q)) * val_c 
            
            term6 = 0.0
            if ld > 0:
                val_d = get_eri(la-1, lb, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1)
                term6 = ld / (2*(p+q)) * val_d
                
            res = term1 + term2 + term3 + term4 + term5 + term6
            memo[key] = res
            return res
            
        # If la done, lb
        if lb > 0:
             # (ab+1|cd)^m = (b+1 a | cd)^m
             # Just swap a and b logic?
             # (Pi - Bi) + ...
             # Simpler: Use transfer equation (a b+1 | ...) = (a+1 b | ...) + (Bi - Ai)(ab|...)
             # Or just implement recursion directly for lb.
             
             # Let's reduce lb
             Pi_minus_Bi_x = (Rp[0] - b.origin[0])
             Wi_minus_Pi_x = - (q / (p + q)) * (Rp[0] - Rq[0])
             
             term1 = Pi_minus_Bi_x * get_eri(la, lb-1, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m)
             term2 = Wi_minus_Pi_x * get_eri(la, lb-1, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
             
             term3 = 0.0
             if la > 0:
                 val = get_eri(la-1, lb-1, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m)
                 val_m = get_eri(la-1, lb-1, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                 term3 = la/(2*p) * (val - (rho/p)*val_m)
                 
             term4 = 0.0
             if lb > 1:
                 val = get_eri(la, lb-2, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m)
                 val_m = get_eri(la, lb-2, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                 term4 = (lb-1)/(2*p) * (val - (rho/p)*val_m)
                 
             term5 = 0.0 # c term
             if lc > 0:
                 val = get_eri(la, lb-1, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                 term5 = lc / (2*(p+q)) * val
                 
             term6 = 0.0 # d term
             if ld > 0:
                 val = get_eri(la, lb-1, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1)
                 term6 = ld / (2*(p+q)) * val
             
             res = term1 + term2 + term3 + term4 + term5 + term6
             memo[key] = res
             return res
             
        # If a,b done (s-type on left), recurse on c,d?
        # (ss|c+1 d)
        # We can swap centers (ab|cd) = (cd|ab)
        if lc > 0 or ld > 0 or mc > 0 or md > 0 or nc > 0 or nd > 0:
            # Swap A/B with C/D
            # (la, lb ... | lc, ld ...) -> (lc, ld ... | la, lb ...)
            # Need to update Rp, Rq, etc?
            # Actually easier: just implement the logic for c.
            # But the logic is symmetric.
            
            # Recursing on C:
            # (ab|c+1 d) ...
            # Center Q is (gamma C + delta D)/q
            # W = (pP + qQ)/(p+q)
            # W - Q = p(P-Q)/(p+q) = p/(p+q) * Rpq
            
            if lc > 0:
                Qi_minus_Ci_x = (Rq[0] - c.origin[0])
                Wi_minus_Qi_x = (p / (p + q)) * (Rp[0] - Rq[0])
                
                term1 = Qi_minus_Ci_x * get_eri(la, lb, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m)
                term2 = Wi_minus_Qi_x * get_eri(la, lb, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                
                term3 = 0.0 # from c
                if lc > 1:
                    val = get_eri(la, lb, lc-2, ld, ma, mb, mc, md, na, nb, nc, nd, m)
                    val_m = get_eri(la, lb, lc-2, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                    term3 = (lc-1)/(2*q) * (val - (rho/q)*val_m)
                    
                term4 = 0.0 # from d
                if ld > 0:
                    val = get_eri(la, lb, lc-1, ld-1, ma, mb, mc, md, na, nb, nc, nd, m)
                    val_m = get_eri(la, lb, lc-1, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1)
                    term4 = ld/(2*q) * (val - (rho/q)*val_m)
                    
                term5 = 0.0 # from a
                if la > 0:
                     val = get_eri(la-1, lb, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                     term5 = la / (2*(p+q)) * val
                
                term6 = 0.0 # from b
                if lb > 0:
                    val = get_eri(la, lb-1, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                    term6 = lb / (2*(p+q)) * val
                    
                res = term1 + term2 + term3 + term4 + term5 + term6
                memo[key] = res
                return res
            
            if ld > 0:
                # Same as c but for d
                Qi_minus_Di_x = (Rq[0] - d.origin[0])
                Wi_minus_Qi_x = (p / (p + q)) * (Rp[0] - Rq[0])
                
                term1 = Qi_minus_Di_x * get_eri(la, lb, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m)
                term2 = Wi_minus_Qi_x * get_eri(la, lb, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1)
                
                term3 = 0.0 # c
                if lc > 0:
                    val = get_eri(la, lb, lc-1, ld-1, ma, mb, mc, md, na, nb, nc, nd, m)
                    val_m = get_eri(la, lb, lc-1, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1)
                    term3 = lc/(2*q) * (val - (rho/q)*val_m)
                
                term4 = 0.0 # d
                if ld > 1:
                    val = get_eri(la, lb, lc, ld-2, ma, mb, mc, md, na, nb, nc, nd, m)
                    val_m = get_eri(la, lb, lc, ld-2, ma, mb, mc, md, na, nb, nc, nd, m+1)
                    term4 = (ld-1)/(2*q) * (val - (rho/q)*val_m)
                    
                term5 = 0.0 # a
                if la > 0:
                    val = get_eri(la-1, lb, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1)
                    term5 = la / (2*(p+q)) * val
                    
                term6 = 0.0 # b
                if lb > 0:
                    val = get_eri(la, lb-1, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1)
                    term6 = lb / (2*(p+q)) * val

                res = term1 + term2 + term3 + term4 + term5 + term6
                memo[key] = res
                return res

        # Handle Y component (ma, mb, mc, md)
        if ma > 0:
            # Logic is identical to x but with Y coords
            # Copy paste logic but use index 1
            Pi_minus_Ai_y = (Rp[1] - a.origin[1])
            Wi_minus_Pi_y = - (q / (p + q)) * (Rp[1] - Rq[1])
            
            term1 = Pi_minus_Ai_y * get_eri(la, lb, lc, ld, ma-1, mb, mc, md, na, nb, nc, nd, m)
            term2 = Wi_minus_Pi_y * get_eri(la, lb, lc, ld, ma-1, mb, mc, md, na, nb, nc, nd, m+1)
            
            term3 = 0.0
            if ma > 1:
                val = get_eri(la, lb, lc, ld, ma-2, mb, mc, md, na, nb, nc, nd, m)
                val_m = get_eri(la, lb, lc, ld, ma-2, mb, mc, md, na, nb, nc, nd, m+1)
                term3 = (ma-1)/(2*p) * (val - (rho/p)*val_m)
                
            term4 = 0.0
            if mb > 0:
                val = get_eri(la, lb, lc, ld, ma-1, mb-1, mc, md, na, nb, nc, nd, m)
                val_m = get_eri(la, lb, lc, ld, ma-1, mb-1, mc, md, na, nb, nc, nd, m+1)
                term4 = mb/(2*p) * (val - (rho/p)*val_m)
                
            term5 = 0.0
            if mc > 0:
                val = get_eri(la, lb, lc, ld, ma-1, mb, mc-1, md, na, nb, nc, nd, m+1)
                term5 = mc / (2*(p+q)) * val
                
            term6 = 0.0
            if md > 0:
                val = get_eri(la, lb, lc, ld, ma-1, mb, mc, md-1, na, nb, nc, nd, m+1)
                term6 = md / (2*(p+q)) * val
                
            res = term1 + term2 + term3 + term4 + term5 + term6
            memo[key] = res
            return res
            
        if mb > 0:
            Pi_minus_Bi_y = (Rp[1] - b.origin[1])
            Wi_minus_Pi_y = - (q / (p + q)) * (Rp[1] - Rq[1])
            
            term1 = Pi_minus_Bi_y * get_eri(la, lb, lc, ld, ma, mb-1, mc, md, na, nb, nc, nd, m)
            term2 = Wi_minus_Pi_y * get_eri(la, lb, lc, ld, ma, mb-1, mc, md, na, nb, nc, nd, m+1)
            
            term3 = 0.0
            if ma > 0:
                val = get_eri(la, lb, lc, ld, ma-1, mb-1, mc, md, na, nb, nc, nd, m)
                val_m = get_eri(la, lb, lc, ld, ma-1, mb-1, mc, md, na, nb, nc, nd, m+1)
                term3 = ma/(2*p) * (val - (rho/p)*val_m)
                
            term4 = 0.0
            if mb > 1:
                val = get_eri(la, lb, lc, ld, ma, mb-2, mc, md, na, nb, nc, nd, m)
                val_m = get_eri(la, lb, lc, ld, ma, mb-2, mc, md, na, nb, nc, nd, m+1)
                term4 = (mb-1)/(2*p) * (val - (rho/p)*val_m)
                
            term5 = 0.0
            if mc > 0:
                val = get_eri(la, lb, lc, ld, ma, mb-1, mc-1, md, na, nb, nc, nd, m+1)
                term5 = mc / (2*(p+q)) * val
                
            term6 = 0.0
            if md > 0:
                val = get_eri(la, lb, lc, ld, ma, mb-1, mc, md-1, na, nb, nc, nd, m+1)
                term6 = md / (2*(p+q)) * val

            res = term1 + term2 + term3 + term4 + term5 + term6
            memo[key] = res
            return res

        # Handle Y on C, D
        if mc > 0:
            Qi_minus_Ci_y = (Rq[1] - c.origin[1])
            Wi_minus_Qi_y = (p / (p + q)) * (Rp[1] - Rq[1])
            
            term1 = Qi_minus_Ci_y * get_eri(la, lb, lc, ld, ma, mb, mc-1, md, na, nb, nc, nd, m)
            term2 = Wi_minus_Qi_y * get_eri(la, lb, lc, ld, ma, mb, mc-1, md, na, nb, nc, nd, m+1)
            
            term3 = 0.0
            if mc > 1:
                val = get_eri(la, lb, lc, ld, ma, mb, mc-2, md, na, nb, nc, nd, m)
                val_m = get_eri(la, lb, lc, ld, ma, mb, mc-2, md, na, nb, nc, nd, m+1)
                term3 = (mc-1)/(2*q) * (val - (rho/q)*val_m)
                
            term4 = 0.0
            if md > 0:
                val = get_eri(la, lb, lc, ld, ma, mb, mc-1, md-1, na, nb, nc, nd, m)
                val_m = get_eri(la, lb, lc, ld, ma, mb, mc-1, md-1, na, nb, nc, nd, m+1)
                term4 = md/(2*q) * (val - (rho/q)*val_m)
                
            term5 = 0.0
            if ma > 0:
                val = get_eri(la, lb, lc, ld, ma-1, mb, mc-1, md, na, nb, nc, nd, m+1)
                term5 = ma / (2*(p+q)) * val
                
            term6 = 0.0
            if mb > 0:
                val = get_eri(la, lb, lc, ld, ma, mb-1, mc-1, md, na, nb, nc, nd, m+1)
                term6 = mb / (2*(p+q)) * val
                
            res = term1 + term2 + term3 + term4 + term5 + term6
            memo[key] = res
            return res

        if md > 0:
            Qi_minus_Di_y = (Rq[1] - d.origin[1])
            Wi_minus_Qi_y = (p / (p + q)) * (Rp[1] - Rq[1])
            
            term1 = Qi_minus_Di_y * get_eri(la, lb, lc, ld, ma, mb, mc, md-1, na, nb, nc, nd, m)
            term2 = Wi_minus_Qi_y * get_eri(la, lb, lc, ld, ma, mb, mc, md-1, na, nb, nc, nd, m+1)
            
            term3 = 0.0
            if mc > 0:
                val = get_eri(la, lb, lc, ld, ma, mb, mc-1, md-1, na, nb, nc, nd, m)
                val_m = get_eri(la, lb, lc, ld, ma, mb, mc-1, md-1, na, nb, nc, nd, m+1)
                term3 = mc/(2*q) * (val - (rho/q)*val_m)
                
            term4 = 0.0
            if md > 1:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md-2, na, nb, nc, nd, m)
                val_m = get_eri(la, lb, lc, ld, ma, mb, mc, md-2, na, nb, nc, nd, m+1)
                term4 = (md-1)/(2*q) * (val - (rho/q)*val_m)
                
            term5 = 0.0
            if ma > 0:
                val = get_eri(la, lb, lc, ld, ma-1, mb, mc, md-1, na, nb, nc, nd, m+1)
                term5 = ma / (2*(p+q)) * val
                
            term6 = 0.0
            if mb > 0:
                val = get_eri(la, lb, lc, ld, ma, mb-1, mc, md-1, na, nb, nc, nd, m+1)
                term6 = mb / (2*(p+q)) * val
                
            res = term1 + term2 + term3 + term4 + term5 + term6
            memo[key] = res
            return res

        # Handle Z component (na, nb, nc, nd)
        # Identical to X, Y but with index 2
        # ... Shortened for brevity, but needed for p-orbitals in Z ...
        # If I skip this, Z-oriented p-orbitals will fail. I must include it.
        
        if na > 0:
            Pi_minus_Ai_z = (Rp[2] - a.origin[2])
            Wi_minus_Pi_z = - (q / (p + q)) * (Rp[2] - Rq[2])
            
            term1 = Pi_minus_Ai_z * get_eri(la, lb, lc, ld, ma, mb, mc, md, na-1, nb, nc, nd, m)
            term2 = Wi_minus_Pi_z * get_eri(la, lb, lc, ld, ma, mb, mc, md, na-1, nb, nc, nd, m+1)
            
            term3 = 0.0
            if na > 1:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na-2, nb, nc, nd, m)
                val_m = get_eri(la, lb, lc, ld, ma, mb, mc, md, na-2, nb, nc, nd, m+1)
                term3 = (na-1)/(2*p) * (val - (rho/p)*val_m)
            
            term4 = 0.0
            if nb > 0:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na-1, nb-1, nc, nd, m)
                val_m = get_eri(la, lb, lc, ld, ma, mb, mc, md, na-1, nb-1, nc, nd, m+1)
                term4 = nb/(2*p) * (val - (rho/p)*val_m)
                
            term5 = 0.0
            if nc > 0:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na-1, nb, nc-1, nd, m+1)
                term5 = nc / (2*(p+q)) * val
                
            term6 = 0.0
            if nd > 0:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na-1, nb, nc, nd-1, m+1)
                term6 = nd / (2*(p+q)) * val
                
            res = term1 + term2 + term3 + term4 + term5 + term6
            memo[key] = res
            return res
            
        if nb > 0:
            Pi_minus_Bi_z = (Rp[2] - b.origin[2])
            Wi_minus_Pi_z = - (q / (p + q)) * (Rp[2] - Rq[2])
            
            term1 = Pi_minus_Bi_z * get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb-1, nc, nd, m)
            term2 = Wi_minus_Pi_z * get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb-1, nc, nd, m+1)
            
            term3 = 0.0
            if na > 0:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na-1, nb-1, nc, nd, m)
                val_m = get_eri(la, lb, lc, ld, ma, mb, mc, md, na-1, nb-1, nc, nd, m+1)
                term3 = na/(2*p) * (val - (rho/p)*val_m)
            
            term4 = 0.0
            if nb > 1:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb-2, nc, nd, m)
                val_m = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb-2, nc, nd, m+1)
                term4 = (nb-1)/(2*p) * (val - (rho/p)*val_m)
                
            term5 = 0.0
            if nc > 0:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb-1, nc-1, nd, m+1)
                term5 = nc / (2*(p+q)) * val
                
            term6 = 0.0
            if nd > 0:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb-1, nc, nd-1, m+1)
                term6 = nd / (2*(p+q)) * val
                
            res = term1 + term2 + term3 + term4 + term5 + term6
            memo[key] = res
            return res
            
        if nc > 0:
            Qi_minus_Ci_z = (Rq[2] - c.origin[2])
            Wi_minus_Qi_z = (p / (p + q)) * (Rp[2] - Rq[2])
            
            term1 = Qi_minus_Ci_z * get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-1, nd, m)
            term2 = Wi_minus_Qi_z * get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-1, nd, m+1)
            
            term3 = 0.0
            if mc > 1: # copy-paste error risk here. nc > 1
                pass
            if nc > 1:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-2, nd, m)
                val_m = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-2, nd, m+1)
                term3 = (nc-1)/(2*q) * (val - (rho/q)*val_m)
                
            term4 = 0.0
            if nd > 0:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-1, nd-1, m)
                val_m = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-1, nd-1, m+1)
                term4 = nd/(2*q) * (val - (rho/q)*val_m)
                
            term5 = 0.0
            if na > 0:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na-1, nb, nc-1, nd, m+1)
                term5 = na / (2*(p+q)) * val
                
            term6 = 0.0
            if nb > 0:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb-1, nc-1, nd, m+1)
                term6 = nb / (2*(p+q)) * val
                
            res = term1 + term2 + term3 + term4 + term5 + term6
            memo[key] = res
            return res
            
        if nd > 0:
            Qi_minus_Di_z = (Rq[2] - d.origin[2])
            Wi_minus_Qi_z = (p / (p + q)) * (Rp[2] - Rq[2])
            
            term1 = Qi_minus_Di_z * get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd-1, m)
            term2 = Wi_minus_Qi_z * get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd-1, m+1)
            
            term3 = 0.0
            if nc > 0:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-1, nd-1, m)
                val_m = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-1, nd-1, m+1)
                term3 = nc/(2*q) * (val - (rho/q)*val_m)
                
            term4 = 0.0
            if nd > 1:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd-2, m)
                val_m = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd-2, m+1)
                term4 = (nd-1)/(2*q) * (val - (rho/q)*val_m)
                
            term5 = 0.0
            if na > 0:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na-1, nb, nc, nd-1, m+1)
                term5 = na / (2*(p+q)) * val
                
            term6 = 0.0
            if nb > 0:
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb-1, nc, nd-1, m+1)
                term6 = nb / (2*(p+q)) * val
                
            res = term1 + term2 + term3 + term4 + term5 + term6
            memo[key] = res
            return res
            
        return 0.0

    val = get_eri(a.l, b.l, c.l, d.l, a.m, b.m, c.m, d.m, a.n, b.n, c.n, d.n, 0)
    return a.N * b.N * c.N * d.N * val
