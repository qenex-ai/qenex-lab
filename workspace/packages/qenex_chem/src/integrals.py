"""
Integrals Module
Provides analytical integrals for Gaussian primitives and basis set definitions (STO-3G).
Supports s and p orbitals via Obara-Saika recurrence relations.
Now optimized with Numba JIT compilation.
"""

import numpy as np
from scipy.special import erf, gamma as scipy_gamma, gammainc as scipy_gammainc
import numba
from numba import jit, float64, int64

import math

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
        'Ne': 2.88,
        
        # Core 1s zetas for 2nd row (approximate)
        'Li_1s': 2.69,
        'Be_1s': 3.68,
        'B_1s': 4.68,
        'C_1s': 5.67,
        'N_1s': 6.67,
        'O_1s': 7.66,
        'F_1s': 8.65,
        'Ne_1s': 9.64
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

# JIT-compiled Boys function wrapper
# Numba does not support scipy.special.erf directly in nopython mode easily without extra config in some versions,
# but supports math.erf from python 3.2+. Numpy erf is also supported.
@jit(nopython=True, fastmath=True)
def boys_jit(n, t):
    """
    Boys function F_n(t) = integral_0^1 u^(2n) exp(-t u^2) du
    JIT-optimized version with asymptotic tail.
    """
    if t < 1e-8:
        return 1.0 / (2*n + 1) - t / (2*n + 3)
    
    # Asymptotic expansion for large t
    elif t > 40.0:
        # F_n(t) ~ (2n-1)!! * sqrt(pi) / (2^(n+1) * t^(n+0.5))
        # Compute (2n-1)!!
        df = 1.0
        if n > 0:
            for i in range(1, 2*n, 2):
                df *= i
        
        return (df * np.sqrt(np.pi)) / (2.0**(n+1) * t**(n + 0.5))
        
    else:
        # F0(t) is exact
        val_0 = 0.5 * np.sqrt(np.pi / t) * math.erf(np.sqrt(t))
        if n == 0:
            return val_0
            
        # For N > 0 in intermediate range, use robust numerical integration.
        # Increased steps for precision (500 steps)
        steps = 500
        res = 0.0
        dt = 1.0 / steps
        for i in range(steps):
            u = (i + 0.5) * dt
            res += (u**(2*n)) * np.exp(-t * u*u)
        return res * dt

def boys(n, t):
    """Python wrapper for JIT boys function"""
    return boys_jit(n, float(t))

# ==========================================
# Primitive Gaussian Class
# ==========================================

class BasisFunction:
    def __init__(self, origin, alpha, coeff, lmn=(0,0,0)):
        self.origin = np.array(origin, dtype=np.float64)
        self.alpha = float(alpha)
        self.coeff = float(coeff)
        self.l, self.m, self.n = lmn
        self.L = self.l + self.m + self.n
        self.norm = normalize_primitive(alpha, lmn)
        self.N = self.coeff * self.norm

# ==========================================
# JIT-Optimized Integral Kernels
# ==========================================

@jit(nopython=True, fastmath=True)
def gaussian_product_center(alpha1, A, alpha2, B):
    return (alpha1 * A + alpha2 * B) / (alpha1 + alpha2)

@jit(nopython=True, fastmath=True)
def overlap_1d(l1, l2, xA, xB, alpha, beta):
    """
    1D Overlap Integral between (x-xA)^l1 and (x-xB)^l2
    """
    p = alpha + beta
    rp = (alpha * xA + beta * xB) / p
    
    # We avoid allocating S array dynamically if possible, or use small fixed size
    # Max angular momentum is usually small (d=2, f=3). 5x5 is safe.
    S = np.zeros((l1 + 1, l2 + 1))
    
    S[0, 0] = np.sqrt(np.pi / p)
    
    XP = rp
    XA = xA
    XB = xB
    
    for i in range(l1 + 1):
        for j in range(l2 + 1):
            if i == 0 and j == 0:
                continue
                
            if i > 0:
                term1 = (XP - XA) * S[i-1, j]
                term2 = (i-1) * S[i-2, j] if i > 1 else 0.0
                term3 = j * S[i-1, j-1] if j > 0 else 0.0
                S[i, j] = term1 + (1.0 / (2*p)) * (term2 + term3)
            else:
                term1 = (XP - XB) * S[i, j-1]
                term2 = i * S[i-1, j-1] if i > 0 else 0.0
                term3 = (j-1) * S[i, j-2] if j > 1 else 0.0
                S[i, j] = term1 + (1.0 / (2*p)) * (term2 + term3)
                
    return S[l1, l2]

@jit(nopython=True, fastmath=True)
def overlap_jit(la, ma, na, lb, mb, nb, ra, rb, alpha, beta):
    p = alpha + beta
    diff = ra - rb
    dist2 = np.dot(diff, diff)
    pre = np.exp(- (alpha * beta / p) * dist2)
    
    sx = overlap_1d(la, lb, ra[0], rb[0], alpha, beta)
    sy = overlap_1d(ma, mb, ra[1], rb[1], alpha, beta)
    sz = overlap_1d(na, nb, ra[2], rb[2], alpha, beta)
    
    return pre * sx * sy * sz

def overlap(a, b):
    val = overlap_jit(a.l, a.m, a.n, b.l, b.m, b.n, a.origin, b.origin, a.alpha, b.alpha)
    return a.N * b.N * val

@jit(nopython=True, fastmath=True)
def kinetic_jit(la, ma, na, lb, mb, nb, ra, rb, alpha, beta):
    p = alpha + beta
    
    # Precompute 1D overlaps needed manually
    # We need to compute S values for different lb
    
    # Helper to compute Term 1D
    # T_1d = -0.5 * [ l2(l2-1)<l1|l2-2> - 2beta(2l2+1)<l1|l2> + 4beta^2 <l1|l2+2> ]
    
    def get_term_1d(l1, l2, xA, xB):
        s0 = overlap_1d(l1, l2, xA, xB, alpha, beta)
        t1 = 0.0
        if l2 >= 2:
            s_minus = overlap_1d(l1, l2-2, xA, xB, alpha, beta)
            t1 = l2 * (l2 - 1) * s_minus
            
        t2 = -2.0 * beta * (2.0 * l2 + 1.0) * s0
        
        s_plus = overlap_1d(l1, l2+2, xA, xB, alpha, beta)
        t3 = 4.0 * beta**2 * s_plus
        
        return -0.5 * (t1 + t2 + t3)

    diff = ra - rb
    dist2 = np.dot(diff, diff)
    pre = np.exp(- (alpha * beta / p) * dist2)
    
    Sx = overlap_1d(la, lb, ra[0], rb[0], alpha, beta)
    Sy = overlap_1d(ma, mb, ra[1], rb[1], alpha, beta)
    Sz = overlap_1d(na, nb, ra[2], rb[2], alpha, beta)
    
    Tx = get_term_1d(la, lb, ra[0], rb[0])
    Ty = get_term_1d(ma, mb, ra[1], rb[1])
    Tz = get_term_1d(na, nb, ra[2], rb[2])
    
    val = Tx*Sy*Sz + Sx*Ty*Sz + Sx*Sy*Tz
    return pre * val

def kinetic(a, b):
    val = kinetic_jit(a.l, a.m, a.n, b.l, b.m, b.n, a.origin, b.origin, a.alpha, b.alpha)
    return a.N * b.N * val

# Note: Nuclear Attraction and ERI involve recursion that is harder to flatten for Numba without
# explicit stack management or fixed recursion depth. For now, we keep them in Python but optimize
# the Boys function and primitives.
#
# Optimization Strategy:
# 1. Provide a `nuclear_attraction_jit` that handles just the s-s case efficiently.
# 2. Keep the general recursive one in Python for now, but call the JIT boys function.

def nuclear_attraction(a, b, C, Z):
    """
    Nuclear Attraction Integral (a | -Z/r_C | b)
    """
    alpha = a.alpha
    beta = b.alpha
    p = alpha + beta
    
    Rp = gaussian_product_center(alpha, a.origin, beta, b.origin)
    diff = a.origin - b.origin
    dist2 = np.dot(diff, diff)
    kab = np.exp(- (alpha * beta / p) * dist2)
    
    RPC = Rp - C
    
    L_total = a.L + b.L
    T_val = p * np.dot(RPC, RPC)
    
    # Use JIT boys function
    boys_vals = [boys_jit(n, float(T_val)) for n in range(L_total + 1)]
    
    memo = {}

    def get_v(la, lb, ma, mb, na, nb, m):
        key = (la, lb, ma, mb, na, nb, m)
        if key in memo:
            return memo[key]
        
        if la == 0 and lb == 0 and ma == 0 and mb == 0 and na == 0 and nb == 0:
            return (2.0 * np.pi / p) * kab * boys_vals[m]
        
        if la > 0:
            term1 = (Rp[0] - a.origin[0]) * get_v(la-1, lb, ma, mb, na, nb, m)
            term2 = (Rp[0] - C[0]) * get_v(la-1, lb, ma, mb, na, nb, m+1)
            term3 = (la-1)/(2*p) * ( get_v(la-2, lb, ma, mb, na, nb, m) - get_v(la-2, lb, ma, mb, na, nb, m+1) ) if la > 1 else 0.0
            term4 = lb/(2*p) * ( get_v(la-1, lb-1, ma, mb, na, nb, m) - get_v(la-1, lb-1, ma, mb, na, nb, m+1) ) if lb > 0 else 0.0
            res = term1 - term2 + term3 + term4
            memo[key] = res
            return res
            
        if lb > 0:
            term1 = (Rp[0] - b.origin[0]) * get_v(la, lb-1, ma, mb, na, nb, m)
            term2 = (Rp[0] - C[0]) * get_v(la, lb-1, ma, mb, na, nb, m+1)
            term3 = la/(2*p) * ( get_v(la-1, lb-1, ma, mb, na, nb, m) - get_v(la-1, lb-1, ma, mb, na, nb, m+1) ) if la > 0 else 0.0
            term4 = (lb-1)/(2*p) * ( get_v(la, lb-2, ma, mb, na, nb, m) - get_v(la, lb-2, ma, mb, na, nb, m+1) ) if lb > 1 else 0.0
            res = term1 - term2 + term3 + term4
            memo[key] = res
            return res

        if ma > 0:
            term1 = (Rp[1] - a.origin[1]) * get_v(la, lb, ma-1, mb, na, nb, m)
            term2 = (Rp[1] - C[1]) * get_v(la, lb, ma-1, mb, na, nb, m+1)
            term3 = (ma-1)/(2*p) * ( get_v(la, lb, ma-2, mb, na, nb, m) - get_v(la, lb, ma-2, mb, na, nb, m+1) ) if ma > 1 else 0.0
            term4 = mb/(2*p) * ( get_v(la, lb, ma-1, mb-1, na, nb, m) - get_v(la, lb, ma-1, mb-1, na, nb, m+1) ) if mb > 0 else 0.0
            # print(f"DEBUG ma>0: m={m}, t1={term1}, t2={term2}, t3={term3}, t4={term4}")
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
            
        return 0.0

    val = get_v(a.l, b.l, a.m, b.m, a.n, b.n, 0)
    return -Z * a.N * b.N * val

def eri(a, b, c, d):
    """
    Two-Electron Repulsion Integral (ab|cd)
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
    
    L_total = a.L + b.L + c.L + d.L
    T_val = alpha_Q * np.dot(Rpq, Rpq)
    boys_vals = [boys_jit(n, float(T_val)) for n in range(L_total + 1)]
    
    memo = {}

    def get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m):
        key = (la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m)
        if key in memo: return memo[key]
        
        rho = alpha_Q

        if sum(key[:-1]) == 0:
            return prefactor * boys_vals[m]
            
        if la > 0:
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
                val_c = get_eri(la-1, lb, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                term5 = lc / (2*(p+q)) * val_c 
            
            term6 = 0.0
            if ld > 0:
                val_d = get_eri(la-1, lb, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1)
                term6 = ld / (2*(p+q)) * val_d
                
            res = term1 + term2 + term3 + term4 + term5 + term6
            memo[key] = res
            return res
            
        if lb > 0:
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
                 
             term5 = 0.0
             if lc > 0:
                 val = get_eri(la, lb-1, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                 term5 = lc / (2*(p+q)) * val
                 
             term6 = 0.0
             if ld > 0:
                 val = get_eri(la, lb-1, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1)
                 term6 = ld / (2*(p+q)) * val
             
             res = term1 + term2 + term3 + term4 + term5 + term6
             memo[key] = res
             return res
        
        # Recurse on C (switch centers)
        if lc > 0:
            Qi_minus_Ci_x = (Rq[0] - c.origin[0])
            Wi_minus_Qi_x = (p / (p + q)) * (Rp[0] - Rq[0])
            
            term1 = Qi_minus_Ci_x * get_eri(la, lb, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m)
            term2 = Wi_minus_Qi_x * get_eri(la, lb, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
            
            term3 = 0.0
            if lc > 1:
                val = get_eri(la, lb, lc-2, ld, ma, mb, mc, md, na, nb, nc, nd, m)
                val_m = get_eri(la, lb, lc-2, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                term3 = (lc-1)/(2*q) * (val - (rho/q)*val_m)
                
            term4 = 0.0
            if ld > 0:
                val = get_eri(la, lb, lc-1, ld-1, ma, mb, mc, md, na, nb, nc, nd, m)
                val_m = get_eri(la, lb, lc-1, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1)
                term4 = ld/(2*q) * (val - (rho/q)*val_m)
                
            term5 = 0.0
            if la > 0:
                 val = get_eri(la-1, lb, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                 term5 = la / (2*(p+q)) * val
            
            term6 = 0.0
            if lb > 0:
                val = get_eri(la, lb-1, lc-1, ld, ma, mb, mc, md, na, nb, nc, nd, m+1)
                term6 = lb / (2*(p+q)) * val
                
            res = term1 + term2 + term3 + term4 + term5 + term6
            memo[key] = res
            return res

        if ld > 0:
            Qi_minus_Di_x = (Rq[0] - d.origin[0])
            Wi_minus_Qi_x = (p / (p + q)) * (Rp[0] - Rq[0])
            
            term1 = Qi_minus_Di_x * get_eri(la, lb, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m)
            term2 = Wi_minus_Qi_x * get_eri(la, lb, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1)
            
            term3 = 0.0
            if lc > 0:
                val = get_eri(la, lb, lc-1, ld-1, ma, mb, mc, md, na, nb, nc, nd, m)
                val_m = get_eri(la, lb, lc-1, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1)
                term3 = lc/(2*q) * (val - (rho/q)*val_m)
            
            term4 = 0.0
            if ld > 1:
                val = get_eri(la, lb, lc, ld-2, ma, mb, mc, md, na, nb, nc, nd, m)
                val_m = get_eri(la, lb, lc, ld-2, ma, mb, mc, md, na, nb, nc, nd, m+1)
                term4 = (ld-1)/(2*q) * (val - (rho/q)*val_m)
            
            term5 = 0.0
            if la > 0:
                val = get_eri(la-1, lb, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1)
                term5 = la / (2*(p+q)) * val
                
            term6 = 0.0
            if lb > 0:
                val = get_eri(la, lb-1, lc, ld-1, ma, mb, mc, md, na, nb, nc, nd, m+1)
                term6 = lb / (2*(p+q)) * val
            
            res = term1 + term2 + term3 + term4 + term5 + term6
            memo[key] = res
            return res

        # Handle Y/Z axes
        # Y axis
        if ma > 0:
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
                # FIXED: Decrement mc, not lc
                val = get_eri(la, lb, lc, ld, ma-1, mb, mc-1, md, na, nb, nc, nd, m+1)
                term5 = mc / (2*(p+q)) * val
                
            term6 = 0.0
            if md > 0:
                # FIXED: Decrement md, not ld
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
                # FIXED: Decrement mc, not lc
                val = get_eri(la, lb, lc, ld, ma, mb-1, mc-1, md, na, nb, nc, nd, m+1)
                term5 = mc / (2*(p+q)) * val
                
            term6 = 0.0
            if md > 0:
                # FIXED: Decrement md, not ld
                val = get_eri(la, lb, lc, ld, ma, mb-1, mc, md-1, na, nb, nc, nd, m+1)
                term6 = md / (2*(p+q)) * val

            res = term1 + term2 + term3 + term4 + term5 + term6
            memo[key] = res
            return res
            
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
            
        # Z axis
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
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc-1, nd, m+1)
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
                val = get_eri(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd-1, m+1)
                term6 = nb / (2*(p+q)) * val
            
            res = term1 + term2 + term3 + term4 + term5 + term6
            memo[key] = res
            return res

        return 0.0

    val = get_eri(a.l, b.l, c.l, d.l, a.m, b.m, c.m, d.m, a.n, b.n, c.n, d.n, 0)
    return a.N * b.N * c.N * d.N * val

# ==========================================
# Derivative Helpers
# ==========================================

def get_derivative_primitives(basis_func, coord_idx):
    """
    Returns a list of (factor, new_basis_func) representing the derivative
    of the given primitive basis function with respect to its center coordinate A[coord_idx].
    
    Relation: d/dAx g(l) = 2*alpha*g(l+1) - l*g(l-1)
    
    The returned basis functions are valid BasisFunction objects (normalized),
    so we must adjust the factors by the ratio of normalizations.
    """
    bf = basis_func
    lmn = list(bf.lmn) if hasattr(bf, 'lmn') else [bf.l, bf.m, bf.n] # Helper handles both
    
    res = []
    
    # Term 1: 2 * alpha * g(l+1)
    # New quantum numbers
    lmn_plus = list(lmn)
    lmn_plus[coord_idx] += 1
    
    bf_plus = BasisFunction(bf.origin, bf.alpha, bf.coeff, tuple(lmn_plus))
    
    # Factor adjustment:
    factor_plus = bf.N * 2.0 * bf.alpha / bf_plus.N
    res.append( (factor_plus, bf_plus) )
    
    # Term 2: -l * g(l-1)
    l_val = lmn[coord_idx]
    if l_val > 0:
        lmn_minus = list(lmn)
        lmn_minus[coord_idx] -= 1
        bf_minus = BasisFunction(bf.origin, bf.alpha, bf.coeff, tuple(lmn_minus))
        
        factor_minus = -1.0 * l_val * bf.N / bf_minus.N
        res.append( (factor_minus, bf_minus) )
        
    return res

def overlap_deriv(a, b, atom_idx, molecule_atoms):
    """
    Computes derivative of overlap integral (a|b) with respect to atom_idx coordinates.
    Returns np.array([dS/dx, dS/dy, dS/dz]).
    """
    grad = np.zeros(3)
    target_pos = np.array(molecule_atoms[atom_idx][1])
    is_a = np.linalg.norm(a.origin - target_pos) < 1e-12
    is_b = np.linalg.norm(b.origin - target_pos) < 1e-12
    
    if not is_a and not is_b:
        return grad
        
    for coord in range(3):
        val = 0.0
        # If a is on center: d/dA <a|b>
        if is_a:
            prims = get_derivative_primitives(a, coord)
            for factor, bf_new in prims:
                val += factor * overlap(bf_new, b)
        
        # If b is on center: d/dA <a|b>
        if is_b:
            prims = get_derivative_primitives(b, coord)
            for factor, bf_new in prims:
                val += factor * overlap(a, bf_new)
                
        grad[coord] = val
        
    return grad

def kinetic_deriv(a, b, atom_idx, molecule_atoms):
    """ Derivative of Kinetic Energy Integral """
    grad = np.zeros(3)
    target_pos = np.array(molecule_atoms[atom_idx][1])
    is_a = np.linalg.norm(a.origin - target_pos) < 1e-12
    is_b = np.linalg.norm(b.origin - target_pos) < 1e-12
    
    if not is_a and not is_b: return grad
    
    for coord in range(3):
        val = 0.0
        if is_a:
            for factor, bf_new in get_derivative_primitives(a, coord):
                val += factor * kinetic(bf_new, b)
        if is_b:
            for factor, bf_new in get_derivative_primitives(b, coord):
                val += factor * kinetic(a, bf_new)
        grad[coord] = val
    return grad

def nuclear_attraction_deriv(a, b, atom_idx, molecule_atoms):
    """ 
    Derivative of V_ne integral.
    """
    grad = np.zeros(3)
    target_pos = np.array(molecule_atoms[atom_idx][1])
    is_a = np.linalg.norm(a.origin - target_pos) < 1e-12
    is_b = np.linalg.norm(b.origin - target_pos) < 1e-12
    
    Z_map = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10}
    
    for c_idx, (el, pos) in enumerate(molecule_atoms):
        Z = Z_map.get(el, 0)
        C_pos = np.array(pos)
        is_C = (c_idx == atom_idx)
        
        for coord in range(3):
            term_basis = 0.0
            if is_a:
                for f, bf_new in get_derivative_primitives(a, coord):
                    term_basis += f * nuclear_attraction(bf_new, b, C_pos, Z)
            if is_b:
                for f, bf_new in get_derivative_primitives(b, coord):
                    term_basis += f * nuclear_attraction(a, bf_new, C_pos, Z)
            
            term_op = 0.0
            if is_C:
                # Hellmann-Feynman term essentially, but computed via basis shift trick
                for f, bf_new in get_derivative_primitives(a, coord):
                    term_op -= f * nuclear_attraction(bf_new, b, C_pos, Z)
                for f, bf_new in get_derivative_primitives(b, coord):
                    term_op -= f * nuclear_attraction(a, bf_new, C_pos, Z)
            
            grad[coord] += term_basis + term_op

    return grad

def eri_deriv(a, b, c, d, atom_idx, molecule_atoms):
    """
    Derivative of (ab|cd) wrt atom_idx.
    """
    grad = np.zeros(3)
    target_pos = np.array(molecule_atoms[atom_idx][1])
    is_a = np.linalg.norm(a.origin - target_pos) < 1e-12
    is_b = np.linalg.norm(b.origin - target_pos) < 1e-12
    is_c = np.linalg.norm(c.origin - target_pos) < 1e-12
    is_d = np.linalg.norm(d.origin - target_pos) < 1e-12
    
    if not (is_a or is_b or is_c or is_d): return grad
    
    for coord in range(3):
        val = 0.0
        if is_a:
            for f, bf in get_derivative_primitives(a, coord): val += f * eri(bf, b, c, d)
        if is_b:
            for f, bf in get_derivative_primitives(b, coord): val += f * eri(a, bf, c, d)
        if is_c:
            for f, bf in get_derivative_primitives(c, coord): val += f * eri(a, b, bf, d)
        if is_d:
            for f, bf in get_derivative_primitives(d, coord): val += f * eri(a, b, c, bf)
        grad[coord] = val
        
    return grad

# ==========================================
# Basis Construction
# ==========================================

class ContractedGaussian:
    def __init__(self, primitives, label=""):
        self.primitives = primitives
        self.label = label

def build_basis(molecule):
    """
    Constructs a basis set for the given molecule.
    Returns a list of ContractedGaussian objects.
    """
    basis_set = []
    
    # Map element to STO-3G parameters
    sto3g = STO3G()
    
    for atom_idx, (element, pos) in enumerate(molecule.atoms):
        if element == 'H' or element == 'He':
            # 1s orbital
            # Scale factors (zeta)
            zeta = sto3g.zeta.get(element, 1.0)
            
            # 1s
            prims_1s = []
            for alpha_norm, d_norm in sto3g.basis_1s:
                alpha = alpha_norm * (zeta**2)
                prims_1s.append(BasisFunction(pos, alpha, d_norm, (0,0,0)))
            
            basis_set.append(ContractedGaussian(prims_1s, f"{element}_{atom_idx}_1s"))
            
        elif element in ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']:
            # Row 2: 1s, 2s, 2p
            
            # 1s Core
            zeta_1s = sto3g.zeta.get(f"{element}_1s", sto3g.zeta.get(element, 1.0)) # Fallback if specific 1s not found
            
            prims_1s = []
            for alpha_norm, d_norm in sto3g.basis_1s:
                alpha = alpha_norm * (zeta_1s**2)
                prims_1s.append(BasisFunction(pos, alpha, d_norm, (0,0,0)))
            basis_set.append(ContractedGaussian(prims_1s, f"{element}_{atom_idx}_1s"))
            
            # 2s Valence
            zeta_val = sto3g.zeta.get(element, 1.0)
            prims_2s = []
            for alpha_norm, d_norm in sto3g.basis_2s:
                alpha = alpha_norm * (zeta_val**2)
                prims_2s.append(BasisFunction(pos, alpha, d_norm, (0,0,0)))
            basis_set.append(ContractedGaussian(prims_2s, f"{element}_{atom_idx}_2s"))
            
            # 2p Valence (px, py, pz)
            # 2px
            prims_2px = []
            for alpha_norm, d_norm in sto3g.basis_2p:
                alpha = alpha_norm * (zeta_val**2)
                prims_2px.append(BasisFunction(pos, alpha, d_norm, (1,0,0)))
            basis_set.append(ContractedGaussian(prims_2px, f"{element}_{atom_idx}_2px"))
            
            # 2py
            prims_2py = []
            for alpha_norm, d_norm in sto3g.basis_2p:
                alpha = alpha_norm * (zeta_val**2)
                prims_2py.append(BasisFunction(pos, alpha, d_norm, (0,1,0)))
            basis_set.append(ContractedGaussian(prims_2py, f"{element}_{atom_idx}_2py"))
            
            # 2pz
            prims_2pz = []
            for alpha_norm, d_norm in sto3g.basis_2p:
                alpha = alpha_norm * (zeta_val**2)
                prims_2pz.append(BasisFunction(pos, alpha, d_norm, (0,0,1)))
            basis_set.append(ContractedGaussian(prims_2pz, f"{element}_{atom_idx}_2pz"))
            
        elif element in ['P', 'S']:
             # Row 3 (Minimal/Experimental support for now)
             pass

    return basis_set
