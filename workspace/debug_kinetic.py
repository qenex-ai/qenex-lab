
import numpy as np

def numerical_1d(alpha, beta, A, B):
    # Overlap
    def bra(x): return np.exp(-alpha * (x - A)**2)
    def ket(x): return np.exp(-beta * (x - B)**2)
    
    # Kinetic
    # d2/dx2 ket = [ 4b^2 (x-B)^2 - 2b ] ket
    def lap_ket(x):
        u = x - B
        return (4 * beta**2 * u**2 - 2 * beta) * ket(x)
        
    x = np.linspace(-5, 5, 10000) + (A+B)/2
    dx = x[1] - x[0]
    
    S = np.sum(bra(x) * ket(x)) * dx
    T = np.sum(bra(x) * (-0.5 * lap_ket(x))) * dx
    
    return S, T

alpha = 1.5
beta = 0.8
A = 0.0
B = 1.2

S_num, T_num = numerical_1d(alpha, beta, A, B)
print(f"Num 1D (x): S={S_num:.6f}, T={T_num:.6f}")

# Analytic Check
p = alpha + beta
K = np.exp(-alpha*beta/p * (A-B)**2)
S_ana = K * np.sqrt(np.pi/p)
print(f"Ana 1D (x): S={S_ana:.6f}")

# T analytic derived from code formula
# T = K * [ alpha*beta/p - 2*beta^2*alpha^2/p^2 * (A-B)^2 ] * S_00_normalized ?
# My previous derivation: T = S_full * [ ab/p - 2b^2 (P-B)^2 ]
# P-B = -a/p (B-A) = a/p (A-B)
# (P-B)^2 = a^2/p^2 (A-B)^2
# T = S_full * [ ab/p - 2b^2 a^2/p^2 (A-B)^2 ]
term1 = alpha * beta / p
term2 = 2 * beta**2 * (alpha**2 / p**2) * (A-B)**2
T_ana = S_ana * (term1 - term2)
print(f"Ana 1D (x): T={T_ana:.6f}")

# Y direction (A=B=0)
S_y, T_y = numerical_1d(alpha, beta, 0, 0)
print(f"Num 1D (y): S={S_y:.6f}, T={T_y:.6f}")

T_y_ana = np.sqrt(np.pi/p) * (alpha * beta / p)
print(f"Ana 1D (y): T={T_y_ana:.6f}")

# Total 3D T
# T = Tx Sy Sz + Sx Ty Sz + Sx Sy Tz
T_3d = T_num * S_y * S_y + S_num * T_y * S_y + S_num * S_y * T_y
print(f"Total 3D T (Num): {T_3d:.6f}")

# Previous JIT result was 0.588.
