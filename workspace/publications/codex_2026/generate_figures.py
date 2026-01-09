import numpy as np
import matplotlib.pyplot as plt

# Generate Data simulating the RHF vs FCI gap for H2
r = np.linspace(0.5, 5.0, 100) # Angstroms

# Model functions (Morse-like potential)
# Exact (FCI) goes to -1.0 Hartree
E_exact = -1.0 + np.exp(-2*(r-0.74)) - 2*np.exp(-(r-0.74))

# RHF goes to approx -0.9 Hartree (Ionic limit error)
# It matches near equilibrium but diverges at large R
E_rhf = -0.8 + np.exp(-2*(r-0.74)) - 2*np.exp(-(r-0.74))
# Smooth blend to show the error appearing
weights = 1 / (1 + np.exp(-2*(r-2.0)))
E_rhf_model = E_exact * (1-weights) + (-0.9) * weights
E_rhf_model[r < 1.0] = E_exact[r < 1.0] # Match at equilibrium

plt.figure(figsize=(10, 6))
plt.plot(r, E_rhf_model, 'r--', linewidth=2, label='Restricted Hartree-Fock (RHF)')
plt.plot(r, E_exact, 'g-', linewidth=2, label='Full Configuration Interaction (FCI)')

plt.fill_between(r, E_rhf_model, E_exact, color='gray', alpha=0.2, label='Electron Correlation Gap')

plt.title('The Electron Correlation Problem in Molecular Dissociation ($H_2$)', fontsize=14)
plt.xlabel('Interatomic Distance (Å)', fontsize=12)
plt.ylabel('Energy (Hartree)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.axhline(y=-1.0, color='k', linestyle=':', alpha=0.5, label='Exact Dissociation Limit')

plt.savefig('publications/codex_2026/figures/correlation_gap.png', dpi=300)
print("Figure 1 generated.")
