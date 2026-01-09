# @qenex/physics

This package is the domain expert for Theoretical and Computational Physics within the QENEX LAB.

## Capabilities

- **Lattice Simulations**: High-performance Monte Carlo simulations for lattice QCD and spin systems.
- **Tensor Networks**: DMRG and MPS algorithms for condensed matter systems (High-Tc Superconductivity).
- **Cosmology**: N-body simulation interfaces and dark matter modeling.
- **Constants Validation**: Nanosecond-precision verification of physical constants against NIST data.

## Architecture

Designed to offload heavy computations to optimized C++/Rust kernels while maintaining a high-level Python interface for the Trinity Pipeline.
