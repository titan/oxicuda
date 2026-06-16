//! `oxicuda-pde` — Numerical PDE solvers for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-pde
//! ├── mesh/        — 1D / 2D structured meshes and triangulations
//! ├── fdm/         — Finite-difference: Poisson (1D/2D), heat (1D/2D), wave, advection
//! ├── fem/         — P1 linear-triangle finite-element assembly + Dirichlet apply
//! ├── spectral/    — Chebyshev collocation, periodic pseudo-spectral (DFT)
//! ├── time/        — Forward/backward Euler, Crank-Nicolson, RK4, BDF2, IMEX
//! ├── multigrid/   — Geometric V-cycle (1D & 2D), restriction & prolongation
//! ├── bc/          — Dirichlet, Neumann, Robin boundary-condition helpers
//! ├── solver/      — CG, PCG (Jacobi / ILU(0) / SSOR), Jacobi, sparse CSR
//! ├── dg/          — Discontinuous Galerkin in 1D (LGL nodal basis, upwind / Lax-Friedrichs, BR2 elliptic)
//! ├── amr/         — Adaptive mesh refinement: quadtree octree + gradient/jump error estimator
//! ├── pde/         — Interface tracking: level-set method (upwind HJ + reinitialisation)
//! ├── pde_apps/    — Application solvers: Cahn-Hilliard phase field, wave equation, WENO5 advection
//! └── metrics/     — L2/H1/max norms and convergence-order estimation
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]

pub mod amr;
pub mod bc;
pub mod dg;
pub mod error;
pub mod fdm;
pub mod fem;
pub mod handle;
pub mod mesh;
pub mod metrics;
pub mod multigrid;
pub mod pde;
pub mod pde_apps;
pub mod ptx_kernels;
pub mod solver;
pub mod spectral;
pub mod time;

pub use amr::{
    Aabb, CHILDREN_PER_CELL, Cell, Indicators, MarkedCells, Quadtree, dorfler_mark,
    gradient_indicator, jump_indicator, threshold_mark,
};
pub use dg::{BR2_FACES_PER_ELEMENT, Br2Elliptic, DEFAULT_BR2_PENALTY};
pub use error::{PdeError, PdeResult};
pub use handle::{LcgRng, PdeHandle, SmVersion};
pub use pde::LevelSet;
pub use pde_apps::{
    AdvDiffBoundary, AdvDiffBoundary2d, AdvectionDiffusion1d, AdvectionDiffusion2d, CahnHilliard,
    CahnHilliard2d, DEFAULT_STABILIZATION, DeltaKernel, ImmersedBoundary, Maxwell1d, Maxwell2dTm,
    MaxwellBoundary1d, MaxwellBoundary2d, MaxwellState1d, MaxwellState2dTm, WENO5_EPS,
    WENO5_IDEAL_WEIGHTS, WaveBoundary, WaveEquation, WaveState, Weno5Advection,
    weno5_reconstruct_left, weno5_weights,
};
pub use solver::{
    AmgPcgConfig, AmgPcgResult, AmgPreconditioner, EigenPair, LanczosConfig, LanczosResult, Which,
    amg_pcg, lanczos, lanczos_csr,
};
pub use spectral::{
    Fourier3dConfig, GllBasis, SpectralElementMesh1d, gll_nodes, gll_weights,
    neg_laplacian_3d_spectral, solve_poisson_3d_fft,
};

#[cfg(test)]
mod e2e_tests;
