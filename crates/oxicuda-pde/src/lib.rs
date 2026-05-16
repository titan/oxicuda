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
//! ├── dg/          — Discontinuous Galerkin in 1D (LGL nodal basis, upwind / Lax-Friedrichs)
//! └── metrics/     — L2/H1/max norms and convergence-order estimation
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]

pub mod bc;
pub mod dg;
pub mod error;
pub mod fdm;
pub mod fem;
pub mod handle;
pub mod mesh;
pub mod metrics;
pub mod multigrid;
pub mod ptx_kernels;
pub mod solver;
pub mod spectral;
pub mod time;

pub use error::{PdeError, PdeResult};
pub use handle::{LcgRng, PdeHandle, SmVersion};

#[cfg(test)]
mod e2e_tests;
