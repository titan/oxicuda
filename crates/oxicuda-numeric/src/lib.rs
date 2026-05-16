//! `oxicuda-numeric` — Numerical Analysis primitives for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-numeric
//! ├── root/        — Root finding (bisection, Newton, secant, Brent, Halley, Aberth)
//! ├── quadrature/  — Quadrature: Romberg, Gauss-Legendre/Hermite/Laguerre/Chebyshev,
//! │                  Clenshaw-Curtis, adaptive Simpson, Gauss-Kronrod
//! ├── special/     — Special functions: Bessel J/Y/I/K, Airy, Lambert W,
//! │                  hypergeometric 2F1, elliptic K/E, zeta, dilogarithm, Ei, polygamma
//! ├── ode/         — ODE solvers: Euler, Heun, RK4, DOPRI5, BDF1/2, Rosenbrock-W, IMEX
//! ├── poly/        — Polynomial roots: Durand-Kerner, Jenkins-Traub, companion matrix,
//! │                  Horner evaluation, polynomial deflation
//! ├── diff/        — Numerical differentiation: central diff, Richardson, complex step
//! ├── interp/      — Interpolation: linear, cubic spline, Akima, PCHIP, Lagrange,
//! │                  Hermite, barycentric Lagrange
//! ├── cubature/    — Multi-D cubature: Monte Carlo, quasi-MC Sobol, tensor-product Gauss,
//! │                  Genz-Malik adaptive
//! ├── linalg/      — Linalg helpers (private): Jacobi eig, QR Givens, LU, Householder QR
//! └── metrics/     — Relative error, condition number, residual diagnostics
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]

pub mod cubature;
pub mod diff;
pub mod error;
pub mod handle;
pub mod interp;
pub mod linalg;
pub mod metrics;
pub mod ode;
pub mod poly;
pub mod ptx_kernels;
pub mod quadrature;
pub mod root;
pub mod special;

pub use error::{NumericError, NumericResult};
pub use handle::{LcgRng, NumericHandle, SmVersion};

#[cfg(test)]
mod e2e_tests;
