//! `oxicuda-numeric` — Numerical Analysis primitives for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-numeric
//! ├── root/        — Root finding (bisection, Newton, secant, Brent, Halley, Aberth)
//! ├── quadrature/  — Quadrature: Romberg, Gauss-Legendre/Hermite/Laguerre/Chebyshev,
//! │                  Clenshaw-Curtis, adaptive Simpson, Gauss-Kronrod, tanh-sinh,
//! │                  Smolyak sparse grid (nested Clenshaw-Curtis)
//! ├── special/     — Special functions: Bessel J/Y/I/K, Airy, Lambert W, Wright ω,
//! │                  hypergeometric 2F1, elliptic K/E, zeta, dilogarithm, Ei, polygamma
//! ├── ode/         — ODE solvers: Euler, Heun, RK4, DOPRI5, BDF1/2, Rosenbrock-W, IMEX,
//! │                  Adams-Bashforth-Moulton (orders 1–4), Radau IIA (order 5),
//! │                  SDIRK (order 3, L-stable), index-1 DAE (backward Euler)
//! ├── bvp/         — Two-point boundary-value problems: single shooting (secant on the
//! │                  initial slope), central finite differences (tridiagonal Newton)
//! ├── lsq/         — Least-squares: Levenberg-Marquardt (analytic + numerical Jacobian)
//! ├── nonlinear/   — Nonlinear systems & optimisation: Newton-Krylov (JFNK/GMRES),
//! │                  Broyden quasi-Newton, BFGS minimiser
//! ├── poly/        — Polynomial roots: Durand-Kerner, Jenkins-Traub, companion matrix,
//! │                  Horner evaluation, polynomial deflation
//! ├── diff/        — Numerical differentiation: central diff, Richardson, complex step
//! ├── interp/      — Interpolation: linear, cubic spline, Akima, PCHIP, Lagrange,
//! │                  Hermite, barycentric Lagrange, Floater-Hormann barycentric
//! │                  rational, Chebyshev series, RBF (scattered)
//! ├── approx/      — Function approximation: Padé rational approximants from Taylor series
//! ├── cubature/    — Multi-D cubature: Monte Carlo, quasi-MC Sobol, tensor-product Gauss,
//! │                  Genz-Malik adaptive
//! ├── series/       — Sequence acceleration: Aitken Δ², Wynn ε-algorithm (Shanks)
//! ├── linalg/      — Linalg helpers (private): Jacobi eig, QR Givens, LU, Householder QR
//! └── metrics/     — Relative error, condition number, residual diagnostics
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]

pub mod approx;
pub mod bvp;
pub mod cubature;
pub mod diff;
pub mod error;
pub mod handle;
pub mod interp;
pub mod linalg;
pub mod lsq;
pub mod metrics;
pub mod nonlinear;
pub mod ode;
pub mod poly;
pub mod ptx_kernels;
pub mod quadrature;
pub mod root;
pub mod series;
pub mod special;

pub use approx::PadeApprox;
pub use bvp::{
    FiniteDifferenceConfig, FiniteDifferenceSolution, ShootingConfig, ShootingSolution,
    solve_finite_difference, solve_shooting,
};
pub use error::{NumericError, NumericResult};
pub use handle::{LcgRng, NumericHandle, SmVersion};
pub use interp::barycentric_rational::FloaterHormann;
pub use nonlinear::{
    BfgsConfig, BfgsResult, CgConfig, CgResult, CgVariant, LbfgsConfig, LbfgsResult,
    NelderMeadConfig, NelderMeadResult, NewtonKrylovConfig, bfgs_minimize, bfgs_minimize_numerical,
    broyden, conjugate_gradient_minimize, conjugate_gradient_minimize_numerical, lbfgs_minimize,
    lbfgs_minimize_numerical, nelder_mead, newton_krylov,
};
pub use ode::dae::{DaeConfig, DaeSolution, DaeSolver};
pub use ode::radau_iia::{RadauConfig, RadauIia};
pub use ode::sdirk::{Sdirk, SdirkConfig};
pub use quadrature::sparse_grid::{SparseGrid, smolyak_integrate, smolyak_integrate_unit};
pub use series::acceleration::{
    WynnEpsilon, aitken_accelerate, aitken_sequence, aitken_step, shanks, wynn_epsilon,
};
pub use special::wright_omega::wright_omega;

#[cfg(test)]
mod e2e_tests;
