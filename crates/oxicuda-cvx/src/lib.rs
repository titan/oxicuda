//! `oxicuda-cvx` — Convex Optimization for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-cvx
//! ├── lp/                   — Linear programming (revised simplex, Mehrotra primal-dual IPM)
//! ├── qp/                   — Quadratic programming (active-set, primal-dual IPM)
//! ├── socp/                 — Second-order cone programming (primal-dual IPM)
//! ├── sdp/                  — Semidefinite programming (interior point, log-det barrier)
//! ├── admm/                 — Alternating Direction Method of Multipliers (ADMM, consensus)
//! ├── proximal/             — Proximal gradient, FISTA, accelerated, Douglas-Rachford
//! ├── primal_dual/          — Chambolle-Pock primal-dual saddle-point algorithm
//! ├── prox_ops/             — Closed-form proximal operators (L1, L2, L∞, group lasso,
//! │                           elastic-net, nuclear, 1D-TV Condat, indicator)
//! ├── projection/           — Projections onto simplex, L1/L2 balls, box, PSD cone,
//! │                           SOC, halfspace
//! ├── augmented_lagrangian/ — Method of multipliers / ALM
//! ├── gradient/             — Projected GD, Nesterov accelerated GD, Polyak heavy-ball
//! ├── linesearch/           — Armijo, Wolfe, strong Wolfe, backtracking
//! ├── linalg/               — CG, matvec, Cholesky, QR (Householder), triangular solves
//! └── metrics/              — Duality gap, primal/dual residual, KKT residual, rates
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::useless_vec)]

pub mod admm;
pub mod augmented_lagrangian;
pub mod error;
pub mod gradient;
pub mod handle;
pub mod linalg;
pub mod linesearch;
pub mod lp;
pub mod metrics;
pub mod primal_dual;
pub mod projection;
pub mod prox_ops;
pub mod proximal;
pub mod ptx_kernels;
pub mod qp;
pub mod sdp;
pub mod socp;

pub use error::{CvxError, CvxResult};
pub use handle::{CvxHandle, LcgRng, SmVersion};

#[cfg(test)]
mod e2e_tests;
