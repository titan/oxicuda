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
//! ├── scs/                  — Splitting conic solver (operator-splitting, product cones)
//! ├── dcp/                  — Disciplined convex programming expression trees
//! ├── cut/                  — Kelley cutting-plane (outer approximation)
//! ├── admm/                 — Alternating Direction Method of Multipliers (ADMM, consensus)
//! ├── newton/               — Semismooth (generalized) Newton: LCP via Fischer-Burmeister
//! ├── splitting/            — Davis-Yin three-operator, Tseng forward-backward-forward
//! ├── proximal/             — Proximal gradient, FISTA, accelerated, Douglas-Rachford
//! ├── primal_dual/          — Chambolle-Pock primal-dual saddle-point algorithm
//! ├── prox_ops/             — Closed-form proximal operators (L1, L2, L∞, group lasso,
//! │                           elastic-net, nuclear, 1D-TV Condat, indicator)
//! ├── projection/           — Projections onto simplex, L1/L2 balls, box, PSD cone,
//! │                           SOC, halfspace
//! ├── augmented_lagrangian/ — Method of multipliers / ALM
//! ├── gradient/             — Projected GD, Nesterov accelerated GD, Polyak heavy-ball,
//! │                           Polyak step-size subgradient
//! ├── riemannian/           — Riemannian gradient descent (sphere, SPD, Stiefel)
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
pub mod constrained;
pub mod cut;
pub mod dcp;
pub mod error;
pub mod gradient;
pub mod handle;
pub mod linalg;
pub mod linesearch;
pub mod lp;
pub mod metrics;
pub mod newton;
pub mod primal_dual;
pub mod projection;
pub mod prox_ops;
pub mod proximal;
pub mod ptx_kernels;
pub mod qp;
pub mod riemannian;
pub mod scs;
pub mod sdp;
pub mod socp;
pub mod splitting;

pub use error::{CvxError, CvxResult};
pub use handle::{CvxHandle, LcgRng, SmVersion};

// Wave AAA+68 re-exports.
pub use admm::{AsyncAdmmConfig, AsyncAdmmResult, async_consensus_admm};
pub use gradient::{PolyakConfig, PolyakResult, PolyakTarget, polyak_subgradient};
pub use riemannian::{Manifold, RiemannianConfig, RiemannianResult, riemannian_gradient_descent};

// Wave AAA+78 re-exports.
pub use prox_ops::{max_value, prox_max};
pub use proximal::peaceman_rachford;
pub use socp::{MehrotraSocpConfig, MehrotraSocpResult, SocpStatus, mehrotra_socp};

// Wave AAA+84 re-exports: SCS conic solver, DCP expression trees, Kelley cuts.
pub use cut::{CuttingPlaneConfig, CuttingPlaneResult, CuttingPlaneStatus, kelley_cutting_plane};
pub use dcp::{Constraint, ConstraintKind, Curvature, Expr, Monotonicity, is_dcp};
pub use scs::{Cone, ScsConfig, ScsResult, ScsStatus, scs_solve};

// Semismooth Newton (LCP via Fischer-Burmeister) and operator-splitting
// (Davis-Yin three-operator, Tseng forward-backward-forward) re-exports.
pub use newton::{
    SemismoothConfig, SemismoothNewtonResult, SemismoothStatus, fischer_burmeister,
    fischer_burmeister_gradient, lcp_residual, semismooth_newton_lcp,
};
pub use splitting::{
    DavisYinConfig, DavisYinResult, DavisYinStatus, TsengConfig, TsengResult, TsengStatus,
    davis_yin_three_operator, tseng_fbf,
};

#[cfg(test)]
mod e2e_tests;
