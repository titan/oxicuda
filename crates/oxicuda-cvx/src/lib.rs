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
pub mod builder;
pub mod constrained;
pub mod cut;
pub mod dcp;
pub mod differentiable;
pub mod error;
pub mod gradient;
pub mod handle;
pub mod linalg;
pub mod linesearch;
pub mod lp;
pub mod metrics;
pub mod newton;
pub mod primal_dual;
pub mod problem;
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

// Differentiable convex optimization (OptNet: differentiating through the KKT system).
pub use differentiable::{OptNetConfig, OptNetLayer, QpParamGrads, QpProblem, QpSolution};

// Algorithmic-deepening re-exports: adaptive-ρ ADMM (Boyd residual balancing),
// FISTA with adaptive restart (O'Donoghue-Candès), and the golden-ratio
// primal-dual algorithm (Chang-Yang GRPDA).
pub use admm::{AdaptiveRhoConfig, AdaptiveRhoResult, adaptive_rho_admm};
pub use primal_dual::{GOLDEN_RATIO, GrpdaConfig, GrpdaResult, grpda};
pub use proximal::{FistaRestartConfig, FistaRestartResult, RestartRule, fista_restart};

// ── API polish: fluent LP builder and a unified problem-dispatch trait. ──────
pub use builder::{LpMethod, LpSolution, LpSolverBuilder};
// Note: `problem::QpProblem` (the standard interior-point QP form) is *not*
// re-exported here because the crate root already exposes the OptNet
// `differentiable::QpProblem`.  Reach it as `oxicuda_cvx::problem::QpProblem`.
pub use problem::{
    LpProblem, ProblemForm, ProblemSolution, ProblemSpec, SdpProblem, SocpProblem, solve,
};

pub use projection::{
    project_box, project_halfspace, project_l1_ball, project_l2_ball, project_psd_cone,
    project_simplex, project_soc,
};
pub use prox_ops::indicator::{
    prox_indicator_l1_ball, prox_indicator_l2_ball, prox_indicator_simplex,
};
/// Ergonomic crate-root re-export of the scalar soft-threshold operator
/// (the prox of `λ|·|`), reachable as `oxicuda_cvx::soft_threshold` instead of
/// `oxicuda_cvx::prox_ops::l1::soft_threshold`.  The companion vector prox /
/// projection operators are re-exported alongside it.
///
/// ```
/// // Soft-thresholding shrinks toward zero by λ.
/// assert_eq!(oxicuda_cvx::soft_threshold(2.0, 0.5), 1.5);
/// assert_eq!(oxicuda_cvx::soft_threshold(-2.0, 0.5), -1.5);
/// assert_eq!(oxicuda_cvx::soft_threshold(0.3, 0.5), 0.0);
///
/// // Vector L1 prox is element-wise soft-thresholding.
/// let v = oxicuda_cvx::prox_l1(&[2.0, 0.5, -0.5, -2.0], 1.0).unwrap();
/// assert_eq!(v, vec![1.0, 0.0, 0.0, -1.0]);
///
/// // Projection onto the probability simplex sums to 1.
/// let p = oxicuda_cvx::project_simplex(&[3.0, 1.0, 2.0], 1.0).unwrap();
/// assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// ```
pub use prox_ops::soft_threshold;
pub use prox_ops::{
    prox_elastic_net, prox_group_lasso, prox_indicator_box, prox_l1, prox_l2, prox_linf,
    prox_nuclear, prox_tv_1d,
};

#[cfg(test)]
mod e2e_tests;

/// On-device GPU validation tests (feature-gated): JIT-compile each hand-written
/// PTX kernel, launch it on a real CUDA device, and assert numerical equivalence
/// to the matching CPU reference. Compiled only under `--features gpu-tests` and
/// only in test builds; every test skips gracefully if no GPU is available.
#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests;
