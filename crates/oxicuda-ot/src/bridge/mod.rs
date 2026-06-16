//! Schrödinger Bridge / Iterative Proportional Fitting (IPF) — entropic OT
//! between two prescribed marginals.
//!
//! Solves the static Schrödinger problem
//!
//! ```text
//! min_P  KL(P ‖ K)        s.t.  P 1 = a,  Pᵀ 1 = b
//! ```
//!
//! where `K_ij = exp(−C_ij/ε)` is the Gibbs kernel of the cost matrix `C`. The
//! solution coincides with the Sinkhorn-Knopp transport plan but the IPF
//! formulation lends itself to *direct* alternating projections without the
//! optimal-transport vocabulary.

/// Iterative-Proportional-Fitting Schrödinger Bridge in log-domain.
pub mod schrodinger;

/// Time-Dependent Schrödinger Bridge via log-domain IPF over space-time path.
pub mod tdsb;

// ─── Re-exports ───────────────────────────────────────────────────────────────

pub use tdsb::{TdsbConfig, TdsbResult, tdsb, tdsb_interpolate, tdsb_transition_plan};

/// Conditional Flow Matching (Lipman 2022, Liu 2023) — simulation-free generative model.
pub mod flow_matching;
pub use flow_matching::{
    CfmConfig, CfmFit, CouplingStrategy, VelocityNet, conditional_flow_matching,
    conditional_velocity, flow_interpolate, flow_straightness,
};
