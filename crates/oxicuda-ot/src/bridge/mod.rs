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
