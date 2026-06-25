//! Riemannian optimisation on tensor-network manifolds.
//!
//! The foundational object is the **fixed-rank matrix manifold** `M_r`, the set of
//! `m × n` real matrices of exact rank `r`. It is the local building block of every
//! fixed-rank tensor-network format (a TT / MPS unfolding is a fixed-rank matrix), so a
//! correct, self-contained geometry for `M_r` is the natural first-class citizen here.
//!
//! See [`riemannian_tn`] for the manifold geometry (tangent-space projection, metric
//! projection retraction via truncated SVD, Riemannian gradient) and the
//! gradient-descent / conjugate-gradient solvers built on top of it.
//!
//! ## References
//!
//! - Vandereycken (2013). *Low-rank matrix completion by Riemannian optimization.*
//!   SIAM J. Optim. 23(2), 1214–1236.
//! - Absil, Mahony & Sepulchre (2008). *Optimization Algorithms on Matrix Manifolds.*
//!   Princeton University Press.

pub mod riemannian_tn;

pub use riemannian_tn::{
    FixedRankManifold, RiemannianTn, RiemannianTnConfig, RiemannianTnMethod, TnPoint, TnResultData,
    eckart_young_objective, low_rank_completion_egrad, low_rank_completion_objective,
    low_rank_egrad, low_rank_objective,
};
