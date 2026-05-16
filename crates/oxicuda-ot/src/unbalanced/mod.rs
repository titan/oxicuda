//! KL-relaxed unbalanced optimal transport.
//!
//! Lifts the hard marginal equality constraints `T 1 = a, Tᵀ 1 = b` of
//! standard OT to soft KL penalties of strength `τ_a` and `τ_b`. As `τ → ∞`
//! the unbalanced solution converges to the standard balanced Sinkhorn plan.

/// KL-relaxed unbalanced optimal transport.
pub mod unbalanced_ot;
