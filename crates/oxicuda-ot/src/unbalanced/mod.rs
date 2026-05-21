//! KL-relaxed unbalanced optimal transport and partial OT with TV/L2 relaxations.
//!
//! Lifts the hard marginal equality constraints `T 1 = a, Tᵀ 1 = b` of
//! standard OT to soft KL penalties of strength `τ_a` and `τ_b`. As `τ → ∞`
//! the unbalanced solution converges to the standard balanced Sinkhorn plan.
//!
//! Additional partial OT variants with TV and L2 relaxations are provided in
//! the `partial_ot` sub-module.

/// Partial OT with TV and L2 marginal relaxations.
pub mod partial_ot;
/// KL-relaxed unbalanced optimal transport.
pub mod unbalanced_ot;

pub use partial_ot::{PartialOtConfig, PartialOtResult, UnbalancedRelaxation, partial_ot};
