//! OT-based domain adaptation via barycentric mapping.
//!
//! Given a source dataset `X ∈ ℝ^{m × dim}` and a target dataset
//! `Y ∈ ℝ^{n × dim}`, the entropic OT plan `P` between empirical distributions
//! defines a soft assignment that we collapse into a barycentric map
//!
//! ```text
//! T(x_i) = Σ_j (P_ij / Σ_k P_ik) · y_j
//! ```
//!
//! mapping each source sample to a convex combination of target samples
//! (Courty et al., *Optimal Transport for Domain Adaptation*, IEEE TPAMI
//! 2017). The mapping is differentiable and preserves cluster structure.

/// Barycentric mapping for OT-based domain adaptation.
pub mod mapping;
