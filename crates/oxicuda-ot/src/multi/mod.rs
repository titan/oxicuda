//! Multi-marginal optimal transport via tensor scaling.
//!
//! Generalises the standard 2-marginal OT to `k ≥ 2` marginals. Solves
//!
//! ```text
//! min_T  ⟨C, T⟩ + ε · KL(T ‖ ⊗_i a^{(i)})
//!       s.t.   π_i T = a^{(i)},  i = 1, …, k
//! ```
//!
//! by alternately enforcing each marginal with axis-wise log-sum-exp updates.
//! For `k = 2` this reduces exactly to the standard log-domain Sinkhorn-Knopp
//! algorithm.

/// Multi-marginal OT with structured pairwise-separable cost.
pub mod mmot_structured;
/// Tensor-scaling multi-marginal OT in log-domain.
pub mod multi_marginal;

pub use mmot_structured::{
    MmotBaryConfig, MmotStructuredConfig, MmotStructuredResult, mmot_barycenter, mmot_structured,
};
