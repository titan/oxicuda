//! Evaluation metrics for continual learning.
//!
//! Provides standard continual learning metrics computed from
//! the accuracy matrix (performance on each task after each training step).

pub mod forgetting;
pub mod intransigence;
pub mod verification;

// ─── Verification utilities re-exports ────────────────────────────────────────
pub use verification::{
    DerSensitivityCell, FisherComparison, GemConvergence, der_sensitivity_grid,
    gaussian_fisher_comparison, gem_convergence_profile,
};
