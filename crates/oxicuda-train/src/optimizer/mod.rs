//! CPU-side optimizer implementations.
//!
//! These optimizers operate on host-side `f32` slices and complement the
//! GPU-resident optimizers in [`crate::gpu_optimizer`].

/// AdaBelief optimizer adapting step sizes by belief in gradients (Zhuang et al., 2020).
pub mod adabelief;

/// ADOPT adaptive gradient optimizer (Taniguchi et al., 2024).
pub mod adopt;

/// LAMB layer-wise adaptive moments optimizer (You et al., 2019 / Ginsburg 2019).
pub mod lamb;

/// Lookahead optimizer wrapper with slow weights (Zhang et al., 2019).
pub mod lookahead;

/// Muon optimizer with Newton-Schulz orthogonalization (Jordan et al., 2024).
pub mod muon;

/// Sharpness-Aware Minimization wrapper (Foret et al., 2021).
pub mod sam;

pub use adabelief::{AdaBelief, AdaBeliefConfig};
pub use adopt::{Adopt, AdoptConfig};
pub use lamb::{Lamb, LambConfig};
pub use lookahead::{Lookahead, LookaheadConfig};
pub use muon::{Muon, MuonConfig};
pub use sam::{Sam, SamConfig};
