//! CPU-side optimizer implementations.
//!
//! These optimizers operate on host-side `f32` slices and complement the
//! GPU-resident optimizers in [`crate::gpu_optimizer`].

/// AdaBelief optimizer adapting step sizes by belief in gradients (Zhuang et al., 2020).
pub mod adabelief;

/// Adafactor sublinear-memory adaptive optimizer (Shazeer & Stern, 2018).
pub mod adafactor;

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

/// Shampoo preconditioned tensor optimizer (Gupta et al., 2018).
pub mod shampoo;

/// Sophia scalable stochastic second-order optimizer (Liu et al., 2023).
pub mod sophia;

pub use adabelief::{AdaBelief, AdaBeliefConfig};
pub use adafactor::{Adafactor, AdafactorConfig};
pub use adopt::{Adopt, AdoptConfig};
pub use lamb::{Lamb, LambConfig};
pub use lookahead::{Lookahead, LookaheadConfig};
pub use muon::{Muon, MuonConfig};
pub use sam::{Sam, SamConfig};
pub use shampoo::{Shampoo, ShampooConfig};
pub use sophia::{Sophia, SophiaConfig};
