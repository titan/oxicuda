//! Replay-based continual learning methods.
//!
//! These methods store and replay past experiences to prevent forgetting,
//! ranging from simple buffers to gradient-level episodic memory.

pub mod a_gem;
pub mod dark_exp;
pub mod dark_exp_v2;
pub mod der_plus_plus;
pub mod er;
pub mod gem;
pub mod vectorised_gem;

// ─── Vectorised GEM re-exports ────────────────────────────────────────────────
pub use vectorised_gem::{VectorisedGemConfig, vectorised_gem_project};

// ─── DER V2 re-exports ────────────────────────────────────────────────────────
pub use dark_exp_v2::{DerV2Buffer, DerV2Config};

// ─── DER++ re-exports ─────────────────────────────────────────────────────────
pub use der_plus_plus::DerPpLoss;
