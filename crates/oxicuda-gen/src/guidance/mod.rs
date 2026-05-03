//! Guidance module for diffusion model inference.
//!
//! Provides classifier-free guidance (CFG), perpendicular negative guidance,
//! and adaptive schedule-based guidance for diffusion model inference.

pub mod adaptive;
pub mod cfg;
pub mod perp_neg;

pub use adaptive::{AdaptiveCfgPolicy, AdaptiveCfgScheduler};
pub use cfg::{CfgConfig, CfgGuidance};
pub use perp_neg::PerpNegGuidance;
