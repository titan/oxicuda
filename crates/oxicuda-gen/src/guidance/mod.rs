//! Guidance module for diffusion model inference.
//!
//! Provides classifier-free guidance (CFG), perpendicular negative guidance,
//! adaptive schedule-based guidance, and classifier gradient guidance for
//! diffusion model inference.

pub mod adaptive;
pub mod cfg;
pub mod classifier_guidance;
pub mod perp_neg;

pub use adaptive::{AdaptiveCfgPolicy, AdaptiveCfgScheduler, PolynomialFit};
pub use cfg::{CfgConfig, CfgGuidance};
pub use classifier_guidance::{ClassifierGuidance, ClassifierGuidanceConfig};
pub use perp_neg::PerpNegGuidance;
