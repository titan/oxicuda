//! Hybrid forecasting architectures combining ideas from multiple models.
//!
//! * [`patch_cross`] — PatchTST × Crossformer hybrid: patch tokenisation
//!   (channel-independent) fused with Crossformer router-based cross-dimension
//!   attention for explicit variate mixing.

pub mod patch_cross;

pub use patch_cross::{PatchCrossConfig, PatchCrossformer};
