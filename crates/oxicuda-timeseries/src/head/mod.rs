//! Forecasting head modules.
//!
//! Each head maps an encoded representation to a multi-step forecast vector.
//! Two variants are provided:
//!
//! - [`LinearHead`]: a single linear projection `[in_features] → [out_features]`.
//! - [`MlpHead`]: a two-layer MLP `in → hidden (ReLU) → out`.

pub mod linear_head;
pub mod mlp_head;

pub use linear_head::LinearHead;
pub use mlp_head::MlpHead;
