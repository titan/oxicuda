//! Forecasting head modules.
//!
//! Each head maps an encoded representation to a multi-step forecast vector.
//! Variants provided:
//!
//! - [`LinearHead`]: a single linear projection `[in_features] → [out_features]`.
//! - [`MlpHead`]: a two-layer MLP `in → hidden (ReLU) → out`.
//! - [`QuantileHead`]: quantile regression producing multiple quantile levels.
//! - [`DeepArHead`]: DeepAR-style autoregressive Gaussian decoder via stacked LSTM.

pub mod linear_head;
pub mod mlp_head;
pub mod prob_head;

pub use linear_head::LinearHead;
pub use mlp_head::MlpHead;
pub use prob_head::{
    DeepArConfig, DeepArHead, DeepArWeights, GaussianPrediction, LstmWeights, QuantileConfig,
    QuantileHead, QuantileHeadWeights, QuantilePrediction,
};
