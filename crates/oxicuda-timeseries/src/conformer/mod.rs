//! Conformer-TS: convolution-augmented Transformer block adapted for
//! time-series forecasting (Gulati et al. 2020, INTERSPEECH).
//!
//! Combines macaron feed-forward modules, multi-head self-attention, and a
//! causal convolution module (pointwise → GLU → depthwise causal conv →
//! LayerNorm → SiLU → pointwise) into a forecasting backbone over a
//! time-major `[T, C]` layout.

pub mod conformer;

pub use conformer::{ConformerBlock, ConformerConfig, ConformerEncoder};
