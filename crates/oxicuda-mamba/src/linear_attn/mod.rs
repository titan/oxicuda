//! Linear-attention family: sub-quadratic attention alternatives.
//!
//! This module collects attention mechanisms whose cost is linear in the
//! sequence length, sharing the recurrent-state view with state-space models:
//!
//! - [`retnet`] — RetNet retention (Sun 2023): parallel / recurrent / chunkwise
//!   forms with multi-scale per-head decay `γ`.
//! - [`linear_attention`] — Causal linear attention (Katharopoulos 2020) with
//!   the `elu+1` feature map, plus gated linear attention (GLA, Yang 2023).

pub mod linear_attention;
pub mod retnet;

pub use linear_attention::{
    FeatureMap, LinearAttentionConfig, gated_linear_attention, linear_attention_parallel,
    linear_attention_recurrent,
};
pub use retnet::{
    RetentionConfig, RetentionState, msr_decays, retention_chunkwise, retention_parallel,
    retention_recurrent,
};
