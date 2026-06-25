//! Differentiable structured-prediction layers.
//!
//! Currently the [`sinkhorn_crf`] module: entropy-regularised optimal-transport
//! (Sinkhorn) normalisation as a differentiable replacement for sum-product
//! normalisation in matching / permutation structured prediction.

pub mod sinkhorn_crf;

pub use sinkhorn_crf::{
    SinkhornConfig, SinkhornCrf, SinkhornResult, sinkhorn_normalize,
    sinkhorn_normalize_with_margins,
};
