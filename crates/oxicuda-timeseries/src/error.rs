//! Error types for `oxicuda-timeseries`.

use thiserror::Error;

/// All errors produced by `oxicuda-timeseries`.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum TsError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("shape mismatch: {msg}")]
    ShapeMismatch { msg: String },

    #[error("empty input: {msg}")]
    EmptyInput { msg: String },

    #[error("invalid sequence length: {0}")]
    InvalidSequenceLength(usize),

    #[error("invalid number of variates: {0}")]
    InvalidNumVariates(usize),

    #[error("invalid patch length: {0}")]
    InvalidPatchLen(usize),

    #[error("invalid stride: {0}")]
    InvalidStride(usize),

    #[error("invalid kernel size: {0}")]
    InvalidKernelSize(usize),

    #[error("invalid dilation: {0}")]
    InvalidDilation(usize),

    #[error("invalid number of heads: {0}")]
    InvalidNumHeads(usize),

    #[error("head dimension mismatch: embed_dim={embed_dim} not divisible by n_heads={n_heads}")]
    HeadDimMismatch { embed_dim: usize, n_heads: usize },

    #[error("invalid embed dim: {0}")]
    InvalidEmbedDim(usize),

    #[error("invalid forecast horizon: {0}")]
    InvalidHorizon(usize),

    #[error("invalid pool size: {0}")]
    InvalidPoolSize(usize),

    #[error("invalid number of periods: top_k={0} exceeds FFT length")]
    InvalidTopK(usize),

    #[error("weight shape mismatch: {msg}")]
    WeightShapeMismatch { msg: String },

    #[error("non-finite value encountered")]
    NonFinite,

    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience alias.
pub type TsResult<T> = Result<T, TsError>;
