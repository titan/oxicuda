//! Error types for `oxicuda-tn`.

use thiserror::Error;

/// Top-level error type for tensor network operations.
#[derive(Debug, Error)]
pub enum TnError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("dimension mismatch: a={a}, b={b}")]
    DimensionMismatch { a: usize, b: usize },
    #[error("algorithm did not converge after {iter} iterations")]
    NotConverged { iter: usize },
    #[error("invalid bond dimension: {0}")]
    InvalidBondDimension(usize),
    #[error("empty input")]
    EmptyInput,
    #[error("index {index} out of bounds for length {len}")]
    IndexOutOfBounds { index: usize, len: usize },
    #[error("linear algebra failure: {0}")]
    LinearAlgebraFailure(String),
    #[error("numerical instability: {0}")]
    NumericalInstability(String),
    #[error("invalid rank: {0}")]
    InvalidRank(usize),
    #[error("unsupported SM version: {0}")]
    UnsupportedSmVersion(u32),
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("rank {rank} exceeds limit {max}")]
    RankExceedsLimit { rank: usize, max: usize },
    #[error("invalid contraction path: {0}")]
    ContractionPathInvalid(String),
}

/// Result alias for tensor network operations.
pub type TnResult<T> = Result<T, TnError>;
