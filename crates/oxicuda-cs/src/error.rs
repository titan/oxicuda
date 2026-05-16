//! Error types for `oxicuda-cs`.

use thiserror::Error;

/// Top-level error type for compressed-sensing operations.
#[derive(Debug, Error)]
pub enum CsError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("dimension mismatch: a={a}, b={b}")]
    DimensionMismatch { a: usize, b: usize },
    #[error("algorithm did not converge after {iter} iterations (residual={residual})")]
    NotConverged { iter: usize, residual: f64 },
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("numerical instability: {0}")]
    NumericalInstability(String),
    #[error("unsupported SM version: {0}")]
    UnsupportedSmVersion(u32),
    #[error("singular matrix encountered: {0}")]
    SingularMatrix(String),
    #[error("index {index} out of bounds for length {len}")]
    IndexOutOfBounds { index: usize, len: usize },
    #[error("empty input")]
    EmptyInput,
    #[error("support too large: requested {requested}, max {max}")]
    SupportTooLarge { requested: usize, max: usize },
    #[error("invalid sparsity level: {0}")]
    InvalidSparsity(usize),
    #[error("invalid rank: {0}")]
    InvalidRank(usize),
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("recovery failed: {0}")]
    RecoveryFailed(String),
}

/// Result alias for compressed-sensing operations.
pub type CsResult<T> = Result<T, CsError>;
