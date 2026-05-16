//! Error types for `oxicuda-survival`.

use thiserror::Error;

/// Top-level error type for survival analysis operations.
#[derive(Debug, Error)]
pub enum SurvivalError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("dimension mismatch: a={a}, b={b}")]
    DimensionMismatch { a: usize, b: usize },
    #[error("algorithm did not converge after {iter} iterations")]
    NotConverged { iter: usize },
    #[error("empty dataset")]
    EmptyDataset,
    #[error("no events in dataset")]
    NoEvents,
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("numerical instability: {0}")]
    NumericalInstability(String),
    #[error("unsupported SM version: {0}")]
    UnsupportedSmVersion(u32),
    #[error("negative time encountered: {0}")]
    NegativeTime(f64),
    #[error("singular matrix encountered")]
    SingularMatrix,
    #[error("index {index} out of bounds for length {len}")]
    IndexOutOfBounds { index: usize, len: usize },
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("linear algebra failure: {0}")]
    LinearAlgebraFailure(String),
}

/// Result alias for survival operations.
pub type SurvivalResult<T> = Result<T, SurvivalError>;
