//! Error types for `oxicuda-cvx`.

use thiserror::Error;

/// Top-level error type for convex optimization operations.
#[derive(Debug, Error)]
pub enum CvxError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("dimension mismatch: a={a}, b={b}")]
    DimensionMismatch { a: usize, b: usize },
    #[error("algorithm did not converge after {iter} iterations (residual={residual})")]
    NotConverged { iter: usize, residual: f64 },
    #[error("problem is infeasible: {0}")]
    Infeasible(String),
    #[error("problem is unbounded below: {0}")]
    Unbounded(String),
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
    #[error("line search failed: {0}")]
    LineSearchFailed(String),
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("cone violation: {0}")]
    ConeViolation(String),
}

/// Result alias for convex optimization operations.
pub type CvxResult<T> = Result<T, CvxError>;
