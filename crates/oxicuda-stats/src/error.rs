//! Error types for `oxicuda-stats`.

use thiserror::Error;

/// Top-level error type for statistical operations.
#[derive(Debug, Error)]
pub enum StatsError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("dimension mismatch: a={a}, b={b}")]
    DimensionMismatch { a: usize, b: usize },
    #[error("algorithm did not converge after {iter} iterations (residual {residual})")]
    NotConverged { iter: usize, residual: f64 },
    #[error("empty input")]
    EmptyInput,
    #[error("invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },
    #[error("numerical instability: {0}")]
    NumericalInstability(String),
    #[error("unsupported SM version: {0}")]
    UnsupportedSmVersion(u32),
    #[error("insufficient sample size: got {got}, need at least {need}")]
    InsufficientSampleSize { got: usize, need: usize },
    #[error("degrees of freedom must be > 0")]
    DegreesOfFreedomZero,
    #[error("probability {value} out of range [0, 1]")]
    ProbabilityOutOfRange { value: f64 },
    #[error("singular matrix encountered during {0}")]
    SingularMatrix(String),
    #[error("index {index} out of bounds for length {len}")]
    IndexOutOfBounds { index: usize, len: usize },
    #[error("invalid distribution parameter: {0}")]
    InvalidDistributionParameter(String),
    #[error("linear algebra failure: {0}")]
    LinearAlgebraFailure(String),
    #[error("data contains non-finite value at index {0}")]
    NonFiniteValue(usize),
    #[error("rank deficient design matrix")]
    RankDeficient,
}

/// Result alias for statistical operations.
pub type StatsResult<T> = Result<T, StatsError>;
