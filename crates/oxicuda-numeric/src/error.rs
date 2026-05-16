//! Error types for `oxicuda-numeric`.

use thiserror::Error;

/// Top-level error type for numerical analysis operations.
#[derive(Debug, Error)]
pub enum NumericError {
    #[error("algorithm did not converge after {iter} iterations (residual={residual})")]
    NotConverged { iter: usize, residual: f64 },
    #[error("root is not bracketed on interval [{a}, {b}] (f(a)={fa}, f(b)={fb})")]
    RootNotBracketed { a: f64, b: f64, fa: f64, fb: f64 },
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("numerical instability: {0}")]
    NumericalInstability(String),
    #[error("unsupported SM version: {0}")]
    UnsupportedSmVersion(u32),
    #[error("index {index} out of bounds for length {len}")]
    IndexOutOfBounds { index: usize, len: usize },
    #[error("dimension mismatch: a={a}, b={b}")]
    DimensionMismatch { a: usize, b: usize },
    #[error("empty input")]
    EmptyInput,
    #[error("polynomial degree {degree} exceeds supported limit {limit}")]
    DegreeTooHigh { degree: usize, limit: usize },
    #[error("argument {value} is out of domain for {function}")]
    OutOfDomain { value: f64, function: String },
    #[error("singular matrix encountered: {0}")]
    SingularMatrix(String),
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("step size {step} is non-positive or non-finite")]
    InvalidStepSize { step: f64 },
}

/// Result alias for numerical analysis operations.
pub type NumericResult<T> = Result<T, NumericError>;
