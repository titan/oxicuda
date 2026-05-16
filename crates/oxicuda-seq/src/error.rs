//! Error types for `oxicuda-seq`.

use thiserror::Error;

/// Errors that can be produced by sequence model algorithms.
#[derive(Debug, Error)]
pub enum SeqError {
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("dimension mismatch between operands: a={a}, b={b}")]
    DimensionMismatch { a: usize, b: usize },

    #[error("algorithm did not converge after {iter} iterations")]
    NotConverged { iter: usize },

    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("empty input")]
    EmptyInput,

    #[error("index {index} out of bounds (len = {len})")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("numerical instability detected: {0}")]
    NumericalInstability(String),

    #[error("probability {0} is out of [0, 1]")]
    ProbabilityOutOfRange(f64),

    #[error("invalid parameter `{name}`: value = {value}")]
    InvalidParameter { name: String, value: f64 },

    #[error("unsupported SM version: {0}")]
    UnsupportedSmVersion(u32),

    #[error("invalid observation: {0}")]
    InvalidObservation(String),

    #[error("length mismatch: a={a}, b={b}")]
    LengthMismatch { a: usize, b: usize },

    #[error("graph invariant violated: {0}")]
    GraphInvariantViolated(String),
}

/// Result alias for fallible sequence operations.
pub type SeqResult<T> = Result<T, SeqError>;
