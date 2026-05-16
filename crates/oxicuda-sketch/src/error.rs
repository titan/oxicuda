//! Error types for `oxicuda-sketch`.

use thiserror::Error;

/// Top-level error type for streaming sketch operations.
#[derive(Debug, Error)]
pub enum SketchError {
    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter { name: String, reason: String },
    #[error("empty stream / empty input")]
    EmptyStream,
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("dimension mismatch: a={a}, b={b}")]
    DimensionMismatch { a: usize, b: usize },
    #[error("unsupported SM version: {0}")]
    UnsupportedSmVersion(u32),
    #[error("capacity exceeded: capacity={cap}, attempted={attempted}")]
    CapacityExceeded { cap: usize, attempted: usize },
    #[error("index {index} out of bounds for length {len}")]
    IndexOutOfBounds { index: usize, len: usize },
    #[error("numerical instability: {0}")]
    NumericalInstability(String),
    #[error("hash table full after {tries} attempts")]
    HashTableFull { tries: usize },
    #[error("dimension must be a power of two: got {0}")]
    DimensionMustBePowerOfTwo(usize),
    #[error("invalid precision parameter: {0}")]
    InvalidPrecision(u32),
    #[error("algorithm did not converge after {iter} iterations")]
    NotConverged { iter: usize },
}

/// Result alias for sketch operations.
pub type SketchResult<T> = Result<T, SketchError>;
