//! Error types for `oxicuda-manifold`.

use thiserror::Error;

/// Top-level error type for manifold-learning operations.
#[derive(Debug, Error)]
pub enum ManifoldError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("dimension mismatch: a={a}, b={b}")]
    DimensionMismatch { a: usize, b: usize },
    #[error("algorithm did not converge after {iter} iterations")]
    NotConverged { iter: usize },
    #[error("empty input")]
    EmptyInput,
    #[error("invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },
    #[error("eigendecomposition failure: {0}")]
    EigenFailure(String),
    #[error("numerical instability: {0}")]
    NumericalInstability(String),
    #[error("unsupported SM version: {0}")]
    UnsupportedSmVersion(u32),
    #[error("requested k={k} neighbors exceeds population n={n}")]
    KNeighborsTooLarge { k: usize, n: usize },
    #[error("singular matrix: {0}")]
    SingularMatrix(String),
    #[error("index {index} out of bounds for length {len}")]
    IndexOutOfBounds { index: usize, len: usize },
    #[error("manifold constraint violated: {0}")]
    ManifoldConstraint(String),
    #[error("graph disconnected: component count = {0}")]
    DisconnectedGraph(usize),
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
}

/// Result alias for manifold-learning operations.
pub type ManifoldResult<T> = Result<T, ManifoldError>;
