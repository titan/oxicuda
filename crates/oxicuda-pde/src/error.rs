//! Error types for `oxicuda-pde`.

use thiserror::Error;

/// Top-level error type for numerical PDE operations.
#[derive(Debug, Error)]
pub enum PdeError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("dimension mismatch: a={a}, b={b}")]
    DimensionMismatch { a: usize, b: usize },
    #[error("algorithm did not converge after {iter} iterations (residual {residual})")]
    NotConverged { iter: usize, residual: f64 },
    #[error("empty mesh: {0}")]
    EmptyMesh(String),
    #[error("invalid grid configuration: {0}")]
    InvalidGrid(String),
    #[error("numerical instability: {0}")]
    NumericalInstability(String),
    #[error("unsupported SM version: {0}")]
    UnsupportedSmVersion(u32),
    #[error("invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },
    #[error("CFL stability violation: dt={dt} > dt_max={dt_max}")]
    CflViolation { dt: f64, dt_max: f64 },
    #[error("boundary condition missing for boundary {0}")]
    BoundaryConditionMissing(String),
    #[error("singular matrix encountered during {0}")]
    SingularMatrix(String),
    #[error("index {index} out of bounds for length {len}")]
    IndexOutOfBounds { index: usize, len: usize },
    #[error("invalid order {order}: must satisfy {reason}")]
    InvalidOrder { order: usize, reason: String },
    #[error("linear algebra failure: {0}")]
    LinearAlgebraFailure(String),
    #[error("unsupported degree {0}")]
    UnsupportedDegree(usize),
}

/// Result alias for numerical PDE operations.
pub type PdeResult<T> = Result<T, PdeError>;
