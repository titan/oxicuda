//! Error types for `oxicuda-geom2d`.

use thiserror::Error;

/// Top-level error type for 2D computational geometry operations.
#[derive(Debug, Error)]
pub enum Geom2dError {
    #[error("degenerate polygon: {0}")]
    DegeneratePolygon(String),
    #[error("not enough points: need {needed}, got {got}")]
    NotEnoughPoints { needed: usize, got: usize },
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
    #[error("polygon is not convex: {0}")]
    NotConvex(String),
    #[error("polygon is not simple: {0}")]
    NotSimplePolygon(String),
    #[error("parallel segments: {0}")]
    ParallelSegments(String),
    #[error("collinear points: {0}")]
    CollinearPoints(String),
    #[error("zero-radius circle: {0}")]
    ZeroRadiusCircle(String),
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("algorithm did not converge after {iter} iterations")]
    NotConverged { iter: usize },
}

/// Result alias for 2D computational geometry operations.
pub type Geom2dResult<T> = Result<T, Geom2dError>;
