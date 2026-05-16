use thiserror::Error;

#[derive(Debug, Error)]
pub enum TdaError {
    #[error("empty point cloud")]
    EmptyPointCloud,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("invalid simplex: {0}")]
    InvalidSimplex(String),
    #[error("simplex closure violated: {0}")]
    ClosureViolation(String),
    #[error("empty complex")]
    EmptyComplex,
    #[error("filtration not sorted")]
    FiltrationNotSorted,
    #[error("boundary matrix reduction failed")]
    ReductionFailed,
    #[error("invalid filtration value: NaN")]
    NanFiltrationValue,
    #[error("cover parameter invalid: {0}")]
    InvalidCoverParameter(String),
    #[error("landmark selection failed: {0}")]
    LandmarkSelectionFailed(String),
    #[error("matching failed: {0}")]
    MatchingFailed(String),
    #[error("parameter out of range: {0}")]
    ParameterOutOfRange(String),
    #[error("topological dimension {0} too large")]
    DimensionTooLarge(usize),
}

pub type TdaResult<T> = Result<T, TdaError>;
