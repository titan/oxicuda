use thiserror::Error;

#[derive(Debug, Error)]
pub enum CausalError {
    #[error("empty input")]
    EmptyInput,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("invalid graph size: {n}")]
    InvalidGraphSize { n: usize },
    #[error("graph contains a cycle")]
    CyclicGraph,
    #[error("not a DAG")]
    NotADag,
    #[error("incompatible data shapes")]
    IncompatibleData,
    #[error("invalid parameter: {reason}")]
    InvalidParameter { reason: String },
    #[error("propensity score out of bounds: {value}")]
    PropensityOutOfBounds { value: f32 },
    #[error("invalid number of folds: {k}")]
    InvalidNumFolds { k: usize },
    #[error("matrix is singular")]
    MatrixSingular,
    #[error("NOTEARS did not converge after {iter} iterations")]
    NotearsDidNotConverge { iter: usize },
    #[error("PC orientation failed")]
    PcOrientationFailed,
    #[error("causal effect not backdoor-identifiable")]
    BackdoorNotIdentifiable,
    #[error("model not fitted")]
    NotFitted,
    #[error("internal error: {msg}")]
    Internal { msg: String },
}

pub type CausalResult<T> = std::result::Result<T, CausalError>;
