use thiserror::Error;

#[derive(Debug, Error)]
pub enum PrivacyError {
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("empty input")]
    EmptyInput,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("budget exhausted: spent {spent:.4} of {total:.4}")]
    BudgetExhausted { spent: f64, total: f64 },
    #[error("sensitivity must be positive, got {0}")]
    NonPositiveSensitivity(f64),
    #[error("epsilon must be positive, got {0}")]
    NonPositiveEpsilon(f64),
    #[error("delta must be in (0,1), got {0}")]
    InvalidDelta(f64),
    #[error("score index {0} out of range {1}")]
    IndexOutOfRange(usize, usize),
    #[error("convergence failed after {0} iterations")]
    ConvergenceFailed(usize),
    #[error("PRV convolution requires at least 1 mechanism")]
    EmptyMechanismList,
    #[error("SVT query limit exceeded: asked {asked}, limit {limit}")]
    SvtQueryLimitExceeded { asked: usize, limit: usize },
    #[error("tree depth {0} exceeds maximum supported {1}")]
    TreeDepthExceeded(usize, usize),
}

pub type PrivacyResult<T> = Result<T, PrivacyError>;
