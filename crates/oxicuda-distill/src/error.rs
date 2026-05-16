use thiserror::Error;

/// Errors arising from knowledge-distillation operations.
#[derive(Debug, Error)]
pub enum DistillError {
    /// An input slice or collection was empty when a non-empty one was required.
    #[error("empty input")]
    EmptyInput,

    /// Two tensors/slices did not agree on their dimension.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// A configuration parameter was out of range or logically inconsistent.
    #[error("invalid config: {msg}")]
    InvalidConfig { msg: String },

    /// A floating-point operation produced NaN, Inf, or another illegal value.
    #[error("numerical error: {msg}")]
    NumericalError { msg: String },

    /// An unexpected internal error not covered by the above variants.
    #[error("internal error: {msg}")]
    Internal { msg: String },
}

/// Convenient alias used throughout the crate.
pub type DistillResult<T> = std::result::Result<T, DistillError>;
