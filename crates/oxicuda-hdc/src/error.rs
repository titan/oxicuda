use thiserror::Error;

#[derive(Debug, Error)]
pub enum HdcError {
    #[error("dimension must be > 0")]
    ZeroDimension,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("empty input")]
    EmptyInput,
    #[error("class {0} not found in classifier")]
    ClassNotFound(usize),
    #[error("item {0} not in item memory")]
    ItemNotFound(usize),
    #[error("invalid n-gram order: must be >= 1, got {0}")]
    InvalidNgramOrder(usize),
    #[error("binary value {0} not in {{-1, +1}}")]
    InvalidBinaryValue(i8),
    #[error("feature index {feat} out of range {max}")]
    FeatureIndexOutOfRange { feat: usize, max: usize },
    #[error("empty item memory — no items to query")]
    EmptyItemMemory,
    #[error("associative memory pattern dimension mismatch")]
    AssocDimensionMismatch,
    #[error("capacity exceeded: stored {stored}, capacity estimate {capacity}")]
    CapacityExceeded { stored: usize, capacity: usize },
    #[error("permutation length {perm_len} does not match dimension {dim}")]
    PermutationLengthMismatch { perm_len: usize, dim: usize },
    #[error("no prototype built — call build() first")]
    PrototypeNotBuilt,
    #[error("invalid probability: must be in (0, 1), got {0}")]
    InvalidProbability(f64),
    #[error("division by zero in metric computation")]
    DivisionByZero,
}

pub type HdcResult<T> = Result<T, HdcError>;
