use thiserror::Error;

#[derive(Debug, Error)]
pub enum RecsysError {
    #[error("empty input")]
    EmptyInput,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("invalid number of users: {n}")]
    InvalidNumUsers { n: usize },
    #[error("invalid number of items: {n}")]
    InvalidNumItems { n: usize },
    #[error("invalid embedding dim: {d}")]
    InvalidEmbeddingDim { d: usize },
    #[error("invalid loss weight: {w}")]
    InvalidLossWeight { w: f32 },
    #[error("invalid k={k} with n={n}")]
    InvalidK { k: usize, n: usize },
    #[error("no negative available for user {user}")]
    NoNegativeAvailable { user: usize },
    #[error("ALS did not converge after {iter} iterations")]
    AlsNotConverged { iter: usize },
    #[error("empty interaction data")]
    EmptyInteraction,
    #[error("unknown user id: {id}")]
    UnknownUser { id: usize },
    #[error("unknown item id: {id}")]
    UnknownItem { id: usize },
    #[error("model not fitted")]
    NotFitted,
    #[error("internal error: {msg}")]
    Internal { msg: String },
}

pub type RecsysResult<T> = std::result::Result<T, RecsysError>;
