#[derive(Debug, thiserror::Error)]
pub enum MetaError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("empty support set")]
    EmptySupport,
    #[error("invalid n_way: {n_way} (must be >= 2)")]
    InvalidNWay { n_way: usize },
    #[error("invalid k_shot: {k_shot} (must be >= 1)")]
    InvalidKShot { k_shot: usize },
    #[error("invalid feature dimension: {dim}")]
    InvalidFeatDim { dim: usize },
    #[error("insufficient classes: need {need}, got {got}")]
    InsufficientClasses { need: usize, got: usize },
    #[error("insufficient examples for class {cls}: need {need}, got {got}")]
    InsufficientExamples { cls: usize, need: usize, got: usize },
    #[error("invalid learning rate: {lr}")]
    InvalidLr { lr: f32 },
    #[error("NaN encountered in {context}")]
    NanEncountered { context: String },
    #[error("invalid query size: {size}")]
    InvalidQuerySize { size: usize },
    #[error("invalid episode config: {msg}")]
    InvalidEpisodeConfig { msg: String },
    #[error("gradient computation failed: {msg}")]
    GradientFailure { msg: String },
    #[error("backbone error: {msg}")]
    BackboneError { msg: String },
    #[error("internal error: {msg}")]
    Internal { msg: String },
}

pub type MetaResult<T> = Result<T, MetaError>;
