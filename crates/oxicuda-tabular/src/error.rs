//! Error types for `oxicuda-tabular`.

/// All errors that can arise in tabular deep learning operations.
#[derive(Debug, thiserror::Error)]
pub enum TabularError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty input")]
    EmptyInput,

    #[error("invalid number of features: {n}")]
    InvalidFeatureCount { n: usize },

    #[error("invalid number of steps: {steps}")]
    InvalidStepCount { steps: usize },

    #[error("invalid attention dim: {dim}")]
    InvalidAttentionDim { dim: usize },

    #[error("invalid embedding dim: {dim}")]
    InvalidEmbedDim { dim: usize },

    #[error("NaN encountered in {context}")]
    NanEncountered { context: String },

    #[error("invalid tree depth: {depth}")]
    InvalidTreeDepth { depth: usize },

    #[error("invalid number of trees: {n}")]
    InvalidTreeCount { n: usize },

    #[error("normalization failed: {msg}")]
    NormalizationFailed { msg: String },

    #[error("categorical out of range: feature {feat}, value {val}, n_categories {n}")]
    CategoricalOutOfRange { feat: usize, val: usize, n: usize },

    #[error("label out of range: {label} >= {n_classes}")]
    LabelOutOfRange { label: usize, n_classes: usize },

    #[error("insufficient samples: need {need}, got {got}")]
    InsufficientSamples { need: usize, got: usize },

    #[error("invalid parameter {name}: {msg}")]
    InvalidParameter { name: String, msg: String },

    #[error("internal error: {msg}")]
    Internal { msg: String },
}

/// Convenience alias for `Result<T, TabularError>`.
pub type TabularResult<T> = Result<T, TabularError>;
