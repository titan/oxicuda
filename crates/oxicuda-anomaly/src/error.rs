//! Error types for `oxicuda-anomaly`.

/// Errors produced by anomaly detection operations.
#[derive(Debug, thiserror::Error)]
pub enum AnomalyError {
    /// Feature or representation dimension mismatch.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Empty slice passed where data is required.
    #[error("empty input")]
    EmptyInput,

    /// Attempt to score before calling `fit()`.
    #[error("model not fitted (call fit() first)")]
    NotFitted,

    /// Too few training samples for the requested operation.
    #[error("insufficient training samples: need {need}, got {got}")]
    InsufficientSamples { need: usize, got: usize },

    /// NaN detected in intermediate computation.
    #[error("NaN encountered in {context}")]
    NanEncountered { context: String },

    /// Invalid `k` for LOF / kNN.
    #[error("invalid k for LOF/kNN: {k}")]
    InvalidK { k: usize },

    /// Feature count is zero or otherwise invalid.
    #[error("invalid number of features: {n}")]
    InvalidFeatureCount { n: usize },

    /// Encoder/decoder layer spec is malformed.
    #[error("invalid layer dims: {msg}")]
    InvalidLayerDims { msg: String },

    /// Covariance matrix cannot be inverted (nearly singular).
    #[error("singular covariance matrix")]
    SingularCovariance,

    /// Percentile is outside `[0, 100]`.
    #[error("invalid threshold percentile: {p}")]
    InvalidThresholdPercentile { p: f32 },

    /// Feature count inconsistency between fit and score.
    #[error("feature count mismatch: expected {expected}, got {got}")]
    FeatureCountMismatch { expected: usize, got: usize },

    /// One-Class SVM `nu` parameter out of `(0, 1]`.
    #[error("invalid nu parameter for OCSVM: {nu}")]
    InvalidNu { nu: f32 },

    /// Iterative algorithm did not converge.
    #[error("convergence failed: {msg}")]
    ConvergenceFailed { msg: String },

    /// Internal logic error.
    #[error("internal error: {msg}")]
    Internal { msg: String },
}

/// Convenient `Result` alias for anomaly detection operations.
pub type AnomalyResult<T> = Result<T, AnomalyError>;
