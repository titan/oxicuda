use thiserror::Error;

#[derive(Debug, Error)]
pub enum RlhfError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty input")]
    EmptyInput,

    #[error("invalid beta: {beta}")]
    InvalidBeta { beta: f32 },

    #[error("invalid temperature: {temp}")]
    InvalidTemp { temp: f32 },

    #[error("NaN encountered")]
    NanEncountered,

    #[error("invalid lambda: {lambda}")]
    InvalidLambda { lambda: f32 },

    #[error("log-probs required for this loss")]
    LogProbsRequired,

    #[error("mismatched pair length: chosen {chosen}, rejected {rejected}")]
    MismatchedPairLength { chosen: usize, rejected: usize },

    #[error("invalid margin: {margin}")]
    InvalidMargin { margin: f32 },

    #[error("KL divergence error: {msg}")]
    KlDivergence { msg: String },

    #[error("invalid reference log-prob")]
    InvalidReferenceLogProb,

    #[error("reward normalization failed: {msg}")]
    RewardNormFailed { msg: String },

    #[error("invalid mask value (must be 0 or 1)")]
    InvalidMaskValue,

    #[error("no valid preference pair could be synthesized: {msg}")]
    NoValidPair { msg: String },

    #[error("internal error: {msg}")]
    Internal { msg: String },
}

pub type RlhfResult<T> = Result<T, RlhfError>;
