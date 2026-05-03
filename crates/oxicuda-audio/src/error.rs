//! Error types for `oxicuda-audio`.

use thiserror::Error;

/// All errors that can be returned from `oxicuda-audio`.
#[derive(Debug, Error, PartialEq)]
pub enum AudioError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("shape mismatch: {msg}")]
    ShapeMismatch { msg: String },

    #[error("empty input: {msg}")]
    EmptyInput { msg: String },

    #[error("invalid number of mel bins: {0} (must be > 0)")]
    InvalidNumMels(usize),

    #[error("invalid sequence length: {0} (must be > 0)")]
    InvalidSequenceLength(usize),

    #[error("invalid embed dimension: {0} (must be > 0)")]
    InvalidEmbedDim(usize),

    #[error("invalid number of attention heads: {0} (must be > 0)")]
    InvalidNumHeads(usize),

    #[error(
        "head dimension mismatch: embed_dim {embed_dim} must be divisible by n_heads {n_heads}"
    )]
    HeadDimMismatch { embed_dim: usize, n_heads: usize },

    #[error("invalid vocabulary size: {0} (must be > 0)")]
    InvalidVocabSize(usize),

    #[error("invalid beam width: {0} (must be > 0)")]
    InvalidBeamWidth(usize),

    #[error("invalid dilation: {0} (must be > 0)")]
    InvalidDilation(usize),

    #[error("invalid kernel size: {0} (must be > 0)")]
    InvalidKernelSize(usize),

    #[error("invalid stride: {0} (must be > 0)")]
    InvalidStride(usize),

    #[error("blank index {blank} out of range for vocabulary size {vocab}")]
    BlankOutOfRange { blank: usize, vocab: usize },

    #[error("weight shape mismatch: {msg}")]
    WeightShapeMismatch { msg: String },

    #[error("non-finite value encountered: {msg}")]
    NonFinite { msg: String },

    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience alias.
pub type AudioResult<T> = Result<T, AudioError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = AudioError::DimensionMismatch {
            expected: 64,
            got: 32,
        };
        assert!(e.to_string().contains("64") && e.to_string().contains("32"));
    }

    #[test]
    fn error_display_shape_mismatch() {
        let e = AudioError::ShapeMismatch {
            msg: "bad shape".into(),
        };
        assert!(e.to_string().contains("bad shape"));
    }

    #[test]
    fn error_display_blank_out_of_range() {
        let e = AudioError::BlankOutOfRange { blank: 5, vocab: 4 };
        assert!(e.to_string().contains("5") && e.to_string().contains("4"));
    }

    #[test]
    fn error_display_head_dim_mismatch() {
        let e = AudioError::HeadDimMismatch {
            embed_dim: 64,
            n_heads: 7,
        };
        assert!(e.to_string().contains("64") && e.to_string().contains("7"));
    }

    #[test]
    fn error_display_internal() {
        let e = AudioError::Internal("oops".into());
        assert!(e.to_string().contains("oops"));
    }

    #[test]
    fn error_equality() {
        let a = AudioError::InvalidNumMels(0);
        let b = AudioError::InvalidNumMels(0);
        assert_eq!(a, b);
    }

    #[test]
    fn audio_result_ok() {
        let r: AudioResult<i32> = Ok(42);
        assert!(r.is_ok());
    }

    #[test]
    fn audio_result_err() {
        let r: AudioResult<i32> = Err(AudioError::InvalidBeamWidth(0));
        assert!(r.is_err());
    }
}
