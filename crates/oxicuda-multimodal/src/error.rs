//! Error types for `oxicuda-multimodal`.

use thiserror::Error;

/// All errors that can be returned from `oxicuda-multimodal`.
#[derive(Debug, Error, PartialEq)]
pub enum MultiModalError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty input")]
    EmptyInput,

    #[error("invalid temperature {temp}: must be > 0 and finite")]
    InvalidTemperature { temp: f32 },

    #[error("invalid number of heads {heads}: must divide d_model {d_model}")]
    InvalidHeads { heads: usize, d_model: usize },

    #[error("invalid batch size: must be >= 1")]
    InvalidBatchSize,

    #[error("mismatched sequence lengths: query {q_len}, key/value {kv_len} — kv_len must be >= 1")]
    MismatchedSeqLens { q_len: usize, kv_len: usize },

    #[error("invalid feature dimension: must be > 0")]
    InvalidFeatureDim,

    #[error("non-finite value encountered at: {location}")]
    NanEncountered { location: &'static str },

    #[error("token id {token_id} out of vocab range [0, {vocab_size})")]
    TokenOutOfRange { token_id: u32, vocab_size: usize },

    #[error("invalid number of modalities: got {n}, must be >= 2")]
    InvalidModalityCount { n: usize },

    #[error("invalid k_factor {k_factor}: must be >= 1")]
    InvalidKFactor { k_factor: usize },

    #[error("invalid number of patches: got {n_patches}, must be > 0")]
    InvalidPatchCount { n_patches: usize },

    #[error("layer count must be >= 1")]
    InvalidLayerCount,

    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience result alias.
pub type MmResult<T> = Result<T, MultiModalError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = MultiModalError::DimensionMismatch {
            expected: 64,
            got: 32,
        };
        assert!(e.to_string().contains("64") && e.to_string().contains("32"));
    }

    #[test]
    fn error_display_invalid_temperature() {
        let e = MultiModalError::InvalidTemperature { temp: -1.0 };
        assert!(e.to_string().contains("-1"));
    }

    #[test]
    fn error_display_invalid_heads() {
        let e = MultiModalError::InvalidHeads {
            heads: 3,
            d_model: 8,
        };
        assert!(e.to_string().contains("3") && e.to_string().contains("8"));
    }

    #[test]
    fn error_display_token_out_of_range() {
        let e = MultiModalError::TokenOutOfRange {
            token_id: 999,
            vocab_size: 100,
        };
        assert!(e.to_string().contains("999") && e.to_string().contains("100"));
    }

    #[test]
    fn error_display_mismatched_seq_lens() {
        let e = MultiModalError::MismatchedSeqLens {
            q_len: 10,
            kv_len: 0,
        };
        assert!(e.to_string().contains("10") && e.to_string().contains("0"));
    }

    #[test]
    fn error_display_internal() {
        let e = MultiModalError::Internal("bad shape".into());
        assert!(e.to_string().contains("bad shape"));
    }

    #[test]
    fn error_equality() {
        assert_eq!(MultiModalError::EmptyInput, MultiModalError::EmptyInput);
    }

    #[test]
    fn mm_result_ok() {
        let r: MmResult<i32> = Ok(42);
        assert!(r.is_ok());
    }

    #[test]
    fn mm_result_err() {
        let r: MmResult<i32> = Err(MultiModalError::InvalidBatchSize);
        assert!(r.is_err());
    }
}
