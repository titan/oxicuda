//! Error types for `oxicuda-mamba`.

use thiserror::Error;

/// Errors returned by `oxicuda-mamba` operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum MambaError {
    /// Tensor dimension does not match the expected value.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Shape mismatch between two tensors.
    #[error("shape mismatch: lhs {lhs:?} vs rhs {rhs:?}")]
    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },

    /// The input slice or sequence is empty.
    #[error("empty input: {0}")]
    EmptyInput(&'static str),

    /// Sequence length is zero or invalid.
    #[error("invalid sequence length: {0}")]
    InvalidSeqLen(usize),

    /// State-space order `N` is zero or invalid.
    #[error("invalid SSM order N={0}")]
    InvalidSsmOrder(usize),

    /// Model dimension `D` is zero or invalid.
    #[error("invalid model dimension D={0}")]
    InvalidModelDim(usize),

    /// The requested number of layers is zero.
    #[error("invalid layer count: {0}")]
    InvalidLayerCount(usize),

    /// Vocabulary size is zero.
    #[error("invalid vocab size: {0}")]
    InvalidVocabSize(usize),

    /// Token index exceeds vocabulary size.
    #[error("token id {id} out of vocab range {vocab_size}")]
    TokenOutOfVocab { id: usize, vocab_size: usize },

    /// Delta discretization step is non-positive.
    #[error("non-positive discretization step delta={0}")]
    NonPositiveDelta(f32),

    /// Chunk size is zero or not a power of two.
    #[error("invalid chunk size: {0}")]
    InvalidChunkSize(usize),

    /// Number of attention/SSM heads is incompatible with dimension.
    #[error("head count {n_heads} does not divide model dim {d_model}")]
    HeadDimMismatch { n_heads: usize, d_model: usize },

    /// Weight tensor has wrong shape.
    #[error("weight shape mismatch for '{name}': expected {expected:?}, got {got:?}")]
    WeightShapeMismatch {
        name: &'static str,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// NaN or infinity encountered in intermediate values.
    #[error("non-finite value encountered: {0}")]
    NonFinite(&'static str),

    /// Internal logic error (should not occur in correct usage).
    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience alias for `Result<T, MambaError>`.
pub type MambaResult<T> = Result<T, MambaError>;

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = MambaError::DimensionMismatch {
            expected: 4,
            got: 8,
        };
        assert!(e.to_string().contains("4"));
        assert!(e.to_string().contains("8"));
    }

    #[test]
    fn error_display_empty_input() {
        let e = MambaError::EmptyInput("u tensor");
        assert!(e.to_string().contains("u tensor"));
    }

    #[test]
    fn error_display_token_out_of_vocab() {
        let e = MambaError::TokenOutOfVocab {
            id: 100,
            vocab_size: 50,
        };
        assert!(e.to_string().contains("100"));
        assert!(e.to_string().contains("50"));
    }

    #[test]
    fn error_display_weight_shape_mismatch() {
        let e = MambaError::WeightShapeMismatch {
            name: "A",
            expected: vec![64, 16],
            got: vec![64, 32],
        };
        let s = e.to_string();
        assert!(s.contains("A"));
        assert!(s.contains("[64, 16]") || s.contains("64"));
    }

    #[test]
    fn error_display_non_positive_delta() {
        let e = MambaError::NonPositiveDelta(-0.01);
        assert!(e.to_string().contains("-0.01") || e.to_string().contains("non-positive"));
    }

    #[test]
    fn error_display_head_dim_mismatch() {
        let e = MambaError::HeadDimMismatch {
            n_heads: 3,
            d_model: 64,
        };
        let s = e.to_string();
        assert!(s.contains("3") && s.contains("64"));
    }

    #[test]
    fn error_display_non_finite() {
        let e = MambaError::NonFinite("hidden state h");
        assert!(e.to_string().contains("hidden state h"));
    }

    #[test]
    fn error_display_internal() {
        let e = MambaError::Internal("unexpected branch".into());
        assert!(e.to_string().contains("unexpected branch"));
    }

    #[test]
    fn error_clone_eq() {
        let a = MambaError::InvalidSsmOrder(0);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn result_alias_ok() {
        fn make_ok() -> MambaResult<u32> {
            Ok(42)
        }
        assert_eq!(make_ok().expect("ok result"), 42);
    }

    #[test]
    fn result_alias_err() {
        let r: MambaResult<u32> = Err(MambaError::EmptyInput("test"));
        assert!(r.is_err());
    }
}
