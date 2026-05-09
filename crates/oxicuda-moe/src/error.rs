//! Error types for `oxicuda-moe`.

/// All errors that can be returned from `oxicuda-moe`.
#[derive(Debug, thiserror::Error)]
pub enum MoeError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty input")]
    EmptyInput,

    #[error("invalid number of experts: {n_experts}")]
    InvalidExpertCount { n_experts: usize },

    #[error("invalid top-k: {k} must be <= n_experts {n_experts}")]
    InvalidTopK { k: usize, n_experts: usize },

    #[error("invalid capacity factor: {factor}")]
    InvalidCapacityFactor { factor: f32 },

    #[error("expert index out of range: {idx} >= {n_experts}")]
    ExpertIndexOutOfRange { idx: usize, n_experts: usize },

    #[error("NaN encountered in {context}")]
    NanEncountered { context: String },

    #[error("invalid hidden dimension: {dim}")]
    InvalidHiddenDim { dim: usize },

    #[error("invalid input dimension: {dim}")]
    InvalidInputDim { dim: usize },

    #[error("dispatch failed: {msg}")]
    DispatchFailed { msg: String },

    #[error("router not initialized")]
    RouterNotInitialized,

    #[error("expert FFN error: {msg}")]
    ExpertFfnError { msg: String },

    #[error("slot assignment error: {msg}")]
    SlotAssignmentError { msg: String },

    #[error("internal error: {msg}")]
    Internal { msg: String },
}

/// Convenience result alias.
pub type MoeResult<T> = Result<T, MoeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = MoeError::DimensionMismatch {
            expected: 128,
            got: 64,
        };
        assert!(e.to_string().contains("128") && e.to_string().contains("64"));
    }

    #[test]
    fn error_display_empty_input() {
        let e = MoeError::EmptyInput;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn error_display_invalid_expert_count() {
        let e = MoeError::InvalidExpertCount { n_experts: 0 };
        assert!(e.to_string().contains('0'));
    }

    #[test]
    fn error_display_invalid_top_k() {
        let e = MoeError::InvalidTopK { k: 5, n_experts: 4 };
        assert!(e.to_string().contains('5') && e.to_string().contains('4'));
    }

    #[test]
    fn error_display_nan_encountered() {
        let e = MoeError::NanEncountered {
            context: "softmax".to_string(),
        };
        assert!(e.to_string().contains("softmax"));
    }
}
