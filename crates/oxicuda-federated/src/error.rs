//! Error types for `oxicuda-federated`.

use thiserror::Error;

/// All errors that can be returned from `oxicuda-federated`.
#[derive(Debug, Error, PartialEq)]
pub enum FedError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty client list")]
    EmptyClientList,

    #[error("insufficient clients: need at least {min}, got {got}")]
    InsufficientClients { min: usize, got: usize },

    #[error("invalid weight {weight}: must be non-negative and finite")]
    InvalidWeight { weight: f32 },

    #[error("client weights do not sum to 1.0")]
    WeightSumNotOne,

    #[error("invalid proximal mu: must be positive and finite")]
    InvalidProximalMu,

    #[error("invalid privacy budget: epsilon and delta must be positive")]
    InvalidPrivacyBudget,

    #[error("invalid noise multiplier: must be positive and finite")]
    InvalidNoiseMultiplier,

    #[error("invalid clip norm: must be positive and finite")]
    InvalidClipNorm,

    #[error("invalid quantization levels: must be >= 1")]
    InvalidQuantizationLevels,

    #[error("threshold {threshold} is too large for {parties} parties")]
    ThresholdTooLarge { threshold: usize, parties: usize },

    #[error("invalid share count: need at least {min}, got {got}")]
    InvalidShareCount { min: usize, got: usize },

    #[error("reconstruction failed: shares are inconsistent or insufficient")]
    ReconstructionFailed,

    #[error("invalid rank {rank}: must be less than min(m,n)={dim}")]
    InvalidRank { rank: usize, dim: usize },

    #[error("invalid client utility: stat_utility and sys_utility must be positive and finite")]
    InvalidClientUtility,

    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience alias.
pub type FedResult<T> = Result<T, FedError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = FedError::DimensionMismatch {
            expected: 64,
            got: 32,
        };
        assert!(e.to_string().contains("64") && e.to_string().contains("32"));
    }

    #[test]
    fn error_display_empty_client_list() {
        let e = FedError::EmptyClientList;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn error_display_insufficient_clients() {
        let e = FedError::InsufficientClients { min: 3, got: 1 };
        assert!(e.to_string().contains("3") && e.to_string().contains("1"));
    }

    #[test]
    fn error_display_invalid_weight() {
        let e = FedError::InvalidWeight { weight: -1.0 };
        assert!(e.to_string().contains("-1"));
    }

    #[test]
    fn error_display_threshold_too_large() {
        let e = FedError::ThresholdTooLarge {
            threshold: 5,
            parties: 3,
        };
        assert!(e.to_string().contains("5") && e.to_string().contains("3"));
    }

    #[test]
    fn error_display_invalid_rank() {
        let e = FedError::InvalidRank { rank: 10, dim: 5 };
        assert!(e.to_string().contains("10") && e.to_string().contains("5"));
    }

    #[test]
    fn error_display_internal() {
        let e = FedError::Internal("oops".into());
        assert!(e.to_string().contains("oops"));
    }

    #[test]
    fn error_equality() {
        let a = FedError::EmptyClientList;
        let b = FedError::EmptyClientList;
        assert_eq!(a, b);
    }

    #[test]
    fn fed_result_ok() {
        let r: FedResult<i32> = Ok(42);
        assert!(r.is_ok());
    }

    #[test]
    fn fed_result_err() {
        let r: FedResult<i32> = Err(FedError::EmptyClientList);
        assert!(r.is_err());
    }
}
