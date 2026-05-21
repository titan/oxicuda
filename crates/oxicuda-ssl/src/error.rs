//! Error types for `oxicuda-ssl`.

use thiserror::Error;

/// All errors that can be returned from `oxicuda-ssl`.
#[derive(Debug, Error, PartialEq)]
pub enum SslError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty input")]
    EmptyInput,

    #[error("invalid temperature {temp}: must be > 0 and finite")]
    InvalidTemperature { temp: f32 },

    #[error("invalid momentum {momentum}: must be in [0, 1]")]
    InvalidMomentum { momentum: f32 },

    #[error("invalid mask ratio {ratio}: must be in [0, 1)")]
    InvalidMaskRatio { ratio: f32 },

    #[error("invalid number of crops: must be >= 1")]
    InvalidNumCrops,

    #[error("invalid loss weight {weight}: must be finite")]
    InvalidLossWeight { weight: f32 },

    #[error("queue capacity must be >= 1")]
    QueueCapacityTooSmall,

    #[error("queue is empty")]
    QueueEmpty,

    #[error("number of prototypes must be >= 2")]
    NumPrototypesTooSmall,

    #[error("Sinkhorn-Knopp diverged after {iters} iterations")]
    SinkhornDiverged { iters: usize },

    #[error("invalid feature dimension: must be > 0")]
    InvalidFeatureDim,

    #[error("invalid batch size: must be >= 2 (need positive + negative pairs)")]
    BatchTooSmall,

    #[error("non-finite value at: {location}")]
    NanEncountered { location: &'static str },

    #[error("invalid projector layer dim: in/hidden/out must be > 0")]
    InvalidProjectorDim,

    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter { name: String, reason: String },

    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience result alias.
pub type SslResult<T> = Result<T, SslError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = SslError::DimensionMismatch {
            expected: 64,
            got: 32,
        };
        assert!(e.to_string().contains("64") && e.to_string().contains("32"));
    }

    #[test]
    fn error_display_invalid_temperature() {
        let e = SslError::InvalidTemperature { temp: -1.0 };
        assert!(e.to_string().contains("-1"));
    }

    #[test]
    fn error_display_invalid_momentum() {
        let e = SslError::InvalidMomentum { momentum: 1.5 };
        assert!(e.to_string().contains("1.5"));
    }

    #[test]
    fn error_display_sinkhorn() {
        let e = SslError::SinkhornDiverged { iters: 100 };
        assert!(e.to_string().contains("100"));
    }

    #[test]
    fn error_display_internal() {
        let e = SslError::Internal("bad shape".into());
        assert!(e.to_string().contains("bad shape"));
    }

    #[test]
    fn error_equality() {
        assert_eq!(SslError::EmptyInput, SslError::EmptyInput);
    }

    #[test]
    fn ssl_result_ok() {
        let r: SslResult<i32> = Ok(42);
        assert!(r.is_ok());
    }

    #[test]
    fn ssl_result_err() {
        let r: SslResult<i32> = Err(SslError::InvalidNumCrops);
        assert!(r.is_err());
    }
}
