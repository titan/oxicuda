//! Error types for `oxicuda-bayes`.

use thiserror::Error;

/// All errors that can be returned from `oxicuda-bayes`.
#[derive(Debug, Error, PartialEq)]
pub enum BayesError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty inputs provided")]
    EmptyInputs,

    #[error("invalid dropout rate {rate}: must be in [0, 1)")]
    InvalidDropoutRate { rate: f32 },

    #[error("invalid temperature {temp}: must be positive and finite")]
    InvalidTemperature { temp: f32 },

    #[error("invalid prior variance: must be positive and finite")]
    InvalidPriorVariance,

    #[error("non-positive sigma: sigma must be strictly positive")]
    NonPositiveSigma,

    #[error("insufficient samples: need at least {min}, got {got}")]
    InsufficientSamples { min: usize, got: usize },

    #[error("insufficient ensemble members: need at least {min}, got {got}")]
    InsufficientEnsembleMembers { min: usize, got: usize },

    #[error("calibration set is empty")]
    CalibrationSetEmpty,

    #[error("number of calibration bins is too small (must be >= 1)")]
    NCalibBinsTooSmall,

    #[error("isotonic regression: fitted function is not monotone")]
    IsotonicNotMonotone,

    #[error("Platt scaling fit failed to converge")]
    PlattFitFailed,

    #[error("temperature scaling produced a non-finite temperature")]
    TemperatureNotFinite,

    #[error("flow dimension mismatch between parameters and input")]
    FlowDimensionMismatch,

    #[error("NaN encountered at location: {location}")]
    NanEncountered { location: &'static str },

    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience alias.
pub type BayesResult<T> = Result<T, BayesError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = BayesError::DimensionMismatch {
            expected: 64,
            got: 32,
        };
        assert!(e.to_string().contains("64") && e.to_string().contains("32"));
    }

    #[test]
    fn error_display_invalid_dropout_rate() {
        let e = BayesError::InvalidDropoutRate { rate: 1.5 };
        assert!(e.to_string().contains("1.5"));
    }

    #[test]
    fn error_display_nan_encountered() {
        let e = BayesError::NanEncountered {
            location: "kl_gaussian",
        };
        assert!(e.to_string().contains("kl_gaussian"));
    }

    #[test]
    fn error_display_internal() {
        let e = BayesError::Internal("oops".into());
        assert!(e.to_string().contains("oops"));
    }

    #[test]
    fn error_equality_empty_inputs() {
        assert_eq!(BayesError::EmptyInputs, BayesError::EmptyInputs);
    }

    #[test]
    fn bayes_result_ok() {
        let r: BayesResult<i32> = Ok(42);
        assert!(r.is_ok());
    }

    #[test]
    fn bayes_result_err() {
        let r: BayesResult<i32> = Err(BayesError::CalibrationSetEmpty);
        assert!(r.is_err());
    }
}
