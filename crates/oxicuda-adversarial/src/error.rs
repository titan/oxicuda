//! Error types for `oxicuda-adversarial`.

use thiserror::Error;

/// All errors that can be returned from `oxicuda-adversarial`.
#[derive(Debug, Error, PartialEq)]
pub enum AdvError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty input")]
    EmptyInput,

    #[error("invalid epsilon {eps}: must be > 0 and finite")]
    InvalidEpsilon { eps: f32 },

    #[error("invalid alpha {alpha}: must be > 0 and finite")]
    InvalidAlpha { alpha: f32 },

    #[error("invalid number of steps: must be >= 1")]
    InvalidNumSteps,

    #[error("invalid Lp norm: only L1 / L2 / L_inf supported")]
    InvalidLpNorm,

    #[error("invalid temperature {temp}: must be > 0 and finite")]
    InvalidTemperature { temp: f32 },

    #[error("invalid noise sigma {sigma}: must be >= 0 and finite")]
    InvalidNoiseSigma { sigma: f32 },

    #[error("invalid certification confidence {alpha}: must be in (0, 1)")]
    InvalidConfidence { alpha: f32 },

    #[error("insufficient samples for certification: need {min}, got {got}")]
    InsufficientCertSamples { min: usize, got: usize },

    #[error("invalid loss weight {weight}: must be finite")]
    InvalidLossWeight { weight: f32 },

    #[error("budget exceeded: spent {spent}, max {max}")]
    BudgetExceeded { spent: f32, max: f32 },

    #[error("non-finite value at: {location}")]
    NanEncountered { location: &'static str },

    #[error("optimization diverged")]
    OptimizationDiverged,

    #[error("attack failed for all examples")]
    AttackFailedAll,

    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience result alias.
pub type AdvResult<T> = Result<T, AdvError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = AdvError::DimensionMismatch {
            expected: 64,
            got: 32,
        };
        assert!(e.to_string().contains("64") && e.to_string().contains("32"));
    }

    #[test]
    fn error_display_invalid_epsilon() {
        let e = AdvError::InvalidEpsilon { eps: -0.1 };
        assert!(e.to_string().contains("-0.1"));
    }

    #[test]
    fn error_display_budget_exceeded() {
        let e = AdvError::BudgetExceeded {
            spent: 1.5,
            max: 1.0,
        };
        assert!(e.to_string().contains("1.5") && e.to_string().contains("1"));
    }

    #[test]
    fn error_display_internal() {
        let e = AdvError::Internal("oops".into());
        assert!(e.to_string().contains("oops"));
    }

    #[test]
    fn error_equality() {
        assert_eq!(AdvError::EmptyInput, AdvError::EmptyInput);
    }

    #[test]
    fn adv_result_ok() {
        let r: AdvResult<i32> = Ok(42);
        assert!(r.is_ok());
    }

    #[test]
    fn adv_result_err() {
        let r: AdvResult<i32> = Err(AdvError::AttackFailedAll);
        assert!(r.is_err());
    }
}
