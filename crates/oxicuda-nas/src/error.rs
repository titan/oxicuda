//! Error types for `oxicuda-nas`.

use thiserror::Error;

/// All errors that can be returned from `oxicuda-nas`.
#[derive(Debug, Error, PartialEq)]
pub enum NasError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty search space")]
    EmptySearchSpace,

    #[error("invalid number of nodes: minimum {min}, got {got}")]
    InvalidNumNodes { min: usize, got: usize },

    #[error("invalid number of ops: must be > 0")]
    InvalidNumOps,

    #[error("invalid architecture encoding: length or op-index out of range")]
    InvalidArchEncoding,

    #[error("invalid weight shape")]
    InvalidWeightShape,

    #[error("no feasible architecture found")]
    NoFeasibleArchitecture,

    #[error("population too small: minimum {min}, got {got}")]
    PopulationTooSmall { min: usize, got: usize },

    #[error("invalid tournament size: must be >= 1 and <= population size")]
    InvalidTournamentSize,

    #[error("Pareto front is empty")]
    ParetoFrontEmpty,

    #[error("invalid width multiplier: {value} (must be in (0, 1])")]
    InvalidWidthMultiplier { value: f32 },

    #[error("invalid rank {rank}: dimension size is {dim}")]
    InvalidRank { rank: usize, dim: usize },

    #[error("latency model has not been fitted yet")]
    LatencyModelNotFitted,

    #[error("NaN detected in architecture parameters")]
    NanInArchParams,

    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience result alias for `oxicuda-nas`.
pub type NasResult<T> = Result<T, NasError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = NasError::DimensionMismatch {
            expected: 64,
            got: 32,
        };
        assert!(e.to_string().contains("64") && e.to_string().contains("32"));
    }

    #[test]
    fn error_display_empty_search_space() {
        let e = NasError::EmptySearchSpace;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn error_display_population_too_small() {
        let e = NasError::PopulationTooSmall { min: 10, got: 3 };
        assert!(e.to_string().contains("10") && e.to_string().contains("3"));
    }

    #[test]
    fn error_display_invalid_rank() {
        let e = NasError::InvalidRank { rank: 5, dim: 3 };
        assert!(e.to_string().contains("5") && e.to_string().contains("3"));
    }

    #[test]
    fn error_display_internal() {
        let e = NasError::Internal("something broke".into());
        assert!(e.to_string().contains("something broke"));
    }

    #[test]
    fn error_equality() {
        let a = NasError::EmptySearchSpace;
        let b = NasError::EmptySearchSpace;
        assert_eq!(a, b);
    }

    #[test]
    fn nas_result_ok() {
        let r: NasResult<i32> = Ok(42);
        assert!(r.is_ok());
    }

    #[test]
    fn nas_result_err() {
        let r: NasResult<i32> = Err(NasError::NanInArchParams);
        assert!(r.is_err());
    }
}
