//! Error types for `oxicuda-continual`.

use thiserror::Error;

/// All errors that can be returned from `oxicuda-continual`.
#[derive(Debug, Error, PartialEq)]
pub enum ContinualError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty input")]
    EmptyInput,

    #[error("invalid lambda {lambda}: must be >= 0 and finite")]
    InvalidLambda { lambda: f32 },

    #[error("invalid sparsity fraction {fraction}: must be in [0, 1)")]
    InvalidSparsityFraction { fraction: f32 },

    #[error("invalid threshold {threshold}: must be finite")]
    InvalidThreshold { threshold: f32 },

    #[error("invalid momentum {momentum}: must be in [0, 1]")]
    InvalidMomentum { momentum: f32 },

    #[error("buffer capacity must be >= 1")]
    BufferCapacityTooSmall,

    #[error("buffer is empty — cannot sample")]
    BufferEmpty,

    #[error("requested batch size {requested} exceeds buffer size {available}")]
    BatchExceedsBuffer { requested: usize, available: usize },

    #[error("task index {index} out of range (n_tasks = {n_tasks})")]
    TaskIndexOutOfRange { index: usize, n_tasks: usize },

    #[error("no tasks in stream")]
    NoTasksInStream,

    #[error("column index {index} out of range (n_columns = {n_columns})")]
    ColumnIndexOutOfRange { index: usize, n_columns: usize },

    #[error("non-finite value encountered at: {location}")]
    NanEncountered { location: &'static str },

    #[error("n_layers must be >= 1")]
    InvalidNumLayers,

    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience result alias.
pub type ContinualResult<T> = Result<T, ContinualError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = ContinualError::DimensionMismatch {
            expected: 128,
            got: 64,
        };
        assert!(e.to_string().contains("128") && e.to_string().contains("64"));
    }

    #[test]
    fn error_display_invalid_lambda() {
        let e = ContinualError::InvalidLambda { lambda: -1.0 };
        assert!(e.to_string().contains("-1"));
    }

    #[test]
    fn error_display_invalid_sparsity() {
        let e = ContinualError::InvalidSparsityFraction { fraction: 1.5 };
        assert!(e.to_string().contains("1.5"));
    }

    #[test]
    fn error_display_buffer_empty() {
        let e = ContinualError::BufferEmpty;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn error_display_batch_exceeds_buffer() {
        let e = ContinualError::BatchExceedsBuffer {
            requested: 32,
            available: 10,
        };
        assert!(e.to_string().contains("32") && e.to_string().contains("10"));
    }

    #[test]
    fn error_display_task_out_of_range() {
        let e = ContinualError::TaskIndexOutOfRange {
            index: 5,
            n_tasks: 3,
        };
        assert!(e.to_string().contains("5") && e.to_string().contains("3"));
    }

    #[test]
    fn error_display_nan_encountered() {
        let e = ContinualError::NanEncountered {
            location: "ewc_loss",
        };
        assert!(e.to_string().contains("ewc_loss"));
    }

    #[test]
    fn error_display_internal() {
        let e = ContinualError::Internal("unexpected shape".into());
        assert!(e.to_string().contains("unexpected shape"));
    }

    #[test]
    fn error_equality() {
        assert_eq!(ContinualError::EmptyInput, ContinualError::EmptyInput);
        assert_eq!(ContinualError::BufferEmpty, ContinualError::BufferEmpty);
    }

    #[test]
    fn continual_result_ok() {
        let r: ContinualResult<i32> = Ok(42);
        assert!(r.is_ok());
    }

    #[test]
    fn continual_result_err() {
        let r: ContinualResult<i32> = Err(ContinualError::EmptyInput);
        assert!(r.is_err());
    }
}
