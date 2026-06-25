//! Error and result types for backend operations.

use std::fmt;

/// Error type for backend operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// The requested operation is not supported by this backend.
    Unsupported(String),
    /// A GPU/device error occurred.
    DeviceError(String),
    /// Invalid argument to an operation.
    InvalidArgument(String),
    /// Out of device memory.
    OutOfMemory,
    /// Backend not initialized — call [`ComputeBackend::init`](crate::ComputeBackend::init) first.
    NotInitialized,
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(msg) => write!(f, "unsupported operation: {msg}"),
            Self::DeviceError(msg) => write!(f, "device error: {msg}"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::OutOfMemory => write!(f, "out of device memory"),
            Self::NotInitialized => write!(f, "backend not initialized"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Result type for backend operations.
pub type BackendResult<T> = Result<T, BackendError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_error_display() {
        assert_eq!(
            BackendError::Unsupported("foo".into()).to_string(),
            "unsupported operation: foo"
        );
        assert_eq!(
            BackendError::DeviceError("bar".into()).to_string(),
            "device error: bar"
        );
        assert_eq!(
            BackendError::InvalidArgument("baz".into()).to_string(),
            "invalid argument: baz"
        );
        assert_eq!(
            BackendError::OutOfMemory.to_string(),
            "out of device memory"
        );
        assert_eq!(
            BackendError::NotInitialized.to_string(),
            "backend not initialized"
        );
    }

    #[test]
    fn backend_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BackendError::DeviceError("test".into()));
        assert!(err.to_string().contains("test"));
    }
}
