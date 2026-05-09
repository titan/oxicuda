//! Error types for `oxicuda-pinn`.

use thiserror::Error;

/// All errors that can be returned from `oxicuda-pinn`.
#[derive(Debug, Error, PartialEq)]
pub enum PinnError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty input")]
    EmptyInput,

    #[error("invalid step size {h}: must be > 0 and finite")]
    InvalidStepSize { h: f32 },

    #[error("invalid time interval [{t0}, {t1}]: must satisfy t1 > t0")]
    InvalidTimeInterval { t0: f32, t1: f32 },

    #[error("non-finite value encountered at: {location}")]
    NanEncountered { location: &'static str },

    #[error("invalid grid resolution {n}: must be >= 2")]
    InvalidGridResolution { n: usize },

    #[error("Fourier modes {k_max} must be <= n/2 = {n_half}")]
    TooManyFourierModes { k_max: usize, n_half: usize },

    #[error("layer width must be >= 1")]
    InvalidLayerWidth,

    #[error("invalid network depth {depth}: must be >= 2")]
    InvalidNetworkDepth { depth: usize },

    #[error("invalid loss weight {weight}: must be in [0, 1]")]
    InvalidWeight { weight: f32 },

    #[error("invalid activation: {0}")]
    InvalidActivation(&'static str),

    #[error("solver failed to converge: {reason}")]
    SolverDivergence { reason: &'static str },

    #[error("collocation set is empty")]
    EmptyCollocationSet,

    #[error("tape index {idx} out of range (size = {size})")]
    TapeIndexOutOfRange { idx: usize, size: usize },

    #[error("PDE coefficient {name} = {value}: must be > 0")]
    InvalidPdeCoefficient { name: &'static str, value: f32 },

    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience result alias.
pub type PinnResult<T> = Result<T, PinnError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = PinnError::DimensionMismatch {
            expected: 128,
            got: 64,
        };
        assert!(e.to_string().contains("128") && e.to_string().contains("64"));
    }

    #[test]
    fn error_display_empty_input() {
        let e = PinnError::EmptyInput;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn error_display_invalid_step_size() {
        let e = PinnError::InvalidStepSize { h: -0.1 };
        assert!(e.to_string().contains("-0.1"));
    }

    #[test]
    fn error_display_invalid_time_interval() {
        let e = PinnError::InvalidTimeInterval { t0: 1.0, t1: 0.5 };
        assert!(e.to_string().contains("1") && e.to_string().contains("0.5"));
    }

    #[test]
    fn error_display_nan_encountered() {
        let e = PinnError::NanEncountered {
            location: "rk4_step",
        };
        assert!(e.to_string().contains("rk4_step"));
    }

    #[test]
    fn error_display_invalid_grid_resolution() {
        let e = PinnError::InvalidGridResolution { n: 1 };
        assert!(e.to_string().contains("1"));
    }

    #[test]
    fn error_display_too_many_fourier_modes() {
        let e = PinnError::TooManyFourierModes {
            k_max: 10,
            n_half: 4,
        };
        assert!(e.to_string().contains("10") && e.to_string().contains("4"));
    }

    #[test]
    fn error_display_invalid_layer_width() {
        let e = PinnError::InvalidLayerWidth;
        assert!(e.to_string().contains("1"));
    }

    #[test]
    fn error_display_invalid_network_depth() {
        let e = PinnError::InvalidNetworkDepth { depth: 1 };
        assert!(e.to_string().contains("1"));
    }

    #[test]
    fn error_display_invalid_weight() {
        let e = PinnError::InvalidWeight { weight: 1.5 };
        assert!(e.to_string().contains("1.5"));
    }

    #[test]
    fn error_display_solver_divergence() {
        let e = PinnError::SolverDivergence {
            reason: "NaN encountered",
        };
        assert!(e.to_string().contains("NaN"));
    }

    #[test]
    fn error_display_empty_collocation_set() {
        let e = PinnError::EmptyCollocationSet;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn error_display_tape_index_out_of_range() {
        let e = PinnError::TapeIndexOutOfRange { idx: 5, size: 3 };
        assert!(e.to_string().contains("5") && e.to_string().contains("3"));
    }

    #[test]
    fn error_display_invalid_pde_coefficient() {
        let e = PinnError::InvalidPdeCoefficient {
            name: "alpha",
            value: -1.0,
        };
        assert!(e.to_string().contains("alpha") && e.to_string().contains("-1"));
    }

    #[test]
    fn error_display_internal() {
        let e = PinnError::Internal("unexpected shape".into());
        assert!(e.to_string().contains("unexpected shape"));
    }

    #[test]
    fn error_equality() {
        assert_eq!(PinnError::EmptyInput, PinnError::EmptyInput);
        assert_eq!(
            PinnError::EmptyCollocationSet,
            PinnError::EmptyCollocationSet
        );
        assert_eq!(PinnError::InvalidLayerWidth, PinnError::InvalidLayerWidth);
    }

    #[test]
    fn pinn_result_ok() {
        let r: PinnResult<i32> = Ok(42);
        assert!(r.is_ok());
    }

    #[test]
    fn pinn_result_err() {
        let r: PinnResult<i32> = Err(PinnError::EmptyInput);
        assert!(r.is_err());
    }
}
