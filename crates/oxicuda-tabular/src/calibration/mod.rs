//! Post-hoc probability calibration for tabular classifiers.
//!
//! Currently provides isotonic-regression calibration via the
//! Pool-Adjacent-Violators Algorithm (see [`isotonic`]). This complements the
//! parametric temperature scaling in [`crate::metrics::calibration`].

pub mod isotonic;

pub use isotonic::IsotonicCalibrator;
