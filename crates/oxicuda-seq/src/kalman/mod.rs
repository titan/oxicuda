//! Linear / Extended Kalman filtering, RTS smoothing, and EM parameter learning.

pub mod ekf;
pub mod kalman_em;
pub mod kalman_filter;
pub mod linalg;
pub mod rts_smoother;

pub use ekf::{ExtendedKalmanFilter, ExtendedKalmanResult};
pub use kalman_em::{KalmanEmConfig, kalman_em};
pub use kalman_filter::{KalmanFilter, KalmanResult};
pub use rts_smoother::{RtsResult, rts_smoother};
