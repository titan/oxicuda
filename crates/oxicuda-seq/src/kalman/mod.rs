//! Linear / Extended Kalman filtering, RTS smoothing, EM parameter learning,
//! Unscented Kalman Filter, and Particle Filter.

pub mod ekf;
pub mod kalman_em;
pub mod kalman_filter;
pub mod linalg;
pub mod particle;
pub mod rts_smoother;
pub mod ukf;

pub use ekf::{ExtendedKalmanFilter, ExtendedKalmanResult};
pub use kalman_em::{KalmanEmConfig, kalman_em};
pub use kalman_filter::{KalmanFilter, KalmanResult};
pub use particle::{ParticleConfig, ParticleFilter, ParticleResult};
pub use rts_smoother::{RtsResult, rts_smoother};
pub use ukf::{UkfParams, UkfResult, UnscentedKalmanFilter};
