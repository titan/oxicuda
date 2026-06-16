//! Linear-Gaussian state-space inference.
//!
//! Provides the [`kalman`] module with a numerically stabilised Kalman filter
//! (Joseph-form covariance update) and the Rauch–Tung–Striebel backward smoother
//! for offline conditioning on a full measurement sequence.
//!
//! # References
//! - Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction
//!   Problems." *J. Basic Eng.* 82(1):35-45.
//! - Rauch, Tung & Striebel (1965). "Maximum Likelihood Estimates of Linear
//!   Dynamic Systems." *AIAA Journal* 3(8):1445-1450.

pub mod kalman;

pub use kalman::{
    KalmanFilterResult, KalmanSmootherResult, LinearGaussianModel, kalman_filter, rts_smoother,
};
