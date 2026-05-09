//! Confidence calibration of probabilistic classifier outputs.
//!
//! Calibrated probabilities satisfy: among all predictions with confidence ≈ p,
//! a fraction ≈ p are correct. Modern deep classifiers are typically
//! over-confident (Guo et al. 2017); this module provides post-hoc recalibration
//! and quantitative miscalibration metrics.
//!
//! # Modules
//!
//! - [`temperature`] — temperature scaling: `p̂ = softmax(z / T)` with NLL-optimised T
//! - [`metrics`] — Expected/Maximum/Adaptive Calibration Error, Brier, NLL, reliability diagrams
//! - [`isotonic`] — non-parametric isotonic regression (PAV) for monotone recalibration
//! - [`platt`] — Platt scaling for binary probabilities (logistic recalibration)
//!
//! All routines run on CPU; complementary PTX kernels in [`crate::ptx_kernels`] cover
//! the per-element steps (`temp_scale_logits_ptx`, `ece_bucket_ptx`).

pub mod isotonic;
pub mod metrics;
pub mod platt;
pub mod temperature;
