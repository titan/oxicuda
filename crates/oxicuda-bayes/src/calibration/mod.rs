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

pub mod beta;
pub mod conformal;
pub mod ece_classwise;
pub mod histogram;
pub mod isotonic;
pub mod metrics;
pub mod platt;
pub mod temperature;
pub mod vector_scaling;

pub use beta::{BetaCalibConfig, BetaCalibrator};
pub use conformal::{ConformalClassifier, ConformalRegressor, RapsClassifier, conformal_quantile};
pub use ece_classwise::{
    BinningScheme, BrierDecomposition, ClassReliability, ClasswiseEceConfig, ReliabilityPoint,
    TopLabelCalibration, brier_decomposition, class_wise_eces, classwise_ece,
    multiclass_brier_score, per_class_reliability, top_label_calibration,
};
pub use histogram::{BinStrategy, HistogramBinCalibrator, HistogramBinConfig};
pub use vector_scaling::{ScalingMode, VectorScaler, VectorScalingConfig};
