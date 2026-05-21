//! Conformal prediction for distribution-free uncertainty quantification.
//!
//! Split (inductive) conformal prediction with finite-sample marginal coverage
//! `1 − alpha`: regression intervals, Conformalized Quantile Regression (CQR),
//! and classification prediction sets (APS / LAC).

pub mod split_conformal;

pub use split_conformal::{
    ClassifierScore, ConformalConfig, ConformalizedQuantileRegressor, SplitConformalClassifier,
    SplitConformalRegressor, empirical_quantile,
};
