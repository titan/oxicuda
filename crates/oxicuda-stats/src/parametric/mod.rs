//! Parametric hypothesis tests: t-tests, ANOVA, MANOVA, regression inference.

pub mod anova;
pub mod manova;
pub mod regression_inference;
pub mod t_test;

pub use anova::{AnovaResult, one_way_anova, two_way_anova};
pub use manova::{ManovaResult, manova_wilks};
pub use regression_inference::{RegressionInference, regression_inference};
pub use t_test::{TTestResult, one_sample_t, paired_t, two_sample_t, welch_t};
