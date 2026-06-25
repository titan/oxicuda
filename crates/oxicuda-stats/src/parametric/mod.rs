//! Parametric hypothesis tests: t-tests, ANOVA, MANOVA, regression inference.

pub mod anova;
pub mod manova;
pub mod manova_followup;
pub mod regression_inference;
pub mod t_test;
pub mod test_builder;
pub mod variance_tests;

pub use anova::{AnovaResult, one_way_anova, two_way_anova};
pub use manova::{ManovaResult, manova_wilks};
pub use manova_followup::{ManovaFollowup, UnivariateAnova, manova_followup, w_inv_b};
pub use regression_inference::{RegressionInference, regression_inference};
pub use t_test::{TTestResult, one_sample_t, paired_t, two_sample_t, welch_t};
pub use test_builder::{
    AnovaBuilder, AnovaBuilderResult, BootstrapBuilder, BootstrapCiMethod, TTestBuilder,
    TTestResult as TTestBuilderResult, TailDirection,
};
pub use variance_tests::{BartlettResult, LeveneCenter, LeveneResult, bartlett_test, levene_test};
