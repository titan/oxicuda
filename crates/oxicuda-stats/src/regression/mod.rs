//! Linear, logistic, ridge, generalised linear, robust, quantile, mixed-effects,
//! multinomial logistic, and negative binomial regression.

pub mod diagnostics;
pub mod glm;
pub mod linear;
pub mod logistic;
pub mod mixed_effects;
pub mod multinomial;
pub mod negbinom;
pub mod quantile;
pub mod ridge_lr;
pub mod robust;

pub use diagnostics::{
    breusch_pagan_test, cooks_distance, dffits, durbin_watson_ols, durbin_watson_residuals,
    leverage, ols_standard_errors, standardized_residuals, vif,
};
pub use glm::{
    GlmConfig, GlmFamily, GlmFit, GlmLink, glm_fit, glm_lrt, glm_predict, glm_score_test,
};
pub use linear::{LinearModel, matrix_inverse_lu, matrix_mul, matrix_transpose, ols};
pub use logistic::{LogisticModel, logistic_fit_irls};
pub use mixed_effects::{
    LmmConfig, LmmData, LmmFit, lmm_fit, lmm_icc, lmm_predict, lmm_residuals_by_group,
};
pub use multinomial::{
    MultinomialConfig, MultinomialFit, multinomial_accuracy, multinomial_fit, multinomial_predict,
    multinomial_predict_proba,
};
pub use negbinom::{
    NbMethod, NegBinomConfig, NegBinomFit, negbinom_fit, negbinom_predict,
    negbinom_predict_with_method,
};
pub use quantile::{QuantileConfig, QuantileFit, quantile_band, quantile_fit, quantile_predict};
pub use ridge_lr::{RidgeModel, ridge_regression};
pub use robust::{
    BisquareConfig, HuberConfig, LmsConfig, RansacConfig, RobustFit, ScaleMethod, bisquare_fit,
    estimate_scale_iqr, estimate_scale_mad, huber_fit, lms_fit, lts_fit, median_absolute_deviation,
    ransac_fit, winsorized_scale,
};
