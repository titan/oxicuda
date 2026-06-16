//! Linear, logistic, ridge, generalised linear, robust, quantile, mixed-effects,
//! multinomial logistic, negative binomial regression, and GAM.

pub mod diagnostics;
pub mod gam;
pub mod glm;
pub mod linear;
pub mod logistic;
pub mod mixed_effects;
pub mod multinomial;
pub mod negbinom;
pub mod quantile;
pub mod quantile_regression;
pub mod ridge_lr;
pub mod robust;
pub mod theil_sen;
pub mod tweedie;

pub use diagnostics::{
    breusch_pagan_test, cooks_distance, dffits, durbin_watson_ols, durbin_watson_residuals,
    leverage, ols_standard_errors, standardized_residuals, vif,
};
pub use gam::{GamConfig, GamFit, GamSmoothConfig, gam_fit, gam_partial_effects, gam_predict};
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
pub use quantile_regression::{
    QuantRegConfig, QuantRegFit, coverage_below, pinball_loss, quantile_regression_fit,
    quantile_regression_predict,
};
pub use ridge_lr::{RidgeModel, ridge_regression};
pub use robust::{
    BisquareConfig, HuberConfig, LmsConfig, RansacConfig, RobustFit, ScaleMethod, bisquare_fit,
    estimate_scale_iqr, estimate_scale_mad, huber_fit, lms_fit, lts_fit, median_absolute_deviation,
    ransac_fit, winsorized_scale,
};
pub use theil_sen::{
    SlopeConfidenceInterval, TheilSenFit, siegel_fit, theil_sen_confidence_interval, theil_sen_fit,
};
pub use tweedie::{
    TweedieConfig, TweedieFit, TweedieLink, tweedie_deviance, tweedie_fit, tweedie_predict,
    tweedie_unit_deviance, tweedie_variance,
};
