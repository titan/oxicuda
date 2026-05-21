//! `oxicuda-stats` — Statistical inference, hypothesis testing, and frequentist analysis for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-stats
//! ├── special/         — Special functions: erf, lgamma, digamma, betainc, gammp
//! ├── distributions/   — Normal, Student-t, chi-squared, F, beta, gamma, binomial, Poisson, exponential
//! ├── descriptive/     — Mean, variance, robust statistics, quantiles
//! ├── parametric/      — t-tests, ANOVA, MANOVA, regression inference
//! ├── nonparametric/   — Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman
//! ├── goodness_of_fit/ — KS, Anderson-Darling, Shapiro-Wilk, Jarque-Bera
//! ├── chi_squared/     — Chi-squared independence, Fisher exact, McNemar
//! ├── multiple/        — Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli, Tukey HSD
//! ├── resampling/      — Bootstrap, jackknife, permutation tests
//! ├── ci/              — Confidence intervals (normal, t, bootstrap, proportion)
//! ├── regression/      — OLS, ridge, logistic regression
//! ├── power/           — t-test, ANOVA power analysis and effect sizes
//! └── correlation/     — Pearson, Spearman, Kendall's tau
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]

pub mod bayesian;
pub mod chi_squared;
pub mod ci;
pub mod circular;
pub mod correlation;
pub mod descriptive;
pub mod distributions;
pub mod error;
pub mod goodness_of_fit;
pub mod handle;
pub mod multiple;
pub mod nonparametric;
pub mod parametric;
pub mod power;
pub mod ptx_kernels;
pub mod regression;
pub mod resampling;
pub mod spatial;
pub mod special;
pub mod survey;
pub mod time_series;
pub mod time_series_advanced;

pub use bayesian::conjugate::{
    BayesFactor, BetaPosterior, CiMethod, CredibleInterval, GammaPosterior, NigPosterior,
    NormalNormalPosterior, bayes_factor_coin, beta_binomial_update, beta_credible_interval,
    dirichlet_multinomial_update, gamma_poisson_update, nig_update, normal_normal_ci,
    normal_normal_update,
};
pub use circular::{
    CircularError, CircularResult, RayleighResult, VonMisesFit, circular_mean, circular_std,
    circular_variance, rayleigh_test, von_mises_cdf, von_mises_mle, von_mises_pdf,
};
pub use error::{StatsError, StatsResult};
pub use handle::{LcgRng, SmVersion, StatsHandle};
pub use regression::{
    BisquareConfig, HuberConfig, LmsConfig, RansacConfig, RobustFit, ScaleMethod, bisquare_fit,
    estimate_scale_iqr, estimate_scale_mad, huber_fit, lms_fit, lts_fit, median_absolute_deviation,
    ransac_fit, winsorized_scale,
};
pub use regression::{
    GlmConfig, GlmFamily, GlmFit, GlmLink, glm_fit, glm_lrt, glm_predict, glm_score_test,
};
pub use regression::{
    LmmConfig, LmmData, LmmFit, lmm_fit, lmm_icc, lmm_predict, lmm_residuals_by_group,
};
pub use regression::{
    MultinomialConfig, MultinomialFit, multinomial_accuracy, multinomial_fit, multinomial_predict,
    multinomial_predict_proba,
};
pub use regression::{
    NbMethod, NegBinomConfig, NegBinomFit, negbinom_fit, negbinom_predict,
    negbinom_predict_with_method,
};
pub use regression::{QuantileConfig, QuantileFit, quantile_band, quantile_fit, quantile_predict};
pub use regression::{
    breusch_pagan_test, cooks_distance, dffits, durbin_watson_ols, durbin_watson_residuals,
    leverage, ols_standard_errors, standardized_residuals, vif,
};
pub use spatial::{GearyCResult, MoransIResult, geary_c, moran_i, ripleys_k};
pub use survey::{
    ClusterResult, StratifiedResult, SurveyDesign, cluster_variance, design_effect,
    jackknife_survey_variance, stratified_variance, weighted_mean, weighted_variance,
};
pub use time_series::{
    AdfResult, AdfTrend, KpssTrend, acf, adf_test, box_pierce, durbin_watson, kpss_test, ljung_box,
};
pub use time_series_advanced::{
    arch_test, bai_perron_single_break, chow_test, variance_ratio_test, zivot_andrews_p_value,
    zivot_andrews_test,
};

pub use parametric::test_builder::{
    AnovaBuilder, AnovaBuilderResult, BootstrapBuilder, BootstrapCiMethod, TTestBuilder,
    TTestResult as TTestBuilderResult, TailDirection,
};

#[cfg(test)]
mod e2e_tests;
