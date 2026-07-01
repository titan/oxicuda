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
//! ├── regression/      — OLS, ridge, logistic, robust, Theil-Sen/Siegel regression
//! ├── power/           — t-test, ANOVA power analysis and effect sizes
//! ├── correlation/     — Pearson, Spearman, Kendall's tau
//! ├── time_series/     — ACF/PACF, ADF, KPSS, GARCH, VAR(p)
//! └── trend/           — Mann-Kendall, Sen's slope, seasonal Mann-Kendall
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]

pub mod bayesian;
pub mod chi_squared;
pub mod ci;
pub mod circular;
pub mod copula;
pub mod correlation;
pub mod density;
pub mod descriptive;
pub mod distributions;
pub mod error;
pub mod extremes;
pub mod goodness_of_fit;
pub mod handle;
pub mod mcmc;
pub mod mixture;
pub mod multiple;
pub mod nonparametric;
pub mod parametric;
pub mod point_process;
pub mod power;
pub mod ptx_kernels;
pub mod regression;
pub mod resampling;
pub mod smoothing;
pub mod spatial;
pub mod special;
pub mod state_space;
pub mod survey;
pub mod time_series;
pub mod time_series_advanced;
pub mod trend;

pub use bayesian::conjugate::{
    BayesFactor, BetaPosterior, CiMethod, CredibleInterval, GammaPosterior, NigPosterior,
    NormalNormalPosterior, bayes_factor_coin, beta_binomial_update, beta_credible_interval,
    dirichlet_multinomial_update, gamma_poisson_update, nig_update, normal_normal_ci,
    normal_normal_update,
};
pub use bayesian::dirichlet_mult::{
    DirMultFitConfig, DirichletMultinomial, dirichlet_multinomial_mle,
};
pub use circular::{
    CircularError, CircularResult, MeanDirection, RayleighResult, VonMisesFit,
    WatsonWilliamsResult, circular_mean, circular_std, circular_variance, kappa_mle,
    kappa_mle_from_angles, mean_direction, rayleigh_test, von_mises_cdf, von_mises_mle,
    von_mises_pdf, watson_williams_test,
};
pub use error::{StatsError, StatsResult};
pub use handle::{LcgRng, SmVersion, StatsHandle};
pub use regression::{
    BisquareConfig, HuberConfig, LmsConfig, RansacConfig, RobustFit, ScaleMethod, bisquare_fit,
    estimate_scale_iqr, estimate_scale_mad, huber_fit, lms_fit, lts_fit, median_absolute_deviation,
    ransac_fit, winsorized_scale,
};
pub use regression::{CoxConfig, CoxFit, TieMethod, concordance_index, cox_ph_fit};
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
pub use regression::{
    QuantRegConfig, QuantRegFit, quantile_regression_fit, quantile_regression_predict,
};
pub use regression::{QuantileConfig, QuantileFit, quantile_band, quantile_fit, quantile_predict};
pub use regression::{
    SlopeConfidenceInterval, TheilSenFit, siegel_fit, theil_sen_confidence_interval, theil_sen_fit,
};
pub use regression::{
    TweedieConfig, TweedieFit, TweedieLink, tweedie_deviance, tweedie_fit, tweedie_predict,
    tweedie_unit_deviance, tweedie_variance,
};
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
    AcfSeResult, AdfResult, AdfTrend, KpssTrend, PacfResult, acf, acf_bartlett, adf_test,
    box_pierce, correlogram_bounds, durbin_watson, kpss_test, ljung_box, pacf,
};
pub use time_series::{
    GrangerResult, VarFit, granger_causality, var_fit, var_forecast, var_is_stable,
    var_spectral_radius, var_unconditional_mean,
};
pub use time_series_advanced::{
    arch_test, bai_perron_single_break, chow_test, variance_ratio_test, zivot_andrews_p_value,
    zivot_andrews_test,
};
pub use trend::{
    MannKendallResult, TrendDirection, mann_kendall, seasonal_mann_kendall, sens_slope,
};

pub use parametric::test_builder::{
    AnovaBuilder, AnovaBuilderResult, BootstrapBuilder, BootstrapCiMethod, TTestBuilder,
    TTestResult as TTestBuilderResult, TailDirection,
};

pub use correlation::partial::{
    PartialCorrResult, PointBiserialResult, partial_correlation, point_biserial,
};
pub use nonparametric::sign_cochran::{CochranQResult, SignTestResult, cochran_q, sign_test};
pub use parametric::manova_followup::{ManovaFollowup, UnivariateAnova, manova_followup, w_inv_b};
pub use parametric::variance_tests::{
    BartlettResult, LeveneCenter, LeveneResult, bartlett_test, levene_test,
};

pub use copula::archimedean::{ArchimedeanCopula, ArchimedeanFamily};
pub use copula::copulas::{
    CopulaFamily, CopulaFit, copula_cdf, copula_fit, copula_log_likelihood, copula_pdf,
    copula_sample, copula_tail_dependence, kendall_tau_pairs,
};
pub use copula::gaussian_copula::{GaussianCopula, pseudo_observations};
pub use copula::vine::{PairCopula, VineCopula, VineFitConfig, VineType, vine_fit};
pub use correlation::distance_correlation::{
    DistanceCorrelation, DistanceTestResult, bias_corrected_distance_correlation,
    distance_correlation, distance_correlation_full, distance_covariance, distance_covariance_test,
};
pub use density::{
    BandwidthRule, Kernel, KernelDensity, KernelDensity2d, scott_bandwidth, silverman_bandwidth,
};
pub use nonparametric::dirichlet_process::{
    ChineseRestaurant, DpMixtureConfig, DpMixtureResult, NormalBaseMeasure, StickBreakingWeights,
    crp_simulate, dp_mixture_fit, stick_breaking_weights,
};
pub use nonparametric::isotonic::{
    IsotonicBlock, IsotonicFit, antitonic_regression, antitonic_regression_weighted,
    isotonic_regression, isotonic_regression_weighted, weighted_sse,
};
pub use smoothing::holt_winters::{
    HwConfig, HwResult, HwVariant, hw_aic, hw_bic, hw_fit, hw_forecast, hw_residuals,
};

pub use extremes::extreme_value::{Gev, Gpd};
pub use mcmc::hmc::{
    HmcConfig, HmcSamples, PotentialTarget, hamiltonian, hmc_sample, leapfrog, leapfrog_step,
};
pub use mcmc::nuts::{NutsConfig, NutsSamples, no_u_turn, nuts_sample};
pub use mixture::{
    GmmConfig, GmmCovariance, GmmModel, gmm_aic, gmm_bic, gmm_fit, gmm_predict, gmm_predict_proba,
    gmm_score,
};
pub use point_process::hawkes::{
    HawkesMleConfig, HawkesMleResult, HawkesParams, hawkes_compensator, hawkes_intensity,
    hawkes_log_likelihood, hawkes_log_likelihood_naive, hawkes_mle, hawkes_simulate,
};
pub use state_space::kalman::{
    KalmanFilterResult, KalmanSmootherResult, LinearGaussianModel, kalman_filter, rts_smoother,
};

#[cfg(test)]
mod e2e_tests;

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests;
