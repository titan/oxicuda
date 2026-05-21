//! `oxicuda-survival` — Survival analysis & time-to-event modelling for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-survival
//! ├── data/            — Observation, Dataset, RiskSet primitives
//! ├── nonparametric/   — Kaplan-Meier, Nelson-Aalen, life table, S(t) curves
//! ├── test/            — Log-rank, stratified log-rank, Peto-Peto, Gehan-Breslow
//! ├── cox/             — Cox PH (Breslow/Efron ties, Newton-Raphson, Schoenfeld)
//! ├── aft/             — Parametric AFT (Exp, Weibull, log-normal, log-logistic, GG)
//! ├── time_varying/    — Counting-process Cox with time-varying covariates
//! ├── competing/       — Cumulative incidence, cause-specific Cox, Fine-Gray
//! ├── rmst/            — Restricted mean survival time
//! ├── concordance/     — Harrell's C, Uno's C
//! ├── calibration/     — Brier, IPCW Brier, integrated Brier, time-dependent AUC
//! ├── deep/            — DeepSurv head, Cox partial-likelihood gradient, loss callables
//! ├── special/         — gammaln, digamma
//! ├── linalg/          — Cholesky, Gauss-Jordan inverse, matmul (crate-private)
//! └── metrics/         — Median survival, RMST, S(τ) summaries
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]

pub mod aft;
pub mod bayes;
pub mod calibration;
pub mod competing;
pub mod concordance;
pub mod cox;
pub mod data;
pub mod deep;
pub mod error;
pub mod handle;
pub mod linalg;
pub mod longitudinal;
pub mod metrics;
pub mod nonparametric;
pub mod plot;
pub mod ptx_kernels;
pub mod rmst;
pub mod special;
pub mod test;
pub mod time_varying;

pub use calibration::time_roc::{
    CalibrationResult, DcaResult, TimeRocResult, calibration_analysis, decision_curve_analysis,
    time_roc, time_roc_auc_only,
};
pub use cox::cox_builder::{CoxBuilder, CoxFitResult};
pub use cox::cure_model::{
    CureModelConfig, CureModelFit, fit_cure_model, predict_cure_prob, predict_cure_survival,
};
pub use cox::gradient_boost::{
    GbCoxConfig, GbCoxModel, GbCoxPred, GbCoxTree, GbNode, gb_cox_concordance, gb_cox_fit,
    gb_cox_predict,
};
pub use cox::iptw::{
    AiptwConfig, AiptwResult, IptwConfig, IptwResult, PropensityResult, aiptw_fit,
    compute_iptw_weights, fit_propensity_score, iptw_cox, iptw_fit,
};
pub use cox::line_search::{ArmijoConfig, WolfeConfig, armijo_backtrack, wolfe_line_search};
pub use cox::newton_raphson::TieMethod;
pub use cox::predict::{SurvivalPredict, predict_survival_curve};
pub use cox::trust_region::{TrustRegionConfig, TrustRegionResult, steihaug_cg, trust_region_cox};
pub use data::truncation::{
    IntervalObs, RightTruncatedObs, TruncatedKmResult, TruncatedObs, TurnbullResult,
    conditional_survival, effective_sample_size, to_counting_process, truncated_km, turnbull_em,
    validate_truncated,
};
pub use error::{SurvivalError, SurvivalResult};
pub use handle::{LcgRng, SmVersion, SurvivalHandle};
pub use longitudinal::joint_model::{
    JointModelConfig, JointModelFit, JointObs, joint_model_fit, joint_model_predict_survival,
    joint_model_predict_trajectory,
};
pub use nonparametric::frailty::{
    FrailtyConfig, FrailtyFit, fit_gamma_frailty, predict_frailty_survival,
};
pub use nonparametric::multi_state::{
    MultiStateConfig, MultiStateFit, MultiStateObs, fit_multi_state, predict_occupation,
    predict_transition_probs,
};
pub use nonparametric::net_survival::{
    NetSurvivalMethod, NetSurvivalResult, PopulationLifeTable, RelSurvObs, ederer_i, ederer_ii,
    net_survival_log_rank, pohar_perme,
};
pub use nonparametric::recurrent::{
    AgConfig, AgFit, RecurrentGroupTest, RecurrentMcfResult, RecurrentObs, fit_andersen_gill,
    predict_cumulative_mean, recurrent_mcf, recurrent_two_sample,
};
pub use nonparametric::survival_meta::{
    CombinedLogRankResult, FixedEffectsResult, GuyotReconstruction, PooledKmResult,
    RandomEffectsResult, StudyHazardRatio, StudyKm, combined_log_rank, compute_study_km,
    fixed_effects_meta, guyot_reconstruct, pool_km_curves, random_effects_meta,
};
pub use nonparametric::survival_rf::{
    SurvivalNode, SurvivalRf, SurvivalRfConfig, SurvivalRfPred, SurvivalTree, survival_rf_fit,
    survival_rf_importance, survival_rf_predict,
};
pub use plot::step_functions::{
    StepFunction, cif_to_step_function, km_to_step_function, median_survival, na_to_step_function,
    rmst_from_step, step_plot_arrays,
};
pub use rmst::pseudo_obs::{
    PseudoObsConfig, PseudoObsOutcome, PseudoObsRegression, PseudoObsResult, pseudo_obs_fit,
};
pub use test::ph_lr_test::{PhLrTestResult, PhWaldResult, ph_lr_test, ph_score_test, ph_wald_test};
pub use test::power_sample_size::{
    FreedmanConfig, FreedmanResult, PowerFromEventsConfig, SchoenefeldConfig, SchoenefeldResult,
    expected_events, freedman_sample_size, power_from_events, schoenfeld_sample_size,
};

#[cfg(test)]
mod e2e_tests;
