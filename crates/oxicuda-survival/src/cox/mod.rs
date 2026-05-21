//! Cox proportional hazards model and supporting routines.

pub mod baseline_hazard;
pub mod breslow_ties;
pub mod cox_builder;
pub mod cox_ph;
pub mod cure_model;
pub mod efron_ties;
pub mod gradient_boost;
pub mod iptw;
pub mod line_search;
pub mod newton_raphson;
pub mod penalised_cox;
pub mod penalized_cox;
pub mod predict;
pub mod schoenfeld;
pub mod stratified_cox;
pub mod time_dep_cox;
pub mod trust_region;

pub use baseline_hazard::breslow_baseline_hazard;
pub use breslow_ties::breslow_log_likelihood;
pub use cox_builder::{CoxBuilder, CoxFitResult};
pub use cox_ph::{CoxFit, CoxPhConfig, TieMethod, fit_cox_ph};
pub use cure_model::{
    CureModelConfig, CureModelFit, fit_cure_model, predict_cure_prob, predict_cure_survival,
};
pub use efron_ties::efron_log_likelihood;
pub use gradient_boost::{
    GbCoxConfig, GbCoxModel, GbCoxPred, GbCoxTree, GbNode, gb_cox_concordance, gb_cox_fit,
    gb_cox_predict,
};
pub use iptw::{
    AiptwConfig, AiptwResult, IptwConfig, IptwResult, PropensityResult, aiptw_fit,
    compute_iptw_weights, fit_propensity_score, iptw_cox, iptw_fit,
};
pub use line_search::{ArmijoConfig, WolfeConfig, armijo_backtrack, wolfe_line_search};
pub use newton_raphson::newton_raphson_cox;
pub use penalised_cox::{
    PenalisedCoxConfig, PenalisedCoxFit, PenaltyKind, penalised_cox_cv_score, penalised_cox_fit,
    penalised_cox_path, penalised_cox_predict_risk,
};
pub use penalized_cox::{PenalizedCoxConfig, PenalizedCoxFit, PenaltyType, fit_penalized_cox};
pub use schoenfeld::{schoenfeld_residuals, schoenfeld_test};
pub use stratified_cox::{
    StratTieMethod, StratifiedCoxConfig, StratifiedCoxFit, stratified_cox_fit,
    stratified_cox_log_likelihood, stratified_cox_predict_survival, stratified_cox_score_test,
};
pub use time_dep_cox::{
    TimeDepCoxConfig, TimeDepCoxFit, TimeDepRecord, time_dep_cox_baseline_hazard, time_dep_cox_fit,
    time_dep_cox_score_test,
};
pub use trust_region::{TrustRegionConfig, TrustRegionResult, steihaug_cg, trust_region_cox};
