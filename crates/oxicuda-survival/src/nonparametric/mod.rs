//! Non-parametric survival estimators: Kaplan-Meier, Nelson-Aalen, life table,
//! Aalen's additive hazards model, the gamma frailty model for clustered data,
//! the Aalen-Johansen multi-state estimator, recurrent-event MCF/AG models,
//! survival meta-analysis, and net/relative survival (cancer-registry methods).

pub mod aalen;
pub mod frailty;
pub mod kaplan_meier;
pub mod life_table;
pub mod multi_state;
pub mod multi_state_inference;
pub mod nelson_aalen;
pub mod net_survival;
pub mod npsurv_bayes;
pub mod recurrent;
pub mod survival_function;
pub mod survival_meta;
pub mod survival_rf;

pub use aalen::{AalenConfig, AalenFit, fit_aalen};
pub use frailty::{FrailtyConfig, FrailtyFit, fit_gamma_frailty, predict_frailty_survival};
pub use kaplan_meier::{KaplanMeier, kaplan_meier_estimate};
pub use life_table::{LifeTable, life_table};
pub use multi_state::{
    MultiStateConfig, MultiStateFit, MultiStateObs, fit_multi_state, predict_occupation,
    predict_transition_probs,
};
pub use multi_state_inference::{
    AjInference, CifInference, MultiStateData, aalen_johansen_variance, cif_with_variance,
    transition_prob_at,
};
pub use nelson_aalen::{NelsonAalen, nelson_aalen_estimate};
pub use net_survival::{
    NetSurvivalMethod, NetSurvivalResult, PopulationLifeTable, RelSurvObs, ederer_i, ederer_ii,
    net_survival_log_rank, pohar_perme,
};
pub use npsurv_bayes::{
    DpSurvivalConfig, DpSurvivalPosterior, dp_predict_survival, dp_survival_posterior,
};
pub use recurrent::{
    AgConfig, AgFit, RecurrentGroupTest, RecurrentMcfResult, RecurrentObs, fit_andersen_gill,
    predict_cumulative_mean, recurrent_mcf, recurrent_two_sample,
};
pub use survival_function::SurvivalFunction;
pub use survival_meta::{
    CombinedLogRankResult, FixedEffectsResult, GuyotReconstruction, PooledKmResult,
    RandomEffectsResult, StudyHazardRatio, StudyKm, combined_log_rank, compute_study_km,
    fixed_effects_meta, guyot_reconstruct, pool_km_curves, random_effects_meta,
};
pub use survival_rf::{
    SurvivalNode, SurvivalRf, SurvivalRfConfig, SurvivalRfPred, SurvivalTree, survival_rf_fit,
    survival_rf_importance, survival_rf_predict,
};
