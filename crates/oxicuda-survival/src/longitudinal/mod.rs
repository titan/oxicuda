//! Joint longitudinal-survival models.
//!
//! Implements the shared-parameter (current-value) joint model of Wulfsohn & Tsiatis (1997)
//! and Rizopoulos (2010), simultaneously fitting:
//! - A **longitudinal sub-model**: biomarker trajectory via linear mixed effects.
//! - A **survival sub-model**: Cox PH hazard depending on the current value of the biomarker.

pub mod joint_model;

pub use joint_model::{
    JointModelConfig, JointModelFit, JointObs, joint_model_fit, joint_model_predict_survival,
    joint_model_predict_trajectory,
};
