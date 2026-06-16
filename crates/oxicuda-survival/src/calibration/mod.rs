//! Calibration metrics: Brier score, integrated Brier, time-dependent AUC, ROC curves, DCA.

pub mod brier_score;
pub mod integrated_brier;
pub mod ipcw_brier;
pub mod pseudo_r2;
pub mod time_dependent_auc;
pub mod time_roc;

pub use brier_score::brier_score_at;
pub use integrated_brier::integrated_brier_score;
pub use ipcw_brier::ipcw_brier_at;
pub use pseudo_r2::{PseudoR2Result, r2_d_from_d, royston_pseudo_r2};
pub use time_dependent_auc::time_dependent_auc;
pub use time_roc::{
    CalibrationResult, DcaResult, TimeRocResult, calibration_analysis, decision_curve_analysis,
    time_roc, time_roc_auc_only,
};
