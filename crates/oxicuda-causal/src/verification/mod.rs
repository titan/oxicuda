//! Verification & numerical-accuracy harnesses.
//!
//! These modules cross-check the production estimators against independent
//! references and known ground truth, turning the loose "is finite" smoke tests
//! into genuine correctness assertions:
//!
//! * [`mod@reference`] — independent reference numerics (erf-based normal CDF,
//!   Jacobi-eigendecomposition matrix exponential).
//! * [`synthetic`] — linear-Gaussian SEM and treatment-effect data generators
//!   with known ground truth.
//! * [`graph_metrics`] — structure-recovery metrics (skeleton F1, SHD,
//!   orientation accuracy).
//! * [`matrix_exp`] — Padé(1,1) vs eigendecomposition error analysis for the
//!   NOTEARS acyclicity exponential.
//! * [`fisher_z`] — Fisher-Z critical-value calibration and empirical type-I
//!   error.
//! * [`notears_recovery`] — NOTEARS structure recovery vs ground-truth DAGs.
//! * [`pc_orientation`] — PC skeleton/v-structure correctness on benchmark
//!   motifs.
//! * [`dml_coverage`] — Double-ML 95% CI coverage & standard-error study.
//! * [`forest_pehe`] — causal-forest PEHE on heterogeneous-effect DGPs.

pub mod dml_coverage;
pub mod fisher_z;
pub mod forest_pehe;
pub mod graph_metrics;
pub mod matrix_exp;
pub mod notears_recovery;
pub mod pc_orientation;
pub mod reference;
pub mod synthetic;

pub use dml_coverage::{CoverageReport, coverage_study};
pub use fisher_z::{empirical_type_one_error, exact_fisher_z_critical};
pub use forest_pehe::{ForestPeheReport, evaluate as forest_pehe_evaluate};
pub use graph_metrics::{
    SkeletonScore, orientation_accuracy, skeleton_score, structural_hamming_distance,
};
pub use matrix_exp::{ExpmErrorReport, expm_error_report};
pub use notears_recovery::{NotearsRecoveryReport, recover as notears_recover};
pub use pc_orientation::{PcRecoveryReport, recover as pc_recover};
pub use reference::{erf, expm_symmetric_eig, normal_cdf, two_sided_z_quantile};
pub use synthetic::{
    ConfoundedData, HeteroEffectData, LinearSem, chain_sem, collider_sem, confounded_data,
    hetero_effect_data, random_dag_sem,
};
