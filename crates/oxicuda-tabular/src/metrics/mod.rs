//! Evaluation metrics for tabular learning.

pub mod calibration;
pub mod pr_metrics;
pub mod tabular_metrics;

pub use pr_metrics::{
    average_precision, binary_ece, brier_score, f1_at_threshold, log_loss, multiclass_log_loss,
    multiclass_prf, pr_auc, precision_at_threshold, precision_recall_curve, recall_at_threshold,
};

// Note: `calibration::brier_score` is the multi-class Brier score; the binary
// `brier_score` above (from `pr_metrics`) keeps the unqualified name to avoid a
// breaking change, so the multi-class variant is reached via `calibration::`.
pub use calibration::{
    BinningScheme, CalibrationConfig, ReliabilityBin, TemperatureScaler,
    expected_calibration_error, maximum_calibration_error, reliability_bins, temperature_nll,
};
