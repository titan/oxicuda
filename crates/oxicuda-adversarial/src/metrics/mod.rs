//! Robustness evaluation metrics: clean & robust accuracy, attack success rate.

pub mod asr;
pub mod corruption;
pub mod feature_squeezing;
pub mod gradient_masking;
pub mod loss_landscape;
pub mod robust_accuracy;
pub mod stratified_accuracy;
pub mod transferability;

pub use corruption::{
    Corruption, CorruptionErrors, CorruptionSummary, MAX_SEVERITY, box_blur, brightness, contrast,
    gaussian_noise, impulse_noise, shot_noise,
};
pub use feature_squeezing::{FeatureSqueezingConfig, FeatureSqueezingDetector};
pub use gradient_masking::{
    GradMaskingConclusion, GradMaskingConfig, GradientMaskingReport, diagnose_gradient_masking,
    random_perturbation_asr,
};
pub use loss_landscape::{
    Histogram, LandscapeProbeConfig, LandscapeReport, ProbeNorm, loss_profile, multi_restart_probe,
};
pub use stratified_accuracy::{ClassRobustness, StratifiedReport, stratified_robust_accuracy};
pub use transferability::{
    TransferMatrix, transferability_matrix, transferability_matrix_from_predictions,
};
