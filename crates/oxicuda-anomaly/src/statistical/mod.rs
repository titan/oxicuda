//! Statistical anomaly detection methods (MAD, Z-score, percentile threshold, HBOS, ECOD,
//! conformal p-values, concept-drift detectors, and online streaming detectors).
pub mod concept_drift;
pub mod conformal;
pub mod ecod;
pub mod hbos;
pub mod online_stats;
pub mod rock_idec;
pub mod stats;

pub use online_stats::{
    ExponentialZ, OnlineZScore, SlidingMad, StreamMethod, StreamingResult,
    StreamingThresholdDetector,
};
pub use rock_idec::{IdecConfig, IdecDetector, RockConfig, RockDetector};
