//! Density-based anomaly detection (COPOD, Mahalanobis, GMM, KDE, FastMCD).
pub mod copod;
pub mod fast_mcd;
pub mod gmm_detector;
pub mod kde_detector;
pub mod mahalanobis;

pub use kde_detector::{Bandwidth, KdeConfig, KdeDetector, KdeKernel};
