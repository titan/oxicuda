//! Correlation coefficients with inference.

pub mod distance_correlation;
pub mod kendall_tau;
pub mod partial;
pub mod pearson;
pub mod spearman;

pub use distance_correlation::{
    DistanceCorrelation, DistanceTestResult, bias_corrected_distance_correlation,
    distance_correlation, distance_correlation_full, distance_covariance, distance_covariance_test,
};
pub use kendall_tau::{KendallResult, kendall_tau};
pub use partial::{PartialCorrResult, PointBiserialResult, partial_correlation, point_biserial};
pub use pearson::{PearsonResult, pearson_r};
pub use spearman::{SpearmanResult, spearman_rho};
