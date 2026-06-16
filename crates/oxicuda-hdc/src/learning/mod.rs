//! Learning algorithms over hypervector encodings: adaptive (iteratively retrained)
//! HD classification and ridge regression in HD space.

pub mod adaptive_hd;
/// Online HD clustering with streaming binary centroids.
pub mod hd_cluster;
/// HDC k-nearest-neighbour classifier storing individual exemplars.
pub mod hd_knn;
pub mod hd_regression;

pub use adaptive_hd::{AdaptiveHdClassifier, AdaptiveHdConfig};
pub use hd_cluster::HdCluster;
pub use hd_knn::HdKnn;
pub use hd_regression::{HdRegressionConfig, HdRegressor};
