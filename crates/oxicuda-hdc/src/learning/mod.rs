//! Learning algorithms over hypervector encodings: adaptive (iteratively retrained)
//! HD classification and ridge regression in HD space.

pub mod adaptive_hd;
/// Federated (gradient-free) HD learning with DP-style noise and per-client aggregation.
pub mod federated_hd;
/// Multi-pass online HD classifier with an explicit exponential forgetting factor.
pub mod forgetting_hd;
/// Online HD clustering with streaming binary centroids.
pub mod hd_cluster;
/// HDC k-nearest-neighbour classifier storing individual exemplars.
pub mod hd_knn;
/// Hybrid HD + small-MLP classifier (HD features → backprop-trained MLP head).
pub mod hd_mlp;
pub mod hd_regression;

pub use adaptive_hd::{AdaptiveHdClassifier, AdaptiveHdConfig};
pub use federated_hd::{ClientModel, FederatedServer};
pub use forgetting_hd::{ForgettingHdClassifier, ForgettingHdConfig};
pub use hd_cluster::HdCluster;
pub use hd_knn::HdKnn;
pub use hd_mlp::{HdMlp, HdMlpConfig};
pub use hd_regression::{HdRegressionConfig, HdRegressor};
