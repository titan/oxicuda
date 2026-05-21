//! Ensemble anomaly scoring.
pub mod ensemble;
pub mod ext_iforest;
pub mod federated;
pub mod loda;
pub mod suod;

pub use federated::{
    AggregationMethod, ClientModel, FederatedAnomalyFit, FederatedConfig, federated_client_score,
    federated_fit, federated_predict, federated_score,
};
