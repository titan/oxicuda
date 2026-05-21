//! Federated learning algorithms.
//!
//! Implements server-side aggregation and client-side training protocols
//! for the most widely-used federated learning algorithms.

pub mod ditto;
pub mod fedadam;
pub mod fedavg;
pub mod fedbuff;
pub mod fednova;
pub mod fedprox;
pub mod robust_agg;
pub mod scaffold;

pub use ditto::{Ditto, DittoClientUpdate, DittoConfig, DittoState};
pub use fedbuff::{BufferedUpdate, FedBuffConfig, FedBuffState};
pub use fednova::{FedNova, FedNovaClientUpdate, FedNovaConfig, FedNovaState};
pub use robust_agg::{RobustAggConfig, RobustAggResult, RobustAggregator};
