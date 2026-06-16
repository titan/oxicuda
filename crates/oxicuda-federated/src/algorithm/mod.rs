//! Federated learning algorithms.
//!
//! Implements server-side aggregation and client-side training protocols
//! for the most widely-used federated learning algorithms.

pub mod centered_clipping;
pub mod ditto;
pub mod fedadam;
pub mod fedavg;
pub mod fedbuff;
pub mod feddf;
pub mod feddyn;
pub mod fednova;
pub mod fedprox;
pub mod flute;
pub mod geometric_median;
pub mod moon;
pub mod robust_agg;
pub mod scaffold;

pub use centered_clipping::{
    CenteredClippingConfig, CenteredClippingResult, centered_clipping, centered_clipping_at_mean,
};
pub use ditto::{Ditto, DittoClientUpdate, DittoConfig, DittoState};
pub use fedbuff::{BufferedUpdate, FedBuffConfig, FedBuffState};
pub use feddf::{FedDf, FedDfConfig, LinearModel, argmax, softmax_with_temperature};
pub use feddyn::{FedDyn, FedDynClientState, FedDynConfig, FedDynState};
pub use fednova::{FedNova, FedNovaClientUpdate, FedNovaConfig, FedNovaState};
pub use flute::{Flute, FluteClientUpdate, FluteConfig, FluteModel, FluteSample};
pub use geometric_median::{GeometricMedianConfig, GeometricMedianResult, geometric_median};
pub use moon::{MoonConfig, moon_contrastive_grad, moon_contrastive_loss, moon_total_loss};
pub use robust_agg::{RobustAggConfig, RobustAggResult, RobustAggregator};
