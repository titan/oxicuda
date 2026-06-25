//! Differential privacy primitives for federated learning.
//!
//! Provides mechanisms, accountants, and aggregation protocols that satisfy
//! (ε, δ)-differential privacy or Rényi DP guarantees.

pub mod dp_ftrl;
pub mod gaussian;
pub mod laplacian;
pub mod ldp_fl;
pub mod moments;
pub mod pate;
pub mod randomized_response;
pub mod rdp;
pub mod zcdp;

pub use dp_ftrl::{DpFtrl, DpFtrlConfig, DpFtrlResult, DpFtrlState};
pub use ldp_fl::{
    LdpFlConfig, LdpMechanism, amplified_epsilon, ldp_fl_aggregate, privatize_update,
};
pub use pate::{ConfidentGnMaxConfig, confident_gnmax};
pub use randomized_response::{RandomizedResponse, RandomizedResponseConfig};
pub use zcdp::{ZcdpAccountant, rdp_to_zcdp, zcdp_gaussian, zcdp_to_dp, zcdp_to_rdp};
