//! Routing strategies for Mixture of Experts.

pub mod base;
pub mod expert_choice;
pub mod hash;
pub mod multi_gate;
pub mod soft_moe;
pub mod stable_moe;
pub mod switch;
pub mod top_k;

pub use base::{BaseConfig, BaseResult, BaseRouter, row_softmax, sinkhorn_convergence};
pub use hash::{HashRouter, HashRoutingConfig};
pub use multi_gate::{MultiGateConfig, MultiGateRouter};
pub use stable_moe::{
    StableMoeConfig, StableMoeGating, StableMoeResult, StableMoeRouter, load_balance_loss, sigmoid,
    z_loss,
};
