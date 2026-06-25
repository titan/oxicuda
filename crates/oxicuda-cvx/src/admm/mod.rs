//! Alternating Direction Method of Multipliers (ADMM).

pub mod adaptive_rho_admm;
pub mod admm;
pub mod async_admm;
pub mod consensus;
pub mod consensus_admm;
pub mod dual_decomp;

pub use adaptive_rho_admm::{AdaptiveRhoConfig, AdaptiveRhoResult, adaptive_rho_admm};
pub use admm::admm_solve;
pub use async_admm::{AsyncAdmmConfig, AsyncAdmmResult, async_consensus_admm};
pub use consensus::{
    ConsensusAdmmConfig, ConsensusAdmmResult, consensus_admm as consensus_admm_new,
};
pub use consensus_admm::consensus_admm;
pub use dual_decomp::{DualDecompConfig, DualDecompResult, dual_decomp};
