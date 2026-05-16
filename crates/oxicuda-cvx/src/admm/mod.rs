//! Alternating Direction Method of Multipliers (ADMM).

pub mod admm;
pub mod consensus_admm;

pub use admm::admm_solve;
pub use consensus_admm::consensus_admm;
