//! Matrix Product Operator (MPO) representation.
//!
//! An MPO is the operator analogue of an MPS: each site holds a rank-4 tensor with two
//! virtual bonds, an upper (output) physical leg, and a lower (input) physical leg.

pub mod auto_compress;
pub mod contraction;
pub mod long_range;
pub mod mpo;

pub use auto_compress::{
    MpoCompressConfig, MpoData, mpo_bond_dims, mpo_compress, mpo_operator_norm_sq,
};
pub use contraction::apply_mpo_to_mps;
pub use long_range::{
    LongRangeMpo, LongRangeMpoConfig, exponential_fit, fsm_mpo_single_exp,
    heisenberg_long_range_mpo, power_law_mpo,
};
pub use mpo::{Mpo, MpoTensor};
