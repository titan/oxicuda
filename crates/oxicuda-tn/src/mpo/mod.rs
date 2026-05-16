//! Matrix Product Operator (MPO) representation.
//!
//! An MPO is the operator analogue of an MPS: each site holds a rank-4 tensor with two
//! virtual bonds, an upper (output) physical leg, and a lower (input) physical leg.

pub mod contraction;
pub mod mpo;

pub use contraction::apply_mpo_to_mps;
pub use mpo::{Mpo, MpoTensor};
