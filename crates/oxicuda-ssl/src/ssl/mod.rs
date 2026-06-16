//! High-level SSL model structs that own their own weights.
//!
//! This module collects struct-based counterparts to the functional APIs
//! scattered across the other `oxicuda-ssl` submodules. Each struct here owns
//! its weight matrices and provides an ergonomic interface for construction,
//! forward passes, and loss computation without requiring the caller to manage
//! raw weight buffers.

pub mod data2vec_v2;
pub mod jem;
pub mod sim_siam;

pub use data2vec_v2::{Data2VecModel, Data2VecModelConfig};
pub use jem::{Jem, JemConfig};
pub use sim_siam::{SimSiam, SimSiamConfig};
