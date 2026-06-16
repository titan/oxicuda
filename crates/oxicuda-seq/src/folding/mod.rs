//! RNA secondary-structure folding algorithms.
//!
//! Currently this module provides the Nussinov (1978) maximum-base-pairing
//! dynamic program, which finds a pseudoknot-free secondary structure that
//! maximises the number of complementary base pairs.

pub mod nussinov;

pub use nussinov::{NussinovConfig, NussinovResult, can_pair, nussinov_fold};
