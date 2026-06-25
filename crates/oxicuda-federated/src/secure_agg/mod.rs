//! Secure aggregation protocols for federated learning.
//!
//! Provides cryptographic primitives for aggregating client updates
//! without the server learning individual client contributions.

pub mod aggregator;
pub mod key_exchange;
pub mod masking;
pub mod shamir;

pub use key_exchange::{DhKeyPair, pairwise_seed_matrix};
