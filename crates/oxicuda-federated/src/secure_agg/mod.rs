//! Secure aggregation protocols for federated learning.
//!
//! Provides cryptographic primitives for aggregating client updates
//! without the server learning individual client contributions.

pub mod aggregator;
pub mod masking;
pub mod shamir;
