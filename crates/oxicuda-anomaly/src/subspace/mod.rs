//! Subspace anomaly detection via randomized hashing.
//!
//! Exposes RS-Hash (Sathe & Aggarwal 2016): an ensemble of randomized
//! `(subspace × resolution × shift)` grid-hash components scored by cell
//! crowdedness.
pub mod rs_hash;

pub use rs_hash::{HashComponent, RsHash, RsHashConfig};
