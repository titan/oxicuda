//! Streaming anomaly detection for feature-evolving data streams.
//!
//! Exposes xStream (Manzoor et al. 2018): StreamHash sparse projection plus an
//! ensemble of multi-scale half-space chains.
pub mod xstream;

pub use xstream::{HalfSpaceChain, StreamHash, XStream, XStreamConfig};
