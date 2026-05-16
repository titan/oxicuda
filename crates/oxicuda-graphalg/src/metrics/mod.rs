//! Graph metrics: diameter, radius, density, clustering coefficient.

#[allow(clippy::module_inception)]
pub mod metrics;

pub use metrics::{clustering_coefficient_global, density, diameter, radius, transitivity};
