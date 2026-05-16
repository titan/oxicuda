//! Embedding-quality metrics.

pub mod metrics;

pub use metrics::{
    continuity, kl_pq, neighborhood_preservation, pairwise_distances, trustworthiness,
};
