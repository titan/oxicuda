//! Representation-quality metrics for self-supervised learning.
//!
//! Provides post-hoc diagnostic tools for evaluating the geometry of learned
//! feature representations — uniformity, alignment, effective rank, and
//! collapse detection — without requiring downstream labels.
//!
//! Also includes the standard SSL kNN evaluation protocol for measuring
//! downstream classification accuracy from frozen features without any
//! parameter training.

pub mod feature_metrics;
pub mod knn_eval;

pub use feature_metrics::{
    alignment_loss, collapse_score, effective_rank, pairwise_cosine_stats, uniformity_loss,
};
pub use knn_eval::{KnnEvalConfig, KnnEvalResult, knn_eval};
