//! Sketch-evaluation metrics: relative error, accuracy, recall-at-k.

pub mod metrics;

pub use metrics::{absolute_error, accuracy, mean_absolute_error, recall_at_k, relative_error};
