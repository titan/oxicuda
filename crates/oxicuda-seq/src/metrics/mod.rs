//! Evaluation metrics for sequence models.

pub mod metrics;

pub use metrics::{bleu_n, edit_distance, log_loss, perplexity, sequence_accuracy, token_accuracy};
