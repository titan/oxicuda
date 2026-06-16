//! Evaluation metrics for sequence models.

pub mod chrf;
pub mod edit_distance;
pub mod metrics;

pub use chrf::{chrf_plus_plus, chrf_score, corpus_chrf, corpus_chrf_plus_plus};
pub use edit_distance::{
    EditAlignment, EditCounts, EditOp, align, character_error_rate, edit_distance_aligned,
    word_error_rate,
};
pub use metrics::{bleu_n, edit_distance, log_loss, perplexity, sequence_accuracy, token_accuracy};
