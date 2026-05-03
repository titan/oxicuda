//! Connectionist Temporal Classification (CTC) algorithms.
//!
//! Provides:
//! - **`forward`**: CTC forward algorithm in log domain (log-likelihood loss).
//! - **`beam_search`**: CTC prefix beam-search decoder.

pub mod beam_search;
pub mod forward;

pub use beam_search::{BeamHypothesis, ctc_beam_search};
pub use forward::ctc_forward_log;
