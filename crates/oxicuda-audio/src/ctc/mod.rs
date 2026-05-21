//! Connectionist Temporal Classification (CTC) algorithms.
//!
//! Provides:
//! - **`forward`**: CTC forward algorithm in log domain (log-likelihood loss).
//! - **`beam_search`**: CTC prefix beam-search decoder.
//! - **`rnnt`**: RNN-T (Transducer) loss (Graves 2012).

pub mod beam_search;
pub mod forward;
pub mod rnnt;

pub use beam_search::{BeamHypothesis, ctc_beam_search};
pub use forward::ctc_forward_log;
pub use rnnt::{RnntConfig, RnntResult, rnnt_loss};
