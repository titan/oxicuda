//! Connectionist Temporal Classification (CTC).
//!
//! CTC (Graves et al. 2006) enables training and decoding of sequence models
//! whose output length differs from, and is not pre-aligned with, the target
//! label sequence. This module provides:
//!
//! * [`mod@ctc_loss`] — the forward-backward CTC negative-log-likelihood loss over a
//!   blank-augmented label lattice, computed in log-space.
//! * [`ctc_log_occupancy`] — per-symbol posterior responsibilities (the basis of
//!   the CTC gradient).
//! * [`ctc_greedy_decode`] — best-path (arg-max) decoding with CTC collapse.
//! * [`ctc_prefix_beam_search`] — prefix-beam-search decoding that sums
//!   alignment multiplicities (Hannun et al. 2014).

pub mod ctc_decode;
pub mod ctc_loss;

pub use ctc_decode::{CtcHypothesis, ctc_greedy_decode, ctc_prefix_beam_search};
pub use ctc_loss::{ctc_log_occupancy, ctc_loss, ctc_loss_batch};
