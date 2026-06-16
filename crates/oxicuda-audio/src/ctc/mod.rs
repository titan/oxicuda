//! Connectionist Temporal Classification (CTC) algorithms.
//!
//! Provides:
//! - **`forward`**: CTC forward algorithm in log domain (log-likelihood loss).
//! - **`beam_search`**: CTC prefix beam-search decoder.
//! - **`rnnt`**: RNN-T (Transducer) loss (Graves 2012).
//! - **`joint_ctc_attention`**: Joint CTC + attention decoding (Watanabe 2017).
//! - **`transducer_decode`**: Streaming RNN-T greedy decoder (Graves 2012/2013).

pub mod beam_search;
pub mod forward;
pub mod joint_ctc_attention;
pub mod rnnt;
pub mod transducer_decode;

pub use beam_search::{BeamHypothesis, ctc_beam_search};
pub use forward::ctc_forward_log;
pub use joint_ctc_attention::JointCtcAttention;
pub use rnnt::{RnntConfig, RnntResult, rnnt_loss};
pub use transducer_decode::TransducerGreedyDecoder;
