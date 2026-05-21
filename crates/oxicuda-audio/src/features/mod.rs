//! Audio feature extraction adapters.

pub mod cmvn;
pub mod delta;
pub mod log_mel_adapter;
pub mod mel_filterbank;

pub use cmvn::{CmvnConfig, apply_cmvn, compute_cmvn};
pub use delta::{compute_delta, compute_delta_delta, stack_delta_features};
pub use log_mel_adapter::LogMelInput;
pub use mel_filterbank::{MelFilterbank, MelFilterbankConfig};
