//! Additional learning rate schedulers.
//!
//! Extends the schedulers in [`crate::lr_scheduler`] with newer variants.

/// Warmup-Stable-Decay (WSD) scheduler (Hu et al., 2024 — MiniCPM).
pub mod wsd;

pub use wsd::{WsdConfig, WsdScheduler};
