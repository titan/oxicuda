//! Additional learning rate schedulers.
//!
//! Extends the schedulers in [`crate::lr_scheduler`] with newer variants.

/// Cosine annealing with warm restarts — SGDR (Loshchilov & Hutter, 2017).
pub mod cosine_restart;

/// Warmup-Stable-Decay (WSD) scheduler (Hu et al., 2024 — MiniCPM).
pub mod wsd;

pub use cosine_restart::{CosineAnnealingWarmRestarts, CosineRestartConfig};
pub use wsd::{WsdConfig, WsdScheduler};
