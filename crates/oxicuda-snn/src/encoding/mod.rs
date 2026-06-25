//! Spike-train encodings for analogue inputs.
//!
//! All encoders write a flat `(t_steps × n)` row-major buffer where row `t`
//! holds the spikes emitted at time step `t`. This layout is consistent across
//! [`rate`], [`temporal`], [`phase`], and [`poisson_input`] so callers can
//! slice trains by row without re-arranging memory.

/// Differentiable / learnable spike encoders (learned rate and TTFS).
pub mod differentiable;
/// Phase-coding via oscillatory reference signal.
pub mod phase;
/// Poisson rate-coded input wrapper.
pub mod poisson_input;
/// Bernoulli rate coding.
pub mod rate;
/// CSR-style sparse spike encoding (compressed per-time-step spike packets).
pub mod sparse_spike;
/// Time-To-First-Spike latency coding.
pub mod temporal;
/// Temporal-contrast (event-camera) spike encoding (Brandli et al. 2014).
pub mod temporal_contrast;

pub use sparse_spike::{SparseSpikes, dense_forward, encode_dense_to_sparse};
pub use temporal_contrast::{
    TemporalContrastConfig, TemporalContrastState, temporal_contrast_encode, temporal_contrast_step,
};
