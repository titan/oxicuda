//! Approximate Message Passing family: AMP, VAMP, EB-AMP.

pub mod amp;
pub mod eb_amp;
pub mod vamp;

pub use amp::amp;
pub use eb_amp::eb_amp;
pub use vamp::vamp;

/// Result of an AMP-family solver.
#[derive(Debug, Clone)]
pub struct AmpResult {
    pub x: Vec<f64>,
    pub residual_norm: f64,
    pub iterations: usize,
}
