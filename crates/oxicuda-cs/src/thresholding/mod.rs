//! Hard-thresholding-based recovery: IHT, NIHT, HTP, AIHT.

pub mod aiht;
pub mod htp;
pub mod iht;
pub mod niht;

pub use aiht::aiht;
pub use htp::htp;
pub use iht::{hard_threshold_k, iht, soft_threshold};
pub use niht::niht;

/// Result of a thresholding-based solver.
#[derive(Debug, Clone)]
pub struct ThresholdingResult {
    pub x: Vec<f64>,
    pub support: Vec<usize>,
    pub residual_norm: f64,
    pub iterations: usize,
}
