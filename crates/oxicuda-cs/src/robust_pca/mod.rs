//! Robust PCA: decompose `M = L + S` into low-rank `L` and sparse `S`.

pub mod godec;
pub mod robust_pca_pcp;

pub use godec::godec;
pub use robust_pca_pcp::robust_pca_pcp;

/// Result of a robust-PCA decomposition.
#[derive(Debug, Clone)]
pub struct RobustPcaResult {
    pub low_rank: Vec<f64>,
    pub sparse: Vec<f64>,
    pub iterations: usize,
}
