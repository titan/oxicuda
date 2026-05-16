//! Low-rank matrix completion methods: SVT, nuclear-norm, ADMM.

pub mod admm_completion;
pub mod nuclear_norm;
pub mod svt;

pub use admm_completion::admm_matrix_completion;
pub use nuclear_norm::nuclear_norm_minimization;
pub use svt::svt;

/// Completion solver result.
#[derive(Debug, Clone)]
pub struct CompletionResult {
    /// Recovered `h × w` matrix (row-major).
    pub x: Vec<f64>,
    pub residual: f64,
    pub iterations: usize,
}
