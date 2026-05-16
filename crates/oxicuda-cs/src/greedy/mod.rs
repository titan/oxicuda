//! Greedy sparse recovery algorithms (OMP, StOMP, ROMP, CoSaMP, SP).

pub mod cosamp;
pub mod omp;
pub mod romp;
pub mod sp;
pub mod stomp;

pub use cosamp::cosamp;
pub use omp::omp;
pub use romp::romp;
pub use sp::subspace_pursuit;
pub use stomp::stomp;

/// Greedy recovery result holding sparse vector, support indices, residual norm and iterations.
#[derive(Debug, Clone)]
pub struct GreedyResult {
    pub x: Vec<f64>,
    pub support: Vec<usize>,
    pub residual_norm: f64,
    pub iterations: usize,
}
