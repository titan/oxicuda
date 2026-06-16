//! Greedy sparse recovery algorithms (OMP, StOMP, ROMP, CoSaMP, SP, SOMP, Block-OMP, LISTA).

pub mod block_omp;
pub mod cosamp;
pub mod lista;
pub mod omp;
pub mod romp;
pub mod somp;
pub mod sp;
pub mod stomp;

pub use block_omp::{BlockOmpResult, block_omp};
pub use cosamp::{CoSampConfig, cosamp, cosamp_with_config};
pub use lista::{Lista, ListaConfig};
pub use omp::omp;
pub use romp::romp;
pub use somp::{MmvResult, somp};
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
