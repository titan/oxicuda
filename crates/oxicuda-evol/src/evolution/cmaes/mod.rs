//! CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

pub mod cmaes;
pub mod linalg;

pub use cmaes::{CmaEsConfig, CmaEsState};
