//! Dictionary learning: K-SVD, MOD, and online dictionary updates.

pub mod k_svd;
pub mod mod_dl;
pub mod online_dl;

pub use k_svd::k_svd;
pub use mod_dl::mod_dl;
pub use online_dl::online_dl;

/// Dictionary learning result.
#[derive(Debug, Clone)]
pub struct DictionaryResult {
    /// `d × k` dictionary row-major (each column is an atom).
    pub dict: Vec<f64>,
    /// `k × n_samples` coefficient matrix row-major.
    pub codes: Vec<f64>,
    pub iterations: usize,
}
