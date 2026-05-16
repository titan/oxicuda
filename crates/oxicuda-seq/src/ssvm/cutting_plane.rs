//! Cutting-plane / sub-gradient SSVM configuration.

/// Hyper-parameters for sub-gradient / cutting-plane SSVM training.
#[derive(Debug, Clone, Copy)]
pub struct CuttingPlaneConfig {
    /// Maximum training epochs.
    pub max_iter: usize,
    /// Initial learning rate.
    pub lr: f64,
    /// Learning-rate decay: `lr_t = lr / (1 + lr_decay · t)`.
    pub lr_decay: f64,
    /// L2 regularisation factor on parameters.
    pub regularisation: f64,
    /// Stop when total Hamming loss falls below this.
    pub tol: f64,
}

impl Default for CuttingPlaneConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            lr: 0.05,
            lr_decay: 0.01,
            regularisation: 1e-3,
            tol: 1e-3,
        }
    }
}
