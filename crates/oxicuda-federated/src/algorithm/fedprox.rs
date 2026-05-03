//! FedProx: Federated Optimization with proximal term.
//!
//! Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
//!
//! FedProx adds a proximal term `(μ/2) * ||w − w_t||²` to the local objective,
//! which stabilizes training on heterogeneous (non-IID) data.

use crate::algorithm::fedavg::FedAvgConfig;
use crate::error::{FedError, FedResult};

/// Configuration for FedProx training.
#[derive(Debug, Clone)]
pub struct FedProxConfig {
    /// Proximal term coefficient μ (must be > 0).
    pub mu: f32,
    /// Underlying FedAvg configuration.
    pub fedavg: FedAvgConfig,
}

impl FedProxConfig {
    /// Create a validated FedProx configuration.
    pub fn new(mu: f32, fedavg: FedAvgConfig) -> FedResult<Self> {
        if !(mu > 0.0 && mu.is_finite()) {
            return Err(FedError::InvalidProximalMu);
        }
        Ok(Self { mu, fedavg })
    }
}

/// Compute the gradient of the proximal term.
///
/// Returns `mu * (local − global)` element-wise, which is the gradient of
/// `(μ/2) * ||local − global||²` with respect to `local`.
///
/// # Errors
/// Returns `DimensionMismatch` if slices have different lengths.
pub fn proximal_gradient(
    local_params: &[f32],
    global_params: &[f32],
    mu: f32,
) -> FedResult<Vec<f32>> {
    if local_params.len() != global_params.len() {
        return Err(FedError::DimensionMismatch {
            expected: global_params.len(),
            got: local_params.len(),
        });
    }
    Ok(local_params
        .iter()
        .zip(global_params.iter())
        .map(|(&l, &g)| mu * (l - g))
        .collect())
}

/// Add the proximal correction in-place to an existing gradient.
///
/// Modifies `grad += mu * (local − global)`, which biases the local update
/// toward the global model, preventing client drift.
///
/// # Errors
/// Returns `DimensionMismatch` if slice lengths differ, or
/// `InvalidProximalMu` if `mu` is non-positive or non-finite.
pub fn fedprox_client_loss_correction(
    grad: &mut [f32],
    local_params: &[f32],
    global_params: &[f32],
    mu: f32,
) -> FedResult<()> {
    if !(mu > 0.0 && mu.is_finite()) {
        return Err(FedError::InvalidProximalMu);
    }
    let n = grad.len();
    if local_params.len() != n {
        return Err(FedError::DimensionMismatch {
            expected: n,
            got: local_params.len(),
        });
    }
    if global_params.len() != n {
        return Err(FedError::DimensionMismatch {
            expected: n,
            got: global_params.len(),
        });
    }
    for ((g, &l), &global) in grad
        .iter_mut()
        .zip(local_params.iter())
        .zip(global_params.iter())
    {
        *g += mu * (l - global);
    }
    Ok(())
}

/// Compute the proximal regularization loss value: `(μ/2) * ||local − global||²`.
///
/// Useful for monitoring how far a client has drifted from the global model.
///
/// # Errors
/// Returns `DimensionMismatch` if slices have different lengths.
pub fn proximal_loss(local_params: &[f32], global_params: &[f32], mu: f32) -> FedResult<f32> {
    if local_params.len() != global_params.len() {
        return Err(FedError::DimensionMismatch {
            expected: global_params.len(),
            got: local_params.len(),
        });
    }
    let sq_diff: f32 = local_params
        .iter()
        .zip(global_params.iter())
        .map(|(&l, &g)| (l - g) * (l - g))
        .sum();
    Ok(0.5 * mu * sq_diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::fedavg::FedAvgConfig;

    #[test]
    fn proximal_gradient_points_toward_global() {
        // When local > global, gradient should be positive (pushes toward global)
        let local = vec![2.0f32, 3.0, 4.0];
        let global = vec![1.0f32, 2.0, 3.0];
        let grad = proximal_gradient(&local, &global, 0.1)
            .expect("test invariant: valid proximal gradient");
        for &g in &grad {
            assert!(g > 0.0, "gradient should be positive when local > global");
        }
        assert!((grad[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn proximal_gradient_dimension_mismatch() {
        let local = vec![1.0f32, 2.0];
        let global = vec![1.0f32, 2.0, 3.0];
        assert!(matches!(
            proximal_gradient(&local, &global, 0.1),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fedprox_loss_correction_in_place() {
        let mut grad = vec![0.0f32, 0.0, 0.0];
        let local = vec![2.0f32, 3.0, 4.0];
        let global = vec![1.0f32, 2.0, 3.0];
        fedprox_client_loss_correction(&mut grad, &local, &global, 0.1)
            .expect("test invariant: valid correction");
        assert!((grad[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn fedprox_loss_correction_invalid_mu() {
        let mut grad = vec![0.0f32; 3];
        let local = vec![1.0f32; 3];
        let global = vec![0.0f32; 3];
        assert!(matches!(
            fedprox_client_loss_correction(&mut grad, &local, &global, -1.0),
            Err(FedError::InvalidProximalMu)
        ));
    }

    #[test]
    fn proximal_loss_zero_when_equal() {
        let params = vec![1.0f32, 2.0, 3.0];
        let loss =
            proximal_loss(&params, &params, 0.5).expect("test invariant: valid proximal loss");
        assert!(loss.abs() < 1e-7);
    }

    #[test]
    fn fedprox_config_valid() {
        let fedavg =
            FedAvgConfig::new(10, 0.5, 5, 0.01).expect("test invariant: valid fedavg config");
        let cfg = FedProxConfig::new(0.01, fedavg).expect("test invariant: valid fedprox config");
        assert!((cfg.mu - 0.01).abs() < 1e-7);
    }

    #[test]
    fn fedprox_config_invalid_mu() {
        let fedavg =
            FedAvgConfig::new(10, 0.5, 5, 0.01).expect("test invariant: valid fedavg config");
        assert!(matches!(
            FedProxConfig::new(-0.1, fedavg),
            Err(FedError::InvalidProximalMu)
        ));
    }
}
