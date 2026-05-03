//! FedAdam / FedYogi / FedAdagrad: Adaptive server-side optimizers for federated learning.
//!
//! Reddi et al., "Adaptive Federated Optimization", ICLR 2021.
//!
//! These algorithms apply adaptive momentum to the pseudo-gradient
//! (the difference between the old and new global model, aggregated from clients)
//! rather than to per-client gradients, enabling stable large-scale federation.

use crate::error::{FedError, FedResult};

/// Choice of adaptive server optimizer variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerOptimizerKind {
    /// FedAdam: `v = β₂*v + (1−β₂)*g²`
    FedAdam,
    /// FedYogi: `v = v + (1−β₂)*g²*sign(g²−v)` (additive, v is monotone increasing)
    FedYogi,
    /// FedAdagrad: `v = v + g²` (accumulate all squared pseudo-gradients)
    FedAdagrad,
}

/// State for adaptive federated server optimizer.
#[derive(Debug, Clone)]
pub struct FedAdamState {
    /// First moment estimate (momentum buffer).
    pub m: Vec<f32>,
    /// Second moment estimate (adaptive learning rate buffer).
    pub v: Vec<f32>,
    /// Number of optimizer steps taken.
    pub step: usize,
    /// Exponential decay rate for first moment (typical: 0.9).
    pub beta1: f32,
    /// Exponential decay rate for second moment (typical: 0.999).
    pub beta2: f32,
    /// Numerical stability constant (typical: 1e-8).
    pub eps: f32,
    /// Server-side learning rate.
    pub lr: f32,
    /// Which adaptive optimizer to use.
    pub kind: ServerOptimizerKind,
}

impl FedAdamState {
    /// Create a new adaptive optimizer state.
    ///
    /// # Arguments
    /// - `n_params` — number of model parameters
    /// - `lr` — server learning rate
    /// - `kind` — optimizer variant
    ///
    /// Uses standard defaults: β₁=0.9, β₂=0.999, ε=1e-8.
    #[must_use]
    pub fn new(n_params: usize, lr: f32, kind: ServerOptimizerKind) -> Self {
        Self {
            m: vec![0.0_f32; n_params],
            v: vec![1e-4_f32; n_params], // small initial v prevents div-by-zero at step 1
            step: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            lr,
            kind,
        }
    }

    /// Create a new state with custom hyperparameters.
    ///
    /// # Errors
    /// Returns `InvalidWeight` if any hyperparameter is out of range.
    pub fn with_hyperparams(
        n_params: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        kind: ServerOptimizerKind,
    ) -> FedResult<Self> {
        if !(lr > 0.0 && lr.is_finite()) {
            return Err(FedError::InvalidWeight { weight: lr });
        }
        if !(beta1 > 0.0 && beta1 < 1.0) {
            return Err(FedError::InvalidWeight { weight: beta1 });
        }
        if !(beta2 > 0.0 && beta2 < 1.0) {
            return Err(FedError::InvalidWeight { weight: beta2 });
        }
        if !(eps > 0.0 && eps.is_finite()) {
            return Err(FedError::InvalidWeight { weight: eps });
        }
        Ok(Self {
            m: vec![0.0_f32; n_params],
            v: vec![eps; n_params],
            step: 0,
            beta1,
            beta2,
            eps,
            lr,
            kind,
        })
    }

    /// Apply one adaptive server optimizer step.
    ///
    /// Updates `global_params` using `pseudo_grad` (typically the mean of
    /// client parameter deltas aggregated by the server).
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if `pseudo_grad` has a different length
    /// from the internal state, or `Internal` if `lr` is non-finite.
    pub fn step(&mut self, pseudo_grad: &[f32], global_params: &mut [f32]) -> FedResult<()> {
        let n = self.m.len();
        if pseudo_grad.len() != n {
            return Err(FedError::DimensionMismatch {
                expected: n,
                got: pseudo_grad.len(),
            });
        }
        if global_params.len() != n {
            return Err(FedError::DimensionMismatch {
                expected: n,
                got: global_params.len(),
            });
        }

        self.step = self.step.saturating_add(1);
        let t = self.step as f32;

        // Bias-correction factors (only used for FedAdam)
        let bias_corr1 = 1.0 - self.beta1.powf(t);
        let bias_corr2 = 1.0 - self.beta2.powf(t);

        for i in 0..n {
            let g = pseudo_grad[i];

            // Update first moment
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;

            // Update second moment based on optimizer kind
            match self.kind {
                ServerOptimizerKind::FedAdam => {
                    self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
                    let m_hat = self.m[i] / bias_corr1;
                    let v_hat = (self.v[i] / bias_corr2).max(self.eps);
                    global_params[i] += self.lr * m_hat / (v_hat.sqrt() + self.eps);
                }
                ServerOptimizerKind::FedYogi => {
                    // FedYogi: v += (1−β₂) * g² * sign(g² − v)
                    // This ensures v is monotonically non-decreasing
                    let g_sq = g * g;
                    let sign_val = if g_sq > self.v[i] { 1.0_f32 } else { -1.0_f32 };
                    self.v[i] += (1.0 - self.beta2) * g_sq * sign_val;
                    self.v[i] = self.v[i].max(self.eps);
                    global_params[i] += self.lr * self.m[i] / (self.v[i].sqrt() + self.eps);
                }
                ServerOptimizerKind::FedAdagrad => {
                    // FedAdagrad: v += g²  (no decay, accumulates all history)
                    self.v[i] += g * g;
                    global_params[i] += self.lr * self.m[i] / (self.v[i].sqrt() + self.eps);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fedadam_step_updates_params() {
        let mut state = FedAdamState::new(3, 0.01, ServerOptimizerKind::FedAdam);
        let mut params = vec![0.0f32; 3];
        let grad = vec![1.0f32, -1.0, 0.5];
        state
            .step(&grad, &mut params)
            .expect("test invariant: valid fedadam step");
        // Params should have moved
        assert!(params.iter().any(|&p| p != 0.0));
        assert_eq!(state.step, 1);
    }

    #[test]
    fn fedyogi_v_increases_when_g_sq_exceeds_v() {
        let mut state = FedAdamState::new(1, 0.01, ServerOptimizerKind::FedYogi);
        let v_before = state.v[0];
        let mut params = vec![0.0f32; 1];
        let grad = vec![100.0f32]; // large gradient, g^2 >> v
        state
            .step(&grad, &mut params)
            .expect("test invariant: valid fedyogi step");
        assert!(
            state.v[0] > v_before,
            "FedYogi v should increase when g^2 > v"
        );
    }

    #[test]
    fn fedadagrad_v_always_increases() {
        let mut state = FedAdamState::new(2, 0.01, ServerOptimizerKind::FedAdagrad);
        let mut params = vec![0.0f32; 2];
        for i in 1..=5 {
            let v_before = state.v.clone();
            let grad = vec![i as f32, -(i as f32)];
            state
                .step(&grad, &mut params)
                .expect("test invariant: valid fedadagrad step");
            for (vb, va) in v_before.iter().zip(state.v.iter()) {
                assert!(va >= vb, "FedAdagrad v should never decrease");
            }
        }
    }

    #[test]
    fn fedadam_dimension_mismatch() {
        let mut state = FedAdamState::new(3, 0.01, ServerOptimizerKind::FedAdam);
        let mut params = vec![0.0f32; 3];
        let grad = vec![1.0f32, 2.0]; // wrong size
        assert!(matches!(
            state.step(&grad, &mut params),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fedadam_with_hyperparams_valid() {
        let state = FedAdamState::with_hyperparams(
            4,
            0.001,
            0.9,
            0.999,
            1e-8,
            ServerOptimizerKind::FedAdam,
        )
        .expect("test invariant: valid hyperparams");
        assert_eq!(state.m.len(), 4);
        assert!((state.beta1 - 0.9).abs() < 1e-7);
    }

    #[test]
    fn fedadam_with_hyperparams_invalid_lr() {
        assert!(matches!(
            FedAdamState::with_hyperparams(
                4,
                -0.001,
                0.9,
                0.999,
                1e-8,
                ServerOptimizerKind::FedAdam
            ),
            Err(FedError::InvalidWeight { .. })
        ));
    }
}
