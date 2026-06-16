//! LoRA-FA — LoRA with Frozen A.
//!
//! Reference: Zhang, L., Zhang, L., Shi, S., Chu, X., & Li, B. (2023).
//! *LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models
//! Fine-tuning*. <https://arxiv.org/abs/2308.03303>
//!
//! LoRA-FA freezes the projection matrix `A` at its random initialisation and only trains
//! the up-projection `B`, halving the trainable-parameter count compared to vanilla LoRA.
//! The forward pass is the standard LoRA delta:
//!
//! ```text
//!   y = s · B · (A · x),     where  s = α / rank
//! ```
//!
//! `A ∈ ℝ^{rank × in_features}` is sampled from `N(0, 1 / √in_features)` (the "Kaiming-style"
//! scaling that keeps the variance of `A · x` independent of `in_features`) and frozen,
//! whereas `B ∈ ℝ^{out_features × rank}` is zero-initialised and updated by SGD.
//!
//! ## Closed-form gradient
//!
//! With `t = A · x ∈ ℝ^{rank}` and loss `L`, the upstream gradient `grad_y = ∂L/∂y` gives
//!
//! ```text
//!   ∂L / ∂B = s · grad_y · tᵀ            (outer product, shape out × rank)
//! ```
//!
//! There is no gradient w.r.t. `A` since it is frozen.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Hyper-parameter bundle for a single LoRA-FA adapter.
#[derive(Debug, Clone)]
pub struct LoraFaConfig {
    /// Number of input features (column count of `A`).
    pub in_features: usize,
    /// Number of output features (row count of `B`).
    pub out_features: usize,
    /// Low-rank dimension shared between `A` and `B`.
    pub rank: usize,
    /// Global scaling factor `α`. The effective multiplier is `s = α / rank`.
    pub alpha: f64,
}

impl LoraFaConfig {
    /// Compute the effective scale `s = α / rank`.
    #[must_use]
    pub fn scale(&self) -> f64 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f64
        }
    }

    /// Validate the configuration without constructing an adapter.
    ///
    /// # Errors
    ///
    /// - [`PeftError::EmptyInput`] if any dimension is zero.
    /// - [`PeftError::RankTooLarge`] if `rank > min(in_features, out_features)`.
    pub fn validate(&self) -> PeftResult<()> {
        if self.in_features == 0 || self.out_features == 0 || self.rank == 0 {
            return Err(PeftError::EmptyInput);
        }
        let dim = self.in_features.min(self.out_features);
        if self.rank > dim {
            return Err(PeftError::RankTooLarge {
                rank: self.rank,
                dim,
            });
        }
        Ok(())
    }
}

/// LoRA-FA adapter with frozen `A` and trainable `B`.
///
/// Layout (all row-major):
/// - `a_frozen` has shape `[rank × in_features]`, drawn from `N(0, 1/√in_features)`.
/// - `b_trainable` has shape `[out_features × rank]`, zero-initialised.
#[derive(Debug, Clone)]
pub struct LoraFaAdapter {
    /// Frozen down-projection, row-major `[rank × in_features]`.
    pub a_frozen: Vec<f64>,
    /// Trainable up-projection, row-major `[out_features × rank]`.
    pub b_trainable: Vec<f64>,
    /// Configuration captured at construction time.
    pub cfg: LoraFaConfig,
}

impl LoraFaAdapter {
    /// Build a fresh adapter.
    ///
    /// `A` is drawn from `N(0, 1/√in_features)` using paired Box-Muller via
    /// [`LcgRng::next_normal_pair`]. `B` is zero-initialised so the adapter contributes
    /// zero at the start of training (mirroring the LoRA convention).
    ///
    /// # Errors
    ///
    /// Forwards [`LoraFaConfig::validate`] errors.
    pub fn new(cfg: LoraFaConfig, rng_seed: u64) -> PeftResult<Self> {
        cfg.validate()?;
        let mut rng = LcgRng::new(rng_seed);
        let std = 1.0_f64 / (cfg.in_features as f64).sqrt();
        let n = cfg.rank * cfg.in_features;
        let mut a_frozen = vec![0.0_f64; n];
        let mut i = 0;
        while i + 1 < n {
            let (u, v) = rng.next_normal_pair();
            a_frozen[i] = (u as f64) * std;
            a_frozen[i + 1] = (v as f64) * std;
            i += 2;
        }
        if i < n {
            a_frozen[i] = (rng.next_normal() as f64) * std;
        }
        let b_trainable = vec![0.0_f64; cfg.out_features * cfg.rank];
        Ok(Self {
            a_frozen,
            b_trainable,
            cfg,
        })
    }

    /// Number of trainable parameters, `out_features × rank` (half of vanilla LoRA's
    /// `rank · (in + out)` for `in ≈ out`).
    #[must_use]
    pub fn n_trainable(&self) -> usize {
        self.cfg.out_features * self.cfg.rank
    }

    /// Effective scale `s = α / rank`.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.cfg.scale()
    }

    /// Compute `y = s · B · (A · x)`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when `x.len() != in_features`.
    pub fn forward(&self, x: &[f64]) -> PeftResult<Vec<f64>> {
        if x.len() != self.cfg.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.cfg.in_features,
                got: x.len(),
            });
        }
        let t = self.compute_ax(x);
        Ok(self.compute_bt_scaled(&t))
    }

    /// Closed-form gradient w.r.t. `B`.
    ///
    /// Returns `∂L/∂B = s · grad_y · (A · x)ᵀ` as a flat row-major matrix of length
    /// `out_features × rank`. There is no gradient w.r.t. `A` (frozen).
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when `x.len() != in_features` or
    /// `grad_y.len() != out_features`.
    pub fn backward(&self, x: &[f64], grad_y: &[f64]) -> PeftResult<Vec<f64>> {
        if x.len() != self.cfg.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.cfg.in_features,
                got: x.len(),
            });
        }
        if grad_y.len() != self.cfg.out_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.cfg.out_features,
                got: grad_y.len(),
            });
        }
        let t = self.compute_ax(x);
        let s = self.scale();
        let r = self.cfg.rank;
        let out = self.cfg.out_features;
        let mut grad_b = vec![0.0_f64; out * r];
        for (i, g_i) in grad_y.iter().enumerate() {
            let row_start = i * r;
            let scaled = s * g_i;
            for (k, t_k) in t.iter().enumerate() {
                grad_b[row_start + k] = scaled * t_k;
            }
        }
        Ok(grad_b)
    }

    /// SGD update `B ← B − lr · grad_b`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when `grad_b.len() != out_features × rank`.
    pub fn apply_b_grad(&mut self, grad_b: &[f64], lr: f64) -> PeftResult<()> {
        let expected = self.cfg.out_features * self.cfg.rank;
        if grad_b.len() != expected {
            return Err(PeftError::DimensionMismatch {
                expected,
                got: grad_b.len(),
            });
        }
        for (b, g) in self.b_trainable.iter_mut().zip(grad_b.iter()) {
            *b -= lr * g;
        }
        Ok(())
    }

    /// `t = A · x`, length `rank`. Internal helper shared by forward and backward.
    fn compute_ax(&self, x: &[f64]) -> Vec<f64> {
        let r = self.cfg.rank;
        let in_f = self.cfg.in_features;
        let mut t = vec![0.0_f64; r];
        for (k, t_k) in t.iter_mut().enumerate() {
            let row_start = k * in_f;
            let mut acc = 0.0_f64;
            for (j, x_j) in x.iter().enumerate() {
                acc += self.a_frozen[row_start + j] * x_j;
            }
            *t_k = acc;
        }
        t
    }

    /// `y = s · B · t`, length `out_features`.
    fn compute_bt_scaled(&self, t: &[f64]) -> Vec<f64> {
        let r = self.cfg.rank;
        let out = self.cfg.out_features;
        let s = self.scale();
        let mut y = vec![0.0_f64; out];
        for (i, y_i) in y.iter_mut().enumerate() {
            let row_start = i * r;
            let mut acc = 0.0_f64;
            for (k, t_k) in t.iter().enumerate() {
                acc += self.b_trainable[row_start + k] * t_k;
            }
            *y_i = s * acc;
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg(in_f: usize, out_f: usize, rank: usize, alpha: f64) -> LoraFaConfig {
        LoraFaConfig {
            in_features: in_f,
            out_features: out_f,
            rank,
            alpha,
        }
    }

    #[test]
    fn initial_forward_is_zero_with_zero_b() {
        let cfg = default_cfg(6, 4, 2, 4.0);
        let adapter = LoraFaAdapter::new(cfg, 7)
            .expect("LoraFaAdapter::new should succeed with valid config");
        let x: Vec<f64> = (0..6).map(|i| i as f64 - 2.5).collect();
        let y = adapter
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        assert_eq!(y.len(), 4);
        for &v in &y {
            assert!(v.abs() < 1e-15, "expected zero output, got {v}");
        }
    }

    #[test]
    fn a_reproducible_across_seeds() {
        let cfg = default_cfg(8, 5, 3, 6.0);
        let a1 = LoraFaAdapter::new(cfg.clone(), 42)
            .expect("LoraFaAdapter::new should succeed with valid config");
        let a2 = LoraFaAdapter::new(cfg, 42)
            .expect("LoraFaAdapter::new should succeed with valid config");
        assert_eq!(a1.a_frozen, a2.a_frozen);
    }

    #[test]
    fn a_differs_for_different_seeds() {
        let cfg = default_cfg(8, 5, 3, 6.0);
        let a1 = LoraFaAdapter::new(cfg.clone(), 1)
            .expect("LoraFaAdapter::new should succeed with valid config");
        let a2 = LoraFaAdapter::new(cfg, 2)
            .expect("LoraFaAdapter::new should succeed with valid config");
        let diff: f64 = a1
            .a_frozen
            .iter()
            .zip(a2.a_frozen.iter())
            .map(|(p, q)| (p - q).abs())
            .sum();
        assert!(diff > 1e-6, "two seeds should produce different A");
    }

    #[test]
    fn forward_dimensions_correct() {
        let cfg = default_cfg(7, 9, 3, 6.0);
        let mut adapter = LoraFaAdapter::new(cfg, 11)
            .expect("LoraFaAdapter::new should succeed with valid config");
        // Inject non-zero B so the output isn't trivially zero
        for (i, b) in adapter.b_trainable.iter_mut().enumerate() {
            *b = (i as f64 + 1.0) * 0.05;
        }
        let x = vec![1.0_f64; 7];
        let y = adapter
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        assert_eq!(y.len(), 9);
    }

    #[test]
    fn backward_grad_shape_correct() {
        let cfg = default_cfg(5, 4, 2, 4.0);
        let adapter = LoraFaAdapter::new(cfg, 3)
            .expect("LoraFaAdapter::new should succeed with valid config");
        let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3];
        let grad_y = vec![0.1_f64, -0.2, 0.3, 0.4];
        let grad_b = adapter
            .backward(&x, &grad_y)
            .expect("backward pass should succeed with valid input");
        assert_eq!(grad_b.len(), 4 * 2);
    }

    #[test]
    fn backward_matches_finite_differences_on_b() {
        let cfg = default_cfg(4, 3, 2, 4.0);
        let mut adapter = LoraFaAdapter::new(cfg, 99)
            .expect("LoraFaAdapter::new should succeed with valid config");
        // Initialise B with non-trivial values (so derivative isn't degenerate)
        for (i, b) in adapter.b_trainable.iter_mut().enumerate() {
            *b = 0.1 * (i as f64 + 1.0);
        }
        let x = vec![0.5_f64, -1.0, 0.25, 0.75];
        let grad_y = vec![1.0_f64, -0.5, 0.25];
        let grad_b = adapter
            .backward(&x, &grad_y)
            .expect("backward pass should succeed with valid input");
        let eps = 1e-6_f64;
        for (k, &g_k) in grad_b.iter().enumerate() {
            let saved = adapter.b_trainable[k];
            adapter.b_trainable[k] = saved + eps;
            let yp = adapter
                .forward(&x)
                .expect("forward pass should succeed with valid input");
            adapter.b_trainable[k] = saved - eps;
            let ym = adapter
                .forward(&x)
                .expect("forward pass should succeed with valid input");
            adapter.b_trainable[k] = saved;
            let lp: f64 = grad_y.iter().zip(yp.iter()).map(|(a, b)| a * b).sum();
            let lm: f64 = grad_y.iter().zip(ym.iter()).map(|(a, b)| a * b).sum();
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - g_k).abs() < 1e-5, "B[{k}] FD={fd} analytic={g_k}");
        }
    }

    #[test]
    fn a_remains_frozen_after_update() {
        let cfg = default_cfg(5, 4, 2, 4.0);
        let mut adapter = LoraFaAdapter::new(cfg, 17)
            .expect("LoraFaAdapter::new should succeed with valid config");
        let a_snap = adapter.a_frozen.clone();
        let x = vec![0.1_f64, 0.2, 0.3, 0.4, 0.5];
        let grad_y = vec![1.0_f64, -0.5, 0.25, 0.1];
        let grad_b = adapter
            .backward(&x, &grad_y)
            .expect("backward pass should succeed with valid input");
        adapter
            .apply_b_grad(&grad_b, 0.1)
            .expect("gradient application should succeed");
        assert_eq!(adapter.a_frozen, a_snap, "A must remain frozen");
    }

    #[test]
    fn sgd_reduces_loss_on_small_fit() {
        // Target: random fixed target vector. Fit y = adapter(x) by gradient descent on B.
        let cfg = default_cfg(6, 4, 2, 4.0);
        let mut adapter = LoraFaAdapter::new(cfg, 21)
            .expect("LoraFaAdapter::new should succeed with valid config");
        let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3, 0.75];
        // Build a target that lies in the adapter's reachable range:
        // pick a B*, compute y* = s · B* · A · x.
        let target = {
            let mut b_star = vec![0.0_f64; 4 * 2];
            for (i, b) in b_star.iter_mut().enumerate() {
                *b = 0.4 * (i as f64 + 1.0);
            }
            let mut probe = adapter.clone();
            probe.b_trainable = b_star;
            probe
                .forward(&x)
                .expect("forward pass should succeed with valid input")
        };
        let mse = |adapter: &LoraFaAdapter| -> f64 {
            let y = adapter
                .forward(&x)
                .expect("forward pass should succeed with valid input");
            y.iter()
                .zip(target.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
        };
        let initial = mse(&adapter);
        for _ in 0..200 {
            let y = adapter
                .forward(&x)
                .expect("forward pass should succeed with valid input");
            // grad of (1/2) ||y - t||^2 w.r.t. y is (y - t)
            let grad_y: Vec<f64> = y
                .iter()
                .zip(target.iter())
                .map(|(yi, ti)| yi - ti)
                .collect();
            let g_b = adapter
                .backward(&x, &grad_y)
                .expect("backward pass should succeed with valid input");
            adapter
                .apply_b_grad(&g_b, 0.05)
                .expect("gradient application should succeed");
        }
        let final_loss = mse(&adapter);
        assert!(
            final_loss * 10.0 < initial,
            "loss {final_loss} should drop >10x from {initial}"
        );
    }

    #[test]
    fn invalid_configs_rejected() {
        assert!(matches!(
            LoraFaAdapter::new(default_cfg(0, 4, 2, 1.0), 0),
            Err(PeftError::EmptyInput)
        ));
        assert!(matches!(
            LoraFaAdapter::new(default_cfg(4, 0, 2, 1.0), 0),
            Err(PeftError::EmptyInput)
        ));
        assert!(matches!(
            LoraFaAdapter::new(default_cfg(4, 4, 0, 1.0), 0),
            Err(PeftError::EmptyInput)
        ));
        assert!(matches!(
            LoraFaAdapter::new(default_cfg(3, 4, 5, 1.0), 0),
            Err(PeftError::RankTooLarge { .. })
        ));
        assert!(matches!(
            LoraFaAdapter::new(default_cfg(4, 3, 5, 1.0), 0),
            Err(PeftError::RankTooLarge { .. })
        ));
    }

    #[test]
    fn dim_mismatch_in_forward_rejected() {
        let adapter = LoraFaAdapter::new(default_cfg(5, 3, 2, 2.0), 0)
            .expect("LoraFaAdapter::new should succeed with valid config");
        let bad_x = vec![1.0_f64, 2.0, 3.0]; // 3 != 5
        assert!(matches!(
            adapter.forward(&bad_x),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn dim_mismatch_in_backward_rejected() {
        let adapter = LoraFaAdapter::new(default_cfg(5, 3, 2, 2.0), 0)
            .expect("LoraFaAdapter::new should succeed with valid config");
        let x = vec![0.1_f64; 5];
        let bad_gy = vec![0.1_f64; 2]; // 2 != 3
        assert!(matches!(
            adapter.backward(&x, &bad_gy),
            Err(PeftError::DimensionMismatch { .. })
        ));
        let bad_x = vec![0.1_f64; 4]; // 4 != 5
        let good_gy = vec![0.1_f64; 3];
        assert!(matches!(
            adapter.backward(&bad_x, &good_gy),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn scale_alpha_over_rank_applied() {
        // Same A (same seed), different alpha → output scales linearly.
        let mut a1 = LoraFaAdapter::new(default_cfg(5, 3, 2, 4.0), 33)
            .expect("LoraFaAdapter::new should succeed with valid config");
        let mut a2 = LoraFaAdapter::new(default_cfg(5, 3, 2, 8.0), 33)
            .expect("LoraFaAdapter::new should succeed with valid config");
        // Both share A from the same seed; force the same non-zero B.
        let b_seed: Vec<f64> = (0..a1.b_trainable.len())
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        a1.b_trainable.copy_from_slice(&b_seed);
        a2.b_trainable.copy_from_slice(&b_seed);
        let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3];
        let y1 = a1
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        let y2 = a2
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        for (v1, v2) in y1.iter().zip(y2.iter()) {
            assert!((2.0 * v1 - v2).abs() < 1e-12, "α doubled → y doubled");
        }
        // explicit scale check
        assert!((a1.scale() - 2.0).abs() < 1e-15);
        assert!((a2.scale() - 4.0).abs() < 1e-15);
    }

    #[test]
    fn alpha_zero_produces_zero_forward() {
        let mut adapter = LoraFaAdapter::new(default_cfg(5, 4, 2, 0.0), 77)
            .expect("LoraFaAdapter::new should succeed with valid config");
        for (i, b) in adapter.b_trainable.iter_mut().enumerate() {
            *b = 0.1 * (i as f64 + 1.0);
        }
        let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3];
        let y = adapter
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        for &v in &y {
            assert!(v.abs() < 1e-15, "α=0 must zero out the adapter, got {v}");
        }
    }

    #[test]
    fn multiple_forward_calls_dont_mutate_state() {
        let mut adapter = LoraFaAdapter::new(default_cfg(6, 5, 3, 6.0), 13)
            .expect("LoraFaAdapter::new should succeed with valid config");
        for (i, b) in adapter.b_trainable.iter_mut().enumerate() {
            *b = 0.1 * (i as f64 + 1.0);
        }
        let a_snap = adapter.a_frozen.clone();
        let b_snap = adapter.b_trainable.clone();
        let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3, 0.75];
        let _ = adapter
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        let _ = adapter
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        let _ = adapter
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        assert_eq!(adapter.a_frozen, a_snap);
        assert_eq!(adapter.b_trainable, b_snap);
    }

    #[test]
    fn apply_b_grad_dim_mismatch_rejected() {
        let mut adapter = LoraFaAdapter::new(default_cfg(5, 3, 2, 2.0), 0)
            .expect("LoraFaAdapter::new should succeed with valid config");
        let bad = vec![0.0_f64; 5]; // expected 3*2=6
        assert!(matches!(
            adapter.apply_b_grad(&bad, 0.01),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn n_trainable_equals_out_times_rank() {
        let adapter = LoraFaAdapter::new(default_cfg(8, 12, 4, 8.0), 0)
            .expect("LoraFaAdapter::new should succeed with valid config");
        assert_eq!(adapter.n_trainable(), 12 * 4);
    }
}
