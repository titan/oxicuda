//! LoRA+ — Low-Rank Adaptation with separate learning rates for A and B.
//!
//! Reference: Hayou, S., Ghosh, N., & Yu, B. (2024). *LoRA+: Efficient Low Rank Adaptation
//! of Large Models*. <https://arxiv.org/abs/2402.12354>
//!
//! The forward pass is identical to vanilla LoRA, `y = s · B · (A · x)` with `s = α / rank`.
//! The training rule differs: the up-projection `B` is updated with a learning rate
//! `η_B = λ · η_A`, where `λ ≥ 16` is recommended. This asymmetry compensates for the
//! different sensitivities of `A` and `B` in low-rank space and provably accelerates
//! convergence (Hayou-Ghosh-Yu, Theorem 4.1).
//!
//! Closed-form gradients with `t = A · x` and upstream `g = ∂L/∂y`:
//!
//! ```text
//!   ∂L/∂A = s · (Bᵀ · g) · xᵀ         (rank × in)
//!   ∂L/∂B = s · g · tᵀ                (out × rank)
//! ```

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// `(dA, dB)` row-major gradients produced by [`LoraPlusAdapter::backward`].
pub type LoraPlusGrads = (Vec<f64>, Vec<f64>);

/// Hyper-parameter bundle for a single LoRA+ adapter.
#[derive(Clone, Debug)]
pub struct LoraPlusConfig {
    /// Input feature count (column count of `A`).
    pub in_features: usize,
    /// Output feature count (row count of `B`).
    pub out_features: usize,
    /// Low-rank dimension shared between `A` and `B`.
    pub rank: usize,
    /// Global LoRA scaling factor `α`; effective scale is `s = α / rank`.
    pub alpha: f64,
    /// Learning rate for the down-projection `A`.
    pub eta_a: f64,
    /// Ratio `λ` so that `η_B = λ · η_A`. Hayou et al. recommend `λ ≥ 16`.
    pub lambda_ratio: f64,
}

impl LoraPlusConfig {
    /// Effective scale `s = α / rank`.
    #[must_use]
    pub fn scale(&self) -> f64 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f64
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`PeftError::EmptyInput`] if any dimension is zero.
    /// - [`PeftError::RankTooLarge`] if `rank > min(in_features, out_features)`.
    /// - [`PeftError::Internal`] if `eta_a < 0` or `lambda_ratio < 0` (training rule
    ///   requires non-negative learning rates).
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
        if self.eta_a < 0.0 {
            return Err(PeftError::Internal {
                msg: format!("eta_a must be >= 0, got {}", self.eta_a),
            });
        }
        if self.lambda_ratio < 0.0 {
            return Err(PeftError::Internal {
                msg: format!("lambda_ratio must be >= 0, got {}", self.lambda_ratio),
            });
        }
        Ok(())
    }
}

/// LoRA+ adapter holding the down-projection `A`, up-projection `B`, and a captured config.
///
/// Layout (all row-major):
/// - `a` shape `[rank × in_features]`, sampled from `N(0, 1/√in_features)`.
/// - `b` shape `[out_features × rank]`, zero-initialised so the adapter is a no-op at start.
pub struct LoraPlusAdapter {
    cfg: LoraPlusConfig,
    a: Vec<f64>,
    b: Vec<f64>,
}

impl LoraPlusAdapter {
    /// Build a fresh adapter.
    ///
    /// `A` is drawn from `N(0, 1/√in_features)` using paired Box-Muller via
    /// [`LcgRng::next_normal_pair`]. `B` starts at zero.
    ///
    /// # Errors
    ///
    /// Forwards [`LoraPlusConfig::validate`] errors.
    pub fn new(cfg: LoraPlusConfig, seed: u64) -> PeftResult<Self> {
        cfg.validate()?;
        let mut rng = LcgRng::new(seed);
        let std_dev = 1.0_f64 / (cfg.in_features as f64).sqrt();
        let n = cfg.rank * cfg.in_features;
        let mut a = vec![0.0_f64; n];
        let mut i = 0;
        while i + 1 < n {
            let (u, v) = rng.next_normal_pair();
            a[i] = (u as f64) * std_dev;
            a[i + 1] = (v as f64) * std_dev;
            i += 2;
        }
        if i < n {
            a[i] = (rng.next_normal() as f64) * std_dev;
        }
        let b = vec![0.0_f64; cfg.out_features * cfg.rank];
        Ok(Self { cfg, a, b })
    }

    /// Borrow the down-projection in row-major `[rank × in_features]` layout.
    #[must_use]
    pub fn a(&self) -> &[f64] {
        &self.a
    }

    /// Borrow the up-projection in row-major `[out_features × rank]` layout.
    #[must_use]
    pub fn b(&self) -> &[f64] {
        &self.b
    }

    /// Effective learning rate for `B`: `η_B = λ · η_A`.
    #[must_use]
    pub fn eta_b(&self) -> f64 {
        self.cfg.eta_a * self.cfg.lambda_ratio
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

    /// Closed-form `(dA, dB)` for `loss = f(y)` with `grad_y = ∂L/∂y`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when `x.len() != in_features` or
    /// `grad_y.len() != out_features`.
    pub fn backward(&self, x: &[f64], grad_y: &[f64]) -> PeftResult<LoraPlusGrads> {
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
        let r = self.cfg.rank;
        let in_f = self.cfg.in_features;
        let out = self.cfg.out_features;
        let s = self.scale();
        let t = self.compute_ax(x);
        let mut grad_b = vec![0.0_f64; out * r];
        for (i, g_i) in grad_y.iter().enumerate() {
            let row = i * r;
            let scaled = s * g_i;
            for (k, t_k) in t.iter().enumerate() {
                grad_b[row + k] = scaled * t_k;
            }
        }
        let mut u = vec![0.0_f64; r];
        for (k, u_k) in u.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for (i, g_i) in grad_y.iter().enumerate() {
                acc += self.b[i * r + k] * g_i;
            }
            *u_k = acc;
        }
        let mut grad_a = vec![0.0_f64; r * in_f];
        for (k, u_k) in u.iter().enumerate() {
            let row = k * in_f;
            let scaled = s * u_k;
            for (j, x_j) in x.iter().enumerate() {
                grad_a[row + j] = scaled * x_j;
            }
        }
        Ok((grad_a, grad_b))
    }

    /// Apply LoRA+ SGD update: `A ← A − η_A · dA`, `B ← B − (λ · η_A) · dB`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when gradient shapes disagree with
    /// the adapter's expected sizes.
    pub fn apply_update(&mut self, grads: &LoraPlusGrads) -> PeftResult<()> {
        let (grad_a, grad_b) = grads;
        let exp_a = self.cfg.rank * self.cfg.in_features;
        let exp_b = self.cfg.out_features * self.cfg.rank;
        if grad_a.len() != exp_a {
            return Err(PeftError::DimensionMismatch {
                expected: exp_a,
                got: grad_a.len(),
            });
        }
        if grad_b.len() != exp_b {
            return Err(PeftError::DimensionMismatch {
                expected: exp_b,
                got: grad_b.len(),
            });
        }
        let eta_a = self.cfg.eta_a;
        let eta_b = self.eta_b();
        for (a, g) in self.a.iter_mut().zip(grad_a.iter()) {
            *a -= eta_a * g;
        }
        for (b, g) in self.b.iter_mut().zip(grad_b.iter()) {
            *b -= eta_b * g;
        }
        Ok(())
    }

    fn compute_ax(&self, x: &[f64]) -> Vec<f64> {
        let r = self.cfg.rank;
        let in_f = self.cfg.in_features;
        let mut t = vec![0.0_f64; r];
        for (k, t_k) in t.iter_mut().enumerate() {
            let row_start = k * in_f;
            let mut acc = 0.0_f64;
            for (j, x_j) in x.iter().enumerate() {
                acc += self.a[row_start + j] * x_j;
            }
            *t_k = acc;
        }
        t
    }

    fn compute_bt_scaled(&self, t: &[f64]) -> Vec<f64> {
        let r = self.cfg.rank;
        let out = self.cfg.out_features;
        let s = self.scale();
        let mut y = vec![0.0_f64; out];
        for (i, y_i) in y.iter_mut().enumerate() {
            let row_start = i * r;
            let mut acc = 0.0_f64;
            for (k, t_k) in t.iter().enumerate() {
                acc += self.b[row_start + k] * t_k;
            }
            *y_i = s * acc;
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(
        in_f: usize,
        out_f: usize,
        rank: usize,
        alpha: f64,
        eta_a: f64,
        lambda_ratio: f64,
    ) -> LoraPlusConfig {
        LoraPlusConfig {
            in_features: in_f,
            out_features: out_f,
            rank,
            alpha,
            eta_a,
            lambda_ratio,
        }
    }

    #[test]
    fn rejects_zero_rank() {
        let bad = cfg(4, 4, 0, 1.0, 0.01, 16.0);
        assert!(matches!(
            LoraPlusAdapter::new(bad, 0),
            Err(PeftError::EmptyInput)
        ));
    }

    #[test]
    fn rejects_zero_in_features() {
        let bad = cfg(0, 4, 2, 1.0, 0.01, 16.0);
        assert!(matches!(
            LoraPlusAdapter::new(bad, 0),
            Err(PeftError::EmptyInput)
        ));
    }

    #[test]
    fn rejects_negative_lambda_ratio() {
        let bad = cfg(4, 4, 2, 1.0, 0.01, -1.0);
        assert!(matches!(
            LoraPlusAdapter::new(bad, 0),
            Err(PeftError::Internal { .. })
        ));
    }

    #[test]
    fn rejects_negative_eta_a() {
        let bad = cfg(4, 4, 2, 1.0, -0.5, 16.0);
        assert!(matches!(
            LoraPlusAdapter::new(bad, 0),
            Err(PeftError::Internal { .. })
        ));
    }

    #[test]
    fn forward_is_zero_with_zero_b() {
        let adapter = LoraPlusAdapter::new(cfg(6, 4, 2, 4.0, 0.01, 16.0), 7).unwrap();
        let x: Vec<f64> = (0..6).map(|i| i as f64 - 2.5).collect();
        let y = adapter.forward(&x).unwrap();
        assert_eq!(y.len(), 4);
        for &v in &y {
            assert!(v.abs() < 1e-15, "expected zero output, got {v}");
        }
    }

    #[test]
    fn forward_changes_after_update_with_nonzero_db() {
        let mut adapter = LoraPlusAdapter::new(cfg(5, 4, 2, 4.0, 0.05, 16.0), 13).unwrap();
        let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3];
        let y_before = adapter.forward(&x).unwrap();
        let grad_a = vec![0.0_f64; 2 * 5];
        let grad_b: Vec<f64> = (0..4 * 2).map(|i| 0.1 * (i as f64 + 1.0)).collect();
        adapter.apply_update(&(grad_a, grad_b)).unwrap();
        let y_after = adapter.forward(&x).unwrap();
        let diff: f64 = y_before
            .iter()
            .zip(y_after.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "forward should change after non-zero dB update"
        );
    }

    #[test]
    fn eta_b_equals_eta_a_times_lambda() {
        let adapter = LoraPlusAdapter::new(cfg(4, 4, 2, 1.0, 0.03, 17.0), 0).unwrap();
        let expected = 0.03_f64 * 17.0;
        assert!((adapter.eta_b() - expected).abs() < 1e-15);
    }

    #[test]
    fn backward_matches_finite_differences_on_a() {
        let mut adapter = LoraPlusAdapter::new(cfg(4, 3, 2, 4.0, 0.01, 16.0), 99).unwrap();
        for (i, b) in adapter.b.iter_mut().enumerate() {
            *b = 0.15 * (i as f64 + 1.0);
        }
        let x = vec![0.5_f64, -1.0, 0.25, 0.75];
        let gy = vec![1.0_f64, -0.5, 0.25];
        let (grad_a, _) = adapter.backward(&x, &gy).unwrap();
        let eps = 1e-6_f64;
        for (k, &g_k) in grad_a.iter().enumerate() {
            let saved = adapter.a[k];
            adapter.a[k] = saved + eps;
            let yp = adapter.forward(&x).unwrap();
            adapter.a[k] = saved - eps;
            let ym = adapter.forward(&x).unwrap();
            adapter.a[k] = saved;
            let lp: f64 = gy.iter().zip(yp.iter()).map(|(a, b)| a * b).sum();
            let lm: f64 = gy.iter().zip(ym.iter()).map(|(a, b)| a * b).sum();
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - g_k).abs() < 1e-4, "A[{k}] FD={fd} analytic={g_k}");
        }
    }

    #[test]
    fn backward_matches_finite_differences_on_b() {
        let mut adapter = LoraPlusAdapter::new(cfg(4, 3, 2, 4.0, 0.01, 16.0), 99).unwrap();
        for (i, b) in adapter.b.iter_mut().enumerate() {
            *b = 0.15 * (i as f64 + 1.0);
        }
        let x = vec![0.5_f64, -1.0, 0.25, 0.75];
        let gy = vec![1.0_f64, -0.5, 0.25];
        let (_, grad_b) = adapter.backward(&x, &gy).unwrap();
        let eps = 1e-6_f64;
        for (k, &g_k) in grad_b.iter().enumerate() {
            let saved = adapter.b[k];
            adapter.b[k] = saved + eps;
            let yp = adapter.forward(&x).unwrap();
            adapter.b[k] = saved - eps;
            let ym = adapter.forward(&x).unwrap();
            adapter.b[k] = saved;
            let lp: f64 = gy.iter().zip(yp.iter()).map(|(a, b)| a * b).sum();
            let lm: f64 = gy.iter().zip(ym.iter()).map(|(a, b)| a * b).sum();
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - g_k).abs() < 1e-4, "B[{k}] FD={fd} analytic={g_k}");
        }
    }

    #[test]
    fn forward_output_length_equals_out_features() {
        let mut adapter = LoraPlusAdapter::new(cfg(7, 9, 3, 6.0, 0.01, 16.0), 11).unwrap();
        for (i, b) in adapter.b.iter_mut().enumerate() {
            *b = (i as f64 + 1.0) * 0.05;
        }
        let x = vec![1.0_f64; 7];
        let y = adapter.forward(&x).unwrap();
        assert_eq!(y.len(), 9);
    }

    #[test]
    fn forward_rejects_wrong_length_x() {
        let adapter = LoraPlusAdapter::new(cfg(5, 3, 2, 2.0, 0.01, 16.0), 0).unwrap();
        assert!(matches!(
            adapter.forward(&[1.0_f64, 2.0, 3.0]),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn backward_rejects_wrong_length_inputs() {
        let adapter = LoraPlusAdapter::new(cfg(5, 3, 2, 2.0, 0.01, 16.0), 0).unwrap();
        assert!(matches!(
            adapter.backward(&[0.1_f64; 5], &[0.1_f64; 2]),
            Err(PeftError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            adapter.backward(&[0.1_f64; 4], &[0.1_f64; 3]),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn b_changes_more_than_a_per_step_for_equal_grad_norms() {
        let mut adapter = LoraPlusAdapter::new(cfg(4, 4, 2, 4.0, 0.05, 16.0), 21).unwrap();
        let a_before = adapter.a.clone();
        let b_before = adapter.b.clone();
        let grad_a = vec![1.0_f64; 2 * 4];
        let grad_b = vec![1.0_f64; 4 * 2];
        adapter.apply_update(&(grad_a, grad_b)).unwrap();
        let da: f64 = adapter
            .a
            .iter()
            .zip(a_before.iter())
            .map(|(p, q)| (p - q).powi(2))
            .sum::<f64>()
            .sqrt();
        let db: f64 = adapter
            .b
            .iter()
            .zip(b_before.iter())
            .map(|(p, q)| (p - q).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            db > da * 10.0,
            "expected ||ΔB|| ≫ ||ΔA|| but got dB={db} dA={da}"
        );
    }

    #[test]
    fn deterministic_given_same_seed() {
        let c = cfg(8, 5, 3, 6.0, 0.01, 16.0);
        let a1 = LoraPlusAdapter::new(c.clone(), 42).unwrap();
        let a2 = LoraPlusAdapter::new(c, 42).unwrap();
        assert_eq!(a1.a, a2.a);
        assert_eq!(a1.b, a2.b);
    }

    #[test]
    fn different_seeds_give_different_a() {
        let c = cfg(8, 5, 3, 6.0, 0.01, 16.0);
        let a1 = LoraPlusAdapter::new(c.clone(), 1).unwrap();
        let a2 = LoraPlusAdapter::new(c, 2).unwrap();
        let diff: f64 =
            a1.a.iter()
                .zip(a2.a.iter())
                .map(|(p, q)| (p - q).abs())
                .sum();
        assert!(diff > 1e-6, "two seeds should yield different A");
    }

    #[test]
    fn apply_update_rejects_wrong_grad_sizes() {
        let mut adapter = LoraPlusAdapter::new(cfg(5, 3, 2, 2.0, 0.01, 16.0), 0).unwrap();
        let bad_a = vec![0.0_f64; 5];
        let good_b = vec![0.0_f64; 6];
        assert!(matches!(
            adapter.apply_update(&(bad_a, good_b)),
            Err(PeftError::DimensionMismatch { .. })
        ));
        let good_a = vec![0.0_f64; 10];
        let bad_b = vec![0.0_f64; 5];
        assert!(matches!(
            adapter.apply_update(&(good_a, bad_b)),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }
}
