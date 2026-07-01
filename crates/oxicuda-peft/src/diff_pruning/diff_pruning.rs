use crate::handle::LcgRng;

/// Configuration for the Diff-Pruning L0 regulariser.
///
/// Default values follow the original paper (β=2/3, γ=−0.1, ζ=1.1, λ_L0=0.01).
#[derive(Debug, Clone)]
pub struct DiffPrunerConfig {
    /// Temperature β for the Hard Concrete distribution.
    pub beta: f32,
    /// Stretch lower bound γ (negative, e.g. −0.1).
    pub gamma: f32,
    /// Stretch upper bound ζ (> 1, e.g. 1.1).
    pub zeta: f32,
    /// L0 regularisation coefficient λ.
    pub lambda_l0: f32,
}

impl Default for DiffPrunerConfig {
    fn default() -> Self {
        Self {
            beta: 2.0 / 3.0,
            gamma: -0.1,
            zeta: 1.1,
            lambda_l0: 0.01,
        }
    }
}

/// Diff-Pruning: learn a sparse binary mask over weight differences using Hard Concrete relaxation.
///
/// For each weight element i the effective update is `z_i * delta_i` where `z_i ∈ [0, 1]`
/// is drawn from the Hard Concrete distribution parameterised by `log_alpha_i`.
/// The base weight `base_w` is fixed; only `delta` and `log_alpha` are trained.
#[derive(Debug, Clone)]
pub struct DiffPruner {
    /// Frozen base weight vector.
    pub base_w: Vec<f32>,
    /// Log-α parameters controlling the per-element mask probability.
    pub log_alpha: Vec<f32>,
    /// Trainable weight difference vector.
    pub delta: Vec<f32>,
    /// Hard Concrete hyperparameters.
    pub config: DiffPrunerConfig,
}

impl DiffPruner {
    /// Construct a `DiffPruner` from a base weight vector.
    ///
    /// `log_alpha` is initialised from N(0, 0.01); `delta` is zero-initialised.
    #[must_use]
    pub fn new(w: &[f32], cfg: DiffPrunerConfig, rng: &mut LcgRng) -> Self {
        let n = w.len();
        let mut log_alpha = vec![0.0_f32; n];
        rng.fill_normal(&mut log_alpha);
        for v in log_alpha.iter_mut() {
            *v *= 0.01;
        }
        let delta = vec![0.0_f32; n];
        Self {
            base_w: w.to_vec(),
            log_alpha,
            delta,
            config: cfg,
        }
    }

    /// Sample a stochastic mask using the Hard Concrete distribution.
    ///
    /// For each element: `s = sigmoid((log_α - log(u/(1-u))) / β)`,
    /// then `s_bar = s * (ζ - γ) + γ`, then `z = clamp(s_bar, 0, 1)`.
    #[must_use]
    pub fn compute_mask(&self, rng: &mut LcgRng) -> Vec<f32> {
        let beta = self.config.beta;
        let gamma = self.config.gamma;
        let zeta = self.config.zeta;
        let stretch = zeta - gamma;
        self.log_alpha
            .iter()
            .map(|&log_a| {
                let u = (rng.next_f32() + 1e-12).min(1.0 - 1e-12);
                let log_odds = (u / (1.0 - u)).ln();
                let s = sigmoid_f32((log_a - log_odds) / beta);
                let s_bar = s * stretch + gamma;
                s_bar.clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Compute the deterministic (inference-time) mask without randomness.
    ///
    /// Uses `sigmoid(log_α) * (ζ - γ) + γ` clamped to `[0, 1]`.
    #[must_use]
    pub fn compute_mask_deterministic(&self) -> Vec<f32> {
        let gamma = self.config.gamma;
        let zeta = self.config.zeta;
        let stretch = zeta - gamma;
        self.log_alpha
            .iter()
            .map(|&log_a| {
                let s = sigmoid_f32(log_a);
                let s_bar = s * stretch + gamma;
                s_bar.clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Compute the masked weight: `base_w + compute_mask(rng) * delta`.
    #[must_use]
    pub fn forward(&self, rng: &mut LcgRng) -> Vec<f32> {
        let mask = self.compute_mask(rng);
        self.base_w
            .iter()
            .zip(self.delta.iter())
            .zip(mask.iter())
            .map(|((&w, &d), &z)| w + z * d)
            .collect()
    }

    /// Compute the expected L0 pseudo-norm (regularisation term).
    ///
    /// `L0 ≈ Σ_i sigmoid(log_α_i - β · log(-γ/ζ))`
    #[must_use]
    pub fn l0_regularizer(&self) -> f32 {
        let beta = self.config.beta;
        let gamma = self.config.gamma;
        let zeta = self.config.zeta;
        // log(-gamma/zeta): gamma is negative so -gamma is positive
        let log_ratio = (-gamma / zeta).ln();
        self.log_alpha
            .iter()
            .map(|&log_a| sigmoid_f32(log_a - beta * log_ratio))
            .sum()
    }
}

/// Numerically stable sigmoid function.
#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // -----------------------------------------------------------------------
    // Test 1: with delta=0 (zero-initialised by construction), forward = base_w
    // -----------------------------------------------------------------------
    #[test]
    fn zero_delta_forward_equals_base() {
        let base = vec![1.0_f32, -2.0, 3.5, 0.0];
        let mut rng_init = LcgRng::new(1);
        // delta is zero-initialised in DiffPruner::new
        let pruner = DiffPruner::new(&base, DiffPrunerConfig::default(), &mut rng_init);
        let mut rng_fwd = LcgRng::new(99);
        let out = pruner.forward(&mut rng_fwd);
        assert_eq!(out.len(), base.len());
        for (i, (&b, &o)) in base.iter().zip(out.iter()).enumerate() {
            assert!(
                (o - b).abs() < 1e-6,
                "index {i}: expected base {b}, got {o}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: very large log_alpha forces mask → 1 and forward ≈ base + delta
    // -----------------------------------------------------------------------
    #[test]
    fn large_log_alpha_mask_saturates_to_one() {
        let base = vec![1.0_f32, 2.0, 3.0];
        let delta = vec![0.5_f32, -1.0, 0.25];
        let mut rng_init = LcgRng::new(2);
        let mut pruner = DiffPruner::new(&base, DiffPrunerConfig::default(), &mut rng_init);
        // Force log_alpha to large positive: s ≈ 1, s_bar ≈ 1.1, clamp → 1.0
        for v in pruner.log_alpha.iter_mut() {
            *v = 100.0;
        }
        pruner.delta = delta.clone();

        // Verify deterministic mask is exactly 1.0
        let mask_det = pruner.compute_mask_deterministic();
        for &m in mask_det.iter() {
            assert!(
                (m - 1.0).abs() < 1e-5,
                "deterministic mask expected 1.0 with log_alpha=100, got {m}"
            );
        }

        // Stochastic mask with any RNG also saturates to 1.0 when log_alpha=100
        let mut rng_fwd = LcgRng::new(50);
        let out = pruner.forward(&mut rng_fwd);
        for (i, (&b, (&d, &o))) in base.iter().zip(delta.iter().zip(out.iter())).enumerate() {
            let expected = b + d;
            assert!(
                (o - expected).abs() < 1e-5,
                "index {i}: expected {expected}, got {o}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: very negative log_alpha forces mask → 0 and forward ≈ base
    // -----------------------------------------------------------------------
    #[test]
    fn negative_log_alpha_mask_saturates_to_zero() {
        let base = vec![2.0_f32, -1.5, 0.5, 3.0];
        let mut rng_init = LcgRng::new(3);
        let mut pruner = DiffPruner::new(&base, DiffPrunerConfig::default(), &mut rng_init);
        for v in pruner.log_alpha.iter_mut() {
            *v = -100.0;
        }
        // Large delta to make any non-zero mask conspicuous
        pruner.delta = vec![10.0_f32, -10.0, 10.0, -10.0];

        // Verify deterministic mask is exactly 0.0
        let mask_det = pruner.compute_mask_deterministic();
        for &m in mask_det.iter() {
            assert!(
                m.abs() < 1e-5,
                "deterministic mask expected 0.0 with log_alpha=-100, got {m}"
            );
        }

        let mut rng_fwd = LcgRng::new(60);
        let out = pruner.forward(&mut rng_fwd);
        for (i, (&b, &o)) in base.iter().zip(out.iter()).enumerate() {
            assert!(
                (o - b).abs() < 1e-5,
                "index {i}: expected base {b}, got {o}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: compute_mask_deterministic returns values in [0, 1]
    // -----------------------------------------------------------------------
    #[test]
    fn deterministic_mask_in_unit_interval() {
        let base = vec![0.1_f32; 20];
        let mut rng = LcgRng::new(4);
        let pruner = DiffPruner::new(&base, DiffPrunerConfig::default(), &mut rng);
        let mask = pruner.compute_mask_deterministic();
        assert_eq!(mask.len(), 20);
        for &m in mask.iter() {
            assert!((0.0..=1.0).contains(&m), "mask value {m} not in [0, 1]");
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: compute_mask_deterministic is reproducible (no RNG dependency)
    // -----------------------------------------------------------------------
    #[test]
    fn deterministic_mask_is_reproducible() {
        let base = vec![0.2_f32; 10];
        let mut rng1 = LcgRng::new(5);
        let mut rng2 = LcgRng::new(5);
        let pruner1 = DiffPruner::new(&base, DiffPrunerConfig::default(), &mut rng1);
        let pruner2 = DiffPruner::new(&base, DiffPrunerConfig::default(), &mut rng2);
        // Same seed → same log_alpha → same deterministic mask
        assert_eq!(
            pruner1.compute_mask_deterministic(),
            pruner2.compute_mask_deterministic()
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: compute_mask_deterministic analytic value for log_alpha = 0
    //
    // Default config: beta=2/3, gamma=-0.1, zeta=1.1, stretch=1.2.
    // s = sigmoid(0) = 0.5, s_bar = 0.5 * 1.2 - 0.1 = 0.5, clamp(0.5) = 0.5.
    // -----------------------------------------------------------------------
    #[test]
    fn deterministic_mask_analytic_for_zero_log_alpha() {
        let base = vec![0.0_f32];
        let mut rng = LcgRng::new(6);
        let mut pruner = DiffPruner::new(&base, DiffPrunerConfig::default(), &mut rng);
        pruner.log_alpha = vec![0.0];
        let mask = pruner.compute_mask_deterministic();
        assert_eq!(mask.len(), 1);
        assert!(
            (mask[0] - 0.5).abs() < 1e-5,
            "log_alpha=0 must give mask=0.5, got {}",
            mask[0]
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: l0_regularizer is finite and in [0, n]
    // -----------------------------------------------------------------------
    #[test]
    fn l0_regularizer_in_valid_range() {
        let n = 16_usize;
        let base = vec![0.0_f32; n];
        let mut rng = LcgRng::new(7);
        let pruner = DiffPruner::new(&base, DiffPrunerConfig::default(), &mut rng);
        let l0 = pruner.l0_regularizer();
        assert!(l0.is_finite(), "l0 must be finite, got {l0}");
        assert!(l0 >= 0.0, "l0 must be non-negative, got {l0}");
        assert!(l0 <= n as f32, "l0 must be ≤ n={n}, got {l0}");
    }

    // -----------------------------------------------------------------------
    // Test 8: forward outputs are all finite for random initialisation
    // -----------------------------------------------------------------------
    #[test]
    fn forward_finite_outputs() {
        let base: Vec<f32> = (0..8).map(|i| i as f32 * 0.5 - 2.0).collect();
        let mut rng_init = LcgRng::new(8);
        let mut pruner = DiffPruner::new(&base, DiffPrunerConfig::default(), &mut rng_init);
        pruner.delta = base.iter().map(|&v| v * 0.1).collect();
        let mut rng_fwd = LcgRng::new(9);
        for &v in pruner.forward(&mut rng_fwd).iter() {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
    }
}
