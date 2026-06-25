//! DP-MASR — DP optimiser with **Adaptive Sensitivity Refinement** of the
//! gradient-clipping bound.
//!
//! References:
//! - Andrew, Thakkar, McMahan & Ramaswamy (2021), "Differentially Private
//!   Learning with Adaptive Clipping", NeurIPS 2021 — privately estimates the
//!   gradient-norm quantile and adapts the clip via geometric gradient descent
//!   on the quantile-matching loss.
//! - Pichapati, Suresh, Yu, Reddi & Kumar (2019), "AdaCliP" — coordinate-wise
//!   adaptive clipping motivation.
//!
//! # Why refine sensitivity adaptively
//! A DP optimiser's noise scales with the L2-sensitivity `C` (the clip bound):
//! noise std `= σ·C`.  A `C` set too high injects excessive noise; too low
//! over-clips and biases the gradient.  The right `C` is roughly the `γ`-quantile
//! of the per-example gradient-norm distribution — but that distribution is
//! data-dependent and *must itself be estimated privately*.  DP-MASR tracks it
//! online:
//!
//! 1. **Private quantile feedback.** For a target quantile `γ` (e.g. 0.5), each
//!    example reports the *bit* `b_i = 𝟙[‖g_i‖ ≤ C]`.  Their mean
//!    `b̄ = (1/m)Σ b_i ∈ [0, 1]` is privatised by adding Gaussian noise of std
//!    `σ_b/m` (the bit-vector has L2-sensitivity `1` so the mean has
//!    sensitivity `1/m`), giving a private estimate `b̃` of `P(‖g‖ ≤ C)`.
//! 2. **Geometric clip update.** Match `b̃` to `γ` by gradient descent on the
//!    quantile loss in *log-space* (so `C` stays positive):
//!    `C ← C · exp(−η_C · (b̃ − γ))`.  If too many norms fall under `C`
//!    (`b̃ > γ`) the clip shrinks; if too few, it grows.
//! 3. **Refined-sensitivity gradient step.** Clip each `g_i` to the *current* `C`,
//!    average, add Gaussian noise `N(0, σ²C²/m·I)` (sensitivity refined to the
//!    just-updated `C`), and take an SGD step.
//!
//! Both the quantile bit-mean and the gradient mean are released under the
//! Gaussian mechanism, so each step spends a `ρ`-zCDP budget of
//! `ρ = 1/(2σ_b²) + 1/(2σ²)` (two independent Gaussian releases); compose across
//! steps additively via [`crate::accounting::zcdp`].

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for DP-MASR.
#[derive(Debug, Clone)]
pub struct DpMasrConfig {
    /// Initial clip bound `C₀ > 0`.
    pub init_clip: f64,
    /// Target gradient-norm quantile `γ ∈ (0, 1)` the clip should track.
    pub target_quantile: f64,
    /// Clip-adaptation learning rate `η_C > 0` (geometric / log-space step).
    pub clip_lr: f64,
    /// Noise multiplier σ for the *gradient* mean.
    pub sigma: f64,
    /// Noise std `σ_b` for the quantile bit-mean release.
    pub quantile_sigma: f64,
    /// Model learning rate `η > 0`.
    pub learning_rate: f64,
    /// Lower clamp on the clip bound to avoid collapse to zero.
    pub min_clip: f64,
}

impl DpMasrConfig {
    /// Construct and validate the configuration.
    ///
    /// # Errors
    /// - `InvalidParameter` for non-positive `init_clip`, `clip_lr`, `sigma`,
    ///   `quantile_sigma`, `learning_rate`, or `min_clip`; or
    ///   `target_quantile ∉ (0, 1)`; or `init_clip < min_clip`.
    pub fn new(
        init_clip: f64,
        target_quantile: f64,
        clip_lr: f64,
        sigma: f64,
        quantile_sigma: f64,
        learning_rate: f64,
        min_clip: f64,
    ) -> PrivacyResult<Self> {
        for (name, v) in [
            ("init_clip", init_clip),
            ("clip_lr", clip_lr),
            ("sigma", sigma),
            ("quantile_sigma", quantile_sigma),
            ("learning_rate", learning_rate),
            ("min_clip", min_clip),
        ] {
            if v <= 0.0 {
                return Err(PrivacyError::InvalidParameter(format!(
                    "{name} must be positive, got {v}"
                )));
            }
        }
        if !(target_quantile > 0.0 && target_quantile < 1.0) {
            return Err(PrivacyError::InvalidParameter(format!(
                "target_quantile must be in (0,1), got {target_quantile}"
            )));
        }
        if init_clip < min_clip {
            return Err(PrivacyError::InvalidParameter(
                "init_clip must be ≥ min_clip".into(),
            ));
        }
        Ok(Self {
            init_clip,
            target_quantile,
            clip_lr,
            sigma,
            quantile_sigma,
            learning_rate,
            min_clip,
        })
    }

    /// Per-step `ρ`-zCDP cost: `1/(2σ_b²) + 1/(2σ²)` (two Gaussian releases —
    /// the quantile bit-mean and the gradient mean).
    #[must_use]
    pub fn step_zcdp_rho(&self) -> f64 {
        1.0 / (2.0 * self.quantile_sigma * self.quantile_sigma)
            + 1.0 / (2.0 * self.sigma * self.sigma)
    }
}

/// Mutable DP-MASR optimiser state.
pub struct DpMasrState {
    params: Vec<f64>,
    /// Current adaptive clip bound `C`.
    clip: f64,
    n_params: usize,
    step: usize,
    /// Accumulated `ρ`-zCDP spent so far.
    rho_spent: f64,
}

impl DpMasrState {
    /// Construct initial state from a config (params zeroed).
    ///
    /// # Errors
    /// - `InvalidParameter` if `n_params == 0`.
    pub fn new(n_params: usize, cfg: &DpMasrConfig) -> PrivacyResult<Self> {
        if n_params == 0 {
            return Err(PrivacyError::InvalidParameter(
                "n_params must be ≥ 1".into(),
            ));
        }
        Ok(Self {
            params: vec![0.0; n_params],
            clip: cfg.init_clip,
            n_params,
            step: 0,
            rho_spent: 0.0,
        })
    }

    /// Current parameter vector.
    #[must_use]
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// Current adaptive clip bound `C` (the refined sensitivity).
    #[must_use]
    pub fn clip(&self) -> f64 {
        self.clip
    }

    /// Steps taken.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Total `ρ`-zCDP spent across all steps so far.
    #[must_use]
    pub fn rho_spent(&self) -> f64 {
        self.rho_spent
    }

    /// Execute one DP-MASR step over a batch of `m` per-example gradients.
    ///
    /// Refines the clip from a private quantile estimate, then takes a clipped +
    /// noised SGD step at the refined sensitivity.  Returns the private
    /// gradient-norm quantile estimate `b̃` used for the clip update (for
    /// diagnostics).
    ///
    /// # Errors
    /// - `EmptyInput` if `per_example_grads` is empty.
    /// - `DimensionMismatch` if any per-example gradient length `!= n_params`.
    pub fn step(
        &mut self,
        per_example_grads: &[Vec<f64>],
        cfg: &DpMasrConfig,
        rng: &mut LcgRng,
    ) -> PrivacyResult<f64> {
        let m = per_example_grads.len();
        if m == 0 {
            return Err(PrivacyError::EmptyInput);
        }
        for g in per_example_grads {
            if g.len() != self.n_params {
                return Err(PrivacyError::DimensionMismatch {
                    expected: self.n_params,
                    got: g.len(),
                });
            }
        }

        // ── 1. Private quantile feedback at the CURRENT clip ──────────────────
        let mut under = 0.0f64;
        for g in per_example_grads {
            let norm = g.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm <= self.clip {
                under += 1.0;
            }
        }
        let bit_mean = under / m as f64;
        // Privatise the bit-mean: sensitivity 1/m, noise std σ_b/m.
        let (zq, _) = rng.normal_pair();
        let noisy_bit_mean = (bit_mean + zq * cfg.quantile_sigma / m as f64).clamp(0.0, 1.0);

        // ── 2. Geometric clip refinement in log-space ────────────────────────
        let delta = noisy_bit_mean - cfg.target_quantile;
        self.clip = (self.clip * (-cfg.clip_lr * delta).exp()).max(cfg.min_clip);

        // ── 3. Refined-sensitivity gradient step ─────────────────────────────
        let mut grad_sum = vec![0.0f64; self.n_params];
        for g in per_example_grads {
            let norm = g.iter().map(|&x| x * x).sum::<f64>().sqrt();
            let norm_safe = norm.max(f64::EPSILON);
            let scale = (self.clip / norm_safe).min(1.0);
            for j in 0..self.n_params {
                grad_sum[j] += g[j] * scale;
            }
        }
        // Mean of clipped gradients.
        for v in grad_sum.iter_mut() {
            *v /= m as f64;
        }
        // Gaussian noise N(0, σ²C²/m²) on the mean (sensitivity C/m).
        let noise_std = cfg.sigma * self.clip / m as f64;
        let mut i = 0;
        while i < self.n_params {
            let (a, b) = rng.normal_pair();
            grad_sum[i] += a * noise_std;
            if i + 1 < self.n_params {
                grad_sum[i + 1] += b * noise_std;
            }
            i += 2;
        }

        // SGD update.
        for (p, &g) in self.params.iter_mut().zip(grad_sum.iter()) {
            *p -= cfg.learning_rate * g;
        }

        self.step += 1;
        self.rho_spent += cfg.step_zcdp_rho();
        Ok(noisy_bit_mean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DpMasrConfig {
        DpMasrConfig::new(5.0, 0.5, 0.2, 1.0, 1.0, 0.05, 0.01).expect("cfg")
    }

    /// Build a batch of `m` gradients each of fixed norm `r` in `dim` dims.
    fn fixed_norm_batch(m: usize, dim: usize, r: f64) -> Vec<Vec<f64>> {
        // Vector (r/√dim, …): L2 norm = r.
        let comp = r / (dim as f64).sqrt();
        (0..m).map(|_| vec![comp; dim]).collect()
    }

    #[test]
    fn test_clip_shrinks_when_all_norms_below() {
        // If every gradient norm is well below the clip, the (near-1) quantile
        // estimate exceeds γ=0.5, so the clip must shrink over time.
        let c = cfg();
        let mut rng = LcgRng::new(13);
        let mut s = DpMasrState::new(4, &c).expect("state");
        let batch = fixed_norm_batch(64, 4, 0.5); // norm 0.5 << clip 5.0
        let c0 = s.clip();
        for _ in 0..30 {
            s.step(&batch, &c, &mut rng).expect("step");
        }
        assert!(
            s.clip() < c0,
            "clip should shrink toward the data scale: {} < {c0}",
            s.clip()
        );
    }

    #[test]
    fn test_clip_grows_when_all_norms_above() {
        // If every gradient norm is above the clip, the (near-0) quantile
        // estimate is below γ=0.5, so the clip must grow.
        let c = DpMasrConfig::new(0.1, 0.5, 0.2, 1.0, 1.0, 0.05, 0.01).expect("cfg");
        let mut rng = LcgRng::new(21);
        let mut s = DpMasrState::new(4, &c).expect("state");
        let batch = fixed_norm_batch(64, 4, 3.0); // norm 3.0 >> clip 0.1
        let c0 = s.clip();
        for _ in 0..30 {
            s.step(&batch, &c, &mut rng).expect("step");
        }
        assert!(
            s.clip() > c0,
            "clip should grow toward the data scale: {} > {c0}",
            s.clip()
        );
    }

    #[test]
    fn test_clip_converges_near_quantile() {
        // Half the batch has small norm, half large; with γ=0.5 the clip should
        // settle between the two norm levels (so ~50% of examples fall under it).
        let c = DpMasrConfig::new(1.0, 0.5, 0.3, 0.5, 0.5, 0.01, 0.001).expect("cfg");
        let mut rng = LcgRng::new(101);
        let mut s = DpMasrState::new(3, &c).expect("state");
        let mut batch = fixed_norm_batch(50, 3, 0.5);
        batch.extend(fixed_norm_batch(50, 3, 4.0));
        for _ in 0..200 {
            s.step(&batch, &c, &mut rng).expect("step");
        }
        let clip = s.clip();
        assert!(
            clip > 0.5 && clip < 4.0,
            "clip {clip} should land between the 0.5 and 4.0 norm levels"
        );
    }

    #[test]
    fn test_rho_accounting_accumulates() {
        let c = cfg();
        let mut rng = LcgRng::new(3);
        let mut s = DpMasrState::new(2, &c).expect("state");
        let batch = fixed_norm_batch(8, 2, 1.0);
        for _ in 0..10 {
            s.step(&batch, &c, &mut rng).expect("step");
        }
        let expected = 10.0 * c.step_zcdp_rho();
        assert!(
            (s.rho_spent() - expected).abs() < 1e-12,
            "ρ spent {} should equal {expected}",
            s.rho_spent()
        );
    }

    #[test]
    fn test_determinism_same_seed() {
        let c = cfg();
        let batch = fixed_norm_batch(16, 3, 1.5);
        let run = || {
            let mut rng = LcgRng::new(777);
            let mut s = DpMasrState::new(3, &c).expect("state");
            for _ in 0..20 {
                s.step(&batch, &c, &mut rng).expect("step");
            }
            (s.params().to_vec(), s.clip())
        };
        let a = run();
        let b = run();
        assert_eq!(a.0, b.0);
        assert!((a.1 - b.1).abs() < 1e-15);
    }

    #[test]
    fn test_invalid_inputs() {
        assert!(DpMasrConfig::new(-1.0, 0.5, 0.2, 1.0, 1.0, 0.05, 0.01).is_err());
        assert!(DpMasrConfig::new(5.0, 0.0, 0.2, 1.0, 1.0, 0.05, 0.01).is_err());
        assert!(DpMasrConfig::new(5.0, 1.0, 0.2, 1.0, 1.0, 0.05, 0.01).is_err());
        assert!(DpMasrConfig::new(0.001, 0.5, 0.2, 1.0, 1.0, 0.05, 0.01).is_err()); // init < min
        let c = cfg();
        let mut rng = LcgRng::new(0);
        let mut s = DpMasrState::new(3, &c).expect("state");
        assert!(s.step(&[], &c, &mut rng).is_err());
        assert!(s.step(&[vec![1.0, 2.0]], &c, &mut rng).is_err());
    }
}
