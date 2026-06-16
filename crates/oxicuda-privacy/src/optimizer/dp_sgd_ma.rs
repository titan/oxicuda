//! DP-SGD with an integrated Moments Accountant (Abadi et al. 2016).
//!
//! Reference: Abadi M, Chu A, Goodfellow I, McMahan HB, Mironov I, Talwar K,
//! Zhang L (2016), "Deep Learning with Differential Privacy", CCS 2016 —
//! Algorithm 1 (DP-SGD) together with the *moments accountant* (Section 3.2),
//! here realised through the equivalent Rényi-DP accounting of the Sampled
//! Gaussian Mechanism (Mironov-Talwar-Zhang 2019).
//!
//! # Algorithm (one step, Algorithm 1)
//! Given per-sample gradients `g_1, …, g_L` of a Poisson-subsampled lot
//! (expected lot size `L = q · N`, sampling rate `q`):
//! 1. Per-sample clip: `ḡ_i = g_i · min(1, C / ‖g_i‖₂)`.
//! 2. Sum: `G = Σ_i ḡ_i`.
//! 3. Add Gaussian noise: `G̃ = G + 𝒩(0, σ²C²·I)`.
//! 4. Descend: `θ ← θ − η · G̃ / L` where `L` is the (expected) lot size.
//!
//! After each step the accountant composes one more Sampled-Gaussian RDP term
//! `q, σ` so that the spent `(ε, δ)` budget can be queried at any time and the
//! training loop can stop once the target ε is exhausted.
//!
//! Unlike [`super::dp_sgd_microbatch::DpSgdMicrobatch`], this optimiser performs
//! strict *per-sample* clipping (microbatch size 1) and owns a privacy
//! accountant, giving an end-to-end privacy-tracked DP-SGD.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::PrivacyHandle;
use crate::mechanism::sampled_gaussian::{SampledGaussianConfig, SampledGaussianMechanism};

/// Configuration for DP-SGD with a moments accountant.
#[derive(Debug, Clone)]
pub struct DpSgdMaConfig {
    /// Learning rate `η > 0`.
    pub learning_rate: f64,
    /// Per-sample L2 clipping bound `C > 0`.
    pub clip_norm: f64,
    /// Gaussian noise multiplier `σ > 0` (noise std = `σ · C`).
    pub noise_multiplier: f64,
    /// Poisson subsampling rate `q ∈ (0, 1]` (expected lot fraction `L / N`).
    pub sampling_rate: f64,
    /// Maximum integer Rényi order tracked by the accountant (≥ 2).
    pub max_order: usize,
}

impl DpSgdMaConfig {
    /// Construct and validate a [`DpSgdMaConfig`].
    ///
    /// # Errors
    /// - `InvalidParameter` for non-positive learning rate, non-positive /
    ///   non-finite noise multiplier, out-of-range sampling rate, or
    ///   `max_order < 2`.
    /// - `NonPositiveSensitivity` if `clip_norm ≤ 0`.
    pub fn new(
        learning_rate: f64,
        clip_norm: f64,
        noise_multiplier: f64,
        sampling_rate: f64,
        max_order: usize,
    ) -> PrivacyResult<Self> {
        if learning_rate <= 0.0 || !learning_rate.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "learning_rate must be positive and finite, got {learning_rate}"
            )));
        }
        if clip_norm <= 0.0 || !clip_norm.is_finite() {
            return Err(PrivacyError::NonPositiveSensitivity(clip_norm));
        }
        if noise_multiplier <= 0.0 || !noise_multiplier.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "noise_multiplier must be positive and finite, got {noise_multiplier}"
            )));
        }
        if !(sampling_rate > 0.0 && sampling_rate <= 1.0) {
            return Err(PrivacyError::InvalidParameter(format!(
                "sampling_rate must be in (0, 1], got {sampling_rate}"
            )));
        }
        if max_order < 2 {
            return Err(PrivacyError::InvalidParameter(format!(
                "max_order must be ≥ 2, got {max_order}"
            )));
        }
        Ok(Self {
            learning_rate,
            clip_norm,
            noise_multiplier,
            sampling_rate,
            max_order,
        })
    }
}

/// Per-sample L2 gradient clipping (Abadi et al. 2016, step 1).
///
/// Each row is scaled by `min(1, clip_norm / ‖row‖₂)` so its L2 norm is at most
/// `clip_norm`; rows already within the bound are unchanged (direction
/// preserved). The clipped rows are returned as a fresh matrix.
///
/// # Errors
/// - `EmptyInput` if `per_sample_grads` is empty or rows are empty.
/// - `NonPositiveSensitivity` if `clip_norm ≤ 0`.
/// - `DimensionMismatch` if rows are not all the same length.
pub fn clip_gradients(
    per_sample_grads: &[Vec<f64>],
    clip_norm: f64,
) -> PrivacyResult<Vec<Vec<f64>>> {
    if per_sample_grads.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if clip_norm <= 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(clip_norm));
    }
    let dim = per_sample_grads[0].len();
    if dim == 0 {
        return Err(PrivacyError::EmptyInput);
    }
    let mut out = Vec::with_capacity(per_sample_grads.len());
    for row in per_sample_grads {
        if row.len() != dim {
            return Err(PrivacyError::DimensionMismatch {
                expected: dim,
                got: row.len(),
            });
        }
        let norm = row.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > clip_norm {
            let scale = clip_norm / norm;
            out.push(row.iter().map(|&x| x * scale).collect());
        } else {
            out.push(row.clone());
        }
    }
    Ok(out)
}

/// Mutable DP-SGD-MA state: parameters, step counter, and the moments
/// accountant.
#[derive(Debug, Clone)]
pub struct DpSgdMaState {
    /// Current parameter vector `θ`.
    pub theta: Vec<f64>,
    /// Number of completed optimisation steps.
    pub step: usize,
    /// The Sampled-Gaussian moments accountant.
    accountant: SampledGaussianMechanism,
}

impl DpSgdMaState {
    /// Query the spent `(ε, δ)` budget: returns ε for the supplied `delta`
    /// given the RDP composed so far.
    ///
    /// # Errors
    /// - `InvalidDelta` if `delta ∉ (0, 1)`.
    /// - `InvalidParameter` if the accountant produced no finite ε (e.g. before
    ///   any step has been taken).
    pub fn spent_epsilon(&self, delta: f64) -> PrivacyResult<f64> {
        self.accountant.get_epsilon(delta)
    }

    /// Read-only access to the underlying accountant.
    #[must_use]
    pub fn accountant(&self) -> &SampledGaussianMechanism {
        &self.accountant
    }
}

/// DP-SGD optimiser with an integrated moments accountant.
#[derive(Debug, Clone)]
pub struct DpSgdMa {
    cfg: DpSgdMaConfig,
}

impl DpSgdMa {
    /// Construct a DP-SGD-MA optimiser and a fresh zero state of dimension
    /// `dim`.
    ///
    /// # Errors
    /// - `InvalidParameter` if `dim == 0`.
    /// - Propagates configuration validation errors.
    pub fn new(cfg: DpSgdMaConfig, dim: usize) -> PrivacyResult<(Self, DpSgdMaState)> {
        if dim == 0 {
            return Err(PrivacyError::InvalidParameter("dim must be ≥ 1".into()));
        }
        let acct_cfg =
            SampledGaussianConfig::new(cfg.sampling_rate, cfg.noise_multiplier, cfg.max_order)?;
        let accountant = SampledGaussianMechanism::new(&acct_cfg)?;
        let state = DpSgdMaState {
            theta: vec![0.0; dim],
            step: 0,
            accountant,
        };
        Ok((Self { cfg }, state))
    }

    /// Read-only access to the configuration.
    #[must_use]
    pub fn config(&self) -> &DpSgdMaConfig {
        &self.cfg
    }

    /// Add Gaussian noise `𝒩(0, σ²C²·I)` to a summed gradient and average by the
    /// (expected) lot size, returning the descent direction `G̃ / lot_size`.
    ///
    /// # Errors
    /// - `EmptyInput` if `summed_grad` is empty.
    /// - `InvalidParameter` if `lot_size == 0`.
    pub fn add_noise(
        &self,
        summed_grad: &[f64],
        lot_size: usize,
        handle: &mut PrivacyHandle,
    ) -> PrivacyResult<Vec<f64>> {
        if summed_grad.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        if lot_size == 0 {
            return Err(PrivacyError::InvalidParameter(
                "lot_size must be ≥ 1".into(),
            ));
        }
        let dim = summed_grad.len();
        let noise_std = self.cfg.noise_multiplier * self.cfg.clip_norm;
        let noise = handle.generate_gaussian_noise(noise_std, dim)?;
        let inv = 1.0 / (lot_size as f64);
        let out = summed_grad
            .iter()
            .zip(noise)
            .map(|(&g, n)| (g + n) * inv)
            .collect();
        Ok(out)
    }

    /// Execute one DP-SGD step over a Poisson-sampled lot of per-sample
    /// gradients and advance the moments accountant by one composition.
    ///
    /// # Errors
    /// - `EmptyInput` if `lot_grads` is empty.
    /// - `DimensionMismatch` if any gradient row length differs from
    ///   `state.theta.len()`.
    /// - Propagates clipping / noise / accountant errors.
    pub fn step(
        &self,
        state: &mut DpSgdMaState,
        lot_grads: &[Vec<f64>],
        handle: &mut PrivacyHandle,
    ) -> PrivacyResult<()> {
        let dim = state.theta.len();
        if lot_grads.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        for row in lot_grads {
            if row.len() != dim {
                return Err(PrivacyError::DimensionMismatch {
                    expected: dim,
                    got: row.len(),
                });
            }
        }

        // Step 1: per-sample clip.
        let clipped = clip_gradients(lot_grads, self.cfg.clip_norm)?;

        // Step 2: sum.
        let mut g_sum = vec![0.0f64; dim];
        for row in &clipped {
            for (acc, &v) in g_sum.iter_mut().zip(row.iter()) {
                *acc += v;
            }
        }

        // Steps 3-4: add noise + average by lot size, then descend.
        let direction = self.add_noise(&g_sum, lot_grads.len(), handle)?;
        let lr = self.cfg.learning_rate;
        for (t, &d) in state.theta.iter_mut().zip(direction.iter()) {
            *t -= lr * d;
        }

        // Moments-accountant update: compose one Sampled-Gaussian RDP term.
        state.accountant.compose(1)?;
        state.step += 1;
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(lr: f64, c: f64, sigma: f64, q: f64) -> DpSgdMaConfig {
        DpSgdMaConfig::new(lr, c, sigma, q, 32).expect("cfg")
    }

    fn l2(v: &[f64]) -> f64 {
        v.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    // 1. Clipping reduces the norm of large gradients to ≤ C.
    #[test]
    fn clip_reduces_norm() {
        let grads = vec![vec![3.0, 4.0], vec![6.0, 8.0]]; // norms 5, 10
        let clipped = clip_gradients(&grads, 1.0).expect("clip");
        for g in &clipped {
            assert!(l2(g) <= 1.0 + 1e-12, "norm {} > C", l2(g));
        }
    }

    // 2. Clipping preserves direction (clipped grad is a positive multiple).
    #[test]
    fn clip_preserves_direction() {
        let grads = vec![vec![3.0, 4.0]]; // norm 5
        let clipped = clip_gradients(&grads, 1.0).expect("clip");
        // Should be [3,4]·(1/5) = [0.6, 0.8].
        assert!((clipped[0][0] - 0.6).abs() < 1e-12);
        assert!((clipped[0][1] - 0.8).abs() < 1e-12);
    }

    // 3. Small gradients are left unclipped.
    #[test]
    fn small_grad_unclipped() {
        let grads = vec![vec![0.1, 0.1]]; // norm ≈ 0.1414 < 1
        let clipped = clip_gradients(&grads, 1.0).expect("clip");
        assert!((clipped[0][0] - 0.1).abs() < 1e-12);
        assert!((clipped[0][1] - 0.1).abs() < 1e-12);
    }

    // 4. Spent ε increases with the number of steps.
    #[test]
    fn spent_epsilon_increases_with_steps() {
        let (opt, mut state) = DpSgdMa::new(cfg(0.01, 1.0, 1.0, 0.01), 3).expect("new");
        let mut handle = PrivacyHandle::new(80, 1);
        let lot = vec![vec![0.1, 0.2, -0.1]; 4];
        for _ in 0..10 {
            opt.step(&mut state, &lot, &mut handle).expect("step");
        }
        let eps_10 = state.spent_epsilon(1e-5).expect("e");
        for _ in 0..40 {
            opt.step(&mut state, &lot, &mut handle).expect("step");
        }
        let eps_50 = state.spent_epsilon(1e-5).expect("e");
        assert!(
            eps_50 > eps_10,
            "more steps → more budget: {eps_50} > {eps_10}"
        );
    }

    // 5. Larger noise multiplier → smaller spent ε for the same step count.
    #[test]
    fn more_noise_less_budget() {
        let (opt_lo, mut s_lo) = DpSgdMa::new(cfg(0.01, 1.0, 1.0, 0.02), 2).expect("lo");
        let (opt_hi, mut s_hi) = DpSgdMa::new(cfg(0.01, 1.0, 5.0, 0.02), 2).expect("hi");
        let mut h_lo = PrivacyHandle::new(80, 3);
        let mut h_hi = PrivacyHandle::new(80, 3);
        let lot = vec![vec![0.3, -0.4]; 4];
        for _ in 0..20 {
            opt_lo.step(&mut s_lo, &lot, &mut h_lo).expect("s");
            opt_hi.step(&mut s_hi, &lot, &mut h_hi).expect("s");
        }
        let eps_lo = s_lo.spent_epsilon(1e-5).expect("e");
        let eps_hi = s_hi.spent_epsilon(1e-5).expect("e");
        assert!(eps_hi < eps_lo, "more noise → less ε: {eps_hi} < {eps_lo}");
    }

    // 6. The descent direction averages over the lot size.
    #[test]
    fn noise_averaged_over_lot() {
        // Zero-noise edge case is disallowed (σ>0), so use a tiny σ and a large
        // lot of identical clipped grads; the averaged direction ≈ clipped grad.
        let cfg = DpSgdMaConfig::new(0.1, 100.0, 1e-9, 1.0, 16).expect("cfg");
        let (opt, _state) = DpSgdMa::new(cfg, 2).expect("new");
        let mut handle = PrivacyHandle::new(80, 5);
        // Summed grad of a lot of 4 copies of [1, 2] = [4, 8]; lot_size 4.
        let direction = opt.add_noise(&[4.0, 8.0], 4, &mut handle).expect("noise");
        assert!((direction[0] - 1.0).abs() < 1e-4, "dir0 {}", direction[0]);
        assert!((direction[1] - 2.0).abs() < 1e-4, "dir1 {}", direction[1]);
    }

    // 7. clip_norm = 0 errors.
    #[test]
    fn clip_norm_zero_error() {
        let grads = vec![vec![1.0, 1.0]];
        assert!(matches!(
            clip_gradients(&grads, 0.0),
            Err(PrivacyError::NonPositiveSensitivity(_))
        ));
        assert!(DpSgdMaConfig::new(0.01, 0.0, 1.0, 0.1, 16).is_err());
    }

    // 8. lot_size = 0 and empty inputs error.
    #[test]
    fn lot_and_empty_errors() {
        let (opt, mut state) = DpSgdMa::new(cfg(0.01, 1.0, 1.0, 0.1), 2).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        assert!(matches!(
            opt.add_noise(&[1.0, 2.0], 0, &mut handle),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            opt.add_noise(&[], 4, &mut handle),
            Err(PrivacyError::EmptyInput)
        ));
        let empty: Vec<Vec<f64>> = vec![];
        assert!(matches!(
            opt.step(&mut state, &empty, &mut handle),
            Err(PrivacyError::EmptyInput)
        ));
        assert!(matches!(
            clip_gradients(&empty, 1.0),
            Err(PrivacyError::EmptyInput)
        ));
    }

    // 9. add_noise output has the same dimension as the input.
    #[test]
    fn output_shape() {
        let (opt, _state) = DpSgdMa::new(cfg(0.01, 1.0, 1.0, 0.1), 5).expect("new");
        let mut handle = PrivacyHandle::new(80, 7);
        let direction = opt.add_noise(&[0.0; 5], 8, &mut handle).expect("noise");
        assert_eq!(direction.len(), 5);
        for v in &direction {
            assert!(v.is_finite(), "non-finite {v}");
        }
    }

    // 10. Per-sample clipping is independent across rows.
    #[test]
    fn per_sample_independent() {
        // One huge row and one small row: the small one stays untouched.
        let grads = vec![vec![100.0, 0.0], vec![0.05, 0.0]];
        let clipped = clip_gradients(&grads, 1.0).expect("clip");
        assert!(
            (clipped[0][0] - 1.0).abs() < 1e-12,
            "big row clipped to unit"
        );
        assert!((clipped[1][0] - 0.05).abs() < 1e-12, "small row unchanged");
    }

    // 11. The optimiser makes progress toward a quadratic minimum despite the
    //     DP noise, landing in a neighbourhood of the target.
    #[test]
    fn converges_on_quadratic() {
        // Minimise 0.5·‖θ − target‖²; per-sample grad = θ − target.
        // Moderate clip C=2 (rarely clips near the optimum), σ=0.7 with a large
        // lot of 64 keeps the averaged-direction noise std ≈ σC/L ≈ 0.022, so
        // the iterate settles within a loose neighbourhood of the target.
        let target = [0.5f64, -0.25];
        let cfg = DpSgdMaConfig::new(0.05, 2.0, 0.7, 1.0, 16).expect("cfg");
        let (opt, mut state) = DpSgdMa::new(cfg, 2).expect("new");
        let mut handle = PrivacyHandle::new(80, 11);
        for _ in 0..500 {
            let g: Vec<f64> = state
                .theta
                .iter()
                .zip(target.iter())
                .map(|(t, x)| t - x)
                .collect();
            let lot: Vec<Vec<f64>> = (0..64).map(|_| g.clone()).collect();
            opt.step(&mut state, &lot, &mut handle).expect("step");
        }
        for (got, want) in state.theta.iter().zip(target.iter()) {
            assert!(
                (got - want).abs() < 0.2,
                "DP-SGD should reach a neighbourhood: got {got} want {want}"
            );
            assert!(got.is_finite());
        }
        // The accountant must report a finite, positive spent budget.
        let eps = state.spent_epsilon(1e-5).expect("e");
        assert!(eps.is_finite() && eps > 0.0, "ε = {eps}");
    }

    // 12. step advances the accountant; ε is queryable and finite after steps.
    #[test]
    fn accountant_advances_and_finite() {
        let (opt, mut state) = DpSgdMa::new(cfg(0.01, 1.0, 1.1, 0.01), 3).expect("new");
        let mut handle = PrivacyHandle::new(80, 13);
        let lot = vec![vec![0.1, 0.1, 0.1]; 4];
        for _ in 0..100 {
            opt.step(&mut state, &lot, &mut handle).expect("step");
        }
        assert_eq!(state.step, 100);
        let eps = state.spent_epsilon(1e-5).expect("e");
        assert!(eps.is_finite() && eps > 0.0, "ε = {eps}");
        // Accountant should report 100 composed steps' worth of RDP (> 0).
        assert!(state.accountant().rdp_curve().iter().all(|&r| r >= 0.0));
    }

    // 13. Dimension mismatch and invalid config error.
    #[test]
    fn dim_mismatch_and_config_errors() {
        let (opt, mut state) = DpSgdMa::new(cfg(0.01, 1.0, 1.0, 0.1), 4).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        let bad = vec![vec![0.0; 3]];
        assert!(matches!(
            opt.step(&mut state, &bad, &mut handle),
            Err(PrivacyError::DimensionMismatch { .. })
        ));
        assert!(DpSgdMa::new(cfg(0.01, 1.0, 1.0, 0.1), 0).is_err());
        assert!(DpSgdMaConfig::new(-1.0, 1.0, 1.0, 0.1, 16).is_err());
        assert!(DpSgdMaConfig::new(0.01, 1.0, 1.0, 1.5, 16).is_err());
        assert!(DpSgdMaConfig::new(0.01, 1.0, 1.0, 0.1, 1).is_err());
    }
}
