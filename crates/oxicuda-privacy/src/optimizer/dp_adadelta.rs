//! Differentially Private AdaDelta optimizer (per-sample clip + Gaussian noise
//! + AdaDelta running-average accumulators).
//!
//! # References
//! - Zeiler MD (2012),
//!   *"ADADELTA: An Adaptive Learning Rate Method"*, arXiv:1212.5701 — the
//!   non-private update rule with squared-gradient and squared-update
//!   running averages, where the parameter step is
//!   `Δθ = − sqrt(E[Δθ²] + ε) / sqrt(E[g²] + ε) · g`.
//! - Abadi M, Chu A, Goodfellow I, McMahan HB, Mironov I, Talwar K,
//!   Zhang L (2016),
//!   *"Deep Learning with Differential Privacy"*, CCS 2016 — the
//!   per-sample L2 clipping + Gaussian noise framework used to privatise
//!   the aggregate gradient before plugging it into the AdaDelta update.
//!
//! # Algorithm (one step)
//!
//! Given per-sample gradients `g₁, …, g_B` (each in `ℝ^p`):
//!
//! ```text
//!     g̃ᵢ      = gᵢ · min(1, C / ‖gᵢ‖₂)              (per-sample L2 clip)
//!     G        = Σᵢ g̃ᵢ                              (aggregate)
//!     G       += N(0, σ²·C²·I)                       (Gaussian noise)
//!     g_priv   = G / B                               (mean)
//!     E[g²]_t  = ρ · E[g²]_{t−1} + (1−ρ) · g_priv²    (RMS of gradient)
//!     Δθ       = − sqrt(E[Δθ²]_{t−1} + ε) / sqrt(E[g²]_t + ε) · g_priv
//!     E[Δθ²]_t = ρ · E[Δθ²]_{t−1} + (1−ρ) · Δθ²       (RMS of update)
//!     θ_t      = θ_{t−1} + Δθ
//! ```

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::PrivacyHandle;

// ─── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the DP-AdaDelta optimizer.
#[derive(Debug, Clone, Copy)]
pub struct DpAdaDeltaConfig {
    /// Per-sample L2 clipping bound `C > 0`.
    pub clip_norm: f64,
    /// Gaussian noise multiplier `σ ≥ 0` (noise std = `σ · C`).
    pub noise_sigma: f64,
    /// EMA decay `ρ ∈ (0, 1)` (Zeiler 2012 §3.1; typical value 0.95).
    pub rho: f64,
    /// Numerical stabiliser `ε > 0` (typical value `1e-6`).
    pub eps: f64,
}

impl Default for DpAdaDeltaConfig {
    fn default() -> Self {
        Self {
            clip_norm: 1.0,
            noise_sigma: 1.0,
            rho: 0.95,
            eps: 1e-6,
        }
    }
}

// ─── State ────────────────────────────────────────────────────────────────────

/// Mutable DP-AdaDelta state.
#[derive(Debug, Clone)]
pub struct DpAdaDeltaState {
    /// Parameter vector `θ`.
    pub theta: Vec<f64>,
    /// Running average `E[g²]_t` of the squared privatised gradient.
    pub accum_grad: Vec<f64>,
    /// Running average `E[Δθ²]_t` of the squared parameter update.
    pub accum_update: Vec<f64>,
    /// Step counter (number of completed `step` calls).
    pub step: usize,
}

// ─── Optimizer ────────────────────────────────────────────────────────────────

/// Stateless DP-AdaDelta optimizer holding the validated configuration.
#[derive(Debug, Clone, Copy)]
pub struct DpAdaDelta {
    /// Active configuration.
    pub cfg: DpAdaDeltaConfig,
}

impl DpAdaDelta {
    /// Construct a new DP-AdaDelta optimizer plus a fresh zero-initialised
    /// state of dimension `dim`.
    ///
    /// # Errors
    /// `InvalidParameter` if any of:
    /// - `clip_norm ≤ 0` or non-finite,
    /// - `noise_sigma < 0` or non-finite,
    /// - `rho ∉ (0, 1)` or non-finite,
    /// - `eps ≤ 0` or non-finite,
    /// - `dim == 0`.
    pub fn new(cfg: DpAdaDeltaConfig, dim: usize) -> PrivacyResult<(Self, DpAdaDeltaState)> {
        if !cfg.clip_norm.is_finite() || cfg.clip_norm <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "clip_norm must be > 0 and finite, got {}",
                cfg.clip_norm
            )));
        }
        if !cfg.noise_sigma.is_finite() || cfg.noise_sigma < 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "noise_sigma must be >= 0 and finite, got {}",
                cfg.noise_sigma
            )));
        }
        if !cfg.rho.is_finite() || cfg.rho <= 0.0 || cfg.rho >= 1.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "rho must lie in (0, 1) and be finite, got {}",
                cfg.rho
            )));
        }
        if !cfg.eps.is_finite() || cfg.eps <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "eps must be > 0 and finite, got {}",
                cfg.eps
            )));
        }
        if dim == 0 {
            return Err(PrivacyError::InvalidParameter("dim must be >= 1".into()));
        }
        let state = DpAdaDeltaState {
            theta: vec![0.0; dim],
            accum_grad: vec![0.0; dim],
            accum_update: vec![0.0; dim],
            step: 0,
        };
        Ok((Self { cfg }, state))
    }

    /// Borrow the active configuration.
    #[must_use]
    pub fn config(&self) -> &DpAdaDeltaConfig {
        &self.cfg
    }

    /// Execute one DP-AdaDelta step given a batch of per-sample gradients.
    ///
    /// # Arguments
    /// - `state`: mutable optimizer state.
    /// - `per_sample_grads`: per-sample gradient rows; each row length
    ///   must equal `state.theta.len()`.
    /// - `handle`: privacy handle (RNG source for the Gaussian draw).
    ///
    /// # Errors
    /// - `EmptyInput` if `per_sample_grads` is empty.
    /// - `DimensionMismatch` if any per-sample gradient row length
    ///   differs from `state.theta.len()`.
    pub fn step(
        &self,
        state: &mut DpAdaDeltaState,
        per_sample_grads: &[Vec<f64>],
        handle: &mut PrivacyHandle,
    ) -> PrivacyResult<()> {
        let dim = state.theta.len();
        if per_sample_grads.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        for row in per_sample_grads {
            if row.len() != dim {
                return Err(PrivacyError::DimensionMismatch {
                    expected: dim,
                    got: row.len(),
                });
            }
        }

        let c = self.cfg.clip_norm;
        // 1 & 2: per-sample L2 clip + aggregate.
        let mut g_sum = vec![0.0f64; dim];
        for sample_grad in per_sample_grads {
            let norm_sq: f64 = sample_grad.iter().map(|&g| g * g).sum();
            let norm = norm_sq.sqrt();
            let scale = if norm > c {
                c / norm.max(f64::EPSILON)
            } else {
                1.0
            };
            for (acc, &g) in g_sum.iter_mut().zip(sample_grad.iter()) {
                *acc += g * scale;
            }
        }

        // 3: Gaussian noise N(0, σ²·C²·I).
        let noise_std = self.cfg.noise_sigma * c;
        if noise_std > 0.0 {
            let noise = handle.generate_gaussian_noise(noise_std, dim)?;
            for (acc, n) in g_sum.iter_mut().zip(noise) {
                *acc += n;
            }
        }

        // 4: convert to the expected per-sample gradient g_priv = G / B.
        let batch_inv = 1.0 / (per_sample_grads.len() as f64);
        for v in g_sum.iter_mut() {
            *v *= batch_inv;
        }
        let g_priv = g_sum;

        // 5: update E[g²]_t = ρ · E[g²]_{t−1} + (1 − ρ) · g_priv².
        let rho = self.cfg.rho;
        let one_minus_rho = 1.0 - rho;
        for (a, &g) in state.accum_grad.iter_mut().zip(g_priv.iter()) {
            *a = rho * *a + one_minus_rho * g * g;
        }

        // 6: compute Δθ and update E[Δθ²]_t.
        let eps = self.cfg.eps;
        for (((accum_u, &accum_g), theta), &g) in state
            .accum_update
            .iter_mut()
            .zip(state.accum_grad.iter())
            .zip(state.theta.iter_mut())
            .zip(g_priv.iter())
        {
            let rms_update = (*accum_u + eps).sqrt();
            let rms_grad = (accum_g + eps).sqrt();
            let delta = -rms_update / rms_grad * g;
            *accum_u = rho * *accum_u + one_minus_rho * delta * delta;
            *theta += delta;
        }

        state.step += 1;
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(clip_norm: f64, noise_sigma: f64, rho: f64, eps: f64) -> DpAdaDeltaConfig {
        DpAdaDeltaConfig {
            clip_norm,
            noise_sigma,
            rho,
            eps,
        }
    }

    // 1. new with dim == 0 → InvalidParameter.
    #[test]
    fn test_new_dim_zero_errors() {
        let cfg = make_cfg(1.0, 0.0, 0.95, 1e-6);
        let res = DpAdaDelta::new(cfg, 0);
        assert!(matches!(res, Err(PrivacyError::InvalidParameter(_))));
    }

    // 2. clip_norm ≤ 0 → InvalidParameter.
    #[test]
    fn test_new_bad_clip_norm_errors() {
        assert!(matches!(
            DpAdaDelta::new(make_cfg(0.0, 1.0, 0.95, 1e-6), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaDelta::new(make_cfg(-0.5, 1.0, 0.95, 1e-6), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaDelta::new(make_cfg(f64::NAN, 1.0, 0.95, 1e-6), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaDelta::new(make_cfg(f64::INFINITY, 1.0, 0.95, 1e-6), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
    }

    // 3. noise_sigma < 0 → InvalidParameter.
    #[test]
    fn test_new_bad_noise_sigma_errors() {
        assert!(matches!(
            DpAdaDelta::new(make_cfg(1.0, -0.1, 0.95, 1e-6), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaDelta::new(make_cfg(1.0, f64::NAN, 0.95, 1e-6), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        // noise_sigma == 0 is permitted (test-only mode).
        assert!(DpAdaDelta::new(make_cfg(1.0, 0.0, 0.95, 1e-6), 2).is_ok());
    }

    // 4. rho ≤ 0 or ≥ 1 → InvalidParameter.
    #[test]
    fn test_new_bad_rho_errors() {
        assert!(matches!(
            DpAdaDelta::new(make_cfg(1.0, 0.0, 0.0, 1e-6), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaDelta::new(make_cfg(1.0, 0.0, 1.0, 1e-6), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaDelta::new(make_cfg(1.0, 0.0, -0.1, 1e-6), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaDelta::new(make_cfg(1.0, 0.0, 1.5, 1e-6), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaDelta::new(make_cfg(1.0, 0.0, f64::NAN, 1e-6), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
    }

    // 5. eps ≤ 0 → InvalidParameter.
    #[test]
    fn test_new_bad_eps_errors() {
        assert!(matches!(
            DpAdaDelta::new(make_cfg(1.0, 0.0, 0.95, 0.0), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaDelta::new(make_cfg(1.0, 0.0, 0.95, -1e-6), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaDelta::new(make_cfg(1.0, 0.0, 0.95, f64::INFINITY), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
    }

    // 6. step with empty grads → EmptyInput.
    #[test]
    fn test_step_empty_grads_errors() {
        let cfg = make_cfg(1.0, 0.0, 0.95, 1e-6);
        let (opt, mut state) = DpAdaDelta::new(cfg, 3).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        let empty: Vec<Vec<f64>> = vec![];
        let r = opt.step(&mut state, &empty, &mut handle);
        assert!(matches!(r, Err(PrivacyError::EmptyInput)));
    }

    // 7. step with grad length mismatch → DimensionMismatch.
    #[test]
    fn test_step_dim_mismatch_errors() {
        let cfg = make_cfg(1.0, 0.0, 0.95, 1e-6);
        let (opt, mut state) = DpAdaDelta::new(cfg, 4).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        let bad = vec![vec![0.0; 3]];
        let r = opt.step(&mut state, &bad, &mut handle);
        assert!(matches!(r, Err(PrivacyError::DimensionMismatch { .. })));
    }

    // 8. Zero-noise + single grad: theta moves in -g direction.
    #[test]
    fn test_zero_noise_single_grad_sign_matches_minus_grad() {
        let cfg = make_cfg(100.0, 0.0, 0.95, 1e-6);
        let (opt, mut state) = DpAdaDelta::new(cfg, 3).expect("new");
        let mut handle = PrivacyHandle::new(80, 1);
        let grad = vec![1.0f64, -2.0, 0.5];
        opt.step(&mut state, std::slice::from_ref(&grad), &mut handle)
            .expect("step");
        for (i, &g) in grad.iter().enumerate() {
            assert!(
                state.theta[i] * g < 0.0,
                "theta[{i}] = {} should have opposite sign to g = {}",
                state.theta[i],
                g
            );
        }
    }

    // 9. accum_grad monotone in zero-noise case (each coordinate's
    //    accumulator is non-decreasing once g_priv stops shrinking, but
    //    the EMA may briefly drop if the recent grad is smaller than the
    //    running average; we verify the simpler invariant that, given a
    //    *constant* per-sample gradient, accum_grad converges to
    //    g_priv² monotonically from below).
    #[test]
    fn test_accum_grad_converges_under_constant_grad() {
        let cfg = make_cfg(100.0, 0.0, 0.95, 1e-6);
        let (opt, mut state) = DpAdaDelta::new(cfg, 2).expect("new");
        let mut handle = PrivacyHandle::new(80, 11);
        let grad = vec![1.5f64, -2.5];
        let g_priv_sq = [1.5_f64 * 1.5, 2.5 * 2.5];
        let mut prev = state.accum_grad.clone();
        for _ in 0..200 {
            opt.step(&mut state, std::slice::from_ref(&grad), &mut handle)
                .expect("step");
            for (j, &a) in state.accum_grad.iter().enumerate() {
                assert!(
                    a + 1e-15 >= prev[j],
                    "accum_grad decreased at coord {j}: {a} < {}",
                    prev[j]
                );
                assert!(a <= g_priv_sq[j] + 1e-9);
            }
            prev = state.accum_grad.clone();
        }
        for (a, &target) in state.accum_grad.iter().zip(g_priv_sq.iter()) {
            assert!(
                (a - target).abs() < 1e-3,
                "accum_grad {a} did not converge to {target}"
            );
        }
    }

    // 10. Deterministic with fixed RNG seed.
    #[test]
    fn test_deterministic_for_fixed_seed() {
        let cfg = make_cfg(1.0, 0.5, 0.95, 1e-6);
        let (opt_a, mut state_a) = DpAdaDelta::new(cfg, 5).expect("a");
        let (opt_b, mut state_b) = DpAdaDelta::new(cfg, 5).expect("b");
        let mut h_a = PrivacyHandle::new(80, 7777);
        let mut h_b = PrivacyHandle::new(80, 7777);
        let grads: Vec<Vec<f64>> = (0..6)
            .map(|i| (0..5).map(|j| 0.1 * (i + j) as f64).collect())
            .collect();
        for _ in 0..5 {
            opt_a.step(&mut state_a, &grads, &mut h_a).expect("a");
            opt_b.step(&mut state_b, &grads, &mut h_b).expect("b");
        }
        for (a, b) in state_a.theta.iter().zip(state_b.theta.iter()) {
            assert!((a - b).abs() < 1e-15, "theta diverged: {a} vs {b}");
        }
        for (a, b) in state_a.accum_grad.iter().zip(state_b.accum_grad.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
        for (a, b) in state_a.accum_update.iter().zip(state_b.accum_update.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    // 11. Multiple steps: theta updates non-trivially and accumulators
    //     become non-zero.
    #[test]
    fn test_multiple_steps_update_state() {
        let cfg = make_cfg(1.0, 0.0, 0.95, 1e-6);
        let (opt, mut state) = DpAdaDelta::new(cfg, 3).expect("new");
        let mut handle = PrivacyHandle::new(80, 1);
        let grads = vec![vec![0.5f64, -0.3, 0.7]];
        for _ in 0..5 {
            opt.step(&mut state, &grads, &mut handle).expect("step");
        }
        assert_eq!(state.step, 5);
        for &t in &state.theta {
            assert!(t.abs() > 1e-9, "theta unchanged");
        }
        for &a in &state.accum_grad {
            assert!(a > 0.0);
        }
        for &a in &state.accum_update {
            assert!(a > 0.0);
        }
    }

    // 12. eps prevents NaN on the first step (sqrt(0 + eps) is finite).
    #[test]
    fn test_eps_prevents_nan_on_first_step() {
        let cfg = make_cfg(1.0, 0.0, 0.95, 1e-12);
        let (opt, mut state) = DpAdaDelta::new(cfg, 2).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        let grads = vec![vec![1.0f64, -1.0]];
        opt.step(&mut state, &grads, &mut handle).expect("step");
        for v in state
            .theta
            .iter()
            .chain(state.accum_grad.iter())
            .chain(state.accum_update.iter())
        {
            assert!(v.is_finite(), "non-finite {v}");
        }
    }

    // 13. Convergence on a convex quadratic (loss decreases over many
    //     noiseless steps).
    #[test]
    fn test_convex_quadratic_loss_decreases() {
        // f(θ) = 0.5 · ‖θ − x*‖² with x* = [1, −0.5, 0.25].
        let x_star = [1.0f64, -0.5, 0.25];
        let cfg = make_cfg(100.0, 0.0, 0.95, 1e-6);
        let (opt, mut state) = DpAdaDelta::new(cfg, 3).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        let loss = |t: &[f64]| -> f64 {
            t.iter()
                .zip(x_star.iter())
                .map(|(a, b)| 0.5 * (a - b).powi(2))
                .sum::<f64>()
        };
        let loss_before = loss(&state.theta);
        for _ in 0..300 {
            let grad: Vec<f64> = state
                .theta
                .iter()
                .zip(x_star.iter())
                .map(|(t, x)| t - x)
                .collect();
            opt.step(&mut state, &[grad], &mut handle).expect("step");
        }
        let loss_after = loss(&state.theta);
        assert!(
            loss_after < loss_before * 0.5,
            "loss did not decrease enough: {loss_after} vs {loss_before}"
        );
    }

    // 14. accum_update tracks Δθ² (becomes positive after at least one
    //     non-zero Δθ).
    #[test]
    fn test_accum_update_tracks_delta_squared() {
        let cfg = make_cfg(1.0, 0.0, 0.95, 1e-6);
        let (opt, mut state) = DpAdaDelta::new(cfg, 4).expect("new");
        let mut handle = PrivacyHandle::new(80, 3);
        // First, zero gradients ⇒ no update, accum_update stays 0.
        let zero_g = vec![vec![0.0; 4]];
        opt.step(&mut state, &zero_g, &mut handle).expect("step");
        for &a in &state.accum_update {
            assert!(a.abs() < 1e-15);
        }
        // Now a non-zero gradient ⇒ accum_update[j] > 0 wherever Δθ ≠ 0.
        let grads = vec![vec![0.4f64, -0.7, 0.1, 0.0]];
        opt.step(&mut state, &grads, &mut handle).expect("step");
        assert!(state.accum_update[0] > 0.0);
        assert!(state.accum_update[1] > 0.0);
        assert!(state.accum_update[2] > 0.0);
        // Coord 3 received a zero gradient ⇒ no update yet.
        assert!(state.accum_update[3].abs() < 1e-15);
    }

    // 15. step counter increments correctly.
    #[test]
    fn test_step_counter_increments() {
        let cfg = make_cfg(1.0, 0.0, 0.95, 1e-6);
        let (opt, mut state) = DpAdaDelta::new(cfg, 2).expect("new");
        let mut handle = PrivacyHandle::new(80, 1);
        let grads = vec![vec![0.1f64, 0.2]];
        for k in 1..=10 {
            opt.step(&mut state, &grads, &mut handle).expect("step");
            assert_eq!(state.step, k);
        }
    }

    // 16. State remains finite after many noisy steps.
    #[test]
    fn test_many_noisy_steps_finite() {
        let cfg = make_cfg(1.0, 0.5, 0.9, 1e-6);
        let (opt, mut state) = DpAdaDelta::new(cfg, 5).expect("new");
        let mut handle = PrivacyHandle::new(80, 9_001);
        let grads: Vec<Vec<f64>> = (0..8)
            .map(|i| (0..5).map(|j| 0.05 * (i + j) as f64).collect())
            .collect();
        for _ in 0..200 {
            opt.step(&mut state, &grads, &mut handle).expect("step");
        }
        for v in state
            .theta
            .iter()
            .chain(state.accum_grad.iter())
            .chain(state.accum_update.iter())
        {
            assert!(v.is_finite(), "non-finite {v}");
        }
        assert_eq!(state.step, 200);
    }

    // 17. Per-sample L2 clip is respected for antipodal cancellation.
    #[test]
    fn test_per_sample_clip_respected() {
        let cfg = make_cfg(0.5, 0.0, 0.95, 1e-6);
        let (opt, mut state) = DpAdaDelta::new(cfg, 2).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        let antipodal = vec![vec![3.0f64, 4.0], vec![-3.0f64, -4.0]];
        opt.step(&mut state, &antipodal, &mut handle).expect("step");
        for &t in &state.theta {
            assert!(t.abs() < 1e-12, "theta {t} should be ~0 (antipodal cancel)");
        }
    }

    // 18. config() accessor returns the input config.
    #[test]
    fn test_config_accessor_returns_input() {
        let cfg = make_cfg(2.0, 0.5, 0.9, 1e-5);
        let (opt, _) = DpAdaDelta::new(cfg, 3).expect("new");
        let back = opt.config();
        assert!((back.clip_norm - cfg.clip_norm).abs() < 1e-12);
        assert!((back.noise_sigma - cfg.noise_sigma).abs() < 1e-12);
        assert!((back.rho - cfg.rho).abs() < 1e-12);
        assert!((back.eps - cfg.eps).abs() < 1e-12);
    }
}
