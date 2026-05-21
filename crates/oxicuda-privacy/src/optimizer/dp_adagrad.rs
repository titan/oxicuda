//! Differentially Private AdaGrad optimizer (per-sample clip + Gaussian noise
//! + AdaGrad coordinate-adaptive scaling).
//!
//! References:
//! - Duchi JC, Hazan E, Singer Y (2011) "Adaptive Subgradient Methods for
//!   Online Learning and Stochastic Optimization", JMLR 12:2121–2159 — the
//!   non-private AdaGrad accumulator and step rule we adopt below.
//! - Abadi M, Chu A, Goodfellow I, McMahan HB, Mironov I, Talwar K, Zhang L
//!   (2016) "Deep Learning with Differential Privacy", CCS 2016 — the
//!   per-sample L2 clipping + Gaussian noise framework that wraps the
//!   privatised aggregate gradient.
//! - Asi H, Ullman J (2021) "Adaptive Differentially Private Algorithms",
//!   COLT 2021 — analysis of adaptive (per-coordinate) variants of the
//!   DP-SGD aggregate, motivating the AdaGrad denominator.
//!
//! # Algorithm (one step)
//! Given per-sample gradients g_1, ..., g_B (each in R^p):
//! 1. Per-sample L2 clip: g_i_tilde = g_i * min(1, C / ||g_i||_2).
//! 2. Aggregate: G = sum_i g_i_tilde.
//! 3. Gaussian noise: G += N(0, sigma^2 * C^2 * I), where sigma is the
//!    multiplier and C is `clip_norm` — matching the convention in `dp_adam`
//!    so that noise standard deviation equals `noise_sigma * clip_norm`.
//! 4. Convert to expected per-sample gradient: g_priv = G / B.
//! 5. AdaGrad accumulator: `accumulator[j] += g_priv[j]^2`.
//! 6. Adaptive parameter update:
//!    `theta[j] -= learning_rate * g_priv[j] / (sqrt(accumulator[j]) + eps)`.
//!
//! The `initial_accumulator` knob (Duchi-Hazan-Singer 2011 §3) lets callers
//! pre-warm the denominator and prevent the sqrt(0)+eps blow-up that
//! otherwise occurs on the first step at very small `eps`.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::PrivacyHandle;

/// Configuration for the DP-AdaGrad optimizer.
#[derive(Debug, Clone, Copy)]
pub struct DpAdaGradConfig {
    /// Learning rate eta > 0.
    pub learning_rate: f64,
    /// Per-sample L2 clipping bound C > 0.
    pub clip_norm: f64,
    /// Gaussian noise multiplier sigma >= 0 (noise std = sigma * clip_norm).
    pub noise_sigma: f64,
    /// Numerical stability term in the denominator: sqrt(accumulator) + eps;
    /// must be > 0.
    pub eps: f64,
    /// Initial value for every entry of the AdaGrad accumulator; must be >= 0.
    pub initial_accumulator: f64,
}

impl Default for DpAdaGradConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-2,
            clip_norm: 1.0,
            noise_sigma: 1.0,
            eps: 1e-8,
            initial_accumulator: 0.0,
        }
    }
}

/// Mutable DP-AdaGrad state.
#[derive(Debug, Clone)]
pub struct DpAdaGradState {
    /// Parameter vector theta.
    pub theta: Vec<f64>,
    /// Per-coordinate accumulator G_t = sum_{tau<=t} g_priv_tau^2 + initial.
    pub accumulator: Vec<f64>,
    /// Step counter (number of completed `step` calls).
    pub step: usize,
}

/// Stateless DP-AdaGrad optimizer holding the configuration.
#[derive(Debug, Clone, Copy)]
pub struct DpAdaGrad {
    /// Active configuration.
    pub cfg: DpAdaGradConfig,
}

impl DpAdaGrad {
    /// Construct a new DP-AdaGrad optimizer plus a fresh zero state.
    ///
    /// # Errors
    /// - `InvalidParameter` if `learning_rate <= 0`, `clip_norm <= 0`,
    ///   `noise_sigma < 0`, `eps <= 0`, `initial_accumulator < 0`, any of
    ///   the parameters is non-finite, or `dim == 0`.
    pub fn new(cfg: DpAdaGradConfig, dim: usize) -> PrivacyResult<(Self, DpAdaGradState)> {
        if !cfg.learning_rate.is_finite() || cfg.learning_rate <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "learning_rate must be > 0 and finite, got {}",
                cfg.learning_rate
            )));
        }
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
        if !cfg.eps.is_finite() || cfg.eps <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "eps must be > 0 and finite, got {}",
                cfg.eps
            )));
        }
        if !cfg.initial_accumulator.is_finite() || cfg.initial_accumulator < 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "initial_accumulator must be >= 0 and finite, got {}",
                cfg.initial_accumulator
            )));
        }
        if dim == 0 {
            return Err(PrivacyError::InvalidParameter("dim must be >= 1".into()));
        }
        let state = DpAdaGradState {
            theta: vec![0.0; dim],
            accumulator: vec![cfg.initial_accumulator; dim],
            step: 0,
        };
        Ok((Self { cfg }, state))
    }

    /// Borrow the active configuration.
    #[must_use]
    pub fn config(&self) -> &DpAdaGradConfig {
        &self.cfg
    }

    /// Execute one DP-AdaGrad step given a batch of per-sample gradients.
    ///
    /// # Arguments
    /// - `state`: mutable optimizer state.
    /// - `per_sample_grads`: per-sample gradient rows; each row must match
    ///   `state.theta.len()`.
    /// - `handle`: privacy handle (RNG source for Gaussian noise).
    ///
    /// # Errors
    /// - `EmptyInput` if `per_sample_grads` is empty.
    /// - `DimensionMismatch` if any per-sample gradient row length differs
    ///   from `state.theta.len()`.
    pub fn step(
        &self,
        state: &mut DpAdaGradState,
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

        // Step 1 & 2: per-sample L2 clip + sum.
        let c = self.cfg.clip_norm;
        let mut g_sum = vec![0.0f64; dim];
        for sample_grad in per_sample_grads {
            let norm_sq: f64 = sample_grad.iter().map(|&g| g * g).sum();
            let norm = norm_sq.sqrt();
            // scale = min(1, C / ||g||); when ||g|| == 0 keep the zero vector.
            let scale = if norm > c {
                c / norm.max(f64::EPSILON)
            } else {
                1.0
            };
            for (acc, &g) in g_sum.iter_mut().zip(sample_grad.iter()) {
                *acc += g * scale;
            }
        }

        // Step 3: Gaussian noise N(0, sigma^2 * C^2 * I).
        let noise_std = self.cfg.noise_sigma * c;
        if noise_std > 0.0 {
            let noise = handle.generate_gaussian_noise(noise_std, dim)?;
            for (acc, n) in g_sum.iter_mut().zip(noise) {
                *acc += n;
            }
        }

        // Step 4: convert to expected per-sample gradient g_priv = G / B.
        let batch_inv = 1.0 / (per_sample_grads.len() as f64);
        for v in g_sum.iter_mut() {
            *v *= batch_inv;
        }
        let g_priv = g_sum;

        // Step 5: AdaGrad accumulator update.
        for (a, &g) in state.accumulator.iter_mut().zip(g_priv.iter()) {
            *a += g * g;
        }

        // Step 6: coordinate-adaptive update.
        let lr = self.cfg.learning_rate;
        let eps = self.cfg.eps;
        for (j, t) in state.theta.iter_mut().enumerate() {
            let denom = state.accumulator[j].sqrt() + eps;
            *t -= lr * g_priv[j] / denom;
        }

        state.step += 1;
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(
        learning_rate: f64,
        clip_norm: f64,
        noise_sigma: f64,
        eps: f64,
        initial_accumulator: f64,
    ) -> DpAdaGradConfig {
        DpAdaGradConfig {
            learning_rate,
            clip_norm,
            noise_sigma,
            eps,
            initial_accumulator,
        }
    }

    // 1. dim == 0 -> InvalidParameter.
    #[test]
    fn test_new_dim_zero_errors() {
        let cfg = make_cfg(1e-2, 1.0, 1.0, 1e-8, 0.0);
        let res = DpAdaGrad::new(cfg, 0);
        assert!(matches!(res, Err(PrivacyError::InvalidParameter(_))));
    }

    // 2. learning_rate <= 0 -> InvalidParameter.
    #[test]
    fn test_new_bad_learning_rate_errors() {
        assert!(matches!(
            DpAdaGrad::new(make_cfg(0.0, 1.0, 1.0, 1e-8, 0.0), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaGrad::new(make_cfg(-1.0, 1.0, 1.0, 1e-8, 0.0), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaGrad::new(make_cfg(f64::NAN, 1.0, 1.0, 1e-8, 0.0), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaGrad::new(make_cfg(f64::INFINITY, 1.0, 1.0, 1e-8, 0.0), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
    }

    // 3. clip_norm <= 0 -> InvalidParameter.
    #[test]
    fn test_new_bad_clip_norm_errors() {
        assert!(matches!(
            DpAdaGrad::new(make_cfg(1e-2, 0.0, 1.0, 1e-8, 0.0), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaGrad::new(make_cfg(1e-2, -0.5, 1.0, 1e-8, 0.0), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaGrad::new(make_cfg(1e-2, f64::NAN, 1.0, 1e-8, 0.0), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
    }

    // 4. noise_sigma < 0 -> InvalidParameter (zero is allowed for testing).
    #[test]
    fn test_new_bad_noise_sigma_errors() {
        assert!(matches!(
            DpAdaGrad::new(make_cfg(1e-2, 1.0, -0.1, 1e-8, 0.0), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaGrad::new(make_cfg(1e-2, 1.0, f64::NAN, 1e-8, 0.0), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        // noise_sigma == 0 is permitted.
        assert!(DpAdaGrad::new(make_cfg(1e-2, 1.0, 0.0, 1e-8, 0.0), 2).is_ok());
    }

    // 5. eps <= 0 -> InvalidParameter.
    #[test]
    fn test_new_bad_eps_errors() {
        assert!(matches!(
            DpAdaGrad::new(make_cfg(1e-2, 1.0, 1.0, 0.0, 0.0), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaGrad::new(make_cfg(1e-2, 1.0, 1.0, -1e-3, 0.0), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaGrad::new(make_cfg(1e-2, 1.0, 1.0, f64::INFINITY, 0.0), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
    }

    // 6. initial_accumulator < 0 -> InvalidParameter.
    #[test]
    fn test_new_bad_initial_accumulator_errors() {
        assert!(matches!(
            DpAdaGrad::new(make_cfg(1e-2, 1.0, 1.0, 1e-8, -0.5), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            DpAdaGrad::new(make_cfg(1e-2, 1.0, 1.0, 1e-8, f64::NAN), 2),
            Err(PrivacyError::InvalidParameter(_))
        ));
        // initial_accumulator == 0 is permitted.
        assert!(DpAdaGrad::new(make_cfg(1e-2, 1.0, 1.0, 1e-8, 0.0), 2).is_ok());
    }

    // 7. step with empty grads -> EmptyInput.
    #[test]
    fn test_step_empty_grads_errors() {
        let cfg = make_cfg(1e-2, 1.0, 0.0, 1e-8, 0.0);
        let (opt, mut state) = DpAdaGrad::new(cfg, 3).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        let empty: Vec<Vec<f64>> = vec![];
        let r = opt.step(&mut state, &empty, &mut handle);
        assert!(matches!(r, Err(PrivacyError::EmptyInput)));
    }

    // 8. step with grad length mismatch -> DimensionMismatch.
    #[test]
    fn test_step_dim_mismatch_errors() {
        let cfg = make_cfg(1e-2, 1.0, 0.0, 1e-8, 0.0);
        let (opt, mut state) = DpAdaGrad::new(cfg, 4).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        let bad = vec![vec![0.0; 3]];
        let r = opt.step(&mut state, &bad, &mut handle);
        assert!(matches!(r, Err(PrivacyError::DimensionMismatch { .. })));
    }

    // 9. Zero-noise + single grad: theta moves in direction -g.
    #[test]
    fn test_zero_noise_single_grad_sign_matches_minus_grad() {
        let cfg = make_cfg(0.1, 100.0, 0.0, 1e-8, 0.0);
        let (opt, mut state) = DpAdaGrad::new(cfg, 3).expect("new");
        let mut handle = PrivacyHandle::new(80, 1);
        let grad = vec![1.0f64, -2.0, 0.5];
        opt.step(&mut state, std::slice::from_ref(&grad), &mut handle)
            .expect("step");
        // The first step with initial_accumulator = 0 produces denom ≈ |g| + eps
        // and so the move in coordinate j is -lr * sign(g_j) * (1 / (1 + eps/|g|)).
        for (i, &g) in grad.iter().enumerate() {
            assert!(
                state.theta[i] * g < 0.0,
                "theta[{i}] = {} should have opposite sign to g = {}",
                state.theta[i],
                g
            );
        }
    }

    // 10. Accumulator non-decreasing per coordinate.
    #[test]
    fn test_accumulator_non_decreasing() {
        let cfg = make_cfg(0.05, 1.0, 0.0, 1e-8, 0.0);
        let (opt, mut state) = DpAdaGrad::new(cfg, 4).expect("new");
        let mut handle = PrivacyHandle::new(80, 11);
        let grads: Vec<Vec<f64>> = (0..3).map(|i| vec![0.5 * (i as f64 + 1.0); 4]).collect();
        let mut prev = state.accumulator.clone();
        for _ in 0..10 {
            opt.step(&mut state, &grads, &mut handle).expect("step");
            for (j, &a) in state.accumulator.iter().enumerate() {
                assert!(
                    a + 1e-15 >= prev[j],
                    "accumulator decreased at coord {j}: {a} < {}",
                    prev[j]
                );
            }
            prev = state.accumulator.clone();
        }
    }

    // 11. Deterministic with fixed RNG seed.
    #[test]
    fn test_deterministic_for_fixed_seed() {
        let cfg = make_cfg(1e-2, 1.0, 1.0, 1e-8, 0.0);
        let (opt_a, mut state_a) = DpAdaGrad::new(cfg, 5).expect("a");
        let (opt_b, mut state_b) = DpAdaGrad::new(cfg, 5).expect("b");
        let mut h_a = PrivacyHandle::new(80, 1234);
        let mut h_b = PrivacyHandle::new(80, 1234);
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
        for (a, b) in state_a.accumulator.iter().zip(state_b.accumulator.iter()) {
            assert!((a - b).abs() < 1e-15, "accumulator diverged: {a} vs {b}");
        }
    }

    // 12. After many steps with non-zero gradients, accumulator > 0 on every
    //     coordinate where a non-zero gradient was observed.
    #[test]
    fn test_accumulator_strictly_positive_after_observation() {
        let cfg = make_cfg(1e-2, 1.0, 0.0, 1e-8, 0.0);
        let (opt, mut state) = DpAdaGrad::new(cfg, 4).expect("new");
        let mut handle = PrivacyHandle::new(80, 7);
        // Non-zero gradient on coords 0 and 2 only.
        let grads = vec![vec![0.5f64, 0.0, -0.7, 0.0]];
        for _ in 0..20 {
            opt.step(&mut state, &grads, &mut handle).expect("step");
        }
        assert!(state.accumulator[0] > 0.0);
        assert!(state.accumulator[2] > 0.0);
        // Coords 1 and 3 stay at the initial_accumulator value (0.0 here).
        assert!((state.accumulator[1] - 0.0).abs() < 1e-15);
        assert!((state.accumulator[3] - 0.0).abs() < 1e-15);
    }

    // 13. Convex quadratic: loss decreases over 100 noiseless steps.
    #[test]
    fn test_convex_quadratic_loss_decreases() {
        // Minimise f(theta) = 0.5 * ||theta - x*||^2 with x* = [1, -0.5, 0.25].
        let x_star = [1.0f64, -0.5, 0.25];
        let cfg = make_cfg(0.5, 100.0, 0.0, 1e-8, 0.1);
        let (opt, mut state) = DpAdaGrad::new(cfg, 3).expect("new");
        let mut handle = PrivacyHandle::new(80, 33);

        let loss = |t: &[f64]| -> f64 {
            t.iter()
                .zip(x_star.iter())
                .map(|(a, b)| 0.5 * (a - b).powi(2))
                .sum::<f64>()
        };
        let loss_before = loss(&state.theta);
        for _ in 0..100 {
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

    // 14. initial_accumulator > 0 prevents extreme first-step blowups.
    #[test]
    fn test_initial_accumulator_warmstart_first_step_bounded() {
        let cfg_zero = make_cfg(1e-2, 1.0, 0.0, 1e-12, 0.0);
        let cfg_warm = make_cfg(1e-2, 1.0, 0.0, 1e-12, 1.0);
        let (opt_zero, mut state_zero) = DpAdaGrad::new(cfg_zero, 2).expect("zero");
        let (opt_warm, mut state_warm) = DpAdaGrad::new(cfg_warm, 2).expect("warm");
        let mut h_zero = PrivacyHandle::new(80, 0);
        let mut h_warm = PrivacyHandle::new(80, 0);
        // A tiny gradient creates a denom ≈ |g| + 1e-12 in the zero-init case,
        // so the move ≈ lr * sign(g). With initial_accumulator = 1 the denom
        // starts at sqrt(1 + g^2) ≈ 1, so the first move ≈ lr * g, which is
        // far smaller than lr in magnitude.
        let tiny = vec![vec![1e-6f64, 1e-6]];
        opt_zero
            .step(&mut state_zero, &tiny, &mut h_zero)
            .expect("zero step");
        opt_warm
            .step(&mut state_warm, &tiny, &mut h_warm)
            .expect("warm step");
        for (z, w) in state_zero.theta.iter().zip(state_warm.theta.iter()) {
            assert!(
                z.abs() > w.abs(),
                "warm-start should produce smaller |theta|: zero {z} vs warm {w}"
            );
            assert!(z.is_finite() && w.is_finite());
        }
    }

    // 15. Per-sample L2 clip is respected for the aggregate magnitude.
    #[test]
    fn test_per_sample_clip_respected() {
        // Per-sample grad norm = 5; clip_norm C = 0.5 => each clipped to 0.5;
        // with 2 samples B=2 the priv grad before noise is bounded by C in norm
        // (means of two unit-norm directions can be arbitrary). With zero noise
        // and one step, the parameter update magnitude per coordinate is at
        // most lr * C / (sqrt(initial + (priv_grad_j)^2) + eps).
        let cfg = make_cfg(0.1, 0.5, 0.0, 1e-8, 0.0);
        let (opt, mut state) = DpAdaGrad::new(cfg, 2).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        let big = vec![
            vec![3.0f64, 4.0],   // norm 5 -> clipped to 0.5 by scale 0.1
            vec![-3.0f64, -4.0], // norm 5
        ];
        opt.step(&mut state, &big, &mut handle).expect("step");
        // The two clipped samples are antipodal so their sum is the zero
        // vector — the priv gradient is exactly 0, so theta stays at origin.
        for &t in &state.theta {
            assert!(t.abs() < 1e-12, "theta {t} should be ~0 (antipodal cancel)");
        }
        // Now apply two parallel large gradients: the clip puts each
        // contribution at exactly L2 0.5 along [0.6, 0.8].
        let parallel = vec![vec![3.0f64, 4.0], vec![3.0f64, 4.0]];
        opt.step(&mut state, &parallel, &mut handle)
            .expect("step parallel");
        // Each parallel sample's contribution to g_sum is (3,4) * (0.5/5) = (0.3, 0.4).
        // Sum over B=2: (0.6, 0.8). Priv grad = (0.3, 0.4). Accumulator picks up
        // (0.09, 0.16). The update magnitude is bounded by lr * |g_priv_j| /
        // (|g_priv_j| + eps), i.e. roughly lr — so theta lies within lr of 0
        // per coordinate.
        let lr = 0.1;
        for &t in &state.theta {
            assert!(t.abs() <= lr + 1e-9, "theta {t} exceeded lr bound {lr}");
        }
    }

    // 16. step counter increments correctly.
    #[test]
    fn test_step_counter_increments() {
        let cfg = make_cfg(1e-2, 1.0, 0.0, 1e-8, 0.0);
        let (opt, mut state) = DpAdaGrad::new(cfg, 2).expect("new");
        let mut handle = PrivacyHandle::new(80, 1);
        let grads = vec![vec![0.1f64, 0.2]];
        for k in 1..=10 {
            opt.step(&mut state, &grads, &mut handle).expect("step");
            assert_eq!(state.step, k);
        }
    }

    // 17. State remains finite after many noisy steps.
    #[test]
    fn test_many_noisy_steps_finite() {
        let cfg = make_cfg(1e-3, 1.0, 0.5, 1e-8, 1e-6);
        let (opt, mut state) = DpAdaGrad::new(cfg, 5).expect("new");
        let mut handle = PrivacyHandle::new(80, 9_001);
        let grads: Vec<Vec<f64>> = (0..8)
            .map(|i| (0..5).map(|j| 0.05 * (i + j) as f64).collect())
            .collect();
        for _ in 0..200 {
            opt.step(&mut state, &grads, &mut handle).expect("step");
        }
        for v in state.theta.iter().chain(state.accumulator.iter()) {
            assert!(v.is_finite(), "non-finite {v}");
        }
        assert_eq!(state.step, 200);
    }
}
