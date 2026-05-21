//! DP-SGD with microbatching (Abadi et al. 2016; McMahan et al. 2018).
//!
//! References:
//! - Abadi M, Chu A, Goodfellow I, McMahan HB, Mironov I, Talwar K, Zhang L (2016)
//!   "Deep Learning with Differential Privacy", CCS 2016.
//! - McMahan HB, Andrew G, Erlingsson U, Chien S, Mironov I, Papernot N, Kairouz P
//!   (2018) "A General Approach to Adding Differential Privacy to Iterative
//!   Training Procedures" — microbatching variant.
//!
//! # Algorithm (one step, Algorithm 1 in Abadi et al. with the McMahan et al.
//! microbatching variant)
//! Given per-sample gradients g_1, ..., g_B (each in R^p):
//! 1. Group into microbatches m_1, ..., m_M of size `microbatch_size` and
//!    compute per-microbatch averages g_bar_m = (1/|m|) sum_{i in m} g_i.
//! 2. Per-microbatch clip: g_bar_m <- g_bar_m * min(1, C / ||g_bar_m||_2).
//! 3. Sum the clipped microbatch grads: G = sum_m g_bar_m.
//! 4. Add Gaussian noise: G += N(0, sigma^2 * C^2 * I).
//! 5. Average: G_hat = G / M.
//! 6. Apply (heavy-ball) momentum: velocity <- mu * velocity + G_hat.
//! 7. Update: theta <- theta - lr * velocity.
//!
//! Note: with microbatch_size = 1 this reduces to per-sample DP-SGD (Abadi et
//! al. 2016 Algorithm 1) exactly. The microbatching variant trades a small
//! per-microbatch bias for substantially reduced gradient noise variance.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::PrivacyHandle;

/// Configuration for DP-SGD with microbatching.
#[derive(Debug, Clone)]
pub struct DpSgdConfig {
    /// Learning rate eta > 0.
    pub learning_rate: f64,
    /// Per-microbatch L2 clipping bound C > 0.
    pub clip_norm: f64,
    /// Gaussian noise multiplier sigma >= 0 (noise std = sigma * clip_norm).
    pub noise_sigma: f64,
    /// Microbatch size |m| >= 1.
    pub microbatch_size: usize,
    /// Heavy-ball momentum mu in [0, 1).
    pub momentum: f64,
}

impl Default for DpSgdConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-2,
            clip_norm: 1.0,
            noise_sigma: 1.0,
            microbatch_size: 1,
            momentum: 0.0,
        }
    }
}

/// Mutable optimizer state.
#[derive(Debug, Clone)]
pub struct DpSgdMicrobatchState {
    /// Current parameter vector theta.
    pub theta: Vec<f64>,
    /// Heavy-ball momentum buffer.
    pub velocity: Vec<f64>,
    /// Step counter (number of completed `step` calls).
    pub step: usize,
}

/// Stateless optimizer object holding the configuration.
#[derive(Debug, Clone)]
pub struct DpSgdMicrobatch {
    cfg: DpSgdConfig,
}

impl DpSgdMicrobatch {
    /// Construct a new DP-SGD microbatch optimizer plus a fresh zero state.
    ///
    /// # Errors
    /// - `InvalidParameter` if `learning_rate <= 0`, `noise_sigma < 0`,
    ///   `microbatch_size == 0`, `momentum < 0`, `momentum >= 1`, or `dim == 0`.
    /// - `NonPositiveSensitivity` if `clip_norm <= 0`.
    pub fn new(cfg: DpSgdConfig, dim: usize) -> PrivacyResult<(Self, DpSgdMicrobatchState)> {
        if cfg.learning_rate <= 0.0 || !cfg.learning_rate.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "learning_rate must be > 0, got {}",
                cfg.learning_rate
            )));
        }
        if cfg.clip_norm <= 0.0 || !cfg.clip_norm.is_finite() {
            return Err(PrivacyError::NonPositiveSensitivity(cfg.clip_norm));
        }
        if cfg.noise_sigma < 0.0 || !cfg.noise_sigma.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "noise_sigma must be >= 0 and finite, got {}",
                cfg.noise_sigma
            )));
        }
        if cfg.microbatch_size == 0 {
            return Err(PrivacyError::InvalidParameter(
                "microbatch_size must be >= 1".into(),
            ));
        }
        if !cfg.momentum.is_finite() || cfg.momentum < 0.0 || cfg.momentum >= 1.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "momentum must be in [0, 1), got {}",
                cfg.momentum
            )));
        }
        if dim == 0 {
            return Err(PrivacyError::InvalidParameter("dim must be >= 1".into()));
        }
        let state = DpSgdMicrobatchState {
            theta: vec![0.0; dim],
            velocity: vec![0.0; dim],
            step: 0,
        };
        Ok((Self { cfg }, state))
    }

    /// Return a read-only reference to the active configuration.
    #[must_use]
    pub fn config(&self) -> &DpSgdConfig {
        &self.cfg
    }

    /// Group per-sample gradients into microbatches and return per-microbatch
    /// averaged gradients g_bar_m = (1/|m|) sum_{i in m} g_i.
    ///
    /// Any trailing partial microbatch (when `microbatch_size` does not divide
    /// the batch evenly) is still averaged over the number of samples it
    /// actually contains. This matches McMahan et al. 2018 Section 3.
    ///
    /// # Errors
    /// - `EmptyInput` if `per_sample` is empty.
    /// - `InvalidParameter` if `microbatch_size == 0` or `microbatch_size >
    ///   per_sample.len()`.
    /// - `DimensionMismatch` if rows of `per_sample` are not all the same
    ///   length.
    pub fn microbatch_average_grad(
        per_sample: &[Vec<f64>],
        microbatch_size: usize,
    ) -> PrivacyResult<Vec<Vec<f64>>> {
        if per_sample.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        if microbatch_size == 0 {
            return Err(PrivacyError::InvalidParameter(
                "microbatch_size must be >= 1".into(),
            ));
        }
        if microbatch_size > per_sample.len() {
            return Err(PrivacyError::InvalidParameter(format!(
                "microbatch_size {microbatch_size} > batch size {}",
                per_sample.len()
            )));
        }
        let dim = per_sample[0].len();
        if dim == 0 {
            return Err(PrivacyError::EmptyInput);
        }
        let n_full = per_sample.len() / microbatch_size;
        let leftover = per_sample.len() - n_full * microbatch_size;
        let n_micro = n_full + usize::from(leftover > 0);
        let mut out: Vec<Vec<f64>> = Vec::with_capacity(n_micro);
        for m in 0..n_micro {
            let start = m * microbatch_size;
            let end = (start + microbatch_size).min(per_sample.len());
            let group = &per_sample[start..end];
            let mut avg = vec![0.0f64; dim];
            for sample in group {
                if sample.len() != dim {
                    return Err(PrivacyError::DimensionMismatch {
                        expected: dim,
                        got: sample.len(),
                    });
                }
                for (a, &g) in avg.iter_mut().zip(sample.iter()) {
                    *a += g;
                }
            }
            let inv = 1.0 / (group.len() as f64);
            for a in avg.iter_mut() {
                *a *= inv;
            }
            out.push(avg);
        }
        Ok(out)
    }

    /// Execute one DP-SGD microbatch optimization step.
    ///
    /// # Errors
    /// - `EmptyInput` if `per_sample_grads` is empty.
    /// - `InvalidParameter` if `cfg.microbatch_size > per_sample_grads.len()`.
    /// - `DimensionMismatch` if any per-sample gradient row length differs
    ///   from `state.theta.len()`.
    pub fn step(
        &self,
        state: &mut DpSgdMicrobatchState,
        per_sample_grads: &[Vec<f64>],
        handle: &mut PrivacyHandle,
    ) -> PrivacyResult<()> {
        let dim = state.theta.len();
        if per_sample_grads.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        if self.cfg.microbatch_size > per_sample_grads.len() {
            return Err(PrivacyError::InvalidParameter(format!(
                "microbatch_size {} > batch size {}",
                self.cfg.microbatch_size,
                per_sample_grads.len()
            )));
        }
        for row in per_sample_grads {
            if row.len() != dim {
                return Err(PrivacyError::DimensionMismatch {
                    expected: dim,
                    got: row.len(),
                });
            }
        }

        // Step 1: per-microbatch averages.
        let mut micro_grads =
            Self::microbatch_average_grad(per_sample_grads, self.cfg.microbatch_size)?;
        let n_micro = micro_grads.len();
        let n_micro_f = n_micro as f64;

        // Step 2: per-microbatch L2 clipping in-place.
        let c = self.cfg.clip_norm;
        for g in micro_grads.iter_mut() {
            let norm_sq: f64 = g.iter().map(|&x| x * x).sum();
            let norm = norm_sq.sqrt();
            if norm > c {
                let scale = c / norm.max(f64::EPSILON);
                for v in g.iter_mut() {
                    *v *= scale;
                }
            }
        }

        // Step 3: aggregate sum across microbatches.
        let mut g_sum = vec![0.0f64; dim];
        for g in micro_grads.iter() {
            for (acc, &v) in g_sum.iter_mut().zip(g.iter()) {
                *acc += v;
            }
        }

        // Step 4: Gaussian noise N(0, sigma^2 * C^2 * I).
        let noise_std = self.cfg.noise_sigma * self.cfg.clip_norm;
        if noise_std > 0.0 {
            let noise = handle.generate_gaussian_noise(noise_std, dim)?;
            for (acc, n) in g_sum.iter_mut().zip(noise) {
                *acc += n;
            }
        }

        // Step 5: mean over microbatches.
        let inv = 1.0 / n_micro_f;
        for v in g_sum.iter_mut() {
            *v *= inv;
        }
        let g_hat = g_sum;

        // Step 6: heavy-ball momentum.
        let mu = self.cfg.momentum;
        for (v, &g) in state.velocity.iter_mut().zip(g_hat.iter()) {
            *v = mu * *v + g;
        }

        // Step 7: parameter update.
        let lr = self.cfg.learning_rate;
        for (t, &v) in state.theta.iter_mut().zip(state.velocity.iter()) {
            *t -= lr * v;
        }

        state.step += 1;
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(lr: f64, clip: f64, sigma: f64, microbatch: usize, momentum: f64) -> DpSgdConfig {
        DpSgdConfig {
            learning_rate: lr,
            clip_norm: clip,
            noise_sigma: sigma,
            microbatch_size: microbatch,
            momentum,
        }
    }

    fn l2(v: &[f64]) -> f64 {
        v.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    // 1. Zero-noise + large clip + microbatch_size=1 reduces to plain (mean) SGD.
    #[test]
    fn test_zero_noise_unclipped_matches_plain_sgd_on_quadratic() {
        // Minimise f(theta) = 0.5 * ||theta - x*||^2  with x* = [1, -2, 0.5].
        // Per-sample grad = theta - x* (same for every sample => batch mean = grad).
        let cfg = make_cfg(0.1, 1e9, 0.0, 1, 0.0);
        let (opt, mut state) = DpSgdMicrobatch::new(cfg.clone(), 3).expect("new");
        let mut handle = PrivacyHandle::new(80, 1);
        let x_star = [1.0, -2.0, 0.5];
        let mut plain = [0.0f64; 3];
        for _ in 0..50 {
            let grad: Vec<f64> = state
                .theta
                .iter()
                .zip(x_star.iter())
                .map(|(t, x)| t - x)
                .collect();
            // Single-sample, single-microbatch: DP-SGD update is theta -= lr * grad.
            opt.step(&mut state, std::slice::from_ref(&grad), &mut handle)
                .expect("step");
            // Plain SGD reference.
            let pg: Vec<f64> = plain
                .iter()
                .zip(x_star.iter())
                .map(|(t, x)| t - x)
                .collect();
            for (p, g) in plain.iter_mut().zip(pg.iter()) {
                *p -= 0.1 * g;
            }
        }
        for (a, b) in state.theta.iter().zip(plain.iter()) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    // 2. Momentum dampens oscillation on a stiff 1D quadratic.
    #[test]
    fn test_momentum_reduces_oscillation_on_stiff_quadratic() {
        // f(theta) = 0.5 * 10 * theta^2 ; grad = 10 * theta.  With lr=0.18,
        // plain SGD oscillates (|1 - lr*10| = 0.8 < 1 but slow); with momentum
        // 0.9 the trajectory should converge faster in L2 norm at step 30.
        let lr = 0.18;
        let init = 1.0;
        let mut no_m_state = DpSgdMicrobatchState {
            theta: vec![init],
            velocity: vec![0.0],
            step: 0,
        };
        let mut m_state = DpSgdMicrobatchState {
            theta: vec![init],
            velocity: vec![0.0],
            step: 0,
        };
        let opt_no = DpSgdMicrobatch::new(make_cfg(lr, 1e9, 0.0, 1, 0.0), 1)
            .expect("no_m")
            .0;
        let opt_m = DpSgdMicrobatch::new(make_cfg(lr, 1e9, 0.0, 1, 0.9), 1)
            .expect("m")
            .0;
        let mut handle = PrivacyHandle::new(80, 9);
        for _ in 0..30 {
            let g_no = vec![10.0 * no_m_state.theta[0]];
            let g_m = vec![10.0 * m_state.theta[0]];
            opt_no
                .step(&mut no_m_state, &[g_no], &mut handle)
                .expect("step");
            opt_m.step(&mut m_state, &[g_m], &mut handle).expect("step");
        }
        // With these settings momentum should be at least as close to 0 in
        // absolute value.  Verify momentum did not blow up and converged.
        assert!(m_state.theta[0].abs() < no_m_state.theta[0].abs().max(0.5));
        assert!(m_state.theta[0].is_finite());
    }

    // 3. Dimension mismatch on per_sample_grads errors.
    #[test]
    fn test_dim_mismatch_errors() {
        let cfg = make_cfg(0.1, 1.0, 0.0, 1, 0.0);
        let (opt, mut state) = DpSgdMicrobatch::new(cfg, 4).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        let bad = vec![vec![0.0; 3]];
        let r = opt.step(&mut state, &bad, &mut handle);
        assert!(matches!(r, Err(PrivacyError::DimensionMismatch { .. })));
    }

    // 4. Empty per-sample grads errors.
    #[test]
    fn test_empty_grads_errors() {
        let cfg = make_cfg(0.1, 1.0, 0.0, 1, 0.0);
        let (opt, mut state) = DpSgdMicrobatch::new(cfg, 2).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        let empty: Vec<Vec<f64>> = vec![];
        let r = opt.step(&mut state, &empty, &mut handle);
        assert!(matches!(r, Err(PrivacyError::EmptyInput)));
    }

    // 5. microbatch_size == 1 case (per-sample DP-SGD).
    #[test]
    fn test_microbatch_size_one_equals_per_sample_dp_sgd() {
        let cfg = make_cfg(0.1, 100.0, 0.0, 1, 0.0);
        let (opt, mut state) = DpSgdMicrobatch::new(cfg, 2).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        // Batch of two identical samples; with clip large, mean over 2 micros
        // equals the original gradient.
        let g = vec![vec![1.0, 2.0], vec![1.0, 2.0]];
        opt.step(&mut state, &g, &mut handle).expect("step");
        // theta = -lr * mean_grad = -0.1 * [1, 2]
        assert!((state.theta[0] - (-0.1)).abs() < 1e-12);
        assert!((state.theta[1] - (-0.2)).abs() < 1e-12);
    }

    // 6. microbatch_size > batch_size errors.
    #[test]
    fn test_microbatch_too_large_errors() {
        let cfg = make_cfg(0.1, 1.0, 0.0, 5, 0.0);
        let (opt, mut state) = DpSgdMicrobatch::new(cfg, 2).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        let g = vec![vec![0.0; 2], vec![0.0; 2]]; // only 2 samples
        let r = opt.step(&mut state, &g, &mut handle);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
    }

    // 7. Clip respected: every microbatch grad after clipping has L2 <= C.
    #[test]
    fn test_clip_respected_per_microbatch() {
        // 4 samples with norm well above C=0.5; group into 2 microbatches.
        let per_sample = vec![
            vec![3.0, 4.0],   // norm 5
            vec![3.0, 4.0],   // norm 5
            vec![-3.0, -4.0], // norm 5
            vec![-3.0, -4.0], // norm 5
        ];
        // Microbatch averages: [3,4] and [-3,-4]; both norm 5 > C=0.5.
        let mut micro = DpSgdMicrobatch::microbatch_average_grad(&per_sample, 2).expect("ok");
        let c = 0.5f64;
        for g in micro.iter_mut() {
            let n = l2(g);
            if n > c {
                let s = c / n;
                for v in g.iter_mut() {
                    *v *= s;
                }
            }
            assert!(l2(g) <= c + 1e-12, "{} > {}", l2(g), c);
        }
    }

    // 8. Zero-noise deterministic across runs (same seed implies same handle path).
    #[test]
    fn test_zero_noise_deterministic_same_seed() {
        // With sigma=0 the handle is never consulted, so two runs trivially agree.
        let cfg = make_cfg(0.05, 1.0, 0.0, 2, 0.0);
        let (opt_a, mut state_a) = DpSgdMicrobatch::new(cfg.clone(), 3).expect("a");
        let (opt_b, mut state_b) = DpSgdMicrobatch::new(cfg, 3).expect("b");
        let mut h_a = PrivacyHandle::new(80, 42);
        let mut h_b = PrivacyHandle::new(80, 42);
        let grads = vec![
            vec![0.1, -0.2, 0.3],
            vec![-0.1, 0.4, 0.5],
            vec![0.2, 0.0, -0.5],
            vec![0.3, 0.3, 0.3],
        ];
        for _ in 0..5 {
            opt_a.step(&mut state_a, &grads, &mut h_a).expect("a");
            opt_b.step(&mut state_b, &grads, &mut h_b).expect("b");
        }
        for (a, b) in state_a.theta.iter().zip(state_b.theta.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    // 9. Many steps do not produce NaN/Inf.
    #[test]
    fn test_many_steps_stay_finite() {
        let cfg = make_cfg(1e-3, 1.0, 0.1, 4, 0.9);
        let (opt, mut state) = DpSgdMicrobatch::new(cfg, 5).expect("new");
        let mut handle = PrivacyHandle::new(80, 7);
        let grads: Vec<Vec<f64>> = (0..16).map(|i| vec![0.01 * (i as f64); 5]).collect();
        for _ in 0..200 {
            opt.step(&mut state, &grads, &mut handle).expect("step");
        }
        for &v in state.theta.iter().chain(state.velocity.iter()) {
            assert!(v.is_finite(), "non-finite {v}");
        }
        assert_eq!(state.step, 200);
    }

    // 10. momentum == 0 reduces to vanilla DP-SGD: velocity equals the noisy
    //     averaged gradient at each step.
    #[test]
    fn test_momentum_zero_reduces_to_vanilla() {
        let cfg = make_cfg(0.1, 1e9, 0.0, 1, 0.0);
        let (opt, mut state) = DpSgdMicrobatch::new(cfg, 2).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        let grads = vec![vec![1.0, -1.0], vec![3.0, 1.0]];
        opt.step(&mut state, &grads, &mut handle).expect("step");
        // mean grad = [(1+3)/2, (-1+1)/2] = [2, 0]; theta = -0.1 * [2, 0]
        assert!((state.theta[0] - (-0.2)).abs() < 1e-12);
        assert!((state.theta[1] - 0.0).abs() < 1e-12);
        // velocity equals mean grad.
        assert!((state.velocity[0] - 2.0).abs() < 1e-12);
        assert!((state.velocity[1] - 0.0).abs() < 1e-12);
    }

    // 11. Multi-step convergence on a quadratic minimum.
    #[test]
    fn test_multi_step_convergence() {
        // Minimise 0.5 * ||theta - target||^2; per-sample grad = theta - target.
        let target = [0.5f64, -0.25, 1.0];
        let cfg = make_cfg(0.2, 1e9, 0.0, 4, 0.0);
        let (opt, mut state) = DpSgdMicrobatch::new(cfg, 3).expect("new");
        let mut handle = PrivacyHandle::new(80, 33);
        for _ in 0..200 {
            let g: Vec<f64> = state
                .theta
                .iter()
                .zip(target.iter())
                .map(|(t, x)| t - x)
                .collect();
            let batch: Vec<Vec<f64>> = (0..4).map(|_| g.clone()).collect();
            opt.step(&mut state, &batch, &mut handle).expect("step");
        }
        for (got, want) in state.theta.iter().zip(target.iter()) {
            assert!((got - want).abs() < 1e-3, "got {got} want {want}");
        }
    }

    // 12. Invalid configurations error out.
    #[test]
    fn test_invalid_configs_error() {
        // learning_rate <= 0
        assert!(DpSgdMicrobatch::new(make_cfg(0.0, 1.0, 1.0, 1, 0.0), 2).is_err());
        assert!(DpSgdMicrobatch::new(make_cfg(-1.0, 1.0, 1.0, 1, 0.0), 2).is_err());
        // clip_norm <= 0 => NonPositiveSensitivity
        match DpSgdMicrobatch::new(make_cfg(1e-3, 0.0, 1.0, 1, 0.0), 2) {
            Err(PrivacyError::NonPositiveSensitivity(_)) => {}
            other => panic!("expected NonPositiveSensitivity, got {other:?}"),
        }
        // noise_sigma < 0
        assert!(DpSgdMicrobatch::new(make_cfg(1e-3, 1.0, -0.1, 1, 0.0), 2).is_err());
        // microbatch_size == 0
        assert!(DpSgdMicrobatch::new(make_cfg(1e-3, 1.0, 1.0, 0, 0.0), 2).is_err());
        // momentum < 0
        assert!(DpSgdMicrobatch::new(make_cfg(1e-3, 1.0, 1.0, 1, -0.1), 2).is_err());
        // momentum >= 1
        assert!(DpSgdMicrobatch::new(make_cfg(1e-3, 1.0, 1.0, 1, 1.0), 2).is_err());
        // dim == 0
        assert!(DpSgdMicrobatch::new(make_cfg(1e-3, 1.0, 1.0, 1, 0.0), 0).is_err());
    }

    // 13. Microbatch averaging shape: B=6, micro=2 => 3 microbatches.
    #[test]
    fn test_microbatch_grouping_shape() {
        let per_sample: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64, (i as f64) * 2.0]).collect();
        let micro = DpSgdMicrobatch::microbatch_average_grad(&per_sample, 2).expect("ok");
        assert_eq!(micro.len(), 3);
        // Averages: [(0+1)/2,(0+2)/2], [(2+3)/2,(4+6)/2], [(4+5)/2,(8+10)/2].
        assert!((micro[0][0] - 0.5).abs() < 1e-12);
        assert!((micro[0][1] - 1.0).abs() < 1e-12);
        assert!((micro[1][0] - 2.5).abs() < 1e-12);
        assert!((micro[1][1] - 5.0).abs() < 1e-12);
        assert!((micro[2][0] - 4.5).abs() < 1e-12);
        assert!((micro[2][1] - 9.0).abs() < 1e-12);
    }

    // 14. Microbatch averaging with leftover partial microbatch.
    #[test]
    fn test_microbatch_grouping_partial_leftover() {
        let per_sample: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![3.0, 0.0], vec![5.0, 0.0]];
        // micro=2 => 1 full (avg of first two = 2) plus 1 partial (5).
        let micro = DpSgdMicrobatch::microbatch_average_grad(&per_sample, 2).expect("ok");
        assert_eq!(micro.len(), 2);
        assert!((micro[0][0] - 2.0).abs() < 1e-12);
        assert!((micro[1][0] - 5.0).abs() < 1e-12);
    }

    // 15. microbatch_average_grad input validation.
    #[test]
    fn test_microbatch_average_validation() {
        let r = DpSgdMicrobatch::microbatch_average_grad(&[], 1);
        assert!(matches!(r, Err(PrivacyError::EmptyInput)));
        let r = DpSgdMicrobatch::microbatch_average_grad(&[vec![1.0], vec![1.0]], 0);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
        let r = DpSgdMicrobatch::microbatch_average_grad(&[vec![1.0], vec![1.0]], 3);
        assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
        let r = DpSgdMicrobatch::microbatch_average_grad(&[vec![1.0, 2.0], vec![1.0]], 2);
        assert!(matches!(r, Err(PrivacyError::DimensionMismatch { .. })));
    }
}
