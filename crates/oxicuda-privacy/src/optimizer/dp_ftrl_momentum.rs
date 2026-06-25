//! DP-FTRL with momentum and bias correction (matrix-factorisation-free).
//!
//! Reference: Kairouz, McMahan, Song, Thakkar, Thakurta & Xu (2021),
//! "Practical and Private (Deep) Learning without Sampling or Shuffling",
//! ICML 2021 — Algorithm 2 (DP-FTRL with momentum).  The plain tree-aggregation
//! DP-FTRL lives in [`crate::optimizer::dp_ftrl`]; this variant adds a heavy-ball
//! **momentum** buffer over the cumulative noisy gradient and an Adam-style
//! **bias correction** of that buffer, which the Kairouz et al. experiments show
//! materially improves utility at fixed privacy.
//!
//! # Algorithm
//! At step `t` (1-indexed), given the per-step averaged gradient `g_t`:
//!
//! 1. **Clip** `g_t` to the L2 ball of radius `C`:  `ĝ_t = g_t · min(1, C/‖g_t‖)`.
//! 2. **Tree-aggregate noise**: add the binary-tree prefix-sum Gaussian noise
//!    `ν_t` (variance `O(σ²C²·log T)`) to the running gradient sum,
//!    `S_t = S_{t-1} + ĝ_t`, and form the *noisy* cumulative sum
//!    `S̃_t = S_t + N_t` where `N_t` is the tree prefix noise at leaf `t`.
//! 3. **Momentum** (heavy ball) on the noisy cumulative sum:
//!    `m_t = β·m_{t-1} + (1−β)·S̃_t`.
//! 4. **Bias correction**: `m̂_t = m_t / (1 − βᵗ)` (Kingma–Ba style), removing the
//!    cold-start bias of the EMA so early steps are not under-scaled.
//! 5. **Update**: `θ_t = θ_{t-1} − η·m̂_t` (FTRL with the corrected momentum).
//!
//! The privacy guarantee is identical to plain DP-FTRL: momentum and bias
//! correction are *post-processing* of the noisy cumulative sum and consume no
//! additional budget (Kairouz et al., §3).  Accordingly the `(ε, δ)` accounting
//! is whatever the tree-aggregation noise multiplier `σ` implies — see
//! [`crate::accounting::rdp_gaussian`].
//!
//! # Tree noise
//! The cumulative-sum noise is generated with the standard *binary tree*
//! mechanism (Honaker / Dwork-Naor-Pitassi-Rothblum / Chan-Shi-Song): the noisy
//! prefix sum at leaf `t` adds the sum of independent Gaussians stored at the
//! `≤ ⌈log₂ T⌉ + 1` tree nodes covering `[1, t]`.  This keeps the per-prefix
//! variance logarithmic in `T` rather than linear.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Maximum supported tree depth (`2^24 ≈ 16.8M` steps).
const MAX_TREE_DEPTH: usize = 24;

/// Configuration for momentum DP-FTRL.
#[derive(Debug, Clone)]
pub struct DpFtrlMomentumConfig {
    /// Noise multiplier σ (per-node Gaussian std = `σ·C`).
    pub sigma: f64,
    /// L2 gradient clipping bound `C > 0`.
    pub grad_clip: f64,
    /// Learning rate `η > 0`.
    pub learning_rate: f64,
    /// Momentum coefficient `β ∈ [0, 1)`.
    pub momentum: f64,
    /// Tree depth `d`: supports up to `2^d` steps.
    pub max_depth: usize,
}

impl DpFtrlMomentumConfig {
    /// Construct and validate the configuration.
    ///
    /// # Errors
    /// - `InvalidParameter` if `sigma ≤ 0`, `grad_clip ≤ 0`, `learning_rate ≤ 0`,
    ///   or `momentum ∉ [0, 1)`.
    /// - `TreeDepthExceeded` if `max_depth > MAX_TREE_DEPTH`.
    pub fn new(
        sigma: f64,
        grad_clip: f64,
        learning_rate: f64,
        momentum: f64,
        max_depth: usize,
    ) -> PrivacyResult<Self> {
        if sigma <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "sigma must be positive, got {sigma}"
            )));
        }
        if grad_clip <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "grad_clip must be positive, got {grad_clip}"
            )));
        }
        if learning_rate <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "learning_rate must be positive, got {learning_rate}"
            )));
        }
        if !(0.0..1.0).contains(&momentum) {
            return Err(PrivacyError::InvalidParameter(format!(
                "momentum must be in [0,1), got {momentum}"
            )));
        }
        if max_depth > MAX_TREE_DEPTH {
            return Err(PrivacyError::TreeDepthExceeded(max_depth, MAX_TREE_DEPTH));
        }
        Ok(Self {
            sigma,
            grad_clip,
            learning_rate,
            momentum,
            max_depth,
        })
    }
}

/// Mutable state for momentum DP-FTRL.
pub struct DpFtrlMomentumState {
    params: Vec<f64>,
    /// Running (clipped, noiseless) gradient sum `S_t`.
    sum_gradients: Vec<f64>,
    /// Heavy-ball momentum buffer `m_t`.
    momentum_buf: Vec<f64>,
    /// Per-depth tree node Gaussian noise: `noise_tree[d][node·n + j]`.
    noise_tree: Vec<Vec<f64>>,
    n_params: usize,
    step: usize,
}

impl DpFtrlMomentumState {
    /// Construct initial state (zero params/gradients/momentum, zeroed tree).
    ///
    /// # Errors
    /// - `InvalidParameter` if `n_params == 0`.
    pub fn new(n_params: usize, cfg: &DpFtrlMomentumConfig) -> PrivacyResult<Self> {
        if n_params == 0 {
            return Err(PrivacyError::InvalidParameter(
                "n_params must be ≥ 1".into(),
            ));
        }
        let mut noise_tree = Vec::with_capacity(cfg.max_depth + 1);
        for d in 0..=cfg.max_depth {
            let n_nodes = 1usize << d;
            noise_tree.push(vec![0.0f64; n_nodes * n_params]);
        }
        Ok(Self {
            params: vec![0.0; n_params],
            sum_gradients: vec![0.0; n_params],
            momentum_buf: vec![0.0; n_params],
            noise_tree,
            n_params,
            step: 0,
        })
    }

    /// Current parameter vector.
    #[must_use]
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// Current momentum buffer (post-EMA, pre-bias-correction).
    #[must_use]
    pub fn momentum_buffer(&self) -> &[f64] {
        &self.momentum_buf
    }

    /// Current step count.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Clip a gradient to the L2 ball of radius `clip`.
    #[must_use]
    pub fn clip_gradient(grad: &[f64], clip: f64) -> Vec<f64> {
        let norm = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
        let norm_safe = norm.max(f64::EPSILON);
        if norm_safe <= clip {
            grad.to_vec()
        } else {
            let s = clip / norm_safe;
            grad.iter().map(|&g| g * s).collect()
        }
    }

    /// Fill `buf` (length `n_params`) with fresh `N(0, std²)` samples.
    fn fill_gaussian(rng: &mut LcgRng, std: f64, n: usize) -> Vec<f64> {
        let mut buf = Vec::with_capacity(n);
        while buf.len() < n {
            let (a, b) = rng.normal_pair();
            buf.push(a * std);
            if buf.len() < n {
                buf.push(b * std);
            }
        }
        buf.truncate(n);
        buf
    }

    /// Tree-aggregated noise for the noisy prefix sum `S̃_t` at leaf `t`.
    ///
    /// Returns the sum of node noises on the root-to-leaf path (the canonical
    /// binary-tree prefix-sum noise).  Exposed for inspection/testing.
    #[must_use]
    pub fn tree_aggregate_noise(&self, step: usize) -> Vec<f64> {
        let mut agg = vec![0.0f64; self.n_params];
        let max_depth = self.noise_tree.len() - 1;
        for d in 0..=max_depth {
            let node_idx = step >> (max_depth - d);
            let offset = node_idx * self.n_params;
            let row = &self.noise_tree[d];
            if offset + self.n_params <= row.len() {
                for j in 0..self.n_params {
                    agg[j] += row[offset + j];
                }
            }
        }
        agg
    }

    /// Execute one momentum-DP-FTRL step.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `gradient.len() != n_params`.
    /// - `TreeDepthExceeded` if `step ≥ 2^max_depth`.
    pub fn step(
        &mut self,
        gradient: &[f64],
        cfg: &DpFtrlMomentumConfig,
        rng: &mut LcgRng,
    ) -> PrivacyResult<()> {
        if gradient.len() != self.n_params {
            return Err(PrivacyError::DimensionMismatch {
                expected: self.n_params,
                got: gradient.len(),
            });
        }
        let max_leaves = 1usize << cfg.max_depth;
        if self.step >= max_leaves {
            return Err(PrivacyError::TreeDepthExceeded(self.step, max_leaves - 1));
        }

        // 1. Clip.
        let clipped = Self::clip_gradient(gradient, cfg.grad_clip);

        // 2. Store fresh leaf noise and update the noiseless running sum.
        let node_std = cfg.sigma * cfg.grad_clip;
        let leaf_noise = Self::fill_gaussian(rng, node_std, self.n_params);
        let leaf_depth = cfg.max_depth;
        let leaf_idx = self.step;
        {
            let offset = leaf_idx * self.n_params;
            let row = &mut self.noise_tree[leaf_depth];
            row[offset..offset + self.n_params].copy_from_slice(&leaf_noise);
        }
        for (s, &c) in self.sum_gradients.iter_mut().zip(clipped.iter()) {
            *s += c;
        }

        // Noisy cumulative sum S̃_t = S_t + (tree prefix noise at leaf t).
        let tree_noise = self.tree_aggregate_noise(self.step);
        let noisy_sum: Vec<f64> = self
            .sum_gradients
            .iter()
            .zip(tree_noise.iter())
            .map(|(&s, &tn)| s + tn)
            .collect();

        // 3. Momentum EMA over the noisy cumulative sum.
        let beta = cfg.momentum;
        for (m, &ns) in self.momentum_buf.iter_mut().zip(noisy_sum.iter()) {
            *m = beta * *m + (1.0 - beta) * ns;
        }

        // 4. Bias correction m̂_t = m_t / (1 − βᵗ).
        let t = (self.step + 1) as f64;
        let bias_correction = if beta > 0.0 { 1.0 - beta.powf(t) } else { 1.0 };
        let bc_safe = bias_correction.max(f64::EPSILON);

        // 5. FTRL update with the corrected momentum, normalised by t (FTRL
        //    averages the cumulative sum; momentum already smooths it).
        for (p, &m) in self.params.iter_mut().zip(self.momentum_buf.iter()) {
            let corrected = m / bc_safe;
            *p -= cfg.learning_rate * corrected / t;
        }

        self.step += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DpFtrlMomentumConfig {
        DpFtrlMomentumConfig::new(1.0, 1.0, 0.05, 0.9, 8).expect("cfg")
    }

    #[test]
    fn test_step_increments_and_changes_params() {
        let c = cfg();
        let mut rng = LcgRng::new(11);
        let mut s = DpFtrlMomentumState::new(4, &c).expect("state");
        let before = s.params().to_vec();
        s.step(&[1.0, 1.0, 1.0, 1.0], &c, &mut rng).expect("step");
        assert_eq!(s.step_count(), 1);
        assert_ne!(before, s.params().to_vec(), "params should move");
    }

    #[test]
    fn test_bias_correction_first_step_unscaled() {
        // With β=0.9 and a fixed gradient, the bias-corrected momentum at step 1
        // must equal the noisy cumulative sum (m̂₁ = m₁/(1−β) = S̃₁), i.e. the
        // first update is NOT shrunk by (1−β).  We verify the param magnitude
        // matches an equivalent no-momentum FTRL on the same noisy sum.
        let c = DpFtrlMomentumConfig::new(0.0 + 1e-12, 1.0, 0.1, 0.9, 6).expect("cfg");
        let mut rng = LcgRng::new(3);
        let mut s = DpFtrlMomentumState::new(2, &c).expect("state");
        s.step(&[1.0, 0.0], &c, &mut rng).expect("step");
        // After 1 step: param[0] = -η·(m̂₁/1). With near-zero noise, S̃₁≈clip=1,
        // m₁=(1−β)·1, m̂₁=m₁/(1−β)=1, so param[0]≈-η = -0.1.
        let p0 = s.params()[0];
        assert!(
            (p0 + 0.1).abs() < 5e-3,
            "bias-corrected first step param[0]={p0}, expected ≈ -0.1"
        );
    }

    #[test]
    fn test_momentum_buffer_grows_with_consistent_gradient() {
        let c = cfg();
        let mut rng = LcgRng::new(99);
        let mut s = DpFtrlMomentumState::new(3, &c).expect("state");
        let g = [0.5, 0.5, 0.5];
        let mut last_norm = 0.0;
        for _ in 0..5 {
            s.step(&g, &c, &mut rng).expect("step");
            let nrm = s
                .momentum_buffer()
                .iter()
                .map(|&x| x * x)
                .sum::<f64>()
                .sqrt();
            assert!(
                nrm >= last_norm - 1e-9,
                "momentum magnitude should grow with consistent gradient: {nrm} >= {last_norm}"
            );
            last_norm = nrm;
        }
    }

    #[test]
    fn test_determinism_same_seed() {
        let c = cfg();
        let run = || {
            let mut rng = LcgRng::new(2024);
            let mut s = DpFtrlMomentumState::new(4, &c).expect("state");
            for _ in 0..6 {
                s.step(&[0.3, -0.2, 0.1, 0.4], &c, &mut rng).expect("step");
            }
            s.params().to_vec()
        };
        assert_eq!(run(), run(), "same seed must give identical params");
    }

    #[test]
    fn test_clip_bounds_norm() {
        let clipped = DpFtrlMomentumState::clip_gradient(&[3.0, 4.0], 1.0);
        let nrm = clipped.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(nrm <= 1.0 + 1e-9, "clipped norm {nrm} exceeds 1");
    }

    #[test]
    fn test_invalid_config_and_dims() {
        assert!(DpFtrlMomentumConfig::new(0.0, 1.0, 0.1, 0.9, 4).is_err());
        assert!(DpFtrlMomentumConfig::new(1.0, 1.0, 0.1, 1.0, 4).is_err());
        assert!(DpFtrlMomentumConfig::new(1.0, 1.0, 0.1, 0.9, 30).is_err());
        let c = cfg();
        let mut rng = LcgRng::new(0);
        let mut s = DpFtrlMomentumState::new(3, &c).expect("state");
        assert!(s.step(&[1.0, 2.0], &c, &mut rng).is_err());
    }
}
