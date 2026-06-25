//! Tree-DP-FTRL with **adaptive tree depth** (online tree doubling).
//!
//! Reference: Kairouz, McMahan, Song, Thakkar, Thakurta & Xu (2021),
//! "Practical and Private (Deep) Learning without Sampling or Shuffling",
//! ICML 2021; binary-tree-aggregation mechanism of Chan, Shi & Song (2011),
//! "Private and Continual Release of Statistics", and Dwork, Naor, Pitassi &
//! Rothblum (2010).  The fixed-depth tree-aggregation DP-FTRL lives in
//! [`crate::optimizer::dp_ftrl`]; this variant **does not require the horizon
//! `T` up front** — it grows the binary tree on demand.
//!
//! # The problem with a fixed depth
//! Plain tree-aggregation DP-FTRL pre-allocates a depth-`d` tree and *fails* once
//! `T > 2^d` steps are taken.  In online / streaming training the horizon is
//! often unknown.  This module uses the **doubling trick**: maintain a forest of
//! completed perfect binary trees whose sizes are distinct powers of two (the
//! binary representation of the current count `t`), plus running partial sums.
//! When `t` crosses the next power of two, the structure transparently extends —
//! the *effective depth* is `⌈log₂(t+1)⌉`, adapting to however long training
//! runs.
//!
//! # Streaming prefix-sum noise (binary mechanism)
//! The noisy prefix sum `S̃_t = Σ_{i≤t} ĝ_i + (tree noise)` is produced with the
//! classic streaming binary-counter mechanism:
//!
//! - Represent `t` in binary.  The prefix `[1, t]` decomposes into
//!   `O(log t)` dyadic blocks, one per set bit of `t`.
//! - Each *completed* dyadic block of size `2^k` carries one independent
//!   Gaussian noise term `ζ_k ~ N(0, σ²C²·I)` (drawn once, when the block
//!   completes).  The prefix noise at step `t` is the sum of the `ζ_k` over the
//!   set bits of `t`.
//!
//! The per-prefix variance is `σ²C²·(number of set bits of t) ≤ σ²C²·⌈log₂(t+1)⌉`,
//! i.e. **logarithmic** in `t`, matching the fixed-depth tree but without a
//! pre-set horizon.  Privacy is unchanged from DP-FTRL: any function of `S̃_t` is
//! post-processing of the released noisy prefix sums.
//!
//! # Privacy accounting
//! Each leaf contributes to at most `⌈log₂(t+1)⌉` released block-sums, so the
//! L2-sensitivity of the released vector of prefix sums grows like
//! `C·√⌈log₂ T⌉`; with per-block multiplier `σ` the mechanism is
//! `ρ`-zCDP with `ρ = ⌈log₂ T⌉ / (2σ²)` — convert via
//! [`crate::accounting::zcdp`].

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// A completed dyadic block of size `2^level` covering a contiguous range of
/// past steps, carrying its (already-noised) gradient sum.
#[derive(Debug, Clone)]
struct DyadicBlock {
    /// `log₂` of the block size (number of leaves it covers).
    level: u32,
    /// Noised partial sum `Σ gradients over the block + N(0, σ²C²)`.
    noisy_sum: Vec<f64>,
}

/// Configuration for adaptive-depth tree DP-FTRL.
#[derive(Debug, Clone)]
pub struct AdaptiveFtrlConfig {
    /// Per-block Gaussian noise multiplier σ (block-sum std = `σ·C`).
    pub sigma: f64,
    /// L2 gradient clipping bound `C > 0`.
    pub grad_clip: f64,
    /// Learning rate `η > 0`.
    pub learning_rate: f64,
    /// L2 regularisation coefficient `λ ≥ 0`.
    pub l2_reg: f64,
}

impl AdaptiveFtrlConfig {
    /// Construct and validate the configuration.
    ///
    /// # Errors
    /// - `InvalidParameter` if `sigma ≤ 0`, `grad_clip ≤ 0`, `learning_rate ≤ 0`,
    ///   or `l2_reg < 0`.
    pub fn new(sigma: f64, grad_clip: f64, learning_rate: f64, l2_reg: f64) -> PrivacyResult<Self> {
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
        if l2_reg < 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "l2_reg must be ≥ 0, got {l2_reg}"
            )));
        }
        Ok(Self {
            sigma,
            grad_clip,
            learning_rate,
            l2_reg,
        })
    }
}

/// Streaming adaptive-depth tree DP-FTRL state.
pub struct AdaptiveFtrlState {
    params: Vec<f64>,
    /// Stack of completed dyadic blocks (a "forest"); at any time their levels
    /// are strictly increasing and correspond to the set bits of `step`.
    blocks: Vec<DyadicBlock>,
    /// Noiseless running gradient sum (for diagnostics and the FTRL average).
    sum_gradients: Vec<f64>,
    n_params: usize,
    step: usize,
}

impl AdaptiveFtrlState {
    /// Construct initial state (zero params, empty forest).
    ///
    /// # Errors
    /// - `InvalidParameter` if `n_params == 0`.
    pub fn new(n_params: usize) -> PrivacyResult<Self> {
        if n_params == 0 {
            return Err(PrivacyError::InvalidParameter(
                "n_params must be ≥ 1".into(),
            ));
        }
        Ok(Self {
            params: vec![0.0; n_params],
            blocks: Vec::new(),
            sum_gradients: vec![0.0; n_params],
            n_params,
            step: 0,
        })
    }

    /// Current parameter vector.
    #[must_use]
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// Steps taken so far.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Effective tree depth = number of completed dyadic blocks currently held,
    /// which equals the number of set bits in `step` and is bounded by
    /// `⌈log₂(step+1)⌉`.  This is the adaptive depth.
    #[must_use]
    pub fn effective_depth(&self) -> usize {
        self.blocks.len()
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

    fn add_noise(buf: &mut [f64], rng: &mut LcgRng, std: f64) {
        let mut i = 0;
        while i < buf.len() {
            let (a, b) = rng.normal_pair();
            buf[i] += a * std;
            if i + 1 < buf.len() {
                buf[i + 1] += b * std;
            }
            i += 2;
        }
    }

    /// Noisy prefix sum `S̃_t` at the current step: the sum of the noised
    /// dyadic-block sums currently held.
    #[must_use]
    pub fn noisy_prefix_sum(&self) -> Vec<f64> {
        let mut acc = vec![0.0f64; self.n_params];
        for blk in &self.blocks {
            for (a, &v) in acc.iter_mut().zip(blk.noisy_sum.iter()) {
                *a += v;
            }
        }
        acc
    }

    /// Execute one streaming DP-FTRL step with adaptive tree growth.
    ///
    /// The new clipped gradient becomes a level-0 block; while the two
    /// top-of-stack blocks share a level they are *merged* into a level-`k+1`
    /// block whose noised sum is the sum of the two child noised sums plus one
    /// fresh independent Gaussian (the canonical binary-mechanism node noise).
    /// This is exactly the streaming binary counter, so the tree depth grows as
    /// `⌈log₂(t+1)⌉` with no pre-set horizon.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `gradient.len() != n_params`.
    pub fn step(
        &mut self,
        gradient: &[f64],
        cfg: &AdaptiveFtrlConfig,
        rng: &mut LcgRng,
    ) -> PrivacyResult<()> {
        if gradient.len() != self.n_params {
            return Err(PrivacyError::DimensionMismatch {
                expected: self.n_params,
                got: gradient.len(),
            });
        }
        let clipped = Self::clip_gradient(gradient, cfg.grad_clip);
        for (s, &c) in self.sum_gradients.iter_mut().zip(clipped.iter()) {
            *s += c;
        }

        let node_std = cfg.sigma * cfg.grad_clip;

        // New leaf = level-0 block with its own independent node noise.
        let mut new_sum = clipped.clone();
        Self::add_noise(&mut new_sum, rng, node_std);
        let mut new_block = DyadicBlock {
            level: 0,
            noisy_sum: new_sum,
        };

        // Merge equal-level top blocks (binary carry), adding fresh node noise
        // to each newly-formed internal node.
        while let Some(top) = self.blocks.last() {
            if top.level != new_block.level {
                break;
            }
            let top = self
                .blocks
                .pop()
                .ok_or_else(|| PrivacyError::InvalidParameter("forest underflow".into()))?;
            let mut merged: Vec<f64> = top
                .noisy_sum
                .iter()
                .zip(new_block.noisy_sum.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            Self::add_noise(&mut merged, rng, node_std);
            new_block = DyadicBlock {
                level: new_block.level + 1,
                noisy_sum: merged,
            };
        }
        self.blocks.push(new_block);

        self.step += 1;

        // FTRL update on the noisy prefix sum, averaged by t.
        let noisy_sum = self.noisy_prefix_sum();
        let t = self.step as f64;
        for (p, &ns) in self.params.iter_mut().zip(noisy_sum.iter()) {
            let ftrl_grad = (ns + cfg.l2_reg * *p) / t;
            *p -= cfg.learning_rate * ftrl_grad;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> AdaptiveFtrlConfig {
        AdaptiveFtrlConfig::new(1.0, 1.0, 0.05, 0.0).expect("cfg")
    }

    #[test]
    fn test_unbounded_steps_no_failure() {
        // Run far past any fixed power-of-two horizon without error.
        let c = cfg();
        let mut rng = LcgRng::new(7);
        let mut s = AdaptiveFtrlState::new(3).expect("state");
        for _ in 0..1000 {
            s.step(&[0.1, -0.1, 0.2], &c, &mut rng).expect("step");
        }
        assert_eq!(s.step_count(), 1000);
    }

    #[test]
    fn test_effective_depth_is_popcount() {
        // After t steps the number of held blocks must equal popcount(t).
        let c = cfg();
        let mut rng = LcgRng::new(1);
        let mut s = AdaptiveFtrlState::new(2).expect("state");
        for t in 1..=64usize {
            s.step(&[0.5, 0.5], &c, &mut rng).expect("step");
            assert_eq!(
                s.effective_depth(),
                t.count_ones() as usize,
                "at step {t}, #blocks must equal popcount({t})"
            );
        }
    }

    #[test]
    fn test_depth_bounded_by_log() {
        let c = cfg();
        let mut rng = LcgRng::new(2);
        let mut s = AdaptiveFtrlState::new(2).expect("state");
        for _ in 0..500 {
            s.step(&[0.3, 0.3], &c, &mut rng).expect("step");
        }
        let log_bound = (s.step_count() as f64 + 1.0).log2().ceil() as usize;
        assert!(
            s.effective_depth() <= log_bound,
            "depth {} should be ≤ ⌈log₂(t+1)⌉={log_bound}",
            s.effective_depth()
        );
    }

    #[test]
    fn test_noiseless_prefix_sum_tracks_when_sigma_tiny() {
        // With near-zero noise, the noisy prefix sum ≈ Σ clipped gradients.
        let c = AdaptiveFtrlConfig::new(1e-9, 10.0, 0.01, 0.0).expect("cfg");
        let mut rng = LcgRng::new(5);
        let mut s = AdaptiveFtrlState::new(2).expect("state");
        let g = [0.4, -0.3]; // norm < clip, so no clipping
        for _ in 0..7 {
            s.step(&g, &c, &mut rng).expect("step");
        }
        let ps = s.noisy_prefix_sum();
        assert!((ps[0] - 7.0 * 0.4).abs() < 1e-3, "ps[0]={}", ps[0]);
        assert!((ps[1] - 7.0 * -0.3).abs() < 1e-3, "ps[1]={}", ps[1]);
    }

    #[test]
    fn test_determinism_same_seed() {
        let c = cfg();
        let run = || {
            let mut rng = LcgRng::new(2025);
            let mut s = AdaptiveFtrlState::new(4).expect("state");
            for _ in 0..40 {
                s.step(&[0.2, -0.1, 0.3, 0.0], &c, &mut rng).expect("step");
            }
            s.params().to_vec()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn test_invalid_config_and_dims() {
        assert!(AdaptiveFtrlConfig::new(0.0, 1.0, 0.1, 0.0).is_err());
        assert!(AdaptiveFtrlConfig::new(1.0, 0.0, 0.1, 0.0).is_err());
        assert!(AdaptiveFtrlConfig::new(1.0, 1.0, 0.0, 0.0).is_err());
        assert!(AdaptiveFtrlState::new(0).is_err());
        let c = cfg();
        let mut rng = LcgRng::new(0);
        let mut s = AdaptiveFtrlState::new(3).expect("state");
        assert!(s.step(&[1.0], &c, &mut rng).is_err());
    }
}
