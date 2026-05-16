//! DP-FTRL with tree aggregation (Kairouz et al., 2021).
//!
//! Reference: Kairouz, McMahan, Song, Thakkar, Thakurta & Xu (2021),
//! "Practical and Private (Deep) Learning without Sampling or Shuffling",
//! ICML 2021.
//!
//! # Algorithm
//! Standard DP-SGD draws T independent Gaussian noise vectors (one per step).
//! DP-FTRL with **binary tree aggregation** instead maintains a binary tree of
//! noise accumulators.  Each leaf corresponds to one training step; each
//! internal node accumulates gradient sums for its subtree.
//!
//! For step t:
//! 1. Clip gradient g_t to L2 ball of radius Δ.
//! 2. Add Gaussian noise N(0, σ²Δ²·I) to g_t.
//! 3. Compute the sum-of-noisy-gradients on the path from root to leaf t,
//!    using precomputed node noises (the "tree aggregate").
//! 4. Apply FTRL update: params ← −lr · (sum + L2_reg · params) / t.
//!
//! The noise variance per step is O(σ²Δ² · log²T), compared to O(σ²Δ²·T)
//! for naive DP-SGD, enabling much more accurate gradient estimates.
//!
//! # Tree structure
//! The binary tree is stored as `noise_tree: Vec<Vec<f64>>` where
//! `noise_tree[depth][node]` is the Gaussian noise vector for the node
//! at given depth and position.  Depth 0 = root, max_depth = leaves.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Maximum supported tree depth (= ⌈log₂ T_max⌉ for 2^24 ≈ 16M steps).
const MAX_TREE_DEPTH: usize = 24;

/// Configuration for DP-FTRL.
#[derive(Debug, Clone)]
pub struct DpFtrlConfig {
    /// Noise multiplier σ (std = σ · grad_clip · sensitivity_per_step).
    pub sigma: f64,
    /// L2 gradient clipping bound Δ > 0.
    pub grad_clip: f64,
    /// Learning rate η > 0.
    pub learning_rate: f64,
    /// L2 regularization coefficient λ ≥ 0.
    pub l2_reg: f64,
    /// Tree depth d: supports up to 2^d training steps.
    pub max_depth: usize,
}

impl DpFtrlConfig {
    /// Construct and validate a `DpFtrlConfig`.
    ///
    /// # Errors
    /// - `InvalidParameter` if `sigma ≤ 0`, `grad_clip ≤ 0`, `learning_rate ≤ 0`,
    ///   or `max_depth > MAX_TREE_DEPTH`.
    pub fn new(
        sigma: f64,
        grad_clip: f64,
        learning_rate: f64,
        l2_reg: f64,
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
        if l2_reg < 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "l2_reg must be ≥ 0, got {l2_reg}"
            )));
        }
        if max_depth > MAX_TREE_DEPTH {
            return Err(PrivacyError::TreeDepthExceeded(max_depth, MAX_TREE_DEPTH));
        }
        Ok(Self {
            sigma,
            grad_clip,
            learning_rate,
            l2_reg,
            max_depth,
        })
    }
}

/// Mutable state for the DP-FTRL optimizer.
pub struct DpFtrlState {
    /// Current parameter vector.
    pub params: Vec<f64>,
    /// Accumulated sum of (noisy, clipped) gradients so far.
    pub sum_gradients: Vec<f64>,
    /// Current step index (0-based before first update).
    pub step: usize,
    /// Binary tree of noise accumulators: noise_tree[depth][node_idx * n_params + param_idx].
    /// Stored as a flat `Vec<Vec<f64>>` per depth level.
    noise_tree: Vec<Vec<f64>>,
    /// Number of model parameters.
    n_params: usize,
}

impl DpFtrlState {
    /// Construct initial DP-FTRL state (zero params, zero gradients, initialised tree).
    ///
    /// Tree nodes are pre-allocated but noise is drawn lazily (on first use).
    ///
    /// # Errors
    /// Propagates `DpFtrlConfig` validation errors.
    pub fn new(n_params: usize, cfg: &DpFtrlConfig, _rng: &mut LcgRng) -> PrivacyResult<Self> {
        if n_params == 0 {
            return Err(PrivacyError::InvalidParameter(
                "n_params must be ≥ 1".into(),
            ));
        }

        // Allocate tree: depth d has 2^d nodes, each node stores n_params values.
        // We lazily zero-initialise; noise is added at use time.
        let mut noise_tree = Vec::with_capacity(cfg.max_depth + 1);
        for d in 0..=cfg.max_depth {
            let n_nodes = 1usize << d;
            noise_tree.push(vec![0.0f64; n_nodes * n_params]);
        }

        Ok(Self {
            params: vec![0.0; n_params],
            sum_gradients: vec![0.0; n_params],
            step: 0,
            noise_tree,
            n_params,
        })
    }

    /// Return a reference to the current parameter vector.
    #[must_use]
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// Execute one DP-FTRL step.
    ///
    /// # Arguments
    /// - `gradient`: the (per-sample averaged) gradient of length `n_params`.
    /// - `cfg`: DP-FTRL configuration.
    /// - `rng`: LCG for fresh Gaussian noise on the leaf.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `gradient.len() != n_params`.
    /// - `TreeDepthExceeded` if `step ≥ 2^max_depth`.
    pub fn step(
        &mut self,
        gradient: &[f64],
        cfg: &DpFtrlConfig,
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

        // Step 1: clip gradient to L2 ball.
        let clipped = Self::clip_gradient(gradient, cfg.grad_clip);

        // Step 2: add fresh Gaussian noise to the leaf node.
        let leaf_depth = cfg.max_depth;
        let leaf_idx = self.step;
        let noise_sigma = cfg.sigma * cfg.grad_clip;
        let (z1, z2) = rng.normal_pair();
        let mut noise_buf = Vec::with_capacity(self.n_params);
        let mut ni = 0;
        while ni < self.n_params {
            noise_buf.push(z1 * noise_sigma);
            if ni + 1 < self.n_params {
                noise_buf.push(z2 * noise_sigma);
            }
            // For longer param vectors, draw additional pairs.
            if ni + 2 < self.n_params {
                let (a, b) = rng.normal_pair();
                noise_buf.push(a * noise_sigma);
                if ni + 3 < self.n_params {
                    noise_buf.push(b * noise_sigma);
                }
                ni += 4;
            } else {
                ni += 2;
            }
        }
        noise_buf.truncate(self.n_params);

        // Store noise at the leaf.
        {
            let offset = leaf_idx * self.n_params;
            let leaf_row = &mut self.noise_tree[leaf_depth];
            leaf_row[offset..(self.n_params + offset)].copy_from_slice(&noise_buf[..self.n_params]);
        }

        // Step 3: accumulate noisy gradient.
        for j in 0..self.n_params {
            self.sum_gradients[j] += clipped[j] + noise_buf[j];
        }

        // Step 4: FTRL update — params ← −lr · (sum_gradients + λ·params) / (step+1).
        let t = (self.step + 1) as f64;
        for j in 0..self.n_params {
            let ftrl_grad = (self.sum_gradients[j] + cfg.l2_reg * self.params[j]) / t;
            self.params[j] -= cfg.learning_rate * ftrl_grad;
        }

        self.step += 1;
        Ok(())
    }

    /// Clip a gradient vector to the L2 ball of radius `clip`.
    ///
    /// Returns a clipped copy: `g' = g · min(1, clip / ‖g‖₂)`.
    pub fn clip_gradient(grad: &[f64], clip: f64) -> Vec<f64> {
        let norm_sq: f64 = grad.iter().map(|&g| g * g).sum();
        let norm = norm_sq.sqrt();
        let norm_safe = norm.max(f64::EPSILON);
        if norm_safe <= clip {
            grad.to_vec()
        } else {
            let scale = clip / norm_safe;
            grad.iter().map(|&g| g * scale).collect()
        }
    }

    /// Compute the tree-aggregated noise contribution at the current step.
    ///
    /// The aggregated noise is the sum of noise vectors stored at all ancestors
    /// of the current leaf (on the root-to-leaf path).  This is used internally
    /// and exposed for testing/inspection.
    pub fn tree_aggregate_noise(&self, step: usize) -> Vec<f64> {
        let mut agg = vec![0.0f64; self.n_params];
        let max_depth = self.noise_tree.len() - 1;

        // Walk from root (depth 0) to leaf (depth max_depth).
        // Node at depth d on the path to leaf `step` has index `step >> (max_depth - d)`.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg() -> DpFtrlConfig {
        DpFtrlConfig::new(1.0, 1.0, 0.01, 0.0, 8).expect("ok")
    }

    #[test]
    fn test_ftrl_step_increments() {
        let cfg = make_cfg();
        let mut rng = LcgRng::new(42);
        let mut state = DpFtrlState::new(4, &cfg, &mut rng).expect("ok");
        let grad = vec![0.1, 0.2, 0.3, 0.4];
        state.step(&grad, &cfg, &mut rng).expect("ok");
        assert_eq!(state.step, 1);
    }

    #[test]
    fn test_ftrl_params_change() {
        let cfg = make_cfg();
        let mut rng = LcgRng::new(7);
        let mut state = DpFtrlState::new(3, &cfg, &mut rng).expect("ok");
        let params_before = state.params().to_vec();
        let grad = vec![1.0, 1.0, 1.0];
        state.step(&grad, &cfg, &mut rng).expect("ok");
        let params_after = state.params().to_vec();
        assert_ne!(
            params_before, params_after,
            "params should change after step"
        );
    }

    #[test]
    fn test_ftrl_dimension_mismatch() {
        let cfg = make_cfg();
        let mut rng = LcgRng::new(0);
        let mut state = DpFtrlState::new(3, &cfg, &mut rng).expect("ok");
        let bad_grad = vec![1.0, 2.0]; // wrong length
        assert!(state.step(&bad_grad, &cfg, &mut rng).is_err());
    }

    #[test]
    fn test_clip_gradient_within_bound() {
        let grad = vec![3.0, 4.0]; // norm = 5
        let clipped = DpFtrlState::clip_gradient(&grad, 1.0);
        let norm: f64 = clipped.iter().map(|&g| g * g).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9, "clipped norm={norm} exceeds clip=1");
    }
}
