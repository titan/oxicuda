//! VarOblivious — variable-depth Neural Oblivious Decision Ensembles.
//!
//! This module extends the fixed-depth NODE of [`crate::tree::node_oblivious`]
//! (where every tree shares `cfg.depth`) to a **variable-depth** ensemble: each
//! oblivious tree carries its *own* depth `d_t`, while still routing every level
//! through the entmax/sparsemax/entmoid simplex projections defined (and reused)
//! from the sibling module.  All the routing maths is identical per level; only
//! the per-tree depth (and therefore the leaf-count `2^{d_t}`) varies across the
//! ensemble.
//!
//! ## Why variable depth?
//!
//! A fixed depth forces a single bias/variance trade-off on every tree.  Letting
//! each tree pick its own depth lets the ensemble mix coarse (shallow) and fine
//! (deep) partitions of feature space — the same motivation behind heterogeneous
//! depths in classical gradient-boosting forests, but here fully differentiable.
//!
//! ## Per-level routing (identical to the fixed-depth module)
//!
//! For tree `t` and level `i ∈ [0, d_t)`:
//!
//! 1. **Feature choice.** A learnable row `F_{t,i} ∈ ℝ^{num_features}` is pushed
//!    through **entmax-α** to obtain a sparse soft one-hot selector `s_{t,i}` on
//!    the probability simplex (`α = 2` ⇒ sparsemax).  The chosen feature value is
//!    `f_{t,i}(x) = ⟨s_{t,i}, x⟩`.
//! 2. **Soft split.** The scalar `(f_{t,i}(x) − b_{t,i})·split_scale` is mapped by
//!    the two-class entmax (**entmoid**) to a gate `c_{t,i} ∈ [0, 1]`.
//!
//! ## Leaf-weight tensor product
//!
//! Because the per-level decisions of an oblivious tree are independent, the leaf
//! assignment **factors into an outer product** of the `d_t` per-level
//! `(1 − c_{t,i}, c_{t,i})` factors.  Writing leaf index `j` in binary as
//! `j_{d_t-1} … j_1 j_0` (MSB-first: level 0 owns the most-significant bit, matching
//! the fixed-depth convention), the weight of leaf `j` is
//!
//! ```text
//!   w_j = ∏_{i=0}^{d_t-1} ( c_{t,i}      if bit_i(j) = 1
//!                           1 − c_{t,i}  if bit_i(j) = 0 )
//! ```
//!
//! The `2^{d_t}` weights are a genuine probability distribution:
//!
//! ```text
//!   Σ_j w_j = ∏_i ( (1 − c_{t,i}) + c_{t,i} ) = ∏_i 1 = 1,   w_j ≥ 0.
//! ```
//!
//! The tree output is the leaf-weighted response sum `Σ_j w_j · R_{t,j}` (each
//! `R_{t,j} ∈ ℝ^{response_dim}`), and the ensemble pools the per-tree outputs by
//! mean or sum (see [`EnsembleReduction`]).
//!
//! ## References
//! - Popov, S., Morozov, S. & Babenko, A. (2019). "Neural Oblivious Decision
//!   Ensembles for Deep Learning on Tabular Data." ICLR 2020.
//! - Peters, B., Niculae, V. & Martins, A. F. T. (2019). "Sparse Sequence-to-
//!   Sequence Models." ACL 2019 (α-entmax / bisection).

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;
use crate::tree::node_oblivious::{
    EnsembleReduction, entmax_alpha_f64, entmoid_alpha_f64, fill_normal_f64,
};

/// The widest depth we permit, so that `1usize << depth` never overflows leaf
/// indexing.  Matches the fixed-depth module's guard.
const MAX_DEPTH: usize = usize::BITS as usize;

// ─── Configuration ───────────────────────────────────────────────────────────────

/// Configuration for a [`VarObliviousLayer`].
///
/// Unlike [`crate::tree::node_oblivious::NodeObliviousConfig`] there is no single
/// `depth`; instead `depths[t]` is the depth of tree `t`, and `depths.len()` is
/// the number of trees in the ensemble.
#[derive(Debug, Clone)]
pub struct VarObliviousConfig {
    /// Per-tree depths.  `depths.len()` (≥ 1) is the tree count; each entry
    /// (≥ 1, `< usize::BITS`) gives that tree's `2^{depth}` leaves.
    pub depths: Vec<usize>,
    /// Number of input features (≥ 1).
    pub num_features: usize,
    /// Dimension of each leaf response vector / the model output (≥ 1).
    pub response_dim: usize,
    /// Entmax temperature `α ∈ (1, 2]`.  `1.5` ⇒ entmax-1.5, `2.0` ⇒ sparsemax.
    pub entmax_alpha: f64,
    /// Multiplicative sharpness applied to `(f_i(x) − b_i)` before the entmoid.
    pub split_scale: f64,
    /// Ensemble pooling rule.
    pub reduction: EnsembleReduction,
    /// RNG seed used when initialising parameters via [`VarObliviousLayer::new`].
    pub seed: u64,
}

impl VarObliviousConfig {
    /// A sensible default: depths `[4, 5, 6]` (a coarse-to-fine trio), entmax-1.5,
    /// mean pooling.
    #[must_use]
    pub fn new(num_features: usize, response_dim: usize) -> Self {
        Self {
            depths: vec![4, 5, 6],
            num_features,
            response_dim,
            entmax_alpha: 1.5,
            split_scale: 1.0,
            reduction: EnsembleReduction::Mean,
            seed: 0,
        }
    }

    fn validate(&self) -> TabularResult<()> {
        if self.depths.is_empty() {
            return Err(TabularError::InvalidTreeCount { n: 0 });
        }
        for &d in &self.depths {
            if d == 0 || d >= MAX_DEPTH {
                return Err(TabularError::InvalidTreeDepth { depth: d });
            }
        }
        if self.num_features == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if self.response_dim == 0 {
            return Err(TabularError::InvalidParameter {
                name: "response_dim".into(),
                msg: "must be ≥ 1".into(),
            });
        }
        if !(self.entmax_alpha > 1.0 && self.entmax_alpha <= 2.0) {
            return Err(TabularError::InvalidParameter {
                name: "entmax_alpha".into(),
                msg: format!("must lie in (1, 2], got {}", self.entmax_alpha),
            });
        }
        if !self.split_scale.is_finite() || self.split_scale <= 0.0 {
            return Err(TabularError::InvalidParameter {
                name: "split_scale".into(),
                msg: format!("must be a positive finite value, got {}", self.split_scale),
            });
        }
        Ok(())
    }
}

// ─── Single variable-depth oblivious tree ────────────────────────────────────────

/// A single soft oblivious decision tree of arbitrary depth (entmax-routed).
///
/// Identical maths to [`crate::tree::node_oblivious::ObliviousTree`]; the only
/// difference is that `depth` is a per-tree property rather than an ensemble-wide
/// constant.
#[derive(Debug, Clone)]
pub struct VarObliviousTree {
    /// Feature-selector logits, flattened `[depth * num_features]` (row `i` is the
    /// selector for level `i`).
    feature_selectors: Vec<f64>,
    /// Per-level split thresholds `b_i`, length `depth`.
    thresholds: Vec<f64>,
    /// Leaf responses, flattened `[2^depth * response_dim]`.
    leaf_responses: Vec<f64>,
    depth: usize,
    num_features: usize,
    response_dim: usize,
    entmax_alpha: f64,
    split_scale: f64,
}

impl VarObliviousTree {
    /// Randomly initialise one tree of the given `depth`.
    ///
    /// Uses the same draw scheme as the fixed-depth module (reusing
    /// `fill_normal_f64`): feature-selector logits `N(0, 0.1)` (so entmax starts
    /// near-uniform), thresholds `N(0, 1)`, leaf responses `N(0, 1/√response_dim)`.
    fn new_random(
        depth: usize,
        num_features: usize,
        response_dim: usize,
        entmax_alpha: f64,
        split_scale: f64,
        rng: &mut LcgRng,
    ) -> Self {
        let n_leaves = 1usize << depth;
        let mut feature_selectors = vec![0.0_f64; depth * num_features];
        fill_normal_f64(rng, &mut feature_selectors, 0.1);

        let mut thresholds = vec![0.0_f64; depth];
        fill_normal_f64(rng, &mut thresholds, 1.0);

        let resp_std = 1.0 / (response_dim as f64).sqrt();
        let mut leaf_responses = vec![0.0_f64; n_leaves * response_dim];
        fill_normal_f64(rng, &mut leaf_responses, resp_std);

        Self {
            feature_selectors,
            thresholds,
            leaf_responses,
            depth,
            num_features,
            response_dim,
            entmax_alpha,
            split_scale,
        }
    }

    /// Construct a tree from explicit parameters (useful for hand-built / loaded
    /// trees and closed-form tests).
    ///
    /// # Errors
    /// - [`TabularError::InvalidTreeDepth`] if `depth == 0` or `depth ≥ usize::BITS`.
    /// - [`TabularError::InvalidFeatureCount`] if `num_features == 0`.
    /// - [`TabularError::InvalidParameter`] if `response_dim == 0`, `α ∉ (1, 2]`, or
    ///   `split_scale` is not a positive finite value.
    /// - [`TabularError::DimensionMismatch`] if any of the three parameter buffers
    ///   does not have the length implied by `depth`/`num_features`/`response_dim`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        feature_selectors: Vec<f64>,
        thresholds: Vec<f64>,
        leaf_responses: Vec<f64>,
        depth: usize,
        num_features: usize,
        response_dim: usize,
        entmax_alpha: f64,
        split_scale: f64,
    ) -> TabularResult<Self> {
        if depth == 0 || depth >= MAX_DEPTH {
            return Err(TabularError::InvalidTreeDepth { depth });
        }
        if num_features == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if response_dim == 0 {
            return Err(TabularError::InvalidParameter {
                name: "response_dim".into(),
                msg: "must be ≥ 1".into(),
            });
        }
        if !(entmax_alpha > 1.0 && entmax_alpha <= 2.0) {
            return Err(TabularError::InvalidParameter {
                name: "entmax_alpha".into(),
                msg: format!("must lie in (1, 2], got {entmax_alpha}"),
            });
        }
        if !split_scale.is_finite() || split_scale <= 0.0 {
            return Err(TabularError::InvalidParameter {
                name: "split_scale".into(),
                msg: format!("must be a positive finite value, got {split_scale}"),
            });
        }
        let want_sel = depth * num_features;
        if feature_selectors.len() != want_sel {
            return Err(TabularError::DimensionMismatch {
                expected: want_sel,
                got: feature_selectors.len(),
            });
        }
        if thresholds.len() != depth {
            return Err(TabularError::DimensionMismatch {
                expected: depth,
                got: thresholds.len(),
            });
        }
        let want_resp = (1usize << depth) * response_dim;
        if leaf_responses.len() != want_resp {
            return Err(TabularError::DimensionMismatch {
                expected: want_resp,
                got: leaf_responses.len(),
            });
        }
        Ok(Self {
            feature_selectors,
            thresholds,
            leaf_responses,
            depth,
            num_features,
            response_dim,
            entmax_alpha,
            split_scale,
        })
    }

    /// This tree's depth `d_t`.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Number of leaves, `2^depth`.
    #[must_use]
    pub fn num_leaves(&self) -> usize {
        1usize << self.depth
    }

    /// Output / leaf-response dimension.
    #[must_use]
    pub fn response_dim(&self) -> usize {
        self.response_dim
    }

    /// The entmax-projected feature-selector simplex `s_i` for `level` (independent
    /// of the input `x`).  The returned vector is non-negative and sums to ≈ 1.
    ///
    /// # Errors
    /// - [`TabularError::InvalidParameter`] if `level ≥ depth`.
    /// - propagates entmax solver errors.
    pub fn feature_selector(&self, level: usize) -> TabularResult<Vec<f64>> {
        if level >= self.depth {
            return Err(TabularError::InvalidParameter {
                name: "level".into(),
                msg: format!("must be < depth {}, got {level}", self.depth),
            });
        }
        let row =
            &self.feature_selectors[level * self.num_features..(level + 1) * self.num_features];
        entmax_alpha_f64(row, self.entmax_alpha)
    }

    /// Compute the `depth` per-level gate probabilities `c_i ∈ [0, 1]` for `x`.
    fn level_gates(&self, x: &[f64]) -> TabularResult<Vec<f64>> {
        let mut gates = Vec::with_capacity(self.depth);
        for level in 0..self.depth {
            let selector = self.feature_selector(level)?;
            let f_val: f64 = selector.iter().zip(x.iter()).map(|(&s, &xi)| s * xi).sum();
            let logit = (f_val - self.thresholds[level]) * self.split_scale;
            let gate = entmoid_alpha_f64(logit, self.entmax_alpha)?;
            gates.push(gate);
        }
        Ok(gates)
    }

    /// Compute the `2^depth` leaf weights for `x` (non-negative, summing to 1 up to
    /// floating-point round-off).  See the module-level leaf-weight tensor maths.
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] if `x.len() != num_features`.
    /// - propagates entmax solver errors.
    pub fn leaf_weights(&self, x: &[f64]) -> TabularResult<Vec<f64>> {
        if x.len() != self.num_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.num_features,
                got: x.len(),
            });
        }
        let gates = self.level_gates(x)?;
        let n_leaves = self.num_leaves();
        let mut weights = vec![1.0_f64; n_leaves];
        for (level, &c) in gates.iter().enumerate() {
            // MSB-first: level 0 owns the most-significant bit (matches the
            // fixed-depth module so the two stay drop-in comparable).
            let shift = self.depth - 1 - level;
            let right = c;
            let left = 1.0 - c;
            for (leaf, w) in weights.iter_mut().enumerate() {
                let bit = (leaf >> shift) & 1;
                *w *= if bit == 1 { right } else { left };
            }
        }
        Ok(weights)
    }

    /// Forward pass for one sample, returning the `response_dim` tree output.
    fn forward(&self, x: &[f64]) -> TabularResult<Vec<f64>> {
        let weights = self.leaf_weights(x)?;
        let mut out = vec![0.0_f64; self.response_dim];
        for (leaf, &w) in weights.iter().enumerate() {
            let base = leaf * self.response_dim;
            for (d, o) in out.iter_mut().enumerate() {
                *o += w * self.leaf_responses[base + d];
            }
        }
        Ok(out)
    }
}

// ─── Variable-depth ensemble layer ───────────────────────────────────────────────

/// A NODE ensemble layer of variable-depth oblivious trees pooled together.
///
/// Every tree shares `num_features`, `response_dim`, `entmax_alpha`,
/// `split_scale`, but each carries its own depth.
#[derive(Debug, Clone)]
pub struct VarObliviousLayer {
    trees: Vec<VarObliviousTree>,
    num_features: usize,
    response_dim: usize,
    reduction: EnsembleReduction,
}

impl VarObliviousLayer {
    /// Build a randomly-initialised layer from `config`, seeding from
    /// `config.seed`.
    ///
    /// # Errors
    /// [`TabularError::InvalidTreeCount`] / [`TabularError::InvalidTreeDepth`] /
    /// [`TabularError::InvalidFeatureCount`] / [`TabularError::InvalidParameter`]
    /// for an invalid configuration.
    pub fn new(config: VarObliviousConfig) -> TabularResult<Self> {
        config.validate()?;
        let mut rng = LcgRng::new(config.seed);
        Self::new_with_rng(config, &mut rng)
    }

    /// Build a randomly-initialised layer using a caller-supplied RNG so the
    /// stream can be shared/threaded with sibling layers.
    ///
    /// # Errors
    /// As [`VarObliviousLayer::new`].
    pub fn new_with_rng(config: VarObliviousConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        config.validate()?;
        let mut trees = Vec::with_capacity(config.depths.len());
        for &depth in &config.depths {
            trees.push(VarObliviousTree::new_random(
                depth,
                config.num_features,
                config.response_dim,
                config.entmax_alpha,
                config.split_scale,
                rng,
            ));
        }
        Ok(Self {
            trees,
            num_features: config.num_features,
            response_dim: config.response_dim,
            reduction: config.reduction,
        })
    }

    /// Assemble a layer from pre-built trees (e.g. hand-constructed or loaded).
    /// All trees must agree on `num_features` and `response_dim`.
    ///
    /// # Errors
    /// - [`TabularError::InvalidTreeCount`] if `trees` is empty.
    /// - [`TabularError::DimensionMismatch`] if the trees disagree on
    ///   `num_features` or `response_dim`.
    pub fn from_trees(
        trees: Vec<VarObliviousTree>,
        reduction: EnsembleReduction,
    ) -> TabularResult<Self> {
        let first = trees
            .first()
            .ok_or(TabularError::InvalidTreeCount { n: 0 })?;
        let num_features = first.num_features;
        let response_dim = first.response_dim;
        for tree in &trees {
            if tree.num_features != num_features {
                return Err(TabularError::DimensionMismatch {
                    expected: num_features,
                    got: tree.num_features,
                });
            }
            if tree.response_dim != response_dim {
                return Err(TabularError::DimensionMismatch {
                    expected: response_dim,
                    got: tree.response_dim,
                });
            }
        }
        Ok(Self {
            trees,
            num_features,
            response_dim,
            reduction,
        })
    }

    /// Number of trees in the ensemble.
    #[must_use]
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Output dimension (= `response_dim`).
    #[must_use]
    pub fn response_dim(&self) -> usize {
        self.response_dim
    }

    /// Number of input features expected by [`Self::forward`].
    #[must_use]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// The per-tree depths in ensemble order.
    #[must_use]
    pub fn depths(&self) -> Vec<usize> {
        self.trees.iter().map(VarObliviousTree::depth).collect()
    }

    /// Borrow the underlying trees (e.g. to inspect leaf weights / selectors).
    #[must_use]
    pub fn trees(&self) -> &[VarObliviousTree] {
        &self.trees
    }

    /// Forward pass for a single sample `x` of length `num_features`, returning the
    /// pooled response vector of length `response_dim`.
    ///
    /// # Errors
    /// [`TabularError::DimensionMismatch`] if `x.len() != num_features`; propagates
    /// entmax solver errors.
    pub fn forward(&self, x: &[f64]) -> TabularResult<Vec<f64>> {
        if x.len() != self.num_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.num_features,
                got: x.len(),
            });
        }
        let mut agg = vec![0.0_f64; self.response_dim];
        for tree in &self.trees {
            let out = tree.forward(x)?;
            for (a, &v) in agg.iter_mut().zip(out.iter()) {
                *a += v;
            }
        }
        if self.reduction == EnsembleReduction::Mean {
            let inv = 1.0 / self.trees.len() as f64;
            for a in &mut agg {
                *a *= inv;
            }
        }
        Ok(agg)
    }

    /// Batched forward pass.  `x` is a flat `[batch_size * num_features]` buffer;
    /// the result is a flat `[batch_size * response_dim]` buffer.
    ///
    /// # Errors
    /// [`TabularError::EmptyInput`] if `batch_size == 0`;
    /// [`TabularError::DimensionMismatch`] if `x.len() != batch_size * num_features`.
    pub fn forward_batch(&self, x: &[f64], batch_size: usize) -> TabularResult<Vec<f64>> {
        if batch_size == 0 {
            return Err(TabularError::EmptyInput);
        }
        let in_d = self.num_features;
        if x.len() != batch_size * in_d {
            return Err(TabularError::DimensionMismatch {
                expected: batch_size * in_d,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(batch_size * self.response_dim);
        for b in 0..batch_size {
            let row = &x[b * in_d..(b + 1) * in_d];
            let pred = self.forward(row)?;
            out.extend_from_slice(&pred);
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::tree::node_oblivious::next_normal_f64;

    fn sum(v: &[f64]) -> f64 {
        v.iter().sum()
    }

    fn mixed_config() -> VarObliviousConfig {
        VarObliviousConfig {
            depths: vec![2, 3, 4],
            num_features: 7,
            response_dim: 3,
            entmax_alpha: 1.5,
            split_scale: 1.0,
            reduction: EnsembleReduction::Mean,
            seed: 41,
        }
    }

    fn random_input(seed: u64, n: usize) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n).map(|_| next_normal_f64(&mut rng, 1.0)).collect()
    }

    // Test 1: per-level feature-selector weights form a valid simplex.
    #[test]
    fn feature_selectors_are_simplex_points() {
        let layer = VarObliviousLayer::new(mixed_config()).expect("layer should build");
        for tree in layer.trees() {
            for level in 0..tree.depth() {
                let s = tree
                    .feature_selector(level)
                    .expect("selector should compute");
                assert_eq!(s.len(), layer.num_features());
                assert!(
                    (sum(&s) - 1.0).abs() < 1e-6,
                    "selector sum={} at level {level}",
                    sum(&s)
                );
                assert!(
                    s.iter().all(|&v| v >= -1e-12),
                    "selector must be non-negative"
                );
            }
        }
        // Out-of-range level is rejected.
        let tree0 = &layer.trees()[0];
        assert!(matches!(
            tree0.feature_selector(tree0.depth()),
            Err(TabularError::InvalidParameter { .. })
        ));
    }

    // Test 2: per-tree leaf weights are a probability distribution.
    #[test]
    fn leaf_weights_are_probability_distributions() {
        let layer = VarObliviousLayer::new(mixed_config()).expect("layer should build");
        // Exercise several random inputs to stress the routing.
        for s in 0..5 {
            let x = random_input(1000 + s, layer.num_features());
            for tree in layer.trees() {
                let w = tree.leaf_weights(&x).expect("leaf weights should compute");
                assert_eq!(w.len(), tree.num_leaves());
                assert!(
                    (sum(&w) - 1.0).abs() < 1e-9,
                    "leaf weights sum={} (depth {})",
                    sum(&w),
                    tree.depth()
                );
                assert!(
                    w.iter().all(|&v| v >= -1e-12),
                    "leaf weights must be non-negative"
                );
            }
        }
    }

    // Test 3: trees of different depths coexist; each contributes 2^{d_t} leaves.
    #[test]
    fn mixed_depths_each_contribute_right_leaf_count() {
        let cfg = mixed_config();
        let depths = cfg.depths.clone();
        let layer = VarObliviousLayer::new(cfg).expect("layer should build");
        assert_eq!(layer.num_trees(), depths.len());
        assert_eq!(layer.depths(), depths);
        let x = random_input(7, layer.num_features());
        for (tree, &d) in layer.trees().iter().zip(depths.iter()) {
            assert_eq!(tree.depth(), d);
            assert_eq!(tree.num_leaves(), 1usize << d);
            let w = tree.leaf_weights(&x).expect("leaf weights");
            assert_eq!(
                w.len(),
                1usize << d,
                "tree depth {d} must yield 2^{d} leaves"
            );
        }
        // Total leaf count across the heterogeneous ensemble.
        let total_leaves: usize = layer.trees().iter().map(VarObliviousTree::num_leaves).sum();
        assert_eq!(total_leaves, (1 << 2) + (1 << 3) + (1 << 4));
    }

    // Test 4: output dimension == response_dim, and the output is finite.
    #[test]
    fn forward_output_dimension_and_finiteness() {
        let cfg = mixed_config();
        let response_dim = cfg.response_dim;
        let layer = VarObliviousLayer::new(cfg).expect("layer should build");
        let x = random_input(99, layer.num_features());
        let out = layer.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), response_dim);
        assert!(out.iter().all(|v| v.is_finite()), "outputs must be finite");
    }

    // Test 5: determinism — same seed + same input ⇒ identical output.
    #[test]
    fn determinism_with_same_seed() {
        let a = VarObliviousLayer::new(mixed_config()).expect("layer a");
        let b = VarObliviousLayer::new(mixed_config()).expect("layer b");
        let x = random_input(2024, a.num_features());
        let oa = a.forward(&x).expect("forward a");
        let ob = b.forward(&x).expect("forward b");
        assert_eq!(oa, ob, "same seed must give bit-identical outputs");
    }

    // Test 6a: a hand-built depth-1 entmoid-1.5 tree at logit 0 ⇒ gate exactly 0.5.
    #[test]
    fn handbuilt_depth1_entmoid15_zero_logit_closed_form() {
        // num_features = 1 ⇒ the entmax selector over a single feature is exactly
        // [1.0], so f(x) = x[0].  threshold = 0, scale = 1, x = 0 ⇒ logit = 0 ⇒
        // entmoid-1.5(0) = 0.5 exactly ⇒ leaf weights [0.5, 0.5].
        let tree = VarObliviousTree::from_parts(
            vec![0.5],                // 1 level × 1 feature selector logit
            vec![0.0],                // threshold
            vec![1.0, 2.0, 3.0, 4.0], // 2 leaves × response_dim 2: R0=[1,2], R1=[3,4]
            1,                        // depth
            1,                        // num_features
            2,                        // response_dim
            1.5,                      // entmax alpha
            1.0,                      // split scale
        )
        .expect("tree should build");

        let x = [0.0_f64];
        let w = tree.leaf_weights(&x).expect("leaf weights");
        assert!(
            (w[0] - 0.5).abs() < 1e-9,
            "left weight should be 0.5, got {}",
            w[0]
        );
        assert!(
            (w[1] - 0.5).abs() < 1e-9,
            "right weight should be 0.5, got {}",
            w[1]
        );

        let layer = VarObliviousLayer::from_trees(vec![tree], EnsembleReduction::Sum)
            .expect("layer should build");
        let out = layer.forward(&x).expect("forward");
        // 0.5·[1,2] + 0.5·[3,4] = [2.0, 3.0].
        assert!((out[0] - 2.0).abs() < 1e-9, "out[0]={}", out[0]);
        assert!((out[1] - 3.0).abs() < 1e-9, "out[1]={}", out[1]);
    }

    // Test 6b: a hand-built depth-1 sparsemax (α=2) tree has a *linear* entmoid,
    // gate(t) = (t+1)/2 on t ∈ [-1, 1] — an independent closed form.
    #[test]
    fn handbuilt_depth1_sparsemax_linear_gate_closed_form() {
        // For α = 2 the two-class entmax of [t, 0] is sparsemax, whose first
        // component is (t+1)/2 while both classes carry mass (|t| ≤ 1).  Pick
        // t = 0.4 ⇒ gate = 0.7 exactly ⇒ leaf weights [0.3, 0.7].
        let tree = VarObliviousTree::from_parts(
            vec![1.0],                // single-feature selector ⇒ [1.0]
            vec![0.0],                // threshold
            vec![1.0, 2.0, 3.0, 4.0], // R0=[1,2], R1=[3,4]
            1,
            1,
            2,
            2.0, // sparsemax
            1.0,
        )
        .expect("tree should build");

        let x = [0.4_f64];
        let w = tree.leaf_weights(&x).expect("leaf weights");
        assert!(
            (w[0] - 0.3).abs() < 1e-9,
            "left weight should be 0.3, got {}",
            w[0]
        );
        assert!(
            (w[1] - 0.7).abs() < 1e-9,
            "right weight should be 0.7, got {}",
            w[1]
        );

        let layer = VarObliviousLayer::from_trees(vec![tree], EnsembleReduction::Sum)
            .expect("layer should build");
        let out = layer.forward(&x).expect("forward");
        // 0.3·[1,2] + 0.7·[3,4] = [2.4, 3.4].
        assert!((out[0] - 2.4).abs() < 1e-9, "out[0]={}", out[0]);
        assert!((out[1] - 3.4).abs() < 1e-9, "out[1]={}", out[1]);
    }

    // Mean pooling equals the manual average of per-tree outputs.
    #[test]
    fn mean_pooling_equals_average_of_trees() {
        let layer = VarObliviousLayer::new(mixed_config()).expect("layer should build");
        let x = random_input(555, layer.num_features());
        let pooled = layer.forward(&x).expect("forward");
        let mut manual = vec![0.0_f64; layer.response_dim()];
        for tree in layer.trees() {
            let o = tree.forward(&x).expect("tree forward");
            for (m, &v) in manual.iter_mut().zip(o.iter()) {
                *m += v;
            }
        }
        let inv = 1.0 / layer.num_trees() as f64;
        for m in &mut manual {
            *m *= inv;
        }
        for (p, m) in pooled.iter().zip(manual.iter()) {
            assert!((p - m).abs() < 1e-12, "mean pooling mismatch");
        }
    }

    // Batched forward matches per-sample forward.
    #[test]
    fn batched_matches_per_sample() {
        let layer = VarObliviousLayer::new(mixed_config()).expect("layer should build");
        let nf = layer.num_features();
        let rd = layer.response_dim();
        let batch = 4;
        let x = random_input(31337, batch * nf);
        let batched = layer.forward_batch(&x, batch).expect("batch forward");
        assert_eq!(batched.len(), batch * rd);
        for b in 0..batch {
            let row = &x[b * nf..(b + 1) * nf];
            let single = layer.forward(row).expect("single forward");
            for d in 0..rd {
                assert!(
                    (batched[b * rd + d] - single[d]).abs() < 1e-12,
                    "batched output must match per-sample"
                );
            }
        }
    }

    // Configuration validation rejects malformed setups.
    #[test]
    fn config_validation_errors() {
        let mut c = mixed_config();
        c.depths = vec![];
        assert!(matches!(
            VarObliviousLayer::new(c),
            Err(TabularError::InvalidTreeCount { .. })
        ));

        let mut c = mixed_config();
        c.depths = vec![2, 0, 4];
        assert!(matches!(
            VarObliviousLayer::new(c),
            Err(TabularError::InvalidTreeDepth { .. })
        ));

        let mut c = mixed_config();
        c.num_features = 0;
        assert!(matches!(
            VarObliviousLayer::new(c),
            Err(TabularError::InvalidFeatureCount { .. })
        ));

        let mut c = mixed_config();
        c.response_dim = 0;
        assert!(matches!(
            VarObliviousLayer::new(c),
            Err(TabularError::InvalidParameter { .. })
        ));

        let mut c = mixed_config();
        c.entmax_alpha = 0.5;
        assert!(matches!(
            VarObliviousLayer::new(c),
            Err(TabularError::InvalidParameter { .. })
        ));

        let mut c = mixed_config();
        c.split_scale = 0.0;
        assert!(matches!(
            VarObliviousLayer::new(c),
            Err(TabularError::InvalidParameter { .. })
        ));
    }

    // from_parts and from_trees reject inconsistent shapes.
    #[test]
    fn constructors_reject_bad_shapes() {
        // Wrong feature-selector length.
        assert!(matches!(
            VarObliviousTree::from_parts(
                vec![1.0, 2.0], // should be depth*num_features = 1
                vec![0.0],
                vec![1.0, 2.0],
                1,
                1,
                1,
                1.5,
                1.0,
            ),
            Err(TabularError::DimensionMismatch { .. })
        ));
        // Wrong leaf-response length (depth 2 ⇒ 4 leaves × 1 = 4 expected).
        assert!(matches!(
            VarObliviousTree::from_parts(
                vec![1.0, 2.0],
                vec![0.0, 0.0],
                vec![1.0, 2.0, 3.0],
                2,
                1,
                1,
                1.5,
                1.0,
            ),
            Err(TabularError::DimensionMismatch { .. })
        ));
        // Mismatched trees in a layer.
        let t_a = VarObliviousTree::from_parts(
            vec![0.0],
            vec![0.0],
            vec![1.0, 2.0, 3.0, 4.0],
            1,
            1,
            2,
            1.5,
            1.0,
        )
        .expect("t_a");
        let t_b = VarObliviousTree::from_parts(
            vec![0.0, 0.0],
            vec![0.0],
            vec![1.0, 2.0],
            1,
            2,
            1,
            1.5,
            1.0,
        )
        .expect("t_b");
        assert!(matches!(
            VarObliviousLayer::from_trees(vec![t_a, t_b], EnsembleReduction::Mean),
            Err(TabularError::DimensionMismatch { .. })
        ));
        // Empty layer.
        assert!(matches!(
            VarObliviousLayer::from_trees(vec![], EnsembleReduction::Mean),
            Err(TabularError::InvalidTreeCount { .. })
        ));
    }

    // Forward rejects a wrongly-sized input.
    #[test]
    fn forward_dimension_mismatch_errors() {
        let layer = VarObliviousLayer::new(mixed_config()).expect("layer should build");
        assert!(matches!(
            layer.forward(&[1.0, 2.0]),
            Err(TabularError::DimensionMismatch { .. })
        ));
        let nf = layer.num_features();
        assert!(matches!(
            layer.forward_batch(&vec![0.0; nf * 3], 0),
            Err(TabularError::EmptyInput)
        ));
        assert!(matches!(
            layer.forward_batch(&vec![0.0; nf * 3 + 1], 3),
            Err(TabularError::DimensionMismatch { .. })
        ));
    }
}
