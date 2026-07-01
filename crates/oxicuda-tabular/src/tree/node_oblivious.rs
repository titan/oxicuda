//! NODE — Neural Oblivious Decision Ensembles (Popov, Morozov & Babenko 2019).
//!
//! This module implements the *forward inference* path of a NODE ensemble layer
//! as a numerically careful, `f64` CPU reference.  It is intentionally distinct
//! from [`crate::tree::node`] (an `f32`, sigmoid-gated variant): here the soft
//! routing is driven end-to-end by the **entmax** family of simplex projections,
//! exactly as in the paper.
//!
//! ## Oblivious decision trees
//!
//! An *oblivious* decision tree of depth `d` is a balanced binary tree in which
//! **every internal node at the same level shares the same split**.  A depth-`d`
//! tree therefore has only `d` `(feature-selector, threshold)` pairs and `2^d`
//! leaves, each leaf holding a response vector of dimension `response_dim`.
//! Because the per-level decisions are independent, the leaf assignment factors
//! into an outer product of the `d` gate probabilities — this is what makes the
//! soft (differentiable) version cheap and exact.
//!
//! ## Differentiable routing
//!
//! For level `i`:
//!
//! 1. **Feature choice.** A learnable weight vector `F_i ∈ ℝ^{num_features}` is
//!    pushed through **entmax-α** (`α = 1.5` default; `α = 2` ⇒ *sparsemax*) to
//!    obtain a sparse, soft one-hot selector `s_i` on the probability simplex.
//!    The selected feature value is `f_i(x) = ⟨s_i, x⟩`.
//! 2. **Soft split.** The 2-logit vector `[(f_i(x) − b_i)·scale, 0]` is mapped by
//!    the 2-class entmax (the **entmoid**) to a gate `c_i ∈ [0, 1]`.
//! 3. **Leaf weight.** Leaf `j` (with bits `j_{d-1}…j_0`) receives weight
//!    `∏_i (c_i if bit_i = 1 else 1 − c_i)`.  The `2^d` weights are non-negative
//!    and sum to 1.
//!
//! The tree output is `Σ_j w_j · R_j`; the ensemble averages (or sums) the
//! `num_trees` tree outputs.
//!
//! ## References
//! - Popov, S., Morozov, S. & Babenko, A. (2019). "Neural Oblivious Decision
//!   Ensembles for Deep Learning on Tabular Data." ICLR 2020.
//! - Peters, B., Niculae, V. & Martins, A. F. T. (2019). "Sparse Sequence-to-
//!   Sequence Models." ACL 2019 (entmax bisection / α-entmax).
//! - Martins, A. F. T. & Astudillo, R. F. (2016). "From Softmax to Sparsemax."
//!   ICML 2016.

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── Numeric helpers on the crate RNG ────────────────────────────────────────────

/// Draw a uniform `f64` in `[0, 1)` from the crate's `LcgRng`.
///
/// `LcgRng` exposes `next_u32`; we use the full 32 bits and divide by `2^32`
/// (never the `2^31` work-around).  This yields 32-bit uniform resolution which
/// is ample for weight initialisation.
#[inline]
fn next_f64_unit(rng: &mut LcgRng) -> f64 {
    f64::from(rng.next_u32()) / 4_294_967_296.0
}

/// Draw a single `N(0, std_dev)` sample (Box–Muller) in `f64`.
///
/// `pub(crate)` so the sibling [`crate::tree::var_oblivious`] module can reuse the
/// identical initialiser/sampler stream (behaviour of this fixed-depth module is
/// unchanged — visibility only).
#[inline]
pub(crate) fn next_normal_f64(rng: &mut LcgRng, std_dev: f64) -> f64 {
    // Guard the log argument away from exactly zero.
    let u1 = (next_f64_unit(rng) + 1e-12).min(1.0 - 1e-12);
    let u2 = next_f64_unit(rng);
    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    radius * theta.cos() * std_dev
}

/// Fill `buf` with `N(0, std_dev)` samples in `f64`.
///
/// `pub(crate)` so [`crate::tree::var_oblivious`] can reuse the exact init scheme
/// (visibility only; fixed-depth behaviour unchanged).
pub(crate) fn fill_normal_f64(rng: &mut LcgRng, buf: &mut [f64], std_dev: f64) {
    for slot in buf.iter_mut() {
        *slot = next_normal_f64(rng, std_dev);
    }
}

// ─── entmax / sparsemax simplex projections (f64) ────────────────────────────────

/// Number of bisection iterations used by the general entmax solver.
///
/// 60 halvings of a bracket initially `≤ |z|`-wide drive the residual well below
/// `f64` round-off for any realistic logit range.
const ENTMAX_BISECT_ITERS: usize = 60;

/// Tolerance for the post-hoc simplex-sum sanity check.
const SIMPLEX_TOL: f64 = 1e-6;

/// **Sparsemax** (α = 2): Euclidean projection of `z` onto the probability
/// simplex, `sparsemax(z)_i = max(0, z_i − τ)`.
///
/// Exact, sort-based solver of Martins & Astudillo (2016): find the support
/// size `k*` then set `τ = (Σ_{j≤k*} z_(j) − 1) / k*`.
///
/// # Errors
/// [`TabularError::EmptyInput`] if `z` is empty.
pub fn sparsemax_f64(z: &[f64]) -> TabularResult<Vec<f64>> {
    if z.is_empty() {
        return Err(TabularError::EmptyInput);
    }
    let mut sorted = z.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0_f64;
    let mut k_star = 1usize;
    for (j, &z_j) in sorted.iter().enumerate() {
        cumsum += z_j;
        // 1-based index k = j + 1.
        let k = j as f64 + 1.0;
        if 1.0 + k * z_j - cumsum > 0.0 {
            k_star = j + 1;
        }
    }
    let tau = (sorted.iter().take(k_star).sum::<f64>() - 1.0) / k_star as f64;
    Ok(z.iter().map(|&zi| (zi - tau).max(0.0)).collect())
}

/// **entmax-α** for general `α ∈ (1, 2]`, via the Peters et al. (2019) bisection.
///
/// The α-entmax map is
/// `p_i = [ (α − 1)·z_i − τ ]_+^{1/(α − 1)}`,
/// with the threshold `τ` chosen so that `Σ_i p_i = 1`.  We solve for `τ` by
/// bisection on the monotone-decreasing sum.  `α = 2` reduces to sparsemax and
/// is dispatched to the exact [`sparsemax_f64`] solver.
///
/// # Errors
/// - [`TabularError::EmptyInput`] if `z` is empty.
/// - [`TabularError::InvalidParameter`] if `α ∉ (1, 2]`.
/// - [`TabularError::NormalizationFailed`] if the bisection fails to converge.
pub fn entmax_alpha_f64(z: &[f64], alpha: f64) -> TabularResult<Vec<f64>> {
    if z.is_empty() {
        return Err(TabularError::EmptyInput);
    }
    if !(alpha > 1.0 && alpha <= 2.0) {
        return Err(TabularError::InvalidParameter {
            name: "entmax_alpha".into(),
            msg: format!("alpha must lie in (1, 2], got {alpha}"),
        });
    }
    // α = 2 is exactly sparsemax — use the closed-form sort solver.
    if (alpha - 2.0).abs() < 1e-12 {
        return sparsemax_f64(z);
    }

    let am1 = alpha - 1.0; // (α − 1)
    let inv_am1 = 1.0 / am1; // 1 / (α − 1)
    // Work in the transformed coordinate y_i = (α − 1)·z_i, so p_i = [y_i − τ]_+^{1/(α−1)}.
    let y: Vec<f64> = z.iter().map(|&zi| am1 * zi).collect();

    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);

    // sum(τ) is strictly decreasing in τ.
    //   τ = y_max          ⇒ at most one tiny term  ⇒ sum ≤ (something small)
    //   τ = y_min − margin ⇒ every term active      ⇒ sum large (≥ 1)
    // Choose a lower bracket guaranteeing sum ≥ 1.  With n active terms each
    // contributing ((y_min − τ))^{1/(α−1)}, picking τ low enough forces sum ≥ 1.
    let n = y.len() as f64;
    // Largest single term is bounded by (y_max − τ)^{inv_am1}; require it ≥ 1.
    let mut lo = y_min - (1.0 + n).powf(am1) - 1.0;
    let mut hi = y_max;

    let sum_at = |tau: f64| -> f64 {
        y.iter()
            .map(|&yi| {
                let base = yi - tau;
                if base > 0.0 { base.powf(inv_am1) } else { 0.0 }
            })
            .sum::<f64>()
    };

    // Ensure the bracket really straddles the root.
    let mut guard = 0;
    while sum_at(lo) < 1.0 && guard < 64 {
        lo -= (hi - lo).abs().max(1.0);
        guard += 1;
    }

    for _ in 0..ENTMAX_BISECT_ITERS {
        let mid = 0.5 * (lo + hi);
        if sum_at(mid) > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let tau = 0.5 * (lo + hi);

    let mut out: Vec<f64> = y
        .iter()
        .map(|&yi| {
            let base = yi - tau;
            if base > 0.0 { base.powf(inv_am1) } else { 0.0 }
        })
        .collect();

    // Renormalise to kill residual bisection error, then verify.
    let total: f64 = out.iter().sum();
    if !(total.is_finite()) || total <= 0.0 {
        return Err(TabularError::NormalizationFailed {
            msg: format!("entmax-{alpha} produced a non-positive mass: {total}"),
        });
    }
    if (total - 1.0).abs() > 1e-3 {
        for v in &mut out {
            *v /= total;
        }
    }
    let check: f64 = out.iter().sum();
    if (check - 1.0).abs() > SIMPLEX_TOL {
        return Err(TabularError::NormalizationFailed {
            msg: format!("entmax-{alpha} did not converge: sum={check}"),
        });
    }
    Ok(out)
}

/// **entmoid-α**: the 2-class entmax, returning the probability `p` of the *first*
/// logit (with the second logit pinned at 0).
///
/// For `α = 1.5` this admits the closed form used by the entmax authors; for
/// other `α ∈ (1, 2]` we fall back to the general [`entmax_alpha_f64`] solver on
/// the 2-logit vector `[t, 0]`.  The returned gate is clamped to `[0, 1]`.
///
/// # Errors
/// [`TabularError::InvalidParameter`] if `α ∉ (1, 2]`.
pub fn entmoid_alpha_f64(t: f64, alpha: f64) -> TabularResult<f64> {
    if !(alpha > 1.0 && alpha <= 2.0) {
        return Err(TabularError::InvalidParameter {
            name: "entmax_alpha".into(),
            msg: format!("alpha must lie in (1, 2], got {alpha}"),
        });
    }

    // α = 1.5 closed form (entmoid15).  By symmetry we may assume the input to
    // the closed form is non-negative and reflect afterwards.
    if (alpha - 1.5).abs() < 1e-9 {
        let s = t.signum(); // +1 if t ≥ 0, else −1 (signum(0.0) == 0.0 handled below)
        let a = t.abs();
        // p_pos = entmoid15(a) for a ≥ 0, derived from the α=1.5 two-class entmax.
        //   p = 0.5 * (1 + sqrt(1 − f(a))) where, with the standard derivation,
        //   the gate solves a quadratic; we use the well-known stable expression.
        let p_pos = entmoid15_nonneg(a);
        let p = if s < 0.0 { 1.0 - p_pos } else { p_pos };
        return Ok(p.clamp(0.0, 1.0));
    }

    // General α: solve the 2-logit entmax and take the first component.
    let probs = entmax_alpha_f64(&[t, 0.0], alpha)?;
    Ok(probs[0].clamp(0.0, 1.0))
}

/// Closed-form `entmoid-1.5` evaluated for a **non-negative** argument `a ≥ 0`.
///
/// Solves the two-class α=1.5 entmax `p = [0.5·a − τ]_+²`, `1 − p = [−τ]_+²`,
/// `p + (1−p) = 1`, for the gate `p ∈ [0.5, 1]`.  The algebra reduces to a
/// quadratic whose stable root is returned.
#[inline]
fn entmoid15_nonneg(a: f64) -> f64 {
    // Two-class entmax-1.5 with logits [a, 0], a ≥ 0:
    //   p     = ([0.5 a − τ]_+)²
    //   1 − p = ([0.5·0 − τ]_+)² = ([−τ]_+)²
    // Write u = 0.5 a ≥ 0.  While *both* classes carry mass we need τ < 0, and
    // (u − τ)² + τ² = 1  ⇒  2τ² − 2uτ + (u² − 1) = 0
    //                    ⇒  τ = [u − sqrt(2 − u²)] / 2   (the root ≤ 0).
    // That root is ≤ 0 exactly while u ≤ 1; at u = 1 we have τ = 0 and the
    // second class loses all its mass.  Hence for u ≥ 1 the gate saturates at
    // p = 1 (the second term is clamped to zero), and for u < 1
    //   p = (u − τ)² = ([u + sqrt(2 − u²)] / 2)².
    let u = 0.5 * a;
    if u >= 1.0 {
        // Second class has zero mass: the gate is fully saturated.
        return 1.0;
    }
    let root = (u + (2.0 - u * u).sqrt()) * 0.5;
    (root * root).min(1.0)
}

// ─── Configuration ───────────────────────────────────────────────────────────────

/// How the per-tree outputs are pooled into the ensemble prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnsembleReduction {
    /// Arithmetic mean over the `num_trees` trees (default).
    #[default]
    Mean,
    /// Plain sum over the `num_trees` trees.
    Sum,
}

/// Configuration for a [`NodeObliviousLayer`].
#[derive(Debug, Clone)]
pub struct NodeObliviousConfig {
    /// Number of oblivious trees in the ensemble (≥ 1).
    pub num_trees: usize,
    /// Depth of each tree (≥ 1).  Produces `2^depth` leaves per tree.
    pub depth: usize,
    /// Number of input features (≥ 1).
    pub num_features: usize,
    /// Dimension of each leaf response vector / the model output (≥ 1).
    pub response_dim: usize,
    /// Entmax temperature `α ∈ (1, 2]`.  `1.5` (default) ⇒ entmax-1.5,
    /// `2.0` ⇒ sparsemax.
    pub entmax_alpha: f64,
    /// Multiplicative sharpness applied to `(f_i(x) − b_i)` before the entmoid.
    pub split_scale: f64,
    /// Ensemble pooling rule.
    pub reduction: EnsembleReduction,
    /// RNG seed used when initialising parameters via [`NodeObliviousLayer::new`].
    pub seed: u64,
}

impl NodeObliviousConfig {
    /// A sensible default: 16 trees, depth 6, entmax-1.5, mean pooling.
    #[must_use]
    pub fn new(num_features: usize, response_dim: usize) -> Self {
        Self {
            num_trees: 16,
            depth: 6,
            num_features,
            response_dim,
            entmax_alpha: 1.5,
            split_scale: 1.0,
            reduction: EnsembleReduction::Mean,
            seed: 0,
        }
    }

    fn validate(&self) -> TabularResult<()> {
        if self.num_trees == 0 {
            return Err(TabularError::InvalidTreeCount { n: 0 });
        }
        if self.depth == 0 {
            return Err(TabularError::InvalidTreeDepth { depth: 0 });
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
        // A depth that would overflow `usize` leaf indexing is rejected early.
        if self.depth >= usize::BITS as usize {
            return Err(TabularError::InvalidTreeDepth { depth: self.depth });
        }
        Ok(())
    }
}

// ─── Single oblivious tree ───────────────────────────────────────────────────────

/// A single soft oblivious decision tree (entmax-routed).
#[derive(Debug, Clone)]
pub struct ObliviousTree {
    /// Feature-selector weights, flattened `[depth * num_features]`.
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

impl ObliviousTree {
    /// Randomly initialise one tree.
    ///
    /// Feature-selector logits are drawn `N(0, 0.1)` (small, so entmax starts
    /// near-uniform and gradually sparsifies during training); thresholds
    /// `N(0, 1)`; leaf responses `N(0, 1/√response_dim)` (fan-in scaling).
    fn new(cfg: &NodeObliviousConfig, rng: &mut LcgRng) -> Self {
        let n_leaves = 1usize << cfg.depth;
        let mut feature_selectors = vec![0.0_f64; cfg.depth * cfg.num_features];
        fill_normal_f64(rng, &mut feature_selectors, 0.1);

        let mut thresholds = vec![0.0_f64; cfg.depth];
        fill_normal_f64(rng, &mut thresholds, 1.0);

        let resp_std = 1.0 / (cfg.response_dim as f64).sqrt();
        let mut leaf_responses = vec![0.0_f64; n_leaves * cfg.response_dim];
        fill_normal_f64(rng, &mut leaf_responses, resp_std);

        Self {
            feature_selectors,
            thresholds,
            leaf_responses,
            depth: cfg.depth,
            num_features: cfg.num_features,
            response_dim: cfg.response_dim,
            entmax_alpha: cfg.entmax_alpha,
            split_scale: cfg.split_scale,
        }
    }

    /// Number of leaves, `2^depth`.
    #[must_use]
    pub fn num_leaves(&self) -> usize {
        1usize << self.depth
    }

    /// Compute the `depth` per-level gate probabilities `c_i ∈ [0, 1]` for `x`.
    fn level_gates(&self, x: &[f64]) -> TabularResult<Vec<f64>> {
        let mut gates = Vec::with_capacity(self.depth);
        for level in 0..self.depth {
            let row =
                &self.feature_selectors[level * self.num_features..(level + 1) * self.num_features];
            // Sparse soft one-hot feature selector.
            let selector = entmax_alpha_f64(row, self.entmax_alpha)?;
            // Selected feature value = ⟨selector, x⟩.
            let f_val: f64 = selector.iter().zip(x.iter()).map(|(&s, &xi)| s * xi).sum();
            let logit = (f_val - self.thresholds[level]) * self.split_scale;
            let gate = entmoid_alpha_f64(logit, self.entmax_alpha)?;
            gates.push(gate);
        }
        Ok(gates)
    }

    /// Compute the `2^depth` leaf weights for `x`.  They are non-negative and sum
    /// to 1 (up to floating-point round-off).
    ///
    /// Bit convention: leaf index bit `level` (taken MSB-first so that level 0 is
    /// the most-significant bit) equal to 1 routes *right* with weight `c`, 0
    /// routes *left* with weight `1 − c`.
    fn leaf_weights(&self, x: &[f64]) -> TabularResult<Vec<f64>> {
        let gates = self.level_gates(x)?;
        let n_leaves = self.num_leaves();
        let mut weights = vec![1.0_f64; n_leaves];
        for (level, &c) in gates.iter().enumerate() {
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
        if x.len() != self.num_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.num_features,
                got: x.len(),
            });
        }
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

// ─── Ensemble layer ──────────────────────────────────────────────────────────────

/// A NODE ensemble layer: `num_trees` soft oblivious trees pooled together.
#[derive(Debug, Clone)]
pub struct NodeObliviousLayer {
    trees: Vec<ObliviousTree>,
    config: NodeObliviousConfig,
}

impl NodeObliviousLayer {
    /// Build a randomly-initialised layer from `config`, seeding from
    /// `config.seed`.
    ///
    /// # Errors
    /// [`TabularError::InvalidTreeCount`] / [`TabularError::InvalidTreeDepth`] /
    /// [`TabularError::InvalidFeatureCount`] / [`TabularError::InvalidParameter`]
    /// for an invalid configuration.
    pub fn new(config: NodeObliviousConfig) -> TabularResult<Self> {
        config.validate()?;
        let mut rng = LcgRng::new(config.seed);
        Self::new_with_rng(config, &mut rng)
    }

    /// Build a randomly-initialised layer using a caller-supplied RNG, so the
    /// stream can be shared/threaded with sibling layers.
    ///
    /// # Errors
    /// As [`NodeObliviousLayer::new`].
    pub fn new_with_rng(config: NodeObliviousConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        config.validate()?;
        let mut trees = Vec::with_capacity(config.num_trees);
        for _ in 0..config.num_trees {
            trees.push(ObliviousTree::new(&config, rng));
        }
        Ok(Self { trees, config })
    }

    /// Number of trees in the ensemble.
    #[must_use]
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Output dimension (= `response_dim`).
    #[must_use]
    pub fn response_dim(&self) -> usize {
        self.config.response_dim
    }

    /// Number of input features expected by [`Self::forward`].
    #[must_use]
    pub fn num_features(&self) -> usize {
        self.config.num_features
    }

    /// Borrow the underlying trees (e.g. to inspect leaf weights in tests).
    #[must_use]
    pub fn trees(&self) -> &[ObliviousTree] {
        &self.trees
    }

    /// Forward pass for a single sample `x` of length `num_features`.
    ///
    /// Returns the pooled response vector of length `response_dim`.
    ///
    /// # Errors
    /// [`TabularError::DimensionMismatch`] if `x.len() != num_features`;
    /// propagates entmax solver errors.
    pub fn forward(&self, x: &[f64]) -> TabularResult<Vec<f64>> {
        if x.len() != self.config.num_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.config.num_features,
                got: x.len(),
            });
        }
        let mut agg = vec![0.0_f64; self.config.response_dim];
        for tree in &self.trees {
            let out = tree.forward(x)?;
            for (a, &v) in agg.iter_mut().zip(out.iter()) {
                *a += v;
            }
        }
        if self.config.reduction == EnsembleReduction::Mean {
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
        let in_d = self.config.num_features;
        if x.len() != batch_size * in_d {
            return Err(TabularError::DimensionMismatch {
                expected: batch_size * in_d,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(batch_size * self.config.response_dim);
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

    fn small_config() -> NodeObliviousConfig {
        NodeObliviousConfig {
            num_trees: 6,
            depth: 4,
            num_features: 8,
            response_dim: 3,
            entmax_alpha: 1.5,
            split_scale: 1.0,
            reduction: EnsembleReduction::Mean,
            seed: 31,
        }
    }

    fn sum(v: &[f64]) -> f64 {
        v.iter().sum()
    }

    #[test]
    fn sparsemax_is_a_simplex_point() {
        let z = vec![1.0_f64, 2.0, 3.0, 4.0, -1.0];
        let p = sparsemax_f64(&z).expect("sparsemax should succeed");
        assert!((sum(&p) - 1.0).abs() < 1e-9, "sum={}", sum(&p));
        assert!(p.iter().all(|&v| v >= -1e-12), "non-negativity");
    }

    #[test]
    fn sparsemax_produces_sparsity() {
        // A dominating logit yields an exact one-hot, with true zeros elsewhere.
        let z = vec![10.0_f64, 0.0, 0.0, 0.0];
        let p = sparsemax_f64(&z).expect("sparsemax should succeed");
        assert!((p[0] - 1.0).abs() < 1e-9);
        assert!(p[1..].iter().all(|&v| v == 0.0), "expected exact zeros");
        // A milder spread still zeroes the smallest entries (true sparsity).
        let z2 = vec![3.0_f64, 0.1, 0.0, -2.0];
        let p2 = sparsemax_f64(&z2).expect("sparsemax should succeed");
        assert!(p2.contains(&0.0), "sparsemax must zero some mass");
        assert!((sum(&p2) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn entmax15_is_a_simplex_point() {
        let cases: &[&[f64]] = &[
            &[0.1, 0.5, 0.3, 0.1],
            &[-1.0, 2.0, 3.0, -0.5],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[0.0, 0.0, 0.0],
        ];
        for &z in cases {
            let p = entmax_alpha_f64(z, 1.5).expect("entmax15 should succeed");
            assert!((sum(&p) - 1.0).abs() < 1e-6, "sum={} for {z:?}", sum(&p));
            assert!(p.iter().all(|&v| v >= -1e-12), "non-negativity for {z:?}");
        }
    }

    #[test]
    fn entmax_matches_sparsemax_at_alpha_two() {
        let z = vec![0.4_f64, 2.1, -0.7, 1.3, 0.0];
        let a = entmax_alpha_f64(&z, 2.0).expect("entmax α=2 should succeed");
        let b = sparsemax_f64(&z).expect("sparsemax should succeed");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-9, "α=2 entmax must equal sparsemax");
        }
    }

    #[test]
    fn entmoid_in_unit_interval_and_monotone() {
        let mut prev = -1.0_f64;
        for k in -50..=50 {
            let t = k as f64 * 0.2;
            let p = entmoid_alpha_f64(t, 1.5).expect("entmoid should succeed");
            assert!((0.0..=1.0).contains(&p), "gate {p} out of [0,1] at t={t}");
            assert!(p + 1e-9 >= prev, "entmoid must be non-decreasing");
            prev = p;
        }
        // Symmetry: p(t) + p(-t) = 1.
        for &t in &[0.3_f64, 1.0, 2.5, 5.0] {
            let a = entmoid_alpha_f64(t, 1.5).expect("ok");
            let b = entmoid_alpha_f64(-t, 1.5).expect("ok");
            assert!((a + b - 1.0).abs() < 1e-9, "symmetry failed at t={t}");
        }
        // At t = 0 the gate is exactly 0.5.
        let mid = entmoid_alpha_f64(0.0, 1.5).expect("ok");
        assert!(
            (mid - 0.5).abs() < 1e-9,
            "entmoid(0) should be 0.5, got {mid}"
        );
    }

    #[test]
    fn entmoid_general_alpha_matches_two_logit_entmax() {
        // For α=1.7 the closed form is not used; cross-check the gate against the
        // first component of the 2-logit entmax directly.
        for &t in &[-1.2_f64, 0.0, 0.8, 3.0] {
            let gate = entmoid_alpha_f64(t, 1.7).expect("ok");
            let direct = entmax_alpha_f64(&[t, 0.0], 1.7).expect("ok");
            assert!((gate - direct[0]).abs() < 1e-9, "mismatch at t={t}");
        }
    }

    #[test]
    fn leaf_weights_sum_to_one() {
        let cfg = small_config();
        let layer = NodeObliviousLayer::new(cfg).expect("layer should build");
        let mut rng = LcgRng::new(123);
        let x: Vec<f64> = (0..layer.num_features())
            .map(|_| next_normal_f64(&mut rng, 1.0))
            .collect();
        for tree in layer.trees() {
            let w = tree.leaf_weights(&x).expect("leaf weights should compute");
            assert_eq!(w.len(), tree.num_leaves());
            assert!((sum(&w) - 1.0).abs() < 1e-9, "leaf weights sum={}", sum(&w));
            assert!(w.iter().all(|&v| v >= -1e-12), "leaf weights non-negative");
        }
    }

    #[test]
    fn forward_output_dimension() {
        let cfg = small_config();
        let response_dim = cfg.response_dim;
        let layer = NodeObliviousLayer::new(cfg).expect("layer should build");
        let x = vec![0.25_f64; layer.num_features()];
        let out = layer.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), response_dim);
        assert!(out.iter().all(|v| v.is_finite()), "outputs must be finite");
    }

    #[test]
    fn ensemble_mean_equals_average_of_trees() {
        let cfg = small_config();
        let layer = NodeObliviousLayer::new(cfg).expect("layer should build");
        let x = vec![0.7_f64; layer.num_features()];
        let pooled = layer.forward(&x).expect("forward should succeed");
        // Manually average each tree's output.
        let mut manual = vec![0.0_f64; layer.response_dim()];
        for tree in layer.trees() {
            let o = tree.forward(&x).expect("tree forward should succeed");
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

    #[test]
    fn ensemble_sum_reduction() {
        let mut cfg = small_config();
        cfg.reduction = EnsembleReduction::Sum;
        let layer = NodeObliviousLayer::new(cfg.clone()).expect("layer should build");
        let x = vec![0.5_f64; layer.num_features()];
        let summed = layer.forward(&x).expect("forward should succeed");
        // Compare with the mean layer scaled by num_trees.
        let mut mean_cfg = cfg;
        mean_cfg.reduction = EnsembleReduction::Mean;
        let mean_layer = NodeObliviousLayer::new(mean_cfg).expect("layer should build");
        let meaned = mean_layer.forward(&x).expect("forward should succeed");
        let n = layer.num_trees() as f64;
        for (s, m) in summed.iter().zip(meaned.iter()) {
            assert!((s - m * n).abs() < 1e-9, "sum should be mean * num_trees");
        }
    }

    #[test]
    fn determinism_with_same_seed() {
        let cfg_a = small_config();
        let cfg_b = small_config();
        let a = NodeObliviousLayer::new(cfg_a).expect("layer a should build");
        let b = NodeObliviousLayer::new(cfg_b).expect("layer b should build");
        let x = vec![0.123_f64; a.num_features()];
        let oa = a.forward(&x).expect("forward a should succeed");
        let ob = b.forward(&x).expect("forward b should succeed");
        assert_eq!(oa, ob, "same seed must give identical outputs");
    }

    #[test]
    fn different_seeds_differ() {
        let cfg_a = small_config();
        let mut cfg_b = small_config();
        cfg_b.seed = 999;
        let a = NodeObliviousLayer::new(cfg_a).expect("layer a should build");
        let b = NodeObliviousLayer::new(cfg_b).expect("layer b should build");
        let x = vec![0.4_f64; a.num_features()];
        let oa = a.forward(&x).expect("forward a should succeed");
        let ob = b.forward(&x).expect("forward b should succeed");
        assert!(
            oa.iter().zip(ob.iter()).any(|(p, q)| (p - q).abs() > 1e-9),
            "different seeds should generally differ"
        );
    }

    #[test]
    fn batched_matches_per_sample() {
        let cfg = small_config();
        let layer = NodeObliviousLayer::new(cfg).expect("layer should build");
        let nf = layer.num_features();
        let rd = layer.response_dim();
        let batch = 5;
        let mut rng = LcgRng::new(2024);
        let x: Vec<f64> = (0..batch * nf)
            .map(|_| next_normal_f64(&mut rng, 1.0))
            .collect();
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

    #[test]
    fn sparsemax_alpha_two_routes_through_entmax() {
        // Exercise the α=2 path end-to-end: build a sparsemax-routed layer and
        // confirm leaf weights are still a valid simplex.
        let mut cfg = small_config();
        cfg.entmax_alpha = 2.0;
        let layer = NodeObliviousLayer::new(cfg).expect("layer should build");
        let x = vec![1.5_f64; layer.num_features()];
        for tree in layer.trees() {
            let w = tree.leaf_weights(&x).expect("leaf weights");
            assert!((sum(&w) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn errors_on_invalid_config() {
        // Zero features.
        let mut c = small_config();
        c.num_features = 0;
        assert!(matches!(
            NodeObliviousLayer::new(c),
            Err(TabularError::InvalidFeatureCount { .. })
        ));
        // Zero depth.
        let mut c = small_config();
        c.depth = 0;
        assert!(matches!(
            NodeObliviousLayer::new(c),
            Err(TabularError::InvalidTreeDepth { .. })
        ));
        // Zero trees.
        let mut c = small_config();
        c.num_trees = 0;
        assert!(matches!(
            NodeObliviousLayer::new(c),
            Err(TabularError::InvalidTreeCount { .. })
        ));
        // Bad alpha.
        let mut c = small_config();
        c.entmax_alpha = 0.5;
        assert!(matches!(
            NodeObliviousLayer::new(c),
            Err(TabularError::InvalidParameter { .. })
        ));
        // Zero response dim.
        let mut c = small_config();
        c.response_dim = 0;
        assert!(matches!(
            NodeObliviousLayer::new(c),
            Err(TabularError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn forward_dimension_mismatch_errors() {
        let cfg = small_config();
        let layer = NodeObliviousLayer::new(cfg).expect("layer should build");
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

    #[test]
    fn empty_entmax_inputs_error() {
        assert!(matches!(sparsemax_f64(&[]), Err(TabularError::EmptyInput)));
        assert!(matches!(
            entmax_alpha_f64(&[], 1.5),
            Err(TabularError::EmptyInput)
        ));
        assert!(matches!(
            entmax_alpha_f64(&[1.0, 2.0], 2.5),
            Err(TabularError::InvalidParameter { .. })
        ));
    }
}
