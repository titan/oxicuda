//! Point Transformer — vector self-attention over kNN neighbourhoods.
//!
//! Reference: Zhao, Jiang, Jia, Torr & Koltun, *"Point Transformer"*
//! (ICCV 2021).
//!
//! Unlike the scalar dot-product attention used by language / image
//! transformers, the Point Transformer layer uses **vector attention** with a
//! *subtraction relation*. For a query point `x_i` and a neighbour `x_j` drawn
//! from `x_i`'s k-nearest-neighbour set `N(i)`, the per-channel attention vector
//! is
//!
//! ```text
//! γ( φ(x_i) − ψ(x_j) + δ_ij )          (a vector in R^d, not a scalar)
//! ```
//!
//! where `φ`, `ψ` are linear query / key projections, `δ_ij = θ(p_i − p_j)` is a
//! learned **position encoding** of the *relative* coordinate offset, and `γ` is
//! a small MLP. The weights are normalised with a **per-channel softmax over the
//! neighbourhood** and aggregate the value vectors:
//!
//! ```text
//! y_i = Σ_{j∈N(i)} ρ( γ( φ(x_i) − ψ(x_j) + δ_ij ) ) ⊙ ( α(x_j) + δ_ij )
//! ```
//!
//! with `ρ = softmax_j`, `α` the value projection and `⊙` the Hadamard product.
//! A final linear projection maps `y_i` to the output dimension.
//!
//! ## Tensor layout
//! - `points`:   flat `[n_points × 3]` row-major XYZ coordinates.
//! - `features`: flat `[n_points × in_dim]` row-major.
//! - outputs:    flat `[n_points × out_dim]` row-major.
//!
//! ## Key properties (exercised by the tests)
//! - **Permutation equivariance**: permuting the input points permutes the
//!   outputs identically — the defining property of a point-cloud network.
//! - **Translation invariance of the relations**: because `δ` only sees the
//!   *relative* offset `p_i − p_j`, translating every point by a constant leaves
//!   the attention weights unchanged.
//! - **Per-channel softmax**: the vector-attention weights are non-negative and
//!   sum to 1 over the neighbourhood, independently for every channel.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
    vit::vit_block::linear,
};

/// Spatial coordinate dimension (XYZ).
const COORD_DIM: usize = 3;

// ─── Linear ────────────────────────────────────────────────────────────────────

/// A dense linear projection `y = x W^T + b` with `[n_out × n_in]` weights.
///
/// Reuses the crate-wide [`crate::vit::vit_block::linear`] kernel so the matmul
/// is not duplicated.
#[derive(Debug, Clone)]
struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    n_in: usize,
    n_out: usize,
}

impl Linear {
    /// Random init with `N(0, scale)` weights and zero bias.
    fn new(n_in: usize, n_out: usize, scale: f32, rng: &mut LcgRng) -> Self {
        let mut weight = vec![0.0f32; n_in * n_out];
        rng.fill_normal(&mut weight);
        for w in &mut weight {
            *w *= scale;
        }
        Self {
            weight,
            bias: vec![0.0f32; n_out],
            n_in,
            n_out,
        }
    }

    /// Apply to a single `[n_in]` vector → `[n_out]`.
    #[inline]
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        linear(x, &self.weight, &self.bias, self.n_in, self.n_out)
    }
}

// ─── Mlp (2-layer, ReLU) ────────────────────────────────────────────────────────

/// Two-layer perceptron `Linear → ReLU → Linear` used for both the position
/// encoder `θ` (`δ`) and the attention mapping `γ`.
#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(n_in: usize, hidden: usize, n_out: usize, rng: &mut LcgRng) -> Self {
        // Kaiming-style scale for the ReLU non-linearity.
        let s1 = (2.0 / n_in as f32).sqrt();
        let s2 = (2.0 / hidden as f32).sqrt();
        Self {
            fc1: Linear::new(n_in, hidden, s1, rng),
            fc2: Linear::new(hidden, n_out, s2, rng),
        }
    }

    /// Apply to a single vector.
    #[inline]
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut h = self.fc1.apply(x);
        for v in &mut h {
            *v = v.max(0.0); // ReLU
        }
        self.fc2.apply(&h)
    }
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a [`PointTransformerLayer`].
#[derive(Debug, Clone, PartialEq)]
pub struct PointTransformerConfig {
    /// Input feature dimension `d_in`.
    pub in_dim: usize,
    /// Attention (query / key / value) dimension `d`.
    pub dim: usize,
    /// Output feature dimension `d_out`.
    pub out_dim: usize,
    /// Hidden width of the position-encoding MLP `θ`.
    pub pos_hidden: usize,
    /// Hidden width of the attention MLP `γ`.
    pub attn_hidden: usize,
    /// Number of neighbours `k` (includes the point itself).
    pub k: usize,
}

impl PointTransformerConfig {
    /// Create and validate a configuration.
    ///
    /// # Errors
    /// - [`VisionError::InvalidEmbedDim`] if `in_dim`, `dim` or `out_dim` is 0.
    /// - [`VisionError::EmptyInput`] if `k == 0`, `pos_hidden == 0` or
    ///   `attn_hidden == 0`.
    pub fn new(
        in_dim: usize,
        dim: usize,
        out_dim: usize,
        pos_hidden: usize,
        attn_hidden: usize,
        k: usize,
    ) -> VisionResult<Self> {
        if in_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(in_dim));
        }
        if dim == 0 {
            return Err(VisionError::InvalidEmbedDim(dim));
        }
        if out_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(out_dim));
        }
        if k == 0 {
            return Err(VisionError::EmptyInput("point transformer k"));
        }
        if pos_hidden == 0 {
            return Err(VisionError::EmptyInput("point transformer pos_hidden"));
        }
        if attn_hidden == 0 {
            return Err(VisionError::EmptyInput("point transformer attn_hidden"));
        }
        Ok(Self {
            in_dim,
            dim,
            out_dim,
            pos_hidden,
            attn_hidden,
            k,
        })
    }

    /// A tiny configuration for unit tests:
    /// `in_dim=8, dim=8, out_dim=8, pos_hidden=8, attn_hidden=8, k=4`.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            in_dim: 8,
            dim: 8,
            out_dim: 8,
            pos_hidden: 8,
            attn_hidden: 8,
            k: 4,
        }
    }
}

// ─── Detailed output ──────────────────────────────────────────────────────────

/// Detailed per-point attention output, exposing the neighbourhood and the
/// per-channel vector-attention weights (useful for inspection / tests).
#[derive(Debug, Clone)]
pub struct PointAttention {
    /// Aggregated output features: flat `[n_points × out_dim]`.
    pub features: Vec<f32>,
    /// Neighbour indices: flat `[n_points × k]`, sorted nearest-first.
    pub neighbors: Vec<usize>,
    /// Per-channel softmax weights: flat `[n_points × k × dim]`.
    ///
    /// For point `i`, neighbour slot `s`, channel `c`:
    /// `weights[(i * k + s) * dim + c]`.
    pub weights: Vec<f32>,
    /// Number of points.
    pub n_points: usize,
    /// Neighbours per point.
    pub k: usize,
    /// Attention dimension `d`.
    pub dim: usize,
}

// ─── k-nearest-neighbours ──────────────────────────────────────────────────────

/// Indices of the `k` nearest points to point `i` (including `i` itself),
/// sorted by ascending squared Euclidean distance with the point index as a
/// deterministic tie-break.
///
/// `points` is flat `[n × 3]`.
fn knn(points: &[f32], n: usize, i: usize, k: usize) -> Vec<usize> {
    let pi = &points[i * COORD_DIM..i * COORD_DIM + COORD_DIM];
    let mut dists: Vec<(f32, usize)> = (0..n)
        .map(|j| {
            let pj = &points[j * COORD_DIM..j * COORD_DIM + COORD_DIM];
            let mut d = 0.0f32;
            for c in 0..COORD_DIM {
                let diff = pi[c] - pj[c];
                d += diff * diff;
            }
            (d, j)
        })
        .collect();
    // Ascending by distance, ties broken by lower index → fully deterministic.
    dists.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    let kk = k.min(n);
    dists.into_iter().take(kk).map(|(_, j)| j).collect()
}

// ─── PointTransformerLayer ──────────────────────────────────────────────────────

/// A single Point Transformer vector-attention layer.
pub struct PointTransformerLayer {
    cfg: PointTransformerConfig,
    /// Query projection `φ`: `in_dim → dim`.
    phi: Linear,
    /// Key projection `ψ`: `in_dim → dim`.
    psi: Linear,
    /// Value projection `α`: `in_dim → dim`.
    alpha: Linear,
    /// Position-encoding MLP `θ`: `3 → dim` (`δ`).
    theta: Mlp,
    /// Attention MLP `γ`: `dim → dim`.
    gamma: Mlp,
    /// Output projection: `dim → out_dim`.
    out_proj: Linear,
}

impl PointTransformerLayer {
    /// Construct a layer with randomly-initialised weights.
    pub fn new(cfg: PointTransformerConfig, rng: &mut LcgRng) -> Self {
        let proj_scale = 1.0 / (cfg.in_dim as f32).sqrt();
        let phi = Linear::new(cfg.in_dim, cfg.dim, proj_scale, rng);
        let psi = Linear::new(cfg.in_dim, cfg.dim, proj_scale, rng);
        let alpha = Linear::new(cfg.in_dim, cfg.dim, proj_scale, rng);
        let theta = Mlp::new(COORD_DIM, cfg.pos_hidden, cfg.dim, rng);
        let gamma = Mlp::new(cfg.dim, cfg.attn_hidden, cfg.dim, rng);
        let out_proj = Linear::new(cfg.dim, cfg.out_dim, 1.0 / (cfg.dim as f32).sqrt(), rng);
        Self {
            cfg,
            phi,
            psi,
            alpha,
            theta,
            gamma,
            out_proj,
        }
    }

    /// Read-only configuration access.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &PointTransformerConfig {
        &self.cfg
    }

    /// Forward pass returning only the output features `[n_points × out_dim]`.
    ///
    /// # Errors
    /// See [`PointTransformerLayer::forward_detailed`].
    pub fn forward(
        &self,
        points: &[f32],
        features: &[f32],
        n_points: usize,
    ) -> VisionResult<Vec<f32>> {
        Ok(self.compute(points, features, n_points, true)?.features)
    }

    /// Forward pass exposing neighbourhoods and per-channel attention weights.
    ///
    /// # Errors
    /// - [`VisionError::EmptyInput`] if `n_points == 0`.
    /// - [`VisionError::DimensionMismatch`] if `points.len() != n_points * 3` or
    ///   `features.len() != n_points * in_dim`.
    /// - [`VisionError::NonFinite`] if any output is non-finite.
    pub fn forward_detailed(
        &self,
        points: &[f32],
        features: &[f32],
        n_points: usize,
    ) -> VisionResult<PointAttention> {
        self.compute(points, features, n_points, true)
    }

    /// Forward pass with the position encoding `δ` forced to zero — used to
    /// verify that `δ` genuinely influences the output.
    ///
    /// # Errors
    /// Same as [`PointTransformerLayer::forward_detailed`].
    pub fn forward_zero_position(
        &self,
        points: &[f32],
        features: &[f32],
        n_points: usize,
    ) -> VisionResult<PointAttention> {
        self.compute(points, features, n_points, false)
    }

    /// Core computation. When `use_delta` is false the position encoding `δ` is
    /// dropped from both the attention relation and the value aggregation.
    fn compute(
        &self,
        points: &[f32],
        features: &[f32],
        n_points: usize,
        use_delta: bool,
    ) -> VisionResult<PointAttention> {
        if n_points == 0 {
            return Err(VisionError::EmptyInput("point transformer points"));
        }
        if points.len() != n_points * COORD_DIM {
            return Err(VisionError::DimensionMismatch {
                expected: n_points * COORD_DIM,
                got: points.len(),
            });
        }
        if features.len() != n_points * self.cfg.in_dim {
            return Err(VisionError::DimensionMismatch {
                expected: n_points * self.cfg.in_dim,
                got: features.len(),
            });
        }

        let d = self.cfg.dim;
        let din = self.cfg.in_dim;
        let k = self.cfg.k.min(n_points);

        // Pre-compute φ / ψ / α for all points (each independent of neighbours).
        let mut phi_all = vec![0.0f32; n_points * d];
        let mut psi_all = vec![0.0f32; n_points * d];
        let mut alpha_all = vec![0.0f32; n_points * d];
        for p in 0..n_points {
            let xf = &features[p * din..(p + 1) * din];
            phi_all[p * d..(p + 1) * d].copy_from_slice(&self.phi.apply(xf));
            psi_all[p * d..(p + 1) * d].copy_from_slice(&self.psi.apply(xf));
            alpha_all[p * d..(p + 1) * d].copy_from_slice(&self.alpha.apply(xf));
        }

        let mut out_features = vec![0.0f32; n_points * self.cfg.out_dim];
        let mut all_neighbors = vec![0usize; n_points * k];
        let mut all_weights = vec![0.0f32; n_points * k * d];

        for i in 0..n_points {
            let neighbors = knn(points, n_points, i, self.cfg.k);
            debug_assert_eq!(neighbors.len(), k);
            all_neighbors[i * k..(i + 1) * k].copy_from_slice(&neighbors);

            let phi_i = &phi_all[i * d..(i + 1) * d];
            let pi = &points[i * COORD_DIM..i * COORD_DIM + COORD_DIM];

            // Per-neighbour: position encoding δ, attention logits γ(relation),
            // and the value vector (α(x_j) + δ).
            let mut deltas = vec![0.0f32; k * d];
            let mut logits = vec![0.0f32; k * d];
            let mut values = vec![0.0f32; k * d];

            for (s, &j) in neighbors.iter().enumerate() {
                // δ_ij = θ(p_i − p_j)  (relative offset only).
                let pj = &points[j * COORD_DIM..j * COORD_DIM + COORD_DIM];
                let rel = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]];
                let delta = if use_delta {
                    self.theta.apply(&rel)
                } else {
                    vec![0.0f32; d]
                };

                // relation = φ(x_i) − ψ(x_j) + δ_ij
                let psi_j = &psi_all[j * d..(j + 1) * d];
                let alpha_j = &alpha_all[j * d..(j + 1) * d];
                let mut relation = vec![0.0f32; d];
                for c in 0..d {
                    relation[c] = phi_i[c] - psi_j[c] + delta[c];
                }
                let g = self.gamma.apply(&relation);

                let row = s * d;
                for c in 0..d {
                    logits[row + c] = g[c];
                    values[row + c] = alpha_j[c] + delta[c];
                    deltas[row + c] = delta[c];
                }
            }
            let _ = &deltas; // retained for clarity of the value construction above

            // Per-channel softmax over the k neighbours.
            softmax_over_neighbors(&mut logits, k, d);
            all_weights[i * k * d..(i + 1) * k * d].copy_from_slice(&logits);

            // Aggregate: y_i[c] = Σ_s weight[s, c] · value[s, c].
            let mut y_i = vec![0.0f32; d];
            for s in 0..k {
                let row = s * d;
                for c in 0..d {
                    y_i[c] += logits[row + c] * values[row + c];
                }
            }

            let proj = self.out_proj.apply(&y_i);
            out_features[i * self.cfg.out_dim..(i + 1) * self.cfg.out_dim].copy_from_slice(&proj);
        }

        if out_features.iter().any(|v| !v.is_finite()) {
            return Err(VisionError::NonFinite("point transformer output"));
        }

        Ok(PointAttention {
            features: out_features,
            neighbors: all_neighbors,
            weights: all_weights,
            n_points,
            k,
            dim: d,
        })
    }
}

/// In-place per-channel softmax over the neighbour axis.
///
/// `logits` is `[k × d]` row-major. For each channel `c`, the `k` neighbour
/// values are normalised with a numerically-stable softmax (max subtraction).
fn softmax_over_neighbors(logits: &mut [f32], k: usize, d: usize) {
    for c in 0..d {
        // Stable max over neighbours for this channel.
        let mut mx = f32::NEG_INFINITY;
        for s in 0..k {
            mx = mx.max(logits[s * d + c]);
        }
        let mut sum = 0.0f32;
        for s in 0..k {
            let e = (logits[s * d + c] - mx).exp();
            logits[s * d + c] = e;
            sum += e;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
        for s in 0..k {
            logits[s * d + c] *= inv;
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random point cloud with distinct coordinates.
    fn make_cloud(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = LcgRng::new(seed);
        let mut points = vec![0.0f32; n * COORD_DIM];
        // Spread points out so kNN has no distance ties.
        for (idx, p) in points.iter_mut().enumerate() {
            *p = rng.next_f32() * 10.0 + idx as f32 * 0.01;
        }
        let mut feats = vec![0.0f32; n * 8];
        rng.fill_normal(&mut feats);
        (points, feats)
    }

    // ── Config ─────────────────────────────────────────────────────────────────

    #[test]
    fn config_tiny_valid() {
        let cfg = PointTransformerConfig::tiny();
        assert_eq!(cfg.dim, 8);
        assert_eq!(cfg.k, 4);
    }

    #[test]
    fn config_zero_dim_errors() {
        assert!(matches!(
            PointTransformerConfig::new(0, 8, 8, 8, 8, 4),
            Err(VisionError::InvalidEmbedDim(0))
        ));
        assert!(matches!(
            PointTransformerConfig::new(8, 8, 8, 8, 8, 0),
            Err(VisionError::EmptyInput(_))
        ));
    }

    // ── kNN ────────────────────────────────────────────────────────────────────

    #[test]
    fn knn_picks_genuine_nearest() {
        // Points on the X axis at 0,1,2,3,4. Nearest of point 0 (incl. self):
        // {0,1,2}; nearest of point 2: {2,1 or 3,...}. Hand-checked.
        let points = vec![
            0.0f32, 0.0, 0.0, // 0
            1.0, 0.0, 0.0, // 1
            2.0, 0.0, 0.0, // 2
            3.0, 0.0, 0.0, // 3
            4.0, 0.0, 0.0, // 4
        ];
        let nn0 = knn(&points, 5, 0, 3);
        assert_eq!(nn0, vec![0, 1, 2], "point 0 nearest set");

        let nn2 = knn(&points, 5, 2, 3);
        // distances from 2: self 0, then 1 and 3 both at distance 1 (tie),
        // index tie-break picks 1 before 3.
        assert_eq!(nn2[0], 2, "self is nearest");
        assert!(nn2.contains(&1) && nn2.contains(&3), "both unit neighbours");

        let nn4 = knn(&points, 5, 4, 2);
        assert_eq!(nn4, vec![4, 3], "point 4 nearest set");
    }

    #[test]
    fn knn_clamps_k_to_n() {
        let points = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let nn = knn(&points, 2, 0, 10);
        assert_eq!(nn.len(), 2, "k clamped to n_points");
    }

    // ── Shapes & finiteness ──────────────────────────────────────────────────

    #[test]
    fn forward_shapes_and_finite() {
        let n = 16;
        let (points, feats) = make_cloud(n, 1);
        let mut rng = LcgRng::new(2);
        let layer = PointTransformerLayer::new(PointTransformerConfig::tiny(), &mut rng);
        let out = layer.forward_detailed(&points, &feats, n).expect("ok");
        assert_eq!(out.features.len(), n * 8);
        assert_eq!(out.neighbors.len(), n * 4);
        assert_eq!(out.weights.len(), n * 4 * 8);
        assert!(out.features.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_wrong_feature_len_errors() {
        let n = 8;
        let (points, _) = make_cloud(n, 3);
        let mut rng = LcgRng::new(4);
        let layer = PointTransformerLayer::new(PointTransformerConfig::tiny(), &mut rng);
        let bad = vec![0.0f32; n * 4]; // in_dim is 8
        let r = layer.forward(&points, &bad, n);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    // ── Per-channel softmax weights ──────────────────────────────────────────

    #[test]
    fn attention_weights_nonneg_and_sum_to_one_per_channel() {
        let n = 12;
        let (points, feats) = make_cloud(n, 5);
        let mut rng = LcgRng::new(6);
        let layer = PointTransformerLayer::new(PointTransformerConfig::tiny(), &mut rng);
        let out = layer.forward_detailed(&points, &feats, n).expect("ok");
        let k = out.k;
        let d = out.dim;
        for i in 0..n {
            for c in 0..d {
                let mut sum = 0.0f32;
                for s in 0..k {
                    let w = out.weights[(i * k + s) * d + c];
                    assert!(w >= 0.0, "weight must be non-negative, got {w}");
                    sum += w;
                }
                assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "point {i} channel {c} weights sum {sum} != 1"
                );
            }
        }
    }

    // ── Permutation equivariance ─────────────────────────────────────────────

    #[test]
    fn permutation_equivariance() {
        let n = 16;
        let (points, feats) = make_cloud(n, 7);
        let mut rng = LcgRng::new(8);
        let layer = PointTransformerLayer::new(PointTransformerConfig::tiny(), &mut rng);
        let din = 8;
        let dout = 8;

        let base = layer.forward(&points, &feats, n).expect("ok");

        // A non-trivial permutation of the point indices.
        let mut perm: Vec<usize> = (0..n).collect();
        let mut prng = LcgRng::new(123);
        prng.shuffle(&mut perm);

        // Build permuted point cloud: row r of the new arrays = row perm[r] of old.
        let mut p_points = vec![0.0f32; n * COORD_DIM];
        let mut p_feats = vec![0.0f32; n * din];
        for (r, &src) in perm.iter().enumerate() {
            p_points[r * COORD_DIM..(r + 1) * COORD_DIM]
                .copy_from_slice(&points[src * COORD_DIM..(src + 1) * COORD_DIM]);
            p_feats[r * din..(r + 1) * din].copy_from_slice(&feats[src * din..(src + 1) * din]);
        }

        let permuted = layer.forward(&p_points, &p_feats, n).expect("ok");

        // out_permuted[r] must equal out_base[perm[r]].
        for (r, &src) in perm.iter().enumerate() {
            for c in 0..dout {
                let a = permuted[r * dout + c];
                let b = base[src * dout + c];
                assert!(
                    (a - b).abs() < 1e-4,
                    "equivariance broken at row {r} ch {c}: {a} vs {b}"
                );
            }
        }
    }

    // ── Position encoding δ matters ──────────────────────────────────────────

    #[test]
    fn position_encoding_changes_output() {
        let n = 14;
        let (points, feats) = make_cloud(n, 9);
        let mut rng = LcgRng::new(10);
        let layer = PointTransformerLayer::new(PointTransformerConfig::tiny(), &mut rng);
        let with_pos = layer.forward_detailed(&points, &feats, n).expect("ok");
        let no_pos = layer.forward_zero_position(&points, &feats, n).expect("ok");
        let diff: f32 = with_pos
            .features
            .iter()
            .zip(no_pos.features.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-3,
            "position encoding δ should change the output, diff={diff}"
        );
    }

    // ── Translation invariance of the attention weights ──────────────────────

    #[test]
    fn translation_leaves_relative_attention_unchanged() {
        let n = 16;
        let (points, feats) = make_cloud(n, 11);
        let mut rng = LcgRng::new(12);
        let layer = PointTransformerLayer::new(PointTransformerConfig::tiny(), &mut rng);

        let base = layer.forward_detailed(&points, &feats, n).expect("ok");

        // Translate every point by a constant offset.
        let mut shifted = points.clone();
        let offset = [3.5f32, -2.0, 7.25];
        for p in 0..n {
            for c in 0..COORD_DIM {
                shifted[p * COORD_DIM + c] += offset[c];
            }
        }
        let moved = layer.forward_detailed(&shifted, &feats, n).expect("ok");

        // Relative offsets (p_i − p_j) are unchanged → δ, γ and the softmax
        // weights must be identical, and so must the neighbourhoods.
        assert_eq!(
            base.neighbors, moved.neighbors,
            "kNN changed under translation"
        );
        for (a, b) in base.weights.iter().zip(moved.weights.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "attention weights changed under translation: {a} vs {b}"
            );
        }
        // Outputs are translation-invariant too (features unchanged, δ unchanged).
        for (a, b) in base.features.iter().zip(moved.features.iter()) {
            assert!((a - b).abs() < 1e-4, "output changed under translation");
        }
    }

    // ── Determinism ───────────────────────────────────────────────────────────

    #[test]
    fn deterministic_same_seed() {
        let n = 10;
        let (points, feats) = make_cloud(n, 13);
        let mut rng_a = LcgRng::new(55);
        let mut rng_b = LcgRng::new(55);
        let la = PointTransformerLayer::new(PointTransformerConfig::tiny(), &mut rng_a);
        let lb = PointTransformerLayer::new(PointTransformerConfig::tiny(), &mut rng_b);
        let oa = la.forward(&points, &feats, n).expect("ok");
        let ob = lb.forward(&points, &feats, n).expect("ok");
        assert_eq!(oa, ob, "same seed must produce identical output");
    }
}
