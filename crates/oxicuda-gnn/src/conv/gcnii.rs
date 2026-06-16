//! GCNII — "Simple and Deep Graph Convolutional Networks" (Chen et al., ICML 2020).
//!
//! GCNII augments the vanilla GCN propagation with two ingredients that together
//! defeat over-smoothing and let the network go *deep* (32–64 layers):
//!
//! 1. **Initial residual** — every layer mixes a fixed fraction `α` of the
//!    original (layer-0) representation `H^{(0)}` back into the smoothed signal,
//!    so the output can never drift arbitrarily far from the input.
//! 2. **Identity mapping** — the learnable transform is applied as
//!    `(1 − β_l) I + β_l W^{(l)}`, a residual around the identity whose strength
//!    `β_l = log(θ / l + 1)` *decays* with depth `l`. Deep layers therefore act
//!    almost as the identity on the feature transform, preserving the rank of the
//!    representation.
//!
//! A single GCNII layer computes
//!
//! ```text
//!   H^{(l+1)} = σ( ( (1 − α) · P̃ · H^{(l)} + α · H^{(0)} )
//!                  · ( (1 − β_l) · I + β_l · W^{(l)} ) )
//! ```
//!
//! where `P̃ = D̃^{−1/2} (A + I) D̃^{−1/2}` is the symmetric normalised adjacency
//! **with self-loops** (`D̃` is the degree matrix of `A + I`) and `σ` is ReLU.
//!
//! The forward keeps `H^{(0)}` — the features after the initial input projection
//! — fixed across the whole stack and feeds it to every layer's residual term.
//! This module builds `P̃` once and reuses it for all layers.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a [`Gcnii`] stack.
#[derive(Debug, Clone, Copy)]
pub struct GcniiConfig {
    /// Feature dimension. GCNII keeps a constant hidden width across the stack,
    /// so input, hidden, and output dimensions are all `dim`.
    pub dim: usize,
    /// Number of stacked GCNII layers `L` (`≥ 1`).
    pub num_layers: usize,
    /// Initial-residual strength `α ∈ [0, 1]`.
    ///
    /// `α = 0` recovers a pure smoothed-propagation layer; `α = 1` drops the
    /// propagation term and returns the (transformed) initial features.
    pub alpha: f32,
    /// Identity-mapping hyper-parameter `θ ≥ 0` controlling the decay
    /// `β_l = log(θ / l + 1)`. Larger `θ` keeps the learnable transform stronger
    /// for longer. `θ = 0` makes every `β_l = 0` (pure identity transform).
    pub theta: f32,
}

// ─── Symmetric normalised adjacency with self-loops ───────────────────────────

/// Dense symmetric normalised adjacency `P̃ = D̃^{−1/2} (A + I) D̃^{−1/2}`.
///
/// The supplied CSR graph is treated as **undirected**: an edge `(i, j)` and its
/// transpose `(j, i)` are merged (multiplicities collapsed) before normalisation,
/// guaranteeing a symmetric operator regardless of how the edges were stored.
/// Self-loops are added with weight `1` to every node, so the diagonal of `P̃`
/// is strictly positive.
struct NormAdj {
    n: usize,
    /// Dense row-major `[n × n]` operator.
    p: Vec<f32>,
}

impl NormAdj {
    fn build(graph: &CsrGraph) -> GnnResult<Self> {
        let n = graph.n_nodes();
        // Boolean adjacency of A + I (undirected, self-loops included).
        let mut adj = vec![false; n * n];
        for i in 0..n {
            adj[i * n + i] = true; // self-loop
            for &j in graph.neighbors(i)? {
                adj[i * n + j] = true;
                adj[j * n + i] = true; // symmetrise
            }
        }
        // Degrees of A + I.
        let mut deg = vec![0.0_f32; n];
        for (i, d) in deg.iter_mut().enumerate() {
            let mut count = 0.0_f32;
            for j in 0..n {
                if adj[i * n + j] {
                    count += 1.0;
                }
            }
            *d = count;
        }
        let d_inv_sqrt: Vec<f32> = deg
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        let mut p = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                if adj[i * n + j] {
                    p[i * n + j] = d_inv_sqrt[i] * d_inv_sqrt[j];
                }
            }
        }
        Ok(Self { n, p })
    }

    /// `out = P̃ · X` for `X` of shape `[n × dim]` row-major.
    fn propagate(&self, x: &[f32], dim: usize) -> Vec<f32> {
        let n = self.n;
        let mut out = vec![0.0_f32; n * dim];
        for i in 0..n {
            for j in 0..n {
                let pij = self.p[i * n + j];
                if pij != 0.0 {
                    for k in 0..dim {
                        out[i * dim + k] += pij * x[j * dim + k];
                    }
                }
            }
        }
        out
    }

    /// Dense `[n × n]` operator (row-major), used by structural tests.
    #[inline]
    fn dense(&self) -> &[f32] {
        &self.p
    }
}

// ─── Per-layer decay coefficient ──────────────────────────────────────────────

/// Identity-mapping coefficient `β_l = log(θ / l + 1)` for layer index `l ≥ 1`.
///
/// `β_l` is monotonically decreasing in `l` for any `θ > 0`, and is `0` for all
/// `l` when `θ = 0`.
#[must_use]
pub fn gcnii_beta(theta: f32, layer: usize) -> f32 {
    let l = layer.max(1) as f32;
    (theta / l + 1.0).ln()
}

// ─── GCNII stack ──────────────────────────────────────────────────────────────

/// An `L`-layer GCNII model operating at a constant hidden width.
///
/// Per-layer weights are owned by the model and laid out row-major as
/// `[dim × dim]` each; there are `num_layers` of them.
pub struct Gcnii {
    config: GcniiConfig,
    /// Per-layer weight matrices, each `[dim × dim]` row-major. Length =
    /// `num_layers`.
    weights: Vec<Vec<f32>>,
}

impl Gcnii {
    /// Build a GCNII stack from explicit per-layer weights.
    ///
    /// `weights[l]` must be a `[dim × dim]` row-major matrix; `weights.len()`
    /// must equal `num_layers`.
    ///
    /// # Errors
    ///
    /// - [`GnnError::InvalidLayerConfig`] for a zero `dim`/`num_layers`, an
    ///   out-of-range `alpha`, or a non-finite `theta`.
    /// - [`GnnError::WeightShapeMismatch`] if any weight matrix is mis-shaped.
    /// - [`GnnError::DimensionMismatch`] if the number of weight matrices does
    ///   not match `num_layers`.
    pub fn new(config: GcniiConfig, weights: Vec<Vec<f32>>) -> GnnResult<Self> {
        if config.dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "GCNII: dim must be > 0".to_string(),
            ));
        }
        if config.num_layers == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "GCNII: num_layers must be >= 1".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&config.alpha) {
            return Err(GnnError::InvalidLayerConfig(format!(
                "GCNII: alpha must be in [0, 1], got {}",
                config.alpha
            )));
        }
        if !config.theta.is_finite() || config.theta < 0.0 {
            return Err(GnnError::InvalidLayerConfig(format!(
                "GCNII: theta must be finite and >= 0, got {}",
                config.theta
            )));
        }
        if weights.len() != config.num_layers {
            return Err(GnnError::DimensionMismatch {
                expected: config.num_layers,
                got: weights.len(),
            });
        }
        for w in &weights {
            if w.len() != config.dim * config.dim {
                return Err(GnnError::WeightShapeMismatch {
                    r: config.dim,
                    c: config.dim,
                    d: config.dim,
                });
            }
        }
        Ok(Self { config, weights })
    }

    /// Build a GCNII stack with identity weight matrices on every layer.
    ///
    /// Convenience constructor for analysis and tests: with `W^{(l)} = I` the
    /// identity-mapping term `(1 − β_l) I + β_l W^{(l)} = I` for every `β_l`, so
    /// the model reduces to the pure initial-residual + smoothed-propagation mix.
    ///
    /// # Errors
    ///
    /// Same as [`Gcnii::new`].
    pub fn with_identity_weights(config: GcniiConfig) -> GnnResult<Self> {
        let dim = config.dim;
        let mut id = vec![0.0_f32; dim * dim];
        for i in 0..dim {
            id[i * dim + i] = 1.0;
        }
        let weights = vec![id; config.num_layers];
        Self::new(config, weights)
    }

    /// Feature dimension (constant across the stack).
    #[inline]
    pub fn dim(&self) -> usize {
        self.config.dim
    }

    /// Number of stacked layers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Decay coefficient `β_l` of layer `l` (1-indexed).
    #[inline]
    pub fn beta(&self, layer: usize) -> f32 {
        gcnii_beta(self.config.theta, layer)
    }

    /// Single GCNII layer applied to current state `h` with fixed initial state
    /// `h0`, using `P̃` and the `layer`-th (1-indexed) weight matrix.
    ///
    /// ```text
    ///   m   = (1 − α) · P̃ · h + α · h0
    ///   out = σ( m · ((1 − β_l) I + β_l W) )
    /// ```
    fn layer_forward(&self, norm_adj: &NormAdj, h: &[f32], h0: &[f32], layer: usize) -> Vec<f32> {
        let n = norm_adj.n;
        let dim = self.config.dim;
        let alpha = self.config.alpha;
        let beta = self.beta(layer);
        let w = &self.weights[layer - 1];

        // Propagation + initial residual:  m = (1-α)·P̃·h + α·h0
        let prop = norm_adj.propagate(h, dim);
        let mut m = vec![0.0_f32; n * dim];
        for idx in 0..n * dim {
            m[idx] = (1.0 - alpha) * prop[idx] + alpha * h0[idx];
        }

        // Identity-mapping transform:  out_row = m_row · ((1-β) I + β W)
        //   = (1-β)·m_row + β·(m_row · W)
        // with W row-major [dim × dim] interpreted as out[k] = Σ_j m[j]·W[j,k].
        let mut out = vec![0.0_f32; n * dim];
        for i in 0..n {
            let m_row = &m[i * dim..(i + 1) * dim];
            let out_row = &mut out[i * dim..(i + 1) * dim];
            for k in 0..dim {
                let mut wk = 0.0_f32;
                for (j, &mj) in m_row.iter().enumerate() {
                    wk += mj * w[j * dim + k];
                }
                let val = (1.0 - beta) * m_row[k] + beta * wk;
                out_row[k] = val.max(0.0); // ReLU
            }
        }
        out
    }

    /// Run the full `L`-layer GCNII forward.
    ///
    /// `h0` are the *already projected* initial features (`[n_nodes × dim]`
    /// row-major); they form the initial state `H^{(0)}` and feed every layer's
    /// residual term. The same `h0` is returned to the caller via the residual
    /// path; this function does not perform the input/output linear projections.
    ///
    /// # Errors
    ///
    /// - [`GnnError::NodeFeatureMismatch`] if `h0.len() != n_nodes * dim`.
    /// - [`GnnError::NonFiniteOutput`] if the result contains non-finite values.
    pub fn forward(&self, graph: &CsrGraph, h0: &[f32]) -> GnnResult<Vec<f32>> {
        let n = graph.n_nodes();
        let dim = self.config.dim;
        if h0.len() != n * dim {
            return Err(GnnError::NodeFeatureMismatch(n, h0.len() / dim.max(1)));
        }
        let norm_adj = NormAdj::build(graph)?;
        let mut h = h0.to_vec();
        for layer in 1..=self.config.num_layers {
            h = self.layer_forward(&norm_adj, &h, h0, layer);
        }
        if h.iter().any(|v| !v.is_finite()) {
            return Err(GnnError::NonFiniteOutput("GCNII forward"));
        }
        Ok(h)
    }

    /// Expose the dense symmetric normalised adjacency `P̃` (`[n × n]`,
    /// row-major) for the supplied graph. Primarily for inspection and tests.
    ///
    /// # Errors
    ///
    /// Propagates any error from neighbour lookup.
    pub fn normalized_adjacency_dense(&self, graph: &CsrGraph) -> GnnResult<Vec<f32>> {
        Ok(NormAdj::build(graph)?.dense().to_vec())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph(n: usize) -> CsrGraph {
        // Undirected path 0-1-2-...-(n-1), both directions stored.
        let mut edges = Vec::new();
        for i in 0..n - 1 {
            edges.push((i, i + 1));
            edges.push((i + 1, i));
        }
        CsrGraph::from_edges(n, &edges).expect("path graph")
    }

    fn ring_graph(n: usize) -> CsrGraph {
        let mut edges = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            edges.push((i, j));
            edges.push((j, i));
        }
        CsrGraph::from_edges(n, &edges).expect("ring graph")
    }

    fn random_feats(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut r = crate::handle::LcgRng::new(seed);
        (0..n * dim).map(|_| r.next_f32() * 2.0 - 1.0).collect()
    }

    fn identity(dim: usize) -> Vec<f32> {
        let mut w = vec![0.0_f32; dim * dim];
        for i in 0..dim {
            w[i * dim + i] = 1.0;
        }
        w
    }

    // (a) β_l = log(θ/l + 1) DECREASES with layer index l.
    #[test]
    fn beta_decreases_with_depth() {
        let theta = 1.0_f32;
        let mut prev = f32::INFINITY;
        for l in 1..=32 {
            let b = gcnii_beta(theta, l);
            assert!(b > 0.0, "beta should be positive for theta>0");
            assert!(b < prev, "beta_{l}={b} should be < beta_{}={prev}", l - 1);
            prev = b;
        }
        // Closed-form check: beta_1 = ln(2).
        assert!((gcnii_beta(1.0, 1) - std::f32::consts::LN_2).abs() < 1e-6);
    }

    // (b) IDENTITY-MAPPING limit: with β_l = 0 the linear map ≈ I, so the weight
    //     has no effect — output equals the identity-weight output regardless of W.
    #[test]
    fn identity_mapping_limit_weight_has_no_effect() {
        let g = path_graph(6);
        let dim = 4;
        let h0 = random_feats(6, dim, 11);
        let cfg = GcniiConfig {
            dim,
            num_layers: 3,
            alpha: 0.1,
            theta: 0.0, // β_l = 0 for all l
        };
        // Arbitrary non-identity weights.
        let weights: Vec<Vec<f32>> = (0..3).map(|s| random_feats(dim, dim, 100 + s)).collect();
        let model_w = Gcnii::new(cfg, weights).expect("model");
        let model_id = Gcnii::with_identity_weights(cfg).expect("model id");
        let out_w = model_w.forward(&g, &h0).expect("fwd w");
        let out_id = model_id.forward(&g, &h0).expect("fwd id");
        for (a, b) in out_w.iter().zip(out_id.iter()) {
            assert!((a - b).abs() < 1e-5, "β=0 must ignore W: {a} vs {b}");
        }
    }

    // (c) INITIAL-RESIDUAL: with α=1 the propagation term drops; the output
    //     depends only on H^{(0)} (and the per-layer transforms), never on the
    //     graph structure. Two different graphs give the same output.
    #[test]
    fn initial_residual_alpha_one_ignores_propagation() {
        let dim = 3;
        let n = 5;
        let h0 = random_feats(n, dim, 7);
        let cfg = GcniiConfig {
            dim,
            num_layers: 4,
            alpha: 1.0,
            theta: 1.0,
        };
        // Identity weights so the transform is exactly identity (β does not
        // matter when W = I), isolating the propagation-vs-residual choice.
        let g_path = path_graph(n);
        let g_ring = ring_graph(n);
        let m_path = Gcnii::with_identity_weights(cfg).expect("m");
        let m_ring = Gcnii::with_identity_weights(cfg).expect("m");
        let out_path = m_path.forward(&g_path, &h0).expect("fwd");
        let out_ring = m_ring.forward(&g_ring, &h0).expect("fwd");
        // α=1 ⇒ m = h0 every layer ⇒ output = ReLU(h0) on both graphs.
        for (a, b) in out_path.iter().zip(out_ring.iter()) {
            assert!((a - b).abs() < 1e-6, "α=1 must ignore graph: {a} vs {b}");
        }
        // And it equals ReLU(h0).
        let relu_h0: Vec<f32> = h0.iter().map(|&v| v.max(0.0)).collect();
        for (a, b) in out_path.iter().zip(relu_h0.iter()) {
            assert!((a - b).abs() < 1e-6, "α=1 output should be ReLU(h0)");
        }
    }

    // (d) ANTI-OVERSMOOTHING: stack L=32 layers; GCNII per-node feature variance
    //     stays bounded away from zero and is STRICTLY GREATER than a vanilla
    //     repeated-P̃ propagation baseline (which collapses toward a constant).
    //
    //     Strictly-positive features are used so the layer's ReLU is a no-op and
    //     the comparison isolates the smoothing-vs-residual dynamics: the vanilla
    //     repeated symmetric propagation on a regular ring drives every channel
    //     toward its (constant) node-mean, collapsing the per-node variance, while
    //     GCNII's initial-residual keeps re-injecting the original signal.
    #[test]
    fn anti_oversmoothing_beats_vanilla_propagation() {
        let n = 20;
        let dim = 8;
        let depth = 32;
        let g = ring_graph(n);
        // Positive features in [0.1, 2.1] ⇒ ReLU never clips.
        let h0: Vec<f32> = {
            let mut r = crate::handle::LcgRng::new(2024);
            (0..n * dim).map(|_| r.next_f32() * 2.0 + 0.1).collect()
        };

        // GCNII with identity weights (isolates the residual + identity-map
        // structure; α keeps the initial signal alive).
        let cfg = GcniiConfig {
            dim,
            num_layers: depth,
            alpha: 0.2,
            theta: 1.0,
        };
        let model = Gcnii::with_identity_weights(cfg).expect("model");
        let out = model.forward(&g, &h0).expect("fwd");

        // Vanilla baseline: repeated symmetric propagation H ← P̃ · H, `depth`
        // times (no residual), the textbook over-smoothing dynamic.
        let norm_adj = NormAdj::build(&g).expect("norm");
        let mut hv = h0.clone();
        for _ in 0..depth {
            hv = norm_adj.propagate(&hv, dim);
        }

        // Mean per-node feature variance across the node dimension, averaged
        // over feature channels.
        let feature_variance = |h: &[f32]| -> f32 {
            let mut total = 0.0_f32;
            for k in 0..dim {
                let mut mean = 0.0_f32;
                for i in 0..n {
                    mean += h[i * dim + k];
                }
                mean /= n as f32;
                let mut var = 0.0_f32;
                for i in 0..n {
                    let d = h[i * dim + k] - mean;
                    var += d * d;
                }
                total += var / n as f32;
            }
            total / dim as f32
        };

        let var_gcnii = feature_variance(&out);
        let var_vanilla = feature_variance(&hv);

        // GCNII keeps the per-node variance bounded away from zero...
        assert!(var_gcnii > 1e-3, "GCNII variance collapsed: {var_gcnii}");
        // ...and it is strictly (and substantially) larger than the vanilla
        // repeated-propagation baseline, which is collapsing toward a constant.
        assert!(
            var_gcnii > 2.0 * var_vanilla,
            "GCNII variance {var_gcnii} must exceed vanilla {var_vanilla}"
        );
    }

    // (e) output shapes correct.
    #[test]
    fn output_shape_correct() {
        let g = path_graph(7);
        let dim = 5;
        let cfg = GcniiConfig {
            dim,
            num_layers: 3,
            alpha: 0.1,
            theta: 0.5,
        };
        let model = Gcnii::with_identity_weights(cfg).expect("model");
        let h0 = random_feats(7, dim, 3);
        let out = model.forward(&g, &h0).expect("fwd");
        assert_eq!(out.len(), 7 * dim);
        assert!(out.iter().all(|v| v.is_finite()));
        assert_eq!(model.dim(), dim);
        assert_eq!(model.num_layers(), 3);
    }

    // (f) P̃ structure: symmetric, self-loops (diagonal strictly > 0).
    #[test]
    fn normalized_adjacency_symmetric_with_self_loops() {
        let g = ring_graph(6);
        let cfg = GcniiConfig {
            dim: 2,
            num_layers: 1,
            alpha: 0.1,
            theta: 1.0,
        };
        let model = Gcnii::with_identity_weights(cfg).expect("model");
        let p = model.normalized_adjacency_dense(&g).expect("p");
        let n = g.n_nodes();
        // Diagonal strictly positive (self-loops present).
        for i in 0..n {
            assert!(p[i * n + i] > 0.0, "diag[{i}]={} not >0", p[i * n + i]);
        }
        // Symmetric.
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (p[i * n + j] - p[j * n + i]).abs() < 1e-6,
                    "asymmetry at ({i},{j})"
                );
            }
        }
        // Known value: ring node has degree 2 in A, plus self-loop ⇒ D̃=3.
        // Diagonal = 1/sqrt(3)*1/sqrt(3) = 1/3.
        for i in 0..n {
            assert!((p[i * n + i] - 1.0 / 3.0).abs() < 1e-5);
        }
        // Off-diagonal neighbour entry = 1/sqrt(3)*1/sqrt(3) = 1/3 too.
        // Row 0, column 1 (node 0 ↔ node 1 are adjacent on the ring).
        assert!((p[1] - 1.0 / 3.0).abs() < 1e-5);
    }

    // Extra: with α=0 and identity weights, a single layer equals one symmetric
    // propagation step (pure GCN smoothing), tying P̃ to the documented formula.
    #[test]
    fn alpha_zero_identity_equals_single_propagation() {
        let g = path_graph(5);
        let dim = 3;
        let h0 = random_feats(5, dim, 55);
        let cfg = GcniiConfig {
            dim,
            num_layers: 1,
            alpha: 0.0,
            theta: 0.0, // β=0 ⇒ identity transform
        };
        let model = Gcnii::with_identity_weights(cfg).expect("model");
        let out = model.forward(&g, &h0).expect("fwd");
        let norm_adj = NormAdj::build(&g).expect("norm");
        let prop = norm_adj.propagate(&h0, dim);
        let relu_prop: Vec<f32> = prop.iter().map(|&v| v.max(0.0)).collect();
        for (a, b) in out.iter().zip(relu_prop.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    // Validation errors.
    #[test]
    fn rejects_bad_config_and_weights() {
        let dim = 3;
        let ok_w = vec![identity(dim)];
        assert!(
            Gcnii::new(
                GcniiConfig {
                    dim: 0,
                    num_layers: 1,
                    alpha: 0.1,
                    theta: 1.0
                },
                vec![]
            )
            .is_err()
        );
        assert!(
            Gcnii::new(
                GcniiConfig {
                    dim,
                    num_layers: 1,
                    alpha: 2.0,
                    theta: 1.0
                },
                ok_w.clone()
            )
            .is_err()
        );
        assert!(
            Gcnii::new(
                GcniiConfig {
                    dim,
                    num_layers: 2,
                    alpha: 0.1,
                    theta: 1.0
                },
                ok_w.clone()
            )
            .is_err() // wrong number of weight matrices
        );
        // wrong weight shape
        assert!(
            Gcnii::new(
                GcniiConfig {
                    dim,
                    num_layers: 1,
                    alpha: 0.1,
                    theta: 1.0
                },
                vec![vec![0.0_f32; dim * dim + 1]]
            )
            .is_err()
        );
        // feature mismatch at forward
        let g = path_graph(4);
        let model = Gcnii::with_identity_weights(GcniiConfig {
            dim,
            num_layers: 1,
            alpha: 0.1,
            theta: 1.0,
        })
        .expect("model");
        assert!(model.forward(&g, &vec![0.0_f32; 4 * dim + 2]).is_err());
    }
}
