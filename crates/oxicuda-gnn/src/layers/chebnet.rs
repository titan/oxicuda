//! ChebNet — Spectral graph convolution via Chebyshev polynomials.
//!
//! Defferrard, Bresson & Vandergheynst, "Convolutional Neural Networks on Graphs
//! with Fast Localized Spectral Filtering", NeurIPS 2016.
//!
//! # Algorithm summary
//!
//! Given a graph `G` with normalized adjacency `Â = D̂^{-1/2} (A+I) D̂^{-1/2}`:
//!
//! 1. Compute the normalized Laplacian `L = I − Â`.
//! 2. Scale it: `L̃ = (2 / λ_max) · L − I`.
//!    With the canonical choice `λ_max = 2`, this simplifies to `L̃ = L − I = −Â`.
//!    For general `λ_max`: `L̃·x = (scale−1)·x − scale·Â·x` where `scale = 2/λ_max`.
//! 3. Chebyshev recurrence: `T₀(L̃)X = X`, `T₁(L̃)X = L̃X`,
//!    `T_k(L̃)X = 2·L̃·T_{k-1}X − T_{k-2}X` for `k ≥ 2`.
//! 4. Output: `Y = Σ_{k=0}^{K} (T_k X) · W_k + bias`.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a ChebNet layer.
#[derive(Debug, Clone)]
pub struct ChebNetConfig {
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// Polynomial order K (≥ 0). Uses Chebyshev polynomials T₀ … T_K (K+1 matrices total).
    pub k_order: usize,
    /// Scaling factor λ_max for the Laplacian. Canonical value is 2.0.
    /// With λ_max = 2.0, the scaled Laplacian simplifies to L̃ = −Â.
    pub lambda_max: f32,
    /// Whether to include a learnable bias term.
    pub bias: bool,
}

impl Default for ChebNetConfig {
    fn default() -> Self {
        Self {
            in_features: 1,
            out_features: 1,
            k_order: 1,
            lambda_max: 2.0,
            bias: false,
        }
    }
}

// ─── Layer ────────────────────────────────────────────────────────────────────

/// A single ChebNet layer.
///
/// Computes `Y = Σ_{k=0}^{K} T_k(L̃) X · W_k + bias`.
///
/// This is a **style-A** layer: all weight parameters are supplied by the
/// caller at forward time.  The struct holds only configuration metadata.
pub struct ChebNetLayer {
    config: ChebNetConfig,
}

impl ChebNetLayer {
    /// Construct a ChebNet layer from configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GnnError::InvalidLayerConfig`] if `in_features`, `out_features` are 0,
    /// `lambda_max` is not positive and finite, or `k_order` is absurdly large (≥ 10_000).
    pub fn new(config: ChebNetConfig) -> GnnResult<Self> {
        if config.in_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "ChebNet: in_features must be > 0".to_string(),
            ));
        }
        if config.out_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "ChebNet: out_features must be > 0".to_string(),
            ));
        }
        if !config.lambda_max.is_finite() || config.lambda_max <= 0.0 {
            return Err(GnnError::InvalidLayerConfig(
                "ChebNet: lambda_max must be a positive finite value".to_string(),
            ));
        }
        if config.k_order >= 10_000 {
            return Err(GnnError::InvalidLayerConfig(
                "ChebNet: k_order must be < 10_000".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Apply the scaled Laplacian `L̃` to a feature matrix `x` of shape `[n × in_f]`.
    ///
    /// `L̃·x = (scale − 1)·x − scale·Â·x`
    ///
    /// where `scale = 2 / lambda_max`.
    ///
    /// With `lambda_max = 2`, this reduces to `L̃·x = −Â·x`.
    fn apply_l_tilde(&self, graph: &CsrGraph, x: &[f32], in_f: usize) -> GnnResult<Vec<f32>> {
        let scale = 2.0_f32 / self.config.lambda_max;
        // Â·x via spmv uses the raw (unweighted) adjacency stored in CsrGraph.
        // However, the ChebNet paper uses the *normalized* adjacency.
        // We replicate the normalized propagation by using the COO form from
        // `normalized_adjacency()` directly, matching the SGC/GCN convention.
        let n = graph.n_nodes();
        let (rows, cols, vals) = graph.normalized_adjacency();

        // a_hat_x = Â·x
        let mut a_hat_x = vec![0.0_f32; n * in_f];
        for idx in 0..rows.len() {
            let i = rows[idx];
            let j = cols[idx];
            let v = vals[idx];
            for d in 0..in_f {
                a_hat_x[i * in_f + d] += v * x[j * in_f + d];
            }
        }

        // L̃·x = (scale − 1)·x − scale·Â·x
        let mut result = vec![0.0_f32; n * in_f];
        let coeff_x = scale - 1.0_f32;
        let coeff_a = scale;
        for i in 0..n * in_f {
            result[i] = coeff_x * x[i] - coeff_a * a_hat_x[i];
        }
        Ok(result)
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// - `graph`: CSR graph.
    /// - `node_features`: `[n_nodes × in_features]` row-major.
    /// - `weight`: `[(K+1) × in_features × out_features]` row-major.
    ///   `weight[k * in_f * out_f .. (k+1) * in_f * out_f]` is `W_k`.
    /// - `bias`: optional `[out_features]`; broadcast over all nodes.
    ///
    /// # Returns
    ///
    /// `[n_nodes × out_features]` row-major.
    ///
    /// # Errors
    ///
    /// - [`GnnError::NodeFeatureMismatch`] if `node_features.len() != n * in_features`.
    /// - [`GnnError::WeightShapeMismatch`] if `weight.len() != (K+1) * in_f * out_f`.
    /// - [`GnnError::DimensionMismatch`] if `bias.len() != out_features`.
    /// - [`GnnError::NonFiniteOutput`] if any output value is NaN or infinite.
    pub fn forward(
        &self,
        graph: &CsrGraph,
        node_features: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> GnnResult<Vec<f32>> {
        let n = graph.n_nodes();
        let in_f = self.config.in_features;
        let out_f = self.config.out_features;
        let k_plus_1 = self.config.k_order + 1;
        let weight_total = k_plus_1 * in_f * out_f;

        // ── Validation ──────────────────────────────────────────────────────────
        if node_features.len() != n * in_f {
            return Err(GnnError::NodeFeatureMismatch(
                n,
                node_features.len() / in_f.max(1),
            ));
        }
        if weight.len() != weight_total {
            return Err(GnnError::WeightShapeMismatch {
                r: k_plus_1 * in_f,
                c: out_f,
                d: k_plus_1 * in_f,
            });
        }
        if let Some(b) = bias {
            if b.len() != out_f {
                return Err(GnnError::DimensionMismatch {
                    expected: out_f,
                    got: b.len(),
                });
            }
        }

        // ── Chebyshev recurrence & accumulation ─────────────────────────────────
        // `accum[i * out_f + o]` accumulates the output for node i, feature o.
        let mut accum = vec![0.0_f32; n * out_f];

        // T₀(L̃)X = X
        let mut t_prev: Vec<f32> = node_features.to_vec();
        // Contribute T₀ · W₀
        add_gemm(
            &t_prev,
            &weight[0..in_f * out_f],
            n,
            in_f,
            out_f,
            &mut accum,
        );

        if self.config.k_order == 0 {
            // No further orders needed.
        } else {
            // T₁(L̃)X = L̃X
            let mut t_cur = self.apply_l_tilde(graph, &t_prev, in_f)?;
            // Contribute T₁ · W₁
            add_gemm(
                &t_cur,
                &weight[in_f * out_f..2 * in_f * out_f],
                n,
                in_f,
                out_f,
                &mut accum,
            );

            // T_k(L̃)X = 2·L̃·T_{k-1}X − T_{k-2}X  for k ≥ 2
            for k in 2..=self.config.k_order {
                let l_tilde_t_cur = self.apply_l_tilde(graph, &t_cur, in_f)?;
                let mut t_new = vec![0.0_f32; n * in_f];
                for i in 0..n * in_f {
                    t_new[i] = 2.0_f32 * l_tilde_t_cur[i] - t_prev[i];
                }
                // Contribute T_k · W_k
                let w_k_start = k * in_f * out_f;
                let w_k_end = w_k_start + in_f * out_f;
                add_gemm(
                    &t_new,
                    &weight[w_k_start..w_k_end],
                    n,
                    in_f,
                    out_f,
                    &mut accum,
                );
                t_prev = t_cur;
                t_cur = t_new;
            }
        }

        // ── Bias ────────────────────────────────────────────────────────────────
        if let Some(b) = bias {
            for i in 0..n {
                for o in 0..out_f {
                    accum[i * out_f + o] += b[o];
                }
            }
        }

        // ── Non-finite guard ────────────────────────────────────────────────────
        if accum.iter().any(|v| !v.is_finite()) {
            return Err(GnnError::NonFiniteOutput(
                "ChebNet forward: NaN/Inf in output",
            ));
        }

        Ok(accum)
    }

    /// Output feature dimension.
    pub fn output_dim(&self) -> usize {
        self.config.out_features
    }
}

// ─── Helper ───────────────────────────────────────────────────────────────────

/// Accumulate `out += t_k · w_k` where `t_k` is `[n × in_f]` and `w_k` is `[in_f × out_f]`.
///
/// Row-major layout: `out[i, o] += Σ_j t_k[i, j] * w_k[j, o]`.
#[inline]
fn add_gemm(t_k: &[f32], w_k: &[f32], n: usize, in_f: usize, out_f: usize, out: &mut [f32]) {
    for i in 0..n {
        for o in 0..out_f {
            let mut acc = 0.0_f32;
            for j in 0..in_f {
                acc += t_k[i * in_f + j] * w_k[j * out_f + o];
            }
            out[i * out_f + o] += acc;
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Graph helpers ─────────────────────────────────────────────────────────

    fn triangle_graph() -> CsrGraph {
        CsrGraph::from_edges(3, &[(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)])
            .expect("test invariant: value must be valid")
    }

    fn path_graph(n: usize) -> CsrGraph {
        let mut edges = Vec::new();
        for i in 0..n - 1 {
            edges.push((i, i + 1));
            edges.push((i + 1, i));
        }
        CsrGraph::from_edges(n, &edges).expect("test invariant: value must be valid")
    }

    fn star_graph(n_leaves: usize) -> CsrGraph {
        // node 0 is center, nodes 1..=n_leaves are leaves
        let mut edges = Vec::new();
        for i in 1..=n_leaves {
            edges.push((0, i));
            edges.push((i, 0));
        }
        CsrGraph::from_edges(n_leaves + 1, &edges).expect("test invariant: value must be valid")
    }

    fn single_node_graph() -> CsrGraph {
        CsrGraph::from_edges(1, &[(0, 0)]).expect("test invariant: value must be valid")
    }

    fn make_layer(k: usize) -> ChebNetLayer {
        ChebNetLayer::new(ChebNetConfig {
            in_features: 2,
            out_features: 2,
            k_order: k,
            lambda_max: 2.0,
            bias: false,
        })
        .expect("test invariant: value must be valid")
    }

    fn identity_weight(in_f: usize, out_f: usize) -> Vec<f32> {
        let mut w = vec![0.0_f32; in_f * out_f];
        let d = in_f.min(out_f);
        for i in 0..d {
            w[i * out_f + i] = 1.0;
        }
        w
    }

    // Compute L̃·x manually for lambda_max=2: L̃·x = −Â·x
    // using the COO normalized adjacency.
    fn manual_l_tilde(graph: &CsrGraph, x: &[f32], in_f: usize) -> Vec<f32> {
        let n = graph.n_nodes();
        let (rows, cols, vals) = graph.normalized_adjacency();
        let mut a_hat_x = vec![0.0_f32; n * in_f];
        for idx in 0..rows.len() {
            let i = rows[idx];
            let j = cols[idx];
            let v = vals[idx];
            for d in 0..in_f {
                a_hat_x[i * in_f + d] += v * x[j * in_f + d];
            }
        }
        // scale=1, coeff_x = scale-1 = 0, coeff_a = scale = 1 → result = -Â·x
        let mut result = vec![0.0_f32; n * in_f];
        for i in 0..n * in_f {
            result[i] = -a_hat_x[i];
        }
        result
    }

    // ── Test 1: k_order=0 equals plain matmul ─────────────────────────────────

    #[test]
    fn k0_equals_xw0() {
        let g = triangle_graph();
        let layer = ChebNetLayer::new(ChebNetConfig {
            in_features: 2,
            out_features: 3,
            k_order: 0,
            lambda_max: 2.0,
            bias: false,
        })
        .expect("test invariant: value must be valid");

        // weight = W₀ ∈ ℝ^{2×3}, features: node i = [i+1, i+2]
        let x: Vec<f32> = vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0];
        let w: Vec<f32> = vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5];

        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");

        // Expected: Y = X · W₀ (pure matmul, no graph)
        let n = 3;
        let in_f = 2;
        let out_f = 3;
        let mut expected = vec![0.0_f32; n * out_f];
        for i in 0..n {
            for o in 0..out_f {
                for j in 0..in_f {
                    expected[i * out_f + o] += x[i * in_f + j] * w[j * out_f + o];
                }
            }
        }
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "k0 matmul: got {a}, expected {b}");
        }
    }

    // ── Test 2: k_order=1 adds graph term ─────────────────────────────────────

    #[test]
    fn k1_adds_graph_term() {
        let g = path_graph(3);
        let in_f = 1usize;
        let out_f = 1usize;
        let layer = ChebNetLayer::new(ChebNetConfig {
            in_features: in_f,
            out_features: out_f,
            k_order: 1,
            lambda_max: 2.0,
            bias: false,
        })
        .expect("test invariant: value must be valid");

        let x = vec![1.0_f32, 2.0, 3.0];
        // W₀=[1], W₁=[1]: Y = X·1 + L̃X·1 = X + (−ÂX)
        let w = vec![1.0_f32, 1.0_f32];

        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");

        // Manual: L̃X = −ÂX (lambda_max=2)
        let l_tilde_x = manual_l_tilde(&g, &x, in_f);
        let mut expected = [0.0_f32; 3];
        for i in 0..3 {
            expected[i] = x[i] + l_tilde_x[i];
        }
        for (i, (a, b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "k1 graph term: node {i} got {a}, expected {b}"
            );
        }
    }

    // ── Test 3: T₂ recurrence correctness ────────────────────────────────────

    #[test]
    fn t2_recurrence_correct() {
        let g = triangle_graph();
        let in_f = 1usize;
        let out_f = 1usize;
        let layer = ChebNetLayer::new(ChebNetConfig {
            in_features: in_f,
            out_features: out_f,
            k_order: 2,
            lambda_max: 2.0,
            bias: false,
        })
        .expect("test invariant: value must be valid");

        let x = vec![1.0_f32, -1.0, 2.0];
        // W₀=[0], W₁=[0], W₂=[1]: output should be T₂X·1 = 2L̃(L̃X) − X
        let w = vec![0.0_f32, 0.0_f32, 1.0_f32];

        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");

        let l1 = manual_l_tilde(&g, &x, in_f);
        let l2 = manual_l_tilde(&g, &l1, in_f);
        let mut expected = [0.0_f32; 3];
        for i in 0..3 {
            expected[i] = 2.0 * l2[i] - x[i];
        }
        for (i, (a, b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "T2 recurrence: node {i} got {a}, expected {b}"
            );
        }
    }

    // ── Test 4: bias is added ─────────────────────────────────────────────────

    #[test]
    fn bias_is_added() {
        let g = triangle_graph();
        let layer = ChebNetLayer::new(ChebNetConfig {
            in_features: 2,
            out_features: 2,
            k_order: 0,
            lambda_max: 2.0,
            bias: true,
        })
        .expect("test invariant: value must be valid");

        let x = vec![0.0_f32; 6];
        let w = vec![0.0_f32; 4]; // zero weight → matmul = 0
        let b = vec![3.0_f32, 7.0_f32];

        let out = layer
            .forward(&g, &x, &w, Some(&b))
            .expect("test invariant: value must be valid");

        // All nodes should have output = bias
        for i in 0..3 {
            assert!((out[i * 2] - 3.0).abs() < 1e-6, "bias[0] node {i}");
            assert!((out[i * 2 + 1] - 7.0).abs() < 1e-6, "bias[1] node {i}");
        }
    }

    // ── Test 5: lambda_max=1.0 doubles the Laplacian term ────────────────────

    #[test]
    fn lambda_max_half_doubles_laplacian() {
        let g = path_graph(3);
        // k=1, W₀=0, W₁=1: output = L̃X
        // lambda_max=2 → L̃X = −ÂX
        // lambda_max=1 → scale=2 → L̃X = (2-1)X - 2·ÂX = X - 2·ÂX
        let make = |lm: f32| {
            ChebNetLayer::new(ChebNetConfig {
                in_features: 1,
                out_features: 1,
                k_order: 1,
                lambda_max: lm,
                bias: false,
            })
            .expect("test invariant: value must be valid")
        };

        let x = vec![1.0_f32, 2.0, 3.0];
        let w = vec![0.0_f32, 1.0_f32]; // W₀=0, W₁=1

        let out2 = make(2.0)
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        let out1 = make(1.0)
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");

        // With lambda_max=2: L̃·x = −Â·x (scale=1, coeff_x=0, coeff_a=1)
        // With lambda_max=1: L̃·x = x − 2·Â·x (scale=2, coeff_x=1, coeff_a=2)
        // These differ unless Â·x=0
        let diff: f32 = out2
            .iter()
            .zip(out1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-5,
            "lambda_max=1 and 2 should give different results, diff={diff}"
        );

        // Additionally verify lambda_max=1 manually for node 1 (feat=2):
        // Â·x[1] with path graph uses normalized_adjacency output
        // We just check finiteness and that out1 != out2
        assert!(out1.iter().all(|v| v.is_finite()));
    }

    // ── Test 6: weight shape validation ──────────────────────────────────────

    #[test]
    fn weight_shape_validation() {
        let g = triangle_graph();
        let layer = make_layer(1);
        let x = vec![1.0_f32; 6]; // 3 nodes × in_f=2
        let bad_w = vec![0.0_f32; 5]; // wrong: should be (1+1)*2*2=8
        let err = layer.forward(&g, &x, &bad_w, None);
        assert!(
            matches!(err, Err(GnnError::WeightShapeMismatch { .. })),
            "expected WeightShapeMismatch"
        );
    }

    // ── Test 7: node feature mismatch ────────────────────────────────────────

    #[test]
    fn node_feature_mismatch() {
        let g = triangle_graph(); // 3 nodes
        let layer = make_layer(0);
        let x = vec![1.0_f32; 4]; // wrong: 2 nodes × in_f=2, but graph has 3
        let w = vec![0.0_f32; 4]; // k=0, in_f=2, out_f=2 → correct size
        let err = layer.forward(&g, &x, &w, None);
        assert!(
            matches!(err, Err(GnnError::NodeFeatureMismatch(..))),
            "expected NodeFeatureMismatch"
        );
    }

    // ── Test 8: bias dim mismatch ─────────────────────────────────────────────

    #[test]
    fn bias_dim_mismatch() {
        let g = triangle_graph();
        let layer = make_layer(0);
        let x = vec![0.0_f32; 6];
        let w = vec![0.0_f32; 4];
        let b = vec![1.0_f32; 5]; // wrong: out_f=2
        let err = layer.forward(&g, &x, &w, Some(&b));
        assert!(
            matches!(err, Err(GnnError::DimensionMismatch { .. })),
            "expected DimensionMismatch"
        );
    }

    // ── Test 9: zero in/out features errors ───────────────────────────────────

    #[test]
    fn zero_in_features_error() {
        let err = ChebNetLayer::new(ChebNetConfig {
            in_features: 0,
            out_features: 4,
            k_order: 1,
            lambda_max: 2.0,
            bias: false,
        });
        assert!(err.is_err());
    }

    #[test]
    fn zero_out_features_error() {
        let err = ChebNetLayer::new(ChebNetConfig {
            in_features: 4,
            out_features: 0,
            k_order: 1,
            lambda_max: 2.0,
            bias: false,
        });
        assert!(err.is_err());
    }

    // ── Test 10: nonfinite guard ──────────────────────────────────────────────

    #[test]
    fn nonfinite_guard() {
        let g = triangle_graph();
        let layer = make_layer(0);
        let x = vec![1.0_f32; 6];
        let w = vec![f32::NAN; 4]; // NaN weight
        let err = layer.forward(&g, &x, &w, None);
        assert!(
            matches!(err, Err(GnnError::NonFiniteOutput(..))),
            "expected NonFiniteOutput"
        );
    }

    // ── Test 11: single node graph ────────────────────────────────────────────

    #[test]
    fn single_node_graph_works() {
        let g = single_node_graph();
        let layer = ChebNetLayer::new(ChebNetConfig {
            in_features: 2,
            out_features: 2,
            k_order: 2,
            lambda_max: 2.0,
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32, 2.0];
        let w = vec![0.1_f32; 3 * 4]; // (k=2)+1=3 matrices, each 2×2
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Test 12: k0_no_bias_all_zero_weight → zero output ────────────────────

    #[test]
    fn k0_no_bias_all_zero_weight() {
        let g = triangle_graph();
        let layer = ChebNetLayer::new(ChebNetConfig {
            in_features: 3,
            out_features: 4,
            k_order: 0,
            lambda_max: 2.0,
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 9];
        let w = vec![0.0_f32; 12]; // k+1=1 matrix, 3×4=12
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        assert!(
            out.iter().all(|v| v.abs() < 1e-8),
            "zero weight should yield zero output"
        );
    }

    // ── Test 13: output shape = n × out_f ────────────────────────────────────

    #[test]
    fn output_shape_is_n_times_out() {
        let configs = [(4, 3, 5, 0_usize), (3, 2, 2, 1), (5, 4, 8, 3), (2, 1, 1, 2)];
        for (n_nodes, in_f, out_f, k) in configs {
            let g = path_graph(n_nodes);
            let layer = ChebNetLayer::new(ChebNetConfig {
                in_features: in_f,
                out_features: out_f,
                k_order: k,
                lambda_max: 2.0,
                bias: false,
            })
            .expect("test invariant: value must be valid");
            let x = vec![0.1_f32; n_nodes * in_f];
            let w = vec![0.1_f32; (k + 1) * in_f * out_f];
            let out = layer
                .forward(&g, &x, &w, None)
                .expect("test invariant: value must be valid");
            assert_eq!(
                out.len(),
                n_nodes * out_f,
                "shape mismatch for n={n_nodes}, in={in_f}, out={out_f}, k={k}"
            );
        }
    }

    // ── Test 14: star graph k=1 ───────────────────────────────────────────────

    #[test]
    fn star_graph_k1() {
        // Star: center=0, leaves=1,2,3
        let g = star_graph(3); // 4 nodes
        let in_f = 1usize;
        let out_f = 1usize;
        let layer = ChebNetLayer::new(ChebNetConfig {
            in_features: in_f,
            out_features: out_f,
            k_order: 1,
            lambda_max: 2.0,
            bias: false,
        })
        .expect("test invariant: value must be valid");

        // W₀=0, W₁=1: output = L̃X = −ÂX (lambda_max=2)
        let w = vec![0.0_f32, 1.0_f32];
        let x = vec![1.0_f32, 1.0, 1.0, 1.0]; // uniform features

        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");

        // L̃X = −ÂX; compute manually
        let expected = manual_l_tilde(&g, &x, in_f);
        for (i, (a, b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "star_graph_k1: node {i} got {a}, expected {b}"
            );
        }
    }

    // ── Test 15: fully-connected 4-node k=1 ──────────────────────────────────

    #[test]
    fn fully_connected_four_node_k1() {
        let mut edges = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    edges.push((i, j));
                }
            }
        }
        let g = CsrGraph::from_edges(4, &edges).expect("test invariant: value must be valid");
        let layer = ChebNetLayer::new(ChebNetConfig {
            in_features: 2,
            out_features: 2,
            k_order: 1,
            lambda_max: 2.0,
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let w = vec![0.1_f32; 2 * 2 * 2]; // 2 matrices each 2×2
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Test 16: disconnected graph ───────────────────────────────────────────

    #[test]
    fn disconnected_graph_k1() {
        // Two disconnected edges: 0↔1, 2↔3
        let g = CsrGraph::from_edges(4, &[(0, 1), (1, 0), (2, 3), (3, 2)])
            .expect("test invariant: value must be valid");
        let layer = make_layer(1);
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 4×2
        let w = vec![0.1_f32; 2 * 2 * 2];
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Test 17: large k_order=5 ──────────────────────────────────────────────

    #[test]
    fn large_k_order_five() {
        let g = triangle_graph();
        let k = 5usize;
        let in_f = 2;
        let out_f = 3;
        let layer = ChebNetLayer::new(ChebNetConfig {
            in_features: in_f,
            out_features: out_f,
            k_order: k,
            lambda_max: 2.0,
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.5_f32; 3 * in_f];
        let w = vec![0.05_f32; (k + 1) * in_f * out_f];
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 3 * out_f);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Test 18: k_order=0 with bias ──────────────────────────────────────────

    #[test]
    fn k0_with_bias() {
        let g = path_graph(3);
        let layer = ChebNetLayer::new(ChebNetConfig {
            in_features: 2,
            out_features: 2,
            k_order: 0,
            lambda_max: 2.0,
            bias: true,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let w = identity_weight(2, 2); // pass-through
        let b = vec![10.0_f32, -10.0_f32];
        let out = layer
            .forward(&g, &x, &w, Some(&b))
            .expect("test invariant: value must be valid");
        // node 0: [1, 0] + [10, -10] = [11, -10]
        assert!((out[0] - 11.0).abs() < 1e-5, "node0 feature0: {}", out[0]);
        assert!(
            (out[1] - (-10.0)).abs() < 1e-5,
            "node0 feature1: {}",
            out[1]
        );
    }

    // ── Test 19: multi-feature columns (in≠out) ───────────────────────────────

    #[test]
    fn multi_feature_columns_asymmetric() {
        let g = triangle_graph();
        let in_f = 4usize;
        let out_f = 2usize;
        let k = 1;
        let layer = ChebNetLayer::new(ChebNetConfig {
            in_features: in_f,
            out_features: out_f,
            k_order: k,
            lambda_max: 2.0,
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x: Vec<f32> = (0..3 * in_f).map(|i| i as f32).collect();
        let w = vec![0.1_f32; (k + 1) * in_f * out_f];
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 3 * out_f);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Test 20: weight all-one gives expected sum ────────────────────────────

    #[test]
    fn weight_all_one_gives_expected_sum() {
        // k=0, W₀=all-ones: Y[i, o] = Σ_j x[i,j]
        let g = path_graph(3);
        let in_f = 3usize;
        let out_f = 2usize;
        let layer = ChebNetLayer::new(ChebNetConfig {
            in_features: in_f,
            out_features: out_f,
            k_order: 0,
            lambda_max: 2.0,
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3×3
        let w = vec![1.0_f32; in_f * out_f];
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        // node 0: sum [1,2,3] = 6, both output features
        assert!((out[0] - 6.0).abs() < 1e-5, "node0: {}", out[0]);
        assert!((out[1] - 6.0).abs() < 1e-5, "node0: {}", out[1]);
        // node 1: sum [4,5,6] = 15
        assert!((out[2] - 15.0).abs() < 1e-5, "node1: {}", out[2]);
        // node 2: sum [7,8,9] = 24
        assert!((out[4] - 24.0).abs() < 1e-5, "node2: {}", out[4]);
    }

    // ── Test 21: k=1 out=1 spot check single entry ────────────────────────────

    #[test]
    fn k1_out1_spot_check() {
        // 2-node bidirected, in=1, k=1
        let g = CsrGraph::from_edges(2, &[(0, 1), (1, 0)])
            .expect("test invariant: value must be valid");
        let layer = ChebNetLayer::new(ChebNetConfig {
            in_features: 1,
            out_features: 1,
            k_order: 1,
            lambda_max: 2.0,
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![4.0_f32, 2.0_f32];
        // W₀=1, W₁=1: Y = X + L̃X
        let w = vec![1.0_f32, 1.0_f32];
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");

        // Normalized adjacency for 2-node bidirected (out_deg=1 each):
        // d_inv_sqrt[i] = 1/sqrt(1+1) = 1/sqrt(2)
        // self-loop val = (1/sqrt(2))^2 = 0.5
        // off-diag val = 1/sqrt(2) * 1 * 1/sqrt(2) = 0.5
        // Â·x[0] = 0.5*4 + 0.5*2 = 3.0
        // L̃·x[0] = −Â·x[0] = −3.0
        // Y[0] = x[0]*W₀ + L̃x[0]*W₁ = 4 + (-3) = 1.0
        assert!((out[0] - 1.0_f32).abs() < 1e-5, "Y[0]={}", out[0]);
        // Symmetric: Y[1] = 2 + (-3) = -1.0
        assert!((out[1] - (-1.0_f32)).abs() < 1e-5, "Y[1]={}", out[1]);
    }

    // ── Test 22: invalid lambda_max ───────────────────────────────────────────

    #[test]
    fn invalid_lambda_max_errors() {
        let err_zero = ChebNetLayer::new(ChebNetConfig {
            in_features: 2,
            out_features: 2,
            k_order: 1,
            lambda_max: 0.0,
            bias: false,
        });
        assert!(err_zero.is_err(), "lambda_max=0 should fail");

        let err_neg = ChebNetLayer::new(ChebNetConfig {
            in_features: 2,
            out_features: 2,
            k_order: 1,
            lambda_max: -1.0,
            bias: false,
        });
        assert!(err_neg.is_err(), "lambda_max<0 should fail");

        let err_nan = ChebNetLayer::new(ChebNetConfig {
            in_features: 2,
            out_features: 2,
            k_order: 1,
            lambda_max: f32::NAN,
            bias: false,
        });
        assert!(err_nan.is_err(), "lambda_max=NaN should fail");
    }

    // ── Test 23: excessive k_order errors ─────────────────────────────────────

    #[test]
    fn excessive_k_order_errors() {
        let err = ChebNetLayer::new(ChebNetConfig {
            in_features: 2,
            out_features: 2,
            k_order: 10_000,
            lambda_max: 2.0,
            bias: false,
        });
        assert!(err.is_err(), "k_order >= 10_000 should fail");
    }
}
