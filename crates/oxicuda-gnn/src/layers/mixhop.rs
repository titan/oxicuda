//! MixHop — Higher-order neighborhood mixing.
//!
//! Abu-El-Haija, Perozzi, Kapoor, Alipourfard, Lerman, Harutyunyan, Ver Steeg & Galstyan,
//! "MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing",
//! ICML 2019.
//!
//! # Algorithm summary
//!
//! Given a graph `G` with normalized adjacency `Â` and a set of powers `P = {p₁, p₂, …}`:
//!
//! 1. For each `p ∈ P`, compute `Â^p X` by applying `Â` (via `spmv`) `p` times
//!    (`p=0`: identity, i.e., `Â⁰X = X`).
//! 2. Multiply each result by a separate weight matrix:
//!    `Z_p = Â^p X · W_p ∈ ℝ^{n × out_per_power}`.
//! 3. Concatenate all `Z_p` along the feature dimension:
//!    `Y = [Z_{p₁} ‖ Z_{p₂} ‖ … ] ∈ ℝ^{n × (|P| · out_per_power)}`.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a MixHop layer.
#[derive(Debug, Clone)]
pub struct MixHopConfig {
    /// Input feature dimension.
    pub in_features: usize,
    /// Output features produced **per power**.  The total output width is
    /// `|powers| × out_per_power`.
    pub out_per_power: usize,
    /// List of propagation powers.  Each entry `p` causes `Â^p X` to be computed
    /// and concatenated in the output.  Duplicate values are allowed but will
    /// generate duplicate column blocks.
    pub powers: Vec<usize>,
    /// Whether to include a learnable bias term (one per power, concatenated).
    pub bias: bool,
}

impl Default for MixHopConfig {
    fn default() -> Self {
        Self {
            in_features: 1,
            out_per_power: 1,
            powers: vec![0, 1],
            bias: false,
        }
    }
}

// ─── Layer ────────────────────────────────────────────────────────────────────

/// A single MixHop layer.
///
/// Computes `Y = [Â^{p₁}X·W₁ ‖ Â^{p₂}X·W₂ ‖ …]` where each `W_p` is a
/// separate `[in_features × out_per_power]` weight matrix.
///
/// This is a **style-A** layer: all weights are supplied at forward time.
pub struct MixHopLayer {
    config: MixHopConfig,
}

impl MixHopLayer {
    /// Construct a MixHop layer from configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GnnError::InvalidLayerConfig`] if:
    /// - `powers` is empty,
    /// - `in_features` or `out_per_power` is 0.
    pub fn new(config: MixHopConfig) -> GnnResult<Self> {
        if config.powers.is_empty() {
            return Err(GnnError::InvalidLayerConfig(
                "MixHop: powers must not be empty".to_string(),
            ));
        }
        if config.in_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "MixHop: in_features must be > 0".to_string(),
            ));
        }
        if config.out_per_power == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "MixHop: out_per_power must be > 0".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// - `graph`: CSR graph.
    /// - `node_features`: `[n_nodes × in_features]` row-major.
    /// - `weight`: `[|powers| × in_features × out_per_power]` row-major.
    ///   Power `p_idx` uses `weight[p_idx * in_f * out_p .. (p_idx+1) * in_f * out_p]`.
    /// - `bias`: optional `[|powers| × out_per_power]`.
    ///   Power `p_idx` uses `bias[p_idx * out_p .. (p_idx+1) * out_p]`.
    ///
    /// # Returns
    ///
    /// `[n_nodes × (|powers| × out_per_power)]` row-major, with power blocks
    /// concatenated in the order given by `config.powers`.
    ///
    /// # Errors
    ///
    /// - [`GnnError::NodeFeatureMismatch`] if `node_features.len() != n * in_features`.
    /// - [`GnnError::WeightShapeMismatch`] if `weight.len() != |powers| * in_f * out_p`.
    /// - [`GnnError::DimensionMismatch`] if `bias.len() != |powers| * out_p`.
    /// - [`GnnError::NonFiniteOutput`] if any output is NaN or infinite.
    pub fn forward(
        &self,
        graph: &CsrGraph,
        node_features: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> GnnResult<Vec<f32>> {
        let n = graph.n_nodes();
        let in_f = self.config.in_features;
        let out_p = self.config.out_per_power;
        let n_powers = self.config.powers.len();
        let total_out = n_powers * out_p;
        let weight_total = n_powers * in_f * out_p;

        // ── Validation ──────────────────────────────────────────────────────────
        if node_features.len() != n * in_f {
            return Err(GnnError::NodeFeatureMismatch(
                n,
                node_features.len() / in_f.max(1),
            ));
        }
        if weight.len() != weight_total {
            return Err(GnnError::WeightShapeMismatch {
                r: n_powers * in_f,
                c: out_p,
                d: n_powers * in_f,
            });
        }
        if let Some(b) = bias {
            if b.len() != n_powers * out_p {
                return Err(GnnError::DimensionMismatch {
                    expected: n_powers * out_p,
                    got: b.len(),
                });
            }
        }

        // ── Output buffer ────────────────────────────────────────────────────────
        // Layout: output[i, p_idx * out_p + o] for node i, power block p_idx, dim o.
        let mut output = vec![0.0_f32; n * total_out];

        // ── Per-power computation ─────────────────────────────────────────────────
        for (p_idx, &p) in self.config.powers.iter().enumerate() {
            // Compute Â^p X by applying spmv p times.
            // p=0: identity (Â⁰X = X).
            // spmv uses the RAW adjacency (edge weights = 1.0 by default).
            // The MixHop paper uses the symmetric normalized adjacency (same as
            // normalized_adjacency()), so we replicate that via COO iteration.
            let x_p = self.propagate_p(graph, node_features, in_f, p)?;

            // W_p block: [in_f × out_p]
            let w_start = p_idx * in_f * out_p;
            let w_p = &weight[w_start..w_start + in_f * out_p];

            // Z_p = Â^p X · W_p, write into column block [p_idx*out_p..(p_idx+1)*out_p]
            let col_offset = p_idx * out_p;
            for i in 0..n {
                for o in 0..out_p {
                    let mut acc = 0.0_f32;
                    for j in 0..in_f {
                        acc += x_p[i * in_f + j] * w_p[j * out_p + o];
                    }
                    output[i * total_out + col_offset + o] += acc;
                }
            }
        }

        // ── Bias ─────────────────────────────────────────────────────────────────
        if let Some(b) = bias {
            for i in 0..n {
                for p_idx in 0..n_powers {
                    let col_offset = p_idx * out_p;
                    let b_offset = p_idx * out_p;
                    for o in 0..out_p {
                        output[i * total_out + col_offset + o] += b[b_offset + o];
                    }
                }
            }
        }

        // ── Non-finite guard ──────────────────────────────────────────────────────
        if output.iter().any(|v| !v.is_finite()) {
            return Err(GnnError::NonFiniteOutput(
                "MixHop forward: NaN/Inf in output",
            ));
        }

        Ok(output)
    }

    /// Compute `Â^p X` using the normalized adjacency.
    ///
    /// Uses COO form from `normalized_adjacency()` for each application,
    /// which gives `D̂^{-1/2} (A+I) D̂^{-1/2}` — the same matrix used in GCN/ChebNet.
    ///
    /// - `p = 0`: returns `x` unchanged (identity).
    /// - `p = 1`: one normalized-adjacency multiply.
    /// - `p ≥ 2`: repeated application.
    fn propagate_p(
        &self,
        graph: &CsrGraph,
        x: &[f32],
        in_f: usize,
        p: usize,
    ) -> GnnResult<Vec<f32>> {
        if p == 0 {
            return Ok(x.to_vec());
        }
        let n = graph.n_nodes();
        let (rows, cols, vals) = graph.normalized_adjacency();

        let mut x_cur: Vec<f32> = x.to_vec();
        for _step in 0..p {
            let mut x_next = vec![0.0_f32; n * in_f];
            for idx in 0..rows.len() {
                let i = rows[idx];
                let j = cols[idx];
                let v = vals[idx];
                for d in 0..in_f {
                    x_next[i * in_f + d] += v * x_cur[j * in_f + d];
                }
            }
            x_cur = x_next;
        }
        Ok(x_cur)
    }

    /// Total output feature dimension: `|powers| × out_per_power`.
    pub fn output_dim(&self) -> usize {
        self.config.powers.len() * self.config.out_per_power
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

    /// Apply normalized adjacency once using COO.
    fn apply_norm_adj_once(graph: &CsrGraph, x: &[f32], in_f: usize) -> Vec<f32> {
        let n = graph.n_nodes();
        let (rows, cols, vals) = graph.normalized_adjacency();
        let mut out = vec![0.0_f32; n * in_f];
        for idx in 0..rows.len() {
            let i = rows[idx];
            let j = cols[idx];
            let v = vals[idx];
            for d in 0..in_f {
                out[i * in_f + d] += v * x[j * in_f + d];
            }
        }
        out
    }

    // ── Test 1: powers=[0] equals X·W ────────────────────────────────────────

    #[test]
    fn powers_zero_equals_xw() {
        let g = triangle_graph();
        let in_f = 2usize;
        let out_p = 3usize;
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: in_f,
            out_per_power: out_p,
            powers: vec![0],
            bias: false,
        })
        .expect("test invariant: value must be valid");

        let x: Vec<f32> = (0..3 * in_f).map(|i| i as f32 + 1.0).collect();
        let w: Vec<f32> = (0..in_f * out_p).map(|i| i as f32 * 0.1 + 0.1).collect();

        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");

        // Expected: X · W
        let mut expected = vec![0.0_f32; 3 * out_p];
        for i in 0..3 {
            for o in 0..out_p {
                for j in 0..in_f {
                    expected[i * out_p + o] += x[i * in_f + j] * w[j * out_p + o];
                }
            }
        }
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "powers=[0]: got {a}, expected {b}");
        }
    }

    // ── Test 2: powers=[1] equals ÂX·W ───────────────────────────────────────

    #[test]
    fn powers_one_spmv_correct() {
        let g = triangle_graph();
        let in_f = 2usize;
        let out_p = 2usize;
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: in_f,
            out_per_power: out_p,
            powers: vec![1],
            bias: false,
        })
        .expect("test invariant: value must be valid");

        let x: Vec<f32> = (0..3 * in_f).map(|i| i as f32 + 1.0).collect();
        let w = vec![0.1_f32; in_f * out_p];

        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");

        // Expected: Â·x · W
        let ax = apply_norm_adj_once(&g, &x, in_f);
        let mut expected = vec![0.0_f32; 3 * out_p];
        for i in 0..3 {
            for o in 0..out_p {
                for j in 0..in_f {
                    expected[i * out_p + o] += ax[i * in_f + j] * w[j * out_p + o];
                }
            }
        }
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "powers=[1]: got {a}, expected {b}");
        }
    }

    // ── Test 3: powers=[2] double hop ────────────────────────────────────────

    #[test]
    fn powers_two_double_hop() {
        let g = triangle_graph();
        let in_f = 2usize;
        let out_p = 2usize;
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: in_f,
            out_per_power: out_p,
            powers: vec![2],
            bias: false,
        })
        .expect("test invariant: value must be valid");

        let x: Vec<f32> = (0..3 * in_f).map(|i| i as f32 + 1.0).collect();
        let w = vec![0.1_f32; in_f * out_p];

        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");

        // Expected: Â²·x · W
        let ax1 = apply_norm_adj_once(&g, &x, in_f);
        let ax2 = apply_norm_adj_once(&g, &ax1, in_f);
        let mut expected = vec![0.0_f32; 3 * out_p];
        for i in 0..3 {
            for o in 0..out_p {
                for j in 0..in_f {
                    expected[i * out_p + o] += ax2[i * in_f + j] * w[j * out_p + o];
                }
            }
        }
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "powers=[2]: got {a}, expected {b}");
        }
    }

    // ── Test 4: concatenation width ───────────────────────────────────────────

    #[test]
    fn concatenation_width() {
        // powers=[0,1,2], out_per_power=3, in=2 → output width = 9
        let g = triangle_graph();
        let out_p = 3usize;
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: 2,
            out_per_power: out_p,
            powers: vec![0, 1, 2],
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 6];
        let w = vec![0.1_f32; 3 * 2 * out_p]; // 3 powers × in_f=2 × out_p=3
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 3 * 9, "output width should be 9 (3 nodes × 9)");
    }

    // ── Test 5: column blocks are independent ─────────────────────────────────

    #[test]
    fn column_blocks_are_independent() {
        // powers=[0,1]: left block = X·W₀, right = ÂX·W₁
        let g = triangle_graph();
        let in_f = 2usize;
        let out_p = 2usize;
        let n = 3usize;

        let x: Vec<f32> = (0..n * in_f).map(|i| i as f32 + 1.0).collect();
        // W₀ and W₁ distinct
        let w0: Vec<f32> = (0..in_f * out_p).map(|i| i as f32 * 0.1 + 0.1).collect();
        let w1: Vec<f32> = (0..in_f * out_p).map(|i| i as f32 * 0.2 + 0.05).collect();
        let mut w = Vec::new();
        w.extend_from_slice(&w0);
        w.extend_from_slice(&w1);

        let layer = MixHopLayer::new(MixHopConfig {
            in_features: in_f,
            out_per_power: out_p,
            powers: vec![0, 1],
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");

        // Left block (columns 0..out_p for each row): X·W₀
        let mut xw0 = vec![0.0_f32; n * out_p];
        for i in 0..n {
            for o in 0..out_p {
                for j in 0..in_f {
                    xw0[i * out_p + o] += x[i * in_f + j] * w0[j * out_p + o];
                }
            }
        }
        // Right block (columns out_p..2*out_p): ÂX·W₁
        let ax = apply_norm_adj_once(&g, &x, in_f);
        let mut axw1 = vec![0.0_f32; n * out_p];
        for i in 0..n {
            for o in 0..out_p {
                for j in 0..in_f {
                    axw1[i * out_p + o] += ax[i * in_f + j] * w1[j * out_p + o];
                }
            }
        }

        let total_out = 2 * out_p;
        for i in 0..n {
            for o in 0..out_p {
                let got_left = out[i * total_out + o];
                let exp_left = xw0[i * out_p + o];
                assert!(
                    (got_left - exp_left).abs() < 1e-5,
                    "left block node {i} dim {o}: got {got_left}, exp {exp_left}"
                );
                let got_right = out[i * total_out + out_p + o];
                let exp_right = axw1[i * out_p + o];
                assert!(
                    (got_right - exp_right).abs() < 1e-5,
                    "right block node {i} dim {o}: got {got_right}, exp {exp_right}"
                );
            }
        }
    }

    // ── Test 6: single power=1 ~ unnormalized GCN linear ─────────────────────

    #[test]
    fn single_power_one_same_as_gcn_without_activation() {
        // powers=[1] with identity weight → output = Â·X
        let g = triangle_graph();
        let in_f = 2usize;
        let out_p = in_f;
        let mut w = vec![0.0_f32; in_f * out_p];
        for i in 0..in_f {
            w[i * out_p + i] = 1.0;
        }
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: in_f,
            out_per_power: out_p,
            powers: vec![1],
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x: Vec<f32> = (0..3 * in_f).map(|i| i as f32 + 1.0).collect();
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        let expected = apply_norm_adj_once(&g, &x, in_f);
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "single power=1: got {a}, exp {b}");
        }
    }

    // ── Test 7: bias added correctly ──────────────────────────────────────────

    #[test]
    fn bias_added_correctly() {
        let g = triangle_graph();
        let out_p = 2usize;
        let in_f = 2usize;
        let n_powers = 2usize;
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: in_f,
            out_per_power: out_p,
            powers: vec![0, 1],
            bias: true,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.0_f32; 3 * in_f];
        let w = vec![0.0_f32; n_powers * in_f * out_p]; // zero weight
        let b = vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]; // power0: [1,2], power1: [3,4]
        let out = layer
            .forward(&g, &x, &w, Some(&b))
            .expect("test invariant: value must be valid");
        let total_out = n_powers * out_p;
        // All nodes should have [1, 2, 3, 4]
        for i in 0..3 {
            assert!((out[i * total_out] - 1.0).abs() < 1e-6, "node{i}[0]");
            assert!((out[i * total_out + 1] - 2.0).abs() < 1e-6, "node{i}[1]");
            assert!((out[i * total_out + 2] - 3.0).abs() < 1e-6, "node{i}[2]");
            assert!((out[i * total_out + 3] - 4.0).abs() < 1e-6, "node{i}[3]");
        }
    }

    // ── Test 8: empty powers validation ──────────────────────────────────────

    #[test]
    fn empty_powers_validation() {
        let err = MixHopLayer::new(MixHopConfig {
            in_features: 2,
            out_per_power: 2,
            powers: vec![],
            bias: false,
        });
        assert!(
            matches!(err, Err(GnnError::InvalidLayerConfig(..))),
            "empty powers should error"
        );
    }

    // ── Test 9: weight shape mismatch ─────────────────────────────────────────

    #[test]
    fn weight_shape_mismatch() {
        let g = triangle_graph();
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: 2,
            out_per_power: 2,
            powers: vec![0, 1],
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 6];
        let bad_w = vec![0.0_f32; 3]; // wrong: should be 2*2*2=8
        let err = layer.forward(&g, &x, &bad_w, None);
        assert!(
            matches!(err, Err(GnnError::WeightShapeMismatch { .. })),
            "expected WeightShapeMismatch"
        );
    }

    // ── Test 10: node feature mismatch ────────────────────────────────────────

    #[test]
    fn node_feature_mismatch() {
        let g = triangle_graph(); // 3 nodes
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: 2,
            out_per_power: 2,
            powers: vec![0],
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 4]; // wrong: only 2 nodes
        let w = vec![0.0_f32; 4];
        let err = layer.forward(&g, &x, &w, None);
        assert!(
            matches!(err, Err(GnnError::NodeFeatureMismatch(..))),
            "expected NodeFeatureMismatch"
        );
    }

    // ── Test 11: bias dim mismatch ────────────────────────────────────────────

    #[test]
    fn bias_dim_mismatch() {
        let g = triangle_graph();
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: 2,
            out_per_power: 2,
            powers: vec![0, 1],
            bias: true,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.0_f32; 6];
        let w = vec![0.0_f32; 8]; // 2 powers × 2 × 2
        let b = vec![1.0_f32; 3]; // wrong: should be 2*2=4
        let err = layer.forward(&g, &x, &w, Some(&b));
        assert!(
            matches!(err, Err(GnnError::DimensionMismatch { .. })),
            "expected DimensionMismatch"
        );
    }

    // ── Test 12: nonfinite guard ──────────────────────────────────────────────

    #[test]
    fn nonfinite_guard() {
        let g = triangle_graph();
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: 2,
            out_per_power: 2,
            powers: vec![0],
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 6];
        let w = vec![f32::INFINITY; 4]; // Inf weight
        let err = layer.forward(&g, &x, &w, None);
        assert!(
            matches!(err, Err(GnnError::NonFiniteOutput(..))),
            "expected NonFiniteOutput"
        );
    }

    // ── Test 13: star graph power=2 ──────────────────────────────────────────

    #[test]
    fn star_graph_power_two() {
        // Star: center=0, leaves=1,2
        let g = star_graph(2);
        let in_f = 1usize;
        let out_p = 1usize;
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: in_f,
            out_per_power: out_p,
            powers: vec![2],
            bias: false,
        })
        .expect("test invariant: value must be valid");

        let x = vec![0.0_f32, 1.0, 1.0]; // center=0, leaf1=1, leaf2=1
        let w = vec![1.0_f32]; // identity

        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");

        // Â²X: apply Â twice, verify manually
        let ax1 = apply_norm_adj_once(&g, &x, in_f);
        let ax2 = apply_norm_adj_once(&g, &ax1, in_f);
        for (i, (a, b)) in out.iter().zip(ax2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "star power2 node {i}: got {a}, exp {b}"
            );
        }
    }

    // ── Test 14: disconnected graph high power ────────────────────────────────

    #[test]
    fn disconnected_graph_power_many() {
        // Node 0 isolated, nodes 1↔2 connected
        let g = CsrGraph::from_edges(3, &[(1, 2), (2, 1)])
            .expect("test invariant: value must be valid");
        let in_f = 1usize;
        let out_p = 1usize;
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: in_f,
            out_per_power: out_p,
            powers: vec![5],
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![9.0_f32, 1.0, 1.0];
        let w = vec![1.0_f32]; // identity

        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");

        // Isolated node 0 gets self-loop with normalized weight 1.0,
        // so feature stays 9.0 after any number of hops.
        assert!(
            (out[0] - 9.0).abs() < 1e-5,
            "isolated node should keep value, got {}",
            out[0]
        );
        assert!(out.iter().all(|v| v.is_finite()), "all finite");
    }

    // ── Test 15: power order in config is respected ───────────────────────────

    #[test]
    fn power_order_in_config_respected() {
        // powers=[2,0,1]: blocks in that order (not sorted)
        let g = triangle_graph();
        let in_f = 1usize;
        let out_p = 1usize;
        let n = 3usize;

        let x = vec![1.0_f32, 2.0, 3.0];
        // W for powers=[2,0,1]: w2, w0, w1
        let w = vec![1.0_f32, 1.0_f32, 1.0_f32]; // each is scalar 1

        let layer = MixHopLayer::new(MixHopConfig {
            in_features: in_f,
            out_per_power: out_p,
            powers: vec![2, 0, 1],
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");

        // Compute Â^0·x, Â^1·x, Â^2·x
        let x0 = x.clone();
        let x1 = apply_norm_adj_once(&g, &x, in_f);
        let x2 = apply_norm_adj_once(&g, &x1, in_f);

        // Block order: Â²X·1, Â⁰X·1, Â¹X·1
        let total_out = 3;
        for i in 0..n {
            // block 0 (power=2)
            let exp_b0 = x2[i];
            // block 1 (power=0)
            let exp_b1 = x0[i];
            // block 2 (power=1)
            let exp_b2 = x1[i];
            assert!(
                (out[i * total_out] - exp_b0).abs() < 1e-5,
                "node {i} block2: got {}, exp {exp_b0}",
                out[i * total_out]
            );
            assert!(
                (out[i * total_out + 1] - exp_b1).abs() < 1e-5,
                "node {i} block0: got {}, exp {exp_b1}",
                out[i * total_out + 1]
            );
            assert!(
                (out[i * total_out + 2] - exp_b2).abs() < 1e-5,
                "node {i} block1: got {}, exp {exp_b2}",
                out[i * total_out + 2]
            );
        }
    }

    // ── Test 16: single node any powers ───────────────────────────────────────

    #[test]
    fn single_node_any_powers() {
        let g = single_node_graph();
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: 2,
            out_per_power: 2,
            powers: vec![0, 1, 3],
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32, 2.0];
        let w = vec![0.1_f32; 3 * 2 * 2]; // 3 powers × 2 × 2
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 6, "shape check: 1 node × 3 powers × 2");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Test 17: all-zero features → all-zero output ──────────────────────────

    #[test]
    fn all_zero_features_zero_output() {
        let g = triangle_graph();
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: 3,
            out_per_power: 2,
            powers: vec![0, 1, 2],
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.0_f32; 3 * 3];
        let w = vec![0.5_f32; 3 * 3 * 2];
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        assert!(
            out.iter().all(|v| v.abs() < 1e-9),
            "zero features should give zero output"
        );
    }

    // ── Test 18: out_per_power=1 ──────────────────────────────────────────────

    #[test]
    fn out_per_power_one() {
        let g = path_graph(4);
        let in_f = 3usize;
        let out_p = 1usize;
        let powers = vec![0, 1, 2];
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: in_f,
            out_per_power: out_p,
            powers: powers.clone(),
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x: Vec<f32> = (0..4 * in_f).map(|i| i as f32 * 0.2).collect();
        let w = vec![0.1_f32; powers.len() * in_f * out_p];
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 4 * powers.len(), "n=4, powers=3, out_p=1");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Test 19: in_features=1 ────────────────────────────────────────────────

    #[test]
    fn in_features_one() {
        let g = triangle_graph();
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: 1,
            out_per_power: 2,
            powers: vec![0, 1],
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32, 2.0, 3.0];
        let w = vec![0.5_f32; 2 * 2]; // 2 powers × 1 × 2
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 3 * 2 * 2, "shape: 3 nodes × 4 features");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Test 20: large power=5 ────────────────────────────────────────────────

    #[test]
    fn large_power_five() {
        let g = triangle_graph();
        let layer = MixHopLayer::new(MixHopConfig {
            in_features: 2,
            out_per_power: 2,
            powers: vec![5],
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let x: Vec<f32> = (0..6).map(|i| i as f32 + 1.0).collect();
        let w = vec![0.1_f32; 2 * 2];
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Test 21: triangle full correctness powers=[0,1,2] ────────────────────

    #[test]
    fn triangle_full_correctness() {
        let g = triangle_graph();
        let in_f = 1usize;
        let out_p = 1usize;
        let n = 3usize;

        let x = vec![1.0_f32, 2.0, 3.0];
        // W for each power: scalar 1 → output = Â^p x concatenated
        let w = vec![1.0_f32, 1.0_f32, 1.0_f32]; // one scalar per power

        let layer = MixHopLayer::new(MixHopConfig {
            in_features: in_f,
            out_per_power: out_p,
            powers: vec![0, 1, 2],
            bias: false,
        })
        .expect("test invariant: value must be valid");
        let out = layer
            .forward(&g, &x, &w, None)
            .expect("test invariant: value must be valid");

        let x0 = x.clone();
        let x1 = apply_norm_adj_once(&g, &x, in_f);
        let x2 = apply_norm_adj_once(&g, &x1, in_f);

        let total_out = 3;
        for i in 0..n {
            assert!(
                (out[i * total_out] - x0[i]).abs() < 1e-5,
                "node {i} power=0: got {}, exp {}",
                out[i * total_out],
                x0[i]
            );
            assert!(
                (out[i * total_out + 1] - x1[i]).abs() < 1e-5,
                "node {i} power=1: got {}, exp {}",
                out[i * total_out + 1],
                x1[i]
            );
            assert!(
                (out[i * total_out + 2] - x2[i]).abs() < 1e-5,
                "node {i} power=2: got {}, exp {}",
                out[i * total_out + 2],
                x2[i]
            );
        }
    }

    // ── Test 22: zero in_features error ──────────────────────────────────────

    #[test]
    fn zero_in_features_error() {
        let err = MixHopLayer::new(MixHopConfig {
            in_features: 0,
            out_per_power: 2,
            powers: vec![0],
            bias: false,
        });
        assert!(err.is_err(), "zero in_features should fail");
    }

    // ── Test 23: zero out_per_power error ────────────────────────────────────

    #[test]
    fn zero_out_per_power_error() {
        let err = MixHopLayer::new(MixHopConfig {
            in_features: 2,
            out_per_power: 0,
            powers: vec![0],
            bias: false,
        });
        assert!(err.is_err(), "zero out_per_power should fail");
    }
}
