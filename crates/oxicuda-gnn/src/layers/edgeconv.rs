//! EdgeConv / DGCNN layer — Wang et al. 2019 (TOG/ACM).
//!
//! For each node `i`, computes edge features for each neighbor `j` as
//! `e_{ij} = [h_i, h_j - h_i]` (or a variant), applies a shared MLP,
//! then max-aggregates the results.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

// ─── EdgeConvMode ─────────────────────────────────────────────────────────────

/// How to construct the edge feature `e_{ij}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeConvMode {
    /// `[h_i, h_j - h_i]` — original DGCNN formulation.
    CenterDiff,
    /// `[h_j - h_i]` — pure difference (translation-invariant, 2nd-order).
    DiffOnly,
    /// `[h_i, h_j]` — concatenation without difference.
    Concat,
}

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for an EdgeConv layer.
#[derive(Debug, Clone)]
pub struct EdgeConvConfig {
    /// Input feature dimension.
    pub in_features: usize,
    /// Hidden dimension of the two-layer MLP applied to each edge feature.
    pub hidden_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// If `true`, include self-loop: node `i` is a neighbor of itself.
    pub self_loop: bool,
    /// Edge feature construction mode.
    pub mode: EdgeConvMode,
}

impl Default for EdgeConvConfig {
    fn default() -> Self {
        Self {
            in_features: 16,
            hidden_features: 64,
            out_features: 16,
            self_loop: true,
            mode: EdgeConvMode::CenterDiff,
        }
    }
}

// ─── Layer ────────────────────────────────────────────────────────────────────

/// EdgeConv layer.
pub struct EdgeConvLayer {
    /// Layer configuration.
    pub config: EdgeConvConfig,
}

impl EdgeConvLayer {
    /// Construct an EdgeConv layer from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `GnnError::InvalidLayerConfig` if any dimension is zero.
    pub fn new(config: EdgeConvConfig) -> GnnResult<Self> {
        if config.in_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "in_features must be > 0".to_string(),
            ));
        }
        if config.hidden_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "hidden_features must be > 0".to_string(),
            ));
        }
        if config.out_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "out_features must be > 0".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Compute the edge feature dimension based on the mode.
    ///
    /// - `CenterDiff`: `2 * in_features`
    /// - `DiffOnly`:   `in_features`
    /// - `Concat`:     `2 * in_features`
    #[inline]
    pub fn edge_feature_dim(&self) -> usize {
        match self.config.mode {
            EdgeConvMode::CenterDiff | EdgeConvMode::Concat => 2 * self.config.in_features,
            EdgeConvMode::DiffOnly => self.config.in_features,
        }
    }

    /// EdgeConv forward pass.
    ///
    /// For each node `i`:
    /// 1. Collect neighbors `N(i)` (plus `i` itself if `self_loop = true`)
    /// 2. For each neighbor `j`: compute edge feature `e_{ij}` based on mode
    /// 3. Apply `MLP(e_{ij})`: Linear → ReLU → Linear
    /// 4. Max-aggregate: `h_i^new = max_{j} MLP(e_{ij})`
    /// 5. For isolated nodes (degree 0 and no self-loop): return zeros
    ///
    /// # Arguments
    /// - `graph`: CSR graph (N(i) = out-neighbors)
    /// - `x`: node features `[n_nodes × in_features]`, row-major
    /// - `w1`: `[hidden × edge_feature_dim]`, row-major
    /// - `b1`: `[hidden]`
    /// - `w2`: `[out_features × hidden]`, row-major
    /// - `b2`: `[out_features]`
    ///
    /// # Returns
    /// `[n_nodes × out_features]`
    pub fn forward(
        &self,
        graph: &CsrGraph,
        x: &[f32],
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
    ) -> GnnResult<Vec<f32>> {
        let n = graph.n_nodes();
        let in_f = self.config.in_features;
        let hid = self.config.hidden_features;
        let out_f = self.config.out_features;
        let edge_dim = self.edge_feature_dim();

        // Validate inputs
        if x.len() != n * in_f {
            return Err(GnnError::NodeFeatureMismatch(n, x.len() / in_f.max(1)));
        }
        if w1.len() != hid * edge_dim {
            return Err(GnnError::WeightShapeMismatch {
                r: hid,
                c: edge_dim,
                d: edge_dim,
            });
        }
        if b1.len() != hid {
            return Err(GnnError::DimensionMismatch {
                expected: hid,
                got: b1.len(),
            });
        }
        if w2.len() != out_f * hid {
            return Err(GnnError::WeightShapeMismatch {
                r: out_f,
                c: hid,
                d: hid,
            });
        }
        if b2.len() != out_f {
            return Err(GnnError::DimensionMismatch {
                expected: out_f,
                got: b2.len(),
            });
        }

        let mode = self.config.mode;
        let self_loop = self.config.self_loop;

        let mut output = vec![0.0_f32; n * out_f];

        for i in 0..n {
            let neighbors = graph.neighbors(i)?;
            let h_i = &x[i * in_f..(i + 1) * in_f];

            // Max-aggregate MLP outputs over all edge features.
            // Initialize with -infinity; will fall back to zeros if no edges.
            let mut agg = vec![f32::NEG_INFINITY; out_f];
            let mut has_any_edge = false;

            // Process graph neighbors
            for &j in neighbors {
                let h_j = &x[j * in_f..(j + 1) * in_f];
                let e = edge_feature(h_i, h_j, mode);
                let f = apply_mlp(&e, edge_dim, w1, b1, hid, w2, b2, out_f);
                for k in 0..out_f {
                    if f[k] > agg[k] {
                        agg[k] = f[k];
                    }
                }
                has_any_edge = true;
            }

            // Self-loop: treat h_i itself as a neighbor
            if self_loop {
                // h_j = h_i so diff = 0
                let e = edge_feature(h_i, h_i, mode);
                let f = apply_mlp(&e, edge_dim, w1, b1, hid, w2, b2, out_f);
                for k in 0..out_f {
                    if f[k] > agg[k] {
                        agg[k] = f[k];
                    }
                }
                has_any_edge = true;
            }

            if has_any_edge {
                // Replace any remaining -inf (shouldn't happen but guard anyway)
                for k in 0..out_f {
                    if agg[k].is_infinite() {
                        agg[k] = 0.0;
                    }
                    output[i * out_f + k] = agg[k];
                }
            }
            // else: isolated node with no self_loop → remains zeros (already initialized)
        }

        Ok(output)
    }
}

// ─── Edge feature construction ────────────────────────────────────────────────

/// Compute one edge feature vector for the edge (i, j) given center `h_i` and
/// neighbor `h_j`.
///
/// - `CenterDiff`: `[h_i[0..d], h_j[0..d] - h_i[0..d]]`
/// - `DiffOnly`:   `[h_j[k] - h_i[k] for k in 0..d]`
/// - `Concat`:     `[h_i[0..d], h_j[0..d]]`
pub fn edge_feature(h_i: &[f32], h_j: &[f32], mode: EdgeConvMode) -> Vec<f32> {
    let d = h_i.len();
    match mode {
        EdgeConvMode::CenterDiff => {
            let mut out = Vec::with_capacity(2 * d);
            out.extend_from_slice(h_i);
            for k in 0..d {
                out.push(h_j[k] - h_i[k]);
            }
            out
        }
        EdgeConvMode::DiffOnly => (0..d).map(|k| h_j[k] - h_i[k]).collect(),
        EdgeConvMode::Concat => {
            let mut out = Vec::with_capacity(2 * d);
            out.extend_from_slice(h_i);
            out.extend_from_slice(h_j);
            out
        }
    }
}

// ─── Internal MLP helper ─────────────────────────────────────────────────────

/// Apply two-layer MLP to edge feature vector `e`.
///
/// `h = ReLU(W1 @ e + b1)`, then `f = W2 @ h + b2`.
fn apply_mlp(
    e: &[f32],
    edge_dim: usize,
    w1: &[f32],
    b1: &[f32],
    hid: usize,
    w2: &[f32],
    b2: &[f32],
    out_f: usize,
) -> Vec<f32> {
    // Layer 1: h = ReLU(W1 @ e + b1)  [hid]
    let mut h = vec![0.0_f32; hid];
    for k in 0..hid {
        let mut acc = b1[k];
        for j in 0..edge_dim {
            acc += w1[k * edge_dim + j] * e[j];
        }
        h[k] = acc.max(0.0); // ReLU
    }

    // Layer 2: f = W2 @ h + b2  [out_f]
    let mut f = vec![0.0_f32; out_f];
    for k in 0..out_f {
        let mut acc = b2[k];
        for j in 0..hid {
            acc += w2[k * hid + j] * h[j];
        }
        f[k] = acc;
    }
    f
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 3-node triangle: 0↔1↔2↔0 (bidirectional).
    fn triangle_graph() -> CsrGraph {
        CsrGraph::from_edges(3, &[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
            .expect("test invariant: valid graph")
    }

    fn make_weights(
        edge_dim: usize,
        hid: usize,
        out: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let w1 = vec![0.1_f32; hid * edge_dim];
        let b1 = vec![0.0_f32; hid];
        let w2 = vec![0.1_f32; out * hid];
        let b2 = vec![0.0_f32; out];
        (w1, b1, w2, b2)
    }

    // ─── edge_feature tests ──────────────────────────────────────────────────

    #[test]
    fn edge_feature_center_diff_length() {
        let h_i = vec![1.0_f32, 2.0];
        let h_j = vec![3.0_f32, 4.0];
        let e = edge_feature(&h_i, &h_j, EdgeConvMode::CenterDiff);
        assert_eq!(e.len(), 4); // 2 * in_features
    }

    #[test]
    fn edge_feature_center_diff_values() {
        // [h_i, h_j - h_i] = [1,2, 2,2]
        let h_i = vec![1.0_f32, 2.0];
        let h_j = vec![3.0_f32, 4.0];
        let e = edge_feature(&h_i, &h_j, EdgeConvMode::CenterDiff);
        assert!((e[0] - 1.0).abs() < 1e-6, "e[0] should be 1, got {}", e[0]);
        assert!((e[1] - 2.0).abs() < 1e-6, "e[1] should be 2, got {}", e[1]);
        assert!(
            (e[2] - 2.0).abs() < 1e-6,
            "e[2] should be h_j[0]-h_i[0]=2, got {}",
            e[2]
        );
        assert!(
            (e[3] - 2.0).abs() < 1e-6,
            "e[3] should be h_j[1]-h_i[1]=2, got {}",
            e[3]
        );
    }

    #[test]
    fn edge_feature_diff_only_values() {
        // [h_j - h_i] = [2, 2]
        let h_i = vec![1.0_f32, 2.0];
        let h_j = vec![3.0_f32, 4.0];
        let e = edge_feature(&h_i, &h_j, EdgeConvMode::DiffOnly);
        assert_eq!(e.len(), 2, "DiffOnly length should be in_features");
        assert!((e[0] - 2.0).abs() < 1e-6, "e[0] should be 2, got {}", e[0]);
        assert!((e[1] - 2.0).abs() < 1e-6, "e[1] should be 2, got {}", e[1]);
    }

    #[test]
    fn edge_feature_concat_values() {
        // [h_i, h_j] = [1, 2, 3, 4]
        let h_i = vec![1.0_f32, 2.0];
        let h_j = vec![3.0_f32, 4.0];
        let e = edge_feature(&h_i, &h_j, EdgeConvMode::Concat);
        assert_eq!(e.len(), 4, "Concat length should be 2 * in_features");
        assert!((e[0] - 1.0).abs() < 1e-6);
        assert!((e[1] - 2.0).abs() < 1e-6);
        assert!((e[2] - 3.0).abs() < 1e-6);
        assert!((e[3] - 4.0).abs() < 1e-6);
    }

    // ─── edge_feature_dim tests ──────────────────────────────────────────────

    #[test]
    fn edge_feature_dim_center_diff() {
        let cfg = EdgeConvConfig {
            in_features: 8,
            hidden_features: 16,
            out_features: 4,
            self_loop: true,
            mode: EdgeConvMode::CenterDiff,
        };
        let layer = EdgeConvLayer::new(cfg).expect("valid config");
        assert_eq!(layer.edge_feature_dim(), 16);
    }

    #[test]
    fn edge_feature_dim_diff_only() {
        let cfg = EdgeConvConfig {
            in_features: 8,
            hidden_features: 16,
            out_features: 4,
            self_loop: true,
            mode: EdgeConvMode::DiffOnly,
        };
        let layer = EdgeConvLayer::new(cfg).expect("valid config");
        assert_eq!(layer.edge_feature_dim(), 8);
    }

    #[test]
    fn edge_feature_dim_concat() {
        let cfg = EdgeConvConfig {
            in_features: 8,
            hidden_features: 16,
            out_features: 4,
            self_loop: false,
            mode: EdgeConvMode::Concat,
        };
        let layer = EdgeConvLayer::new(cfg).expect("valid config");
        assert_eq!(layer.edge_feature_dim(), 16);
    }

    // ─── EdgeConvLayer::new error tests ──────────────────────────────────────

    #[test]
    fn edgeconv_new_invalid_in_features() {
        let cfg = EdgeConvConfig {
            in_features: 0,
            ..Default::default()
        };
        assert!(EdgeConvLayer::new(cfg).is_err());
    }

    #[test]
    fn edgeconv_new_invalid_hidden_features() {
        let cfg = EdgeConvConfig {
            hidden_features: 0,
            ..Default::default()
        };
        assert!(EdgeConvLayer::new(cfg).is_err());
    }

    #[test]
    fn edgeconv_new_invalid_out_features() {
        let cfg = EdgeConvConfig {
            out_features: 0,
            ..Default::default()
        };
        assert!(EdgeConvLayer::new(cfg).is_err());
    }

    // ─── forward tests ───────────────────────────────────────────────────────

    #[test]
    fn edgeconv_forward_output_shape() {
        let graph = triangle_graph();
        let cfg = EdgeConvConfig {
            in_features: 2,
            hidden_features: 4,
            out_features: 3,
            self_loop: true,
            mode: EdgeConvMode::CenterDiff,
        };
        let layer = EdgeConvLayer::new(cfg.clone()).expect("valid config");
        let edge_dim = layer.edge_feature_dim();
        let (w1, b1, w2, b2) = make_weights(edge_dim, cfg.hidden_features, cfg.out_features);
        let x = vec![0.5_f32; 3 * cfg.in_features];
        let out = layer
            .forward(&graph, &x, &w1, &b1, &w2, &b2)
            .expect("forward should succeed");
        assert_eq!(out.len(), 3 * cfg.out_features);
    }

    #[test]
    fn edgeconv_forward_self_loop_true_isolated_node_nonzero() {
        // Node 2 has no outgoing graph edges; self_loop=true → self-edge provides output
        let graph = CsrGraph::from_edges(3, &[(0, 1), (1, 0)]).expect("valid graph");
        let cfg = EdgeConvConfig {
            in_features: 2,
            hidden_features: 4,
            out_features: 2,
            self_loop: true,
            mode: EdgeConvMode::CenterDiff,
        };
        let layer = EdgeConvLayer::new(cfg.clone()).expect("valid config");
        let edge_dim = layer.edge_feature_dim();
        let w1 = vec![1.0_f32; cfg.hidden_features * edge_dim];
        let b1 = vec![0.1_f32; cfg.hidden_features];
        let w2 = vec![1.0_f32; cfg.out_features * cfg.hidden_features];
        let b2 = vec![0.1_f32; cfg.out_features];
        let x = vec![1.0_f32; 3 * cfg.in_features];
        let out = layer
            .forward(&graph, &x, &w1, &b1, &w2, &b2)
            .expect("forward should succeed");
        assert_eq!(out.len(), 3 * cfg.out_features);
        // Node 2 (isolated) should produce a finite, non-zero value due to self-loop
        let node2_out = &out[2 * cfg.out_features..(2 + 1) * cfg.out_features];
        assert!(
            node2_out.iter().all(|v| v.is_finite()),
            "self-loop output must be finite"
        );
        let max_val: f32 = node2_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_val > 0.0,
            "self-loop should produce nonzero output, got {max_val}"
        );
    }

    #[test]
    fn edgeconv_forward_self_loop_false_isolated_node_zeros() {
        // Node 2 has no outgoing edges; self_loop=false → output is zeros
        let graph = CsrGraph::from_edges(3, &[(0, 1), (1, 0)]).expect("valid graph");
        let cfg = EdgeConvConfig {
            in_features: 2,
            hidden_features: 4,
            out_features: 2,
            self_loop: false,
            mode: EdgeConvMode::CenterDiff,
        };
        let layer = EdgeConvLayer::new(cfg.clone()).expect("valid config");
        let edge_dim = layer.edge_feature_dim();
        let (w1, b1, w2, b2) = make_weights(edge_dim, cfg.hidden_features, cfg.out_features);
        let x = vec![1.0_f32; 3 * cfg.in_features];
        let out = layer
            .forward(&graph, &x, &w1, &b1, &w2, &b2)
            .expect("forward should succeed");
        let node2_out = &out[2 * cfg.out_features..(2 + 1) * cfg.out_features];
        assert!(
            node2_out.iter().all(|&v| v == 0.0),
            "isolated node with no self_loop should be zeros, got {node2_out:?}"
        );
    }

    #[test]
    fn edgeconv_forward_output_finite() {
        let graph = triangle_graph();
        let cfg = EdgeConvConfig::default();
        let layer = EdgeConvLayer::new(cfg.clone()).expect("valid config");
        let edge_dim = layer.edge_feature_dim();
        let (w1, b1, w2, b2) = make_weights(edge_dim, cfg.hidden_features, cfg.out_features);
        let x: Vec<f32> = (0..3 * cfg.in_features).map(|i| i as f32 * 0.1).collect();
        let out = layer
            .forward(&graph, &x, &w1, &b1, &w2, &b2)
            .expect("forward should succeed");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all outputs should be finite"
        );
    }

    #[test]
    fn edgeconv_forward_wrong_w1_shape_error() {
        let graph = triangle_graph();
        let cfg = EdgeConvConfig {
            in_features: 2,
            hidden_features: 4,
            out_features: 2,
            self_loop: true,
            mode: EdgeConvMode::CenterDiff,
        };
        let layer = EdgeConvLayer::new(cfg.clone()).expect("valid config");
        let x = vec![0.0_f32; 3 * cfg.in_features];
        let w1 = vec![0.1_f32; 3]; // wrong
        let b1 = vec![0.0_f32; cfg.hidden_features];
        let w2 = vec![0.1_f32; cfg.out_features * cfg.hidden_features];
        let b2 = vec![0.0_f32; cfg.out_features];
        assert!(layer.forward(&graph, &x, &w1, &b1, &w2, &b2).is_err());
    }

    #[test]
    fn edgeconv_forward_wrong_x_shape_error() {
        let graph = triangle_graph();
        let cfg = EdgeConvConfig {
            in_features: 4,
            hidden_features: 8,
            out_features: 4,
            self_loop: true,
            mode: EdgeConvMode::CenterDiff,
        };
        let layer = EdgeConvLayer::new(cfg.clone()).expect("valid config");
        let edge_dim = layer.edge_feature_dim();
        let (w1, b1, w2, b2) = make_weights(edge_dim, cfg.hidden_features, cfg.out_features);
        let x = vec![0.0_f32; 2 * cfg.in_features]; // only 2 rows, graph has 3 nodes
        assert!(layer.forward(&graph, &x, &w1, &b1, &w2, &b2).is_err());
    }

    #[test]
    fn edgeconv_center_diff_identical_features_diff_zero() {
        // When all nodes have the same feature, CenterDiff diff=0 → only center part matters
        let graph = triangle_graph();
        let cfg = EdgeConvConfig {
            in_features: 2,
            hidden_features: 4,
            out_features: 2,
            self_loop: false,
            mode: EdgeConvMode::CenterDiff,
        };
        let layer = EdgeConvLayer::new(cfg.clone()).expect("valid config");
        let edge_dim = layer.edge_feature_dim();
        // Use w1 with second half of columns zeroed to verify diff=0 doesn't contribute
        // Simple test: same features → output should be finite and consistent
        let (w1, b1, w2, b2) = make_weights(edge_dim, cfg.hidden_features, cfg.out_features);
        let x = vec![1.0_f32; 3 * cfg.in_features];
        let out = layer
            .forward(&graph, &x, &w1, &b1, &w2, &b2)
            .expect("forward ok");
        // All nodes have same features → all output rows should be equal
        let row0 = &out[0..cfg.out_features];
        let row1 = &out[cfg.out_features..2 * cfg.out_features];
        let row2 = &out[2 * cfg.out_features..3 * cfg.out_features];
        for k in 0..cfg.out_features {
            assert!(
                (row0[k] - row1[k]).abs() < 1e-5,
                "rows 0 and 1 should match"
            );
            assert!(
                (row0[k] - row2[k]).abs() < 1e-5,
                "rows 0 and 2 should match"
            );
        }
    }

    #[test]
    fn edgeconv_single_node_self_loop_works() {
        // n=1 with self_loop=true: h[0]-h[0]=0 → CenterDiff edge = [h[0], 0..]
        let graph = CsrGraph::from_edges(1, &[(0, 0)]).expect("valid graph");
        let cfg = EdgeConvConfig {
            in_features: 2,
            hidden_features: 4,
            out_features: 2,
            self_loop: true,
            mode: EdgeConvMode::CenterDiff,
        };
        let layer = EdgeConvLayer::new(cfg.clone()).expect("valid config");
        let edge_dim = layer.edge_feature_dim();
        let w1 = vec![1.0_f32; cfg.hidden_features * edge_dim];
        let b1 = vec![0.1_f32; cfg.hidden_features];
        let w2 = vec![1.0_f32; cfg.out_features * cfg.hidden_features];
        let b2 = vec![0.1_f32; cfg.out_features];
        let x = vec![1.0_f32, 2.0];
        let out = layer
            .forward(&graph, &x, &w1, &b1, &w2, &b2)
            .expect("forward ok");
        assert_eq!(out.len(), cfg.out_features);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "single-node self-loop output must be finite"
        );
    }

    #[test]
    fn edgeconv_diff_only_translation_equivariance() {
        // DiffOnly: shifting all features by a constant should not change output
        // because h_j - h_i cancels the shift
        let graph = triangle_graph();
        let cfg = EdgeConvConfig {
            in_features: 2,
            hidden_features: 4,
            out_features: 2,
            self_loop: false,
            mode: EdgeConvMode::DiffOnly,
        };
        let layer = EdgeConvLayer::new(cfg.clone()).expect("valid config");
        let edge_dim = layer.edge_feature_dim();
        let (w1, b1, w2, b2) = make_weights(edge_dim, cfg.hidden_features, cfg.out_features);

        let x_base: Vec<f32> = (0..3 * cfg.in_features).map(|i| i as f32).collect();
        let shift = 100.0_f32;
        let x_shifted: Vec<f32> = x_base.iter().map(|v| v + shift).collect();

        let out_base = layer
            .forward(&graph, &x_base, &w1, &b1, &w2, &b2)
            .expect("forward ok");
        let out_shifted = layer
            .forward(&graph, &x_shifted, &w1, &b1, &w2, &b2)
            .expect("forward ok");

        for (a, b) in out_base.iter().zip(out_shifted.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "DiffOnly should be translation-equivariant; diff={}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn edgeconv_concat_mode_output_shape() {
        let graph = triangle_graph();
        let cfg = EdgeConvConfig {
            in_features: 3,
            hidden_features: 6,
            out_features: 4,
            self_loop: true,
            mode: EdgeConvMode::Concat,
        };
        let layer = EdgeConvLayer::new(cfg.clone()).expect("valid config");
        let edge_dim = layer.edge_feature_dim();
        let (w1, b1, w2, b2) = make_weights(edge_dim, cfg.hidden_features, cfg.out_features);
        let x: Vec<f32> = (0..3 * cfg.in_features).map(|i| i as f32 * 0.5).collect();
        let out = layer
            .forward(&graph, &x, &w1, &b1, &w2, &b2)
            .expect("forward ok");
        assert_eq!(out.len(), 3 * cfg.out_features);
    }

    #[test]
    fn edgeconv_default_config() {
        let cfg = EdgeConvConfig::default();
        assert_eq!(cfg.mode, EdgeConvMode::CenterDiff);
        assert!(cfg.self_loop);
        assert!(EdgeConvLayer::new(cfg).is_ok());
    }
}
