//! Graph Attention Network (GAT) layer — Veličković et al. 2018.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;
use crate::message_passing::update::leaky_relu;

/// Configuration for a GAT layer.
#[derive(Debug, Clone)]
pub struct GatConfig {
    /// Input feature dimension.
    pub in_features: usize,
    /// Total output feature dimension (split across heads when `concat_heads = true`).
    pub out_features: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dropout probability on attention coefficients (applied as mask in CPU simulation).
    pub dropout: f32,
    /// Negative slope for LeakyReLU in attention computation.
    pub leaky_relu_slope: f32,
    /// If `true`, concatenate head outputs; if `false`, average them.
    pub concat_heads: bool,
}

/// A single GAT layer.
pub struct GatLayer {
    config: GatConfig,
    head_dim: usize,
}

/// Borrowed bundle of per-edge attention inputs used by
/// [`GatLayer::forward_with_edges`].
///
/// Kept private: the public surface exposes the three slices directly so the
/// edge-free [`GatLayer::forward`] signature is unchanged.
struct EdgeAttention<'a> {
    /// `[n_edges × edge_dim]` row-major edge attributes, in CSR edge order.
    edge_features: &'a [f32],
    /// `[num_heads × head_dim × edge_dim]` per-head edge projection `W_e`.
    edge_weight: &'a [f32],
    /// `[num_heads × head_dim]` per-head edge attention vector `a_edge`.
    edge_attn: &'a [f32],
}

impl GatLayer {
    /// Construct a GAT layer from configuration.
    ///
    /// Requires `out_features % num_heads == 0`.
    pub fn new(config: GatConfig) -> GnnResult<Self> {
        if config.in_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "in_features must be > 0".to_string(),
            ));
        }
        if config.num_heads == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "num_heads must be > 0".to_string(),
            ));
        }
        if config.out_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "out_features must be > 0".to_string(),
            ));
        }
        if config.out_features % config.num_heads != 0 {
            return Err(GnnError::InvalidAttentionHeads {
                dim: config.out_features,
                heads: config.num_heads,
            });
        }
        let head_dim = config.out_features / config.num_heads;
        Ok(Self { config, head_dim })
    }

    /// Multi-head attention forward pass.
    ///
    /// # Arguments
    ///
    /// - `graph`: CSR graph
    /// - `x`: `[n_nodes × in_features]`
    /// - `weight`: `[num_heads × head_dim × in_features]` (linearised row-major)
    ///   — one linear projection per head
    /// - `attn_weight`: `[num_heads × 2 × head_dim]` (linearised)
    ///   — attention vector `a^T = [a_src || a_dst]` per head
    ///
    /// # Returns
    ///
    /// - `concat_heads=true`: `[n_nodes × out_features]`
    /// - `concat_heads=false`: `[n_nodes × head_dim]`
    pub fn forward(
        &self,
        graph: &CsrGraph,
        x: &[f32],
        weight: &[f32],
        attn_weight: &[f32],
    ) -> GnnResult<Vec<f32>> {
        // Delegate to the shared core with no edge features. The result is
        // bit-for-bit identical to the original (edge-free) implementation.
        self.forward_inner(graph, x, weight, attn_weight, None)
    }

    /// Multi-head attention forward pass **with per-edge features**.
    ///
    /// Extends the standard GAT attention logit with a learnable edge term, per
    /// Veličković et al. 2018 §2.1 (the original paper allows the attention
    /// mechanism `a` to consume an additional edge-feature segment):
    ///
    /// ```text
    /// e_ij = LeakyReLU( a_src^T (W x_i) + a_dst^T (W x_j) + a_edge^T (W_e edge_ij) )
    /// ```
    ///
    /// i.e. the edge attribute `edge_ij` is linearly projected by a per-head
    /// `W_e` into the head space and contributes to the attention score before
    /// the per-source-neighbourhood softmax.  The value-side aggregation is
    /// unchanged (`h_i = Σ_j α_ij · W x_j`); only the attention coefficients are
    /// modulated by the edge features.
    ///
    /// Edges are indexed in CSR order: the `idx`-th neighbour of node `i`
    /// (`graph.neighbors(i)[idx]`) is edge `row_ptr[i] + idx`, matching
    /// [`CsrGraph::col_idx`].
    ///
    /// # Arguments
    ///
    /// - `graph`, `x`, `weight`, `attn_weight`: as in [`GatLayer::forward`].
    /// - `edge_features`: `[n_edges × edge_dim]` row-major, in CSR edge order.
    /// - `edge_weight`: `[num_heads × head_dim × edge_dim]` — per-head `W_e`
    ///   projection of an edge attribute into the head space.
    /// - `edge_attn`: `[num_heads × head_dim]` — per-head edge attention vector
    ///   `a_edge`.
    ///
    /// `edge_dim` is inferred from `edge_weight.len() / (num_heads * head_dim)`
    /// and must evenly divide it and be consistent with `edge_features`.
    ///
    /// # Returns
    ///
    /// Same shape as [`GatLayer::forward`].
    pub fn forward_with_edges(
        &self,
        graph: &CsrGraph,
        x: &[f32],
        weight: &[f32],
        attn_weight: &[f32],
        edge_features: &[f32],
        edge_weight: &[f32],
        edge_attn: &[f32],
    ) -> GnnResult<Vec<f32>> {
        self.forward_inner(
            graph,
            x,
            weight,
            attn_weight,
            Some(EdgeAttention {
                edge_features,
                edge_weight,
                edge_attn,
            }),
        )
    }

    /// Shared forward core.  When `edge` is `Some`, the per-edge term is folded
    /// into each attention logit; when `None`, the path is identical to the
    /// classic edge-free GAT.
    fn forward_inner(
        &self,
        graph: &CsrGraph,
        x: &[f32],
        weight: &[f32],
        attn_weight: &[f32],
        edge: Option<EdgeAttention<'_>>,
    ) -> GnnResult<Vec<f32>> {
        let n = graph.n_nodes();
        let in_f = self.config.in_features;
        let hd = self.head_dim;
        let nh = self.config.num_heads;
        let slope = self.config.leaky_relu_slope;

        if x.len() != n * in_f {
            return Err(GnnError::NodeFeatureMismatch(n, x.len() / in_f.max(1)));
        }
        // weight: [nh × hd × in_f]
        if weight.len() != nh * hd * in_f {
            return Err(GnnError::WeightShapeMismatch {
                r: nh * hd,
                c: in_f,
                d: in_f,
            });
        }
        // attn_weight: [nh × 2 × hd]
        if attn_weight.len() != nh * 2 * hd {
            return Err(GnnError::WeightShapeMismatch {
                r: nh * 2,
                c: hd,
                d: hd,
            });
        }

        // Validate the edge-feature inputs and pre-project edge attributes into
        // a per-head, per-edge scalar contribution to the attention logit:
        //   edge_logit[h][e] = a_edge_h^T (W_e_h · edge_e).
        // Flat layout: edge_logit[h * n_edges + e].
        let edge_logit: Option<Vec<f32>> = match edge {
            None => None,
            Some(ea) => {
                let n_edges = graph.n_edges();
                // Infer edge_dim from the projection weight.
                let denom = nh * hd;
                if denom == 0 || ea.edge_weight.len() % denom != 0 {
                    return Err(GnnError::WeightShapeMismatch {
                        r: nh * hd,
                        c: 0,
                        d: ea.edge_weight.len(),
                    });
                }
                let edge_dim = ea.edge_weight.len() / denom;
                if edge_dim == 0 {
                    return Err(GnnError::InvalidLayerConfig(
                        "GAT edge features: edge_dim must be > 0".to_string(),
                    ));
                }
                if ea.edge_features.len() != n_edges * edge_dim {
                    return Err(GnnError::EdgeFeatureMismatch(
                        n_edges,
                        ea.edge_features.len() / edge_dim.max(1),
                    ));
                }
                if ea.edge_attn.len() != nh * hd {
                    return Err(GnnError::WeightShapeMismatch {
                        r: nh,
                        c: hd,
                        d: hd,
                    });
                }

                let mut logits = vec![0.0_f32; nh * n_edges];
                for h in 0..nh {
                    let we_off = h * hd * edge_dim;
                    let aedge_off = h * hd;
                    for e in 0..n_edges {
                        let ef_off = e * edge_dim;
                        // proj_k = Σ_d W_e[h,k,d] * edge[e,d]; logit += a_edge[h,k] * proj_k
                        let mut acc = 0.0_f32;
                        for k in 0..hd {
                            let mut proj = 0.0_f32;
                            for d in 0..edge_dim {
                                proj += ea.edge_weight[we_off + k * edge_dim + d]
                                    * ea.edge_features[ef_off + d];
                            }
                            acc += ea.edge_attn[aedge_off + k] * proj;
                        }
                        logits[h * n_edges + e] = acc;
                    }
                }
                Some(logits)
            }
        };

        // Pre-compute projected features for all nodes and all heads: Wx
        // wx[h][i][k] = Σ_j W[h,k,j] * x[i,j]
        // Flat layout: wx[(h*n + i)*hd + k]
        let mut wx = vec![0.0_f32; nh * n * hd];
        for h in 0..nh {
            let w_off = h * hd * in_f;
            for i in 0..n {
                for k in 0..hd {
                    let mut acc = 0.0_f32;
                    for j in 0..in_f {
                        acc += weight[w_off + k * in_f + j] * x[i * in_f + j];
                    }
                    wx[(h * n + i) * hd + k] = acc;
                }
            }
        }

        // For each head, compute attention and aggregate
        let out_per_head = hd;
        let total_out = if self.config.concat_heads {
            nh * hd
        } else {
            hd
        };
        let mut all_head_out = vec![0.0_f32; nh * n * out_per_head];
        let n_edges = graph.n_edges();

        for h in 0..nh {
            let a_off = h * 2 * hd; // offset into attn_weight for head h
            let wx_off = h * n * hd;

            // Compute attention logits for all edges
            // e_ij = LeakyReLU(a_src^T Wx_i + a_dst^T Wx_j [+ a_edge^T W_e edge_ij])
            // Collect per-node: for each node i, compute unnorm attentions to neighbors
            // Then softmax within each node's neighborhood
            let mut node_out = vec![0.0_f32; n * hd];

            for i in 0..n {
                let neighbors = graph.neighbors(i)?;
                if neighbors.is_empty() {
                    // No neighbors: output is 0
                    continue;
                }
                // CSR edge index of node i's first outgoing edge.
                let edge_base = graph.row_ptr()[i];

                // Compute a_src^T * Wx_i (constant per source node)
                let mut a_src_dot: f32 = 0.0;
                for k in 0..hd {
                    a_src_dot += attn_weight[a_off + k] * wx[wx_off + i * hd + k];
                }

                // Compute edge scores
                let mut scores = Vec::with_capacity(neighbors.len());
                for (idx_j, &j) in neighbors.iter().enumerate() {
                    let mut a_dst_dot: f32 = 0.0;
                    for k in 0..hd {
                        a_dst_dot += attn_weight[a_off + hd + k] * wx[wx_off + j * hd + k];
                    }
                    let mut raw = a_src_dot + a_dst_dot;
                    // Edge-feature term (when present): edge e = edge_base + idx_j.
                    if let Some(ref logits) = edge_logit {
                        raw += logits[h * n_edges + edge_base + idx_j];
                    }
                    // LeakyReLU
                    let score = if raw >= 0.0 { raw } else { slope * raw };
                    scores.push(score);
                }

                // Softmax over scores
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exps.iter().sum();
                let alphas: Vec<f32> = if sum_exp > 0.0 {
                    exps.iter().map(|&e| e / sum_exp).collect()
                } else {
                    vec![1.0 / neighbors.len() as f32; neighbors.len()]
                };

                // Aggregate: h_i^h = Σ_j α_ij * Wx_j
                for (&j, &alpha) in neighbors.iter().zip(alphas.iter()) {
                    for k in 0..hd {
                        node_out[i * hd + k] += alpha * wx[wx_off + j * hd + k];
                    }
                }
            }

            // Copy into all_head_out
            for i in 0..n {
                for k in 0..hd {
                    all_head_out[(h * n + i) * hd + k] = node_out[i * hd + k];
                }
            }
        }

        // Combine heads
        let mut out = vec![0.0_f32; n * total_out];
        if self.config.concat_heads {
            // Interleave heads: out[i, h*hd + k] = all_head_out[h, i, k]
            for h in 0..nh {
                for i in 0..n {
                    for k in 0..hd {
                        out[i * total_out + h * hd + k] = all_head_out[(h * n + i) * hd + k];
                    }
                }
            }
        } else {
            // Average heads
            let inv_nh = 1.0 / nh as f32;
            for h in 0..nh {
                for i in 0..n {
                    for k in 0..hd {
                        out[i * total_out + k] += all_head_out[(h * n + i) * hd + k] * inv_nh;
                    }
                }
            }
        }

        // Suppress unused warning for leaky_relu import
        let _ = leaky_relu;

        Ok(out)
    }

    /// Output feature dimension.
    pub fn output_dim(&self) -> usize {
        if self.config.concat_heads {
            self.config.out_features
        } else {
            self.head_dim
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_graph(n: usize) -> CsrGraph {
        let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
        CsrGraph::from_edges(n, &edges).expect("test invariant: value must be valid")
    }

    #[test]
    fn invalid_heads_not_divisible() {
        let err = GatLayer::new(GatConfig {
            in_features: 4,
            out_features: 6,
            num_heads: 4,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        });
        assert!(matches!(err, Err(GnnError::InvalidAttentionHeads { .. })));
    }

    #[test]
    fn single_head_output_shape_concat() {
        let g = ring_graph(5);
        let n = 5;
        let in_f = 4;
        let out_f = 8;
        let nh = 2;
        let hd = out_f / nh;
        let layer = GatLayer::new(GatConfig {
            in_features: in_f,
            out_features: out_f,
            num_heads: nh,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.1_f32; n * in_f];
        let w = vec![0.01_f32; nh * hd * in_f];
        let aw = vec![0.01_f32; nh * 2 * hd];
        let out = layer
            .forward(&g, &x, &w, &aw)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), n * out_f);
    }

    #[test]
    fn mean_heads_output_shape() {
        let g = ring_graph(4);
        let n = 4;
        let in_f = 4;
        let out_f = 8;
        let nh = 4;
        let hd = out_f / nh;
        let layer = GatLayer::new(GatConfig {
            in_features: in_f,
            out_features: out_f,
            num_heads: nh,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.1_f32; n * in_f];
        let w = vec![0.01_f32; nh * hd * in_f];
        let aw = vec![0.01_f32; nh * 2 * hd];
        let out = layer
            .forward(&g, &x, &w, &aw)
            .expect("test invariant: value must be valid");
        // output_dim = hd (mean, not concat)
        assert_eq!(out.len(), n * hd);
    }

    #[test]
    fn output_dim_concat() {
        let layer = GatLayer::new(GatConfig {
            in_features: 4,
            out_features: 8,
            num_heads: 2,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        })
        .expect("test invariant: value must be valid");
        assert_eq!(layer.output_dim(), 8);
    }

    #[test]
    fn output_dim_mean() {
        let layer = GatLayer::new(GatConfig {
            in_features: 4,
            out_features: 8,
            num_heads: 2,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: false,
        })
        .expect("test invariant: value must be valid");
        assert_eq!(layer.output_dim(), 4); // head_dim = 8/2 = 4
    }

    #[test]
    fn attention_values_finite() {
        let g = ring_graph(5);
        let n = 5;
        let in_f = 3;
        let out_f = 3;
        let nh = 1;
        let hd = 3;
        let layer = GatLayer::new(GatConfig {
            in_features: in_f,
            out_features: out_f,
            num_heads: nh,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        })
        .expect("test invariant: value must be valid");
        let mut x = vec![0.0_f32; n * in_f];
        for i in 0..n {
            x[i * in_f] = i as f32;
        }
        let w = vec![0.5_f32; nh * hd * in_f];
        let aw = vec![0.1_f32; nh * 2 * hd];
        let out = layer
            .forward(&g, &x, &w, &aw)
            .expect("test invariant: value must be valid");
        assert!(out.iter().all(|v| v.is_finite()), "outputs must be finite");
    }

    #[test]
    fn isolated_node_produces_zero() {
        // Node 2 has no outgoing edges
        let g = CsrGraph::from_edges(3, &[(0, 1), (1, 0)])
            .expect("test invariant: value must be valid");
        let n = 3;
        let in_f = 2;
        let out_f = 2;
        let nh = 1;
        let hd = 2;
        let layer = GatLayer::new(GatConfig {
            in_features: in_f,
            out_features: out_f,
            num_heads: nh,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; n * in_f];
        let w = vec![0.1_f32; nh * hd * in_f];
        let aw = vec![0.1_f32; nh * 2 * hd];
        let out = layer
            .forward(&g, &x, &w, &aw)
            .expect("test invariant: value must be valid");
        // Node 2 has no outgoing edges → zero output
        assert!((out[2 * out_f]).abs() < 1e-6);
        assert!((out[2 * out_f + 1]).abs() < 1e-6);
    }

    #[test]
    fn zero_weights_zero_output() {
        let g = ring_graph(4);
        let n = 4;
        let in_f = 4;
        let out_f = 4;
        let nh = 1;
        let hd = 4;
        let layer = GatLayer::new(GatConfig {
            in_features: in_f,
            out_features: out_f,
            num_heads: nh,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; n * in_f];
        let w = vec![0.0_f32; nh * hd * in_f]; // zero projection
        let aw = vec![0.1_f32; nh * 2 * hd];
        let out = layer
            .forward(&g, &x, &w, &aw)
            .expect("test invariant: value must be valid");
        // Wx = 0, so outputs are uniform 0
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn node_feature_mismatch_error() {
        let g = ring_graph(4);
        let layer = GatLayer::new(GatConfig {
            in_features: 4,
            out_features: 4,
            num_heads: 1,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 3 * 4]; // only 3 nodes but graph has 4
        let w = vec![0.1_f32; 4 * 4];
        let aw = vec![0.1_f32; 2 * 4];
        let err = layer.forward(&g, &x, &w, &aw);
        assert!(matches!(err, Err(GnnError::NodeFeatureMismatch(..))));
    }

    #[test]
    fn uniform_features_equal_outputs() {
        // With uniform features and weights, all nodes should have equal output
        let g = ring_graph(4);
        let n = 4;
        let in_f = 2;
        let out_f = 2;
        let nh = 1;
        let hd = 2;
        let layer = GatLayer::new(GatConfig {
            in_features: in_f,
            out_features: out_f,
            num_heads: nh,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; n * in_f];
        let w = vec![1.0_f32; nh * hd * in_f];
        let aw = vec![0.5_f32; nh * 2 * hd];
        let out = layer
            .forward(&g, &x, &w, &aw)
            .expect("test invariant: value must be valid");
        // Each node has exactly one neighbor in ring, all features uniform
        let first = out[0];
        assert!(out.iter().all(|&v| (v - first).abs() < 1e-4));
    }

    #[test]
    fn four_heads_concat_output_shape() {
        let g = ring_graph(6);
        let n = 6;
        let in_f = 8;
        let out_f = 8;
        let nh = 4;
        let hd = out_f / nh;
        let layer = GatLayer::new(GatConfig {
            in_features: in_f,
            out_features: out_f,
            num_heads: nh,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.1_f32; n * in_f];
        let w = vec![0.01_f32; nh * hd * in_f];
        let aw = vec![0.01_f32; nh * 2 * hd];
        let out = layer
            .forward(&g, &x, &w, &aw)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), n * out_f);
    }

    #[test]
    fn invalid_zero_heads() {
        let err = GatLayer::new(GatConfig {
            in_features: 4,
            out_features: 4,
            num_heads: 0,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        });
        assert!(err.is_err());
    }

    // ── Edge-feature attention ────────────────────────────────────────────────

    fn edge_layer() -> (GatLayer, usize, usize, usize, usize) {
        let in_f = 3;
        let out_f = 4;
        let nh = 2;
        let hd = out_f / nh;
        let layer = GatLayer::new(GatConfig {
            in_features: in_f,
            out_features: out_f,
            num_heads: nh,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        })
        .expect("test invariant: value must be valid");
        (layer, in_f, out_f, nh, hd)
    }

    #[test]
    fn edge_features_change_output() {
        // A non-trivial directed graph so attention has real choices to make.
        let g = CsrGraph::from_edges(4, &[(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (2, 0), (3, 0)])
            .expect("test invariant: value must be valid");
        let (layer, in_f, out_f, nh, hd) = edge_layer();
        let n = g.n_nodes();
        let edge_dim = 2;
        let n_edges = g.n_edges();

        let x: Vec<f32> = (0..n * in_f).map(|i| 0.05 * (i as f32 + 1.0)).collect();
        let w: Vec<f32> = (0..nh * hd * in_f)
            .map(|i| 0.1 * (i as f32) - 0.3)
            .collect();
        let aw: Vec<f32> = (0..nh * 2 * hd).map(|i| 0.07 * (i as f32) - 0.2).collect();

        // Distinct, non-degenerate edge attributes per edge.
        let ef: Vec<f32> = (0..n_edges * edge_dim)
            .map(|i| 0.13 * (i as f32) - 0.5)
            .collect();
        let we: Vec<f32> = (0..nh * hd * edge_dim)
            .map(|i| 0.11 * (i as f32) - 0.25)
            .collect();
        let ea: Vec<f32> = (0..nh * hd).map(|i| 0.09 * (i as f32) + 0.15).collect();

        let base = layer
            .forward(&g, &x, &w, &aw)
            .expect("test invariant: value must be valid");
        let with_edges = layer
            .forward_with_edges(&g, &x, &w, &aw, &ef, &we, &ea)
            .expect("test invariant: value must be valid");

        assert_eq!(base.len(), n * out_f);
        assert_eq!(with_edges.len(), n * out_f);
        assert!(with_edges.iter().all(|v| v.is_finite()));

        // Edge features genuinely flow into the attention coefficients, so the
        // output must differ from the edge-free pass.
        let max_diff = base
            .iter()
            .zip(with_edges.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "edge features must change the output (max_diff={max_diff})"
        );
    }

    #[test]
    fn edge_features_deterministic() {
        let g = CsrGraph::from_edges(3, &[(0, 1), (0, 2), (1, 2), (2, 0)])
            .expect("test invariant: value must be valid");
        let (layer, in_f, _out_f, nh, hd) = edge_layer();
        let n = g.n_nodes();
        let edge_dim = 2;
        let n_edges = g.n_edges();
        let x = vec![0.3_f32; n * in_f];
        let w = vec![0.2_f32; nh * hd * in_f];
        let aw = vec![0.1_f32; nh * 2 * hd];
        let ef: Vec<f32> = (0..n_edges * edge_dim).map(|i| 0.1 * i as f32).collect();
        let we = vec![0.15_f32; nh * hd * edge_dim];
        let ea = vec![0.25_f32; nh * hd];
        let a = layer
            .forward_with_edges(&g, &x, &w, &aw, &ef, &we, &ea)
            .expect("test invariant: value must be valid");
        let b = layer
            .forward_with_edges(&g, &x, &w, &aw, &ef, &we, &ea)
            .expect("test invariant: value must be valid");
        assert_eq!(a, b, "edge-feature forward must be deterministic");
    }

    #[test]
    fn zero_edge_weight_matches_edge_free() {
        // With W_e = 0 the edge term vanishes, so the result must be bit-for-bit
        // identical to the classic edge-free GAT (regression-safe path).
        let g = CsrGraph::from_edges(4, &[(0, 1), (0, 2), (0, 3), (1, 0), (2, 3), (3, 1)])
            .expect("test invariant: value must be valid");
        let (layer, in_f, _out_f, nh, hd) = edge_layer();
        let n = g.n_nodes();
        let edge_dim = 3;
        let n_edges = g.n_edges();
        let x: Vec<f32> = (0..n * in_f).map(|i| 0.07 * i as f32 - 0.2).collect();
        let w: Vec<f32> = (0..nh * hd * in_f).map(|i| 0.05 * i as f32 - 0.1).collect();
        let aw: Vec<f32> = (0..nh * 2 * hd).map(|i| 0.03 * i as f32 + 0.1).collect();
        // Non-zero edge features, but a zero projection ⇒ zero edge logit.
        let ef: Vec<f32> = (0..n_edges * edge_dim)
            .map(|i| 0.5 * i as f32 + 1.0)
            .collect();
        let we = vec![0.0_f32; nh * hd * edge_dim];
        let ea = vec![0.9_f32; nh * hd];

        let base = layer
            .forward(&g, &x, &w, &aw)
            .expect("test invariant: value must be valid");
        let with_zero_edge = layer
            .forward_with_edges(&g, &x, &w, &aw, &ef, &we, &ea)
            .expect("test invariant: value must be valid");
        assert_eq!(
            base, with_zero_edge,
            "zero edge projection must reproduce the edge-free output exactly"
        );
    }

    #[test]
    fn edge_attention_coefficients_form_valid_softmax() {
        // Single source node 0 with exactly two neighbours whose value vectors
        // are linearly independent. Then out_0 = α_a·Wx_a + α_b·Wx_b is a
        // convex combination; we recover (α_a, α_b) and assert α_a+α_b == 1 and
        // both lie in [0,1] — i.e. the per-node softmax over the edge-modulated
        // logits is valid. Use a single head with an identity projection so the
        // value vectors equal the node features.
        let g = CsrGraph::from_edges(3, &[(0, 1), (0, 2)])
            .expect("test invariant: value must be valid");
        let in_f = 2;
        let out_f = 2;
        let nh = 1;
        // head_dim = out_f / nh = 2; edge_dim = 2 (both encoded in the literal
        // array sizes below).
        let layer = GatLayer::new(GatConfig {
            in_features: in_f,
            out_features: out_f,
            num_heads: nh,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        })
        .expect("test invariant: value must be valid");
        // Node features: node1 = e0, node2 = e1 (independent value vectors).
        let x = vec![
            0.0, 0.0, // node 0 (source, value irrelevant to its own output)
            1.0, 0.0, // node 1
            0.0, 1.0, // node 2
        ];
        // Identity projection ⇒ Wx_j == x_j.
        let w = vec![1.0_f32, 0.0, 0.0, 1.0];
        let aw = vec![0.3_f32, -0.1, 0.2, 0.05]; // [a_src(2) || a_dst(2)]

        // Two outgoing edges of node 0 (CSR order: →1, →2). Asymmetric attrs.
        let ef = vec![
            1.0, -0.5, // edge 0→1
            -0.7, 0.4, // edge 0→2
        ];
        let we = vec![0.6_f32, -0.2, 0.3, 0.5]; // [hd × edge_dim]
        let ea = vec![0.4_f32, 0.8]; // [hd]

        let out = layer
            .forward_with_edges(&g, &x, &w, &aw, &ef, &we, &ea)
            .expect("test invariant: value must be valid");

        // out_0 = α_a·[1,0] + α_b·[0,1] = [α_a, α_b].
        let alpha_a = out[0];
        let alpha_b = out[1];
        assert!(alpha_a.is_finite() && alpha_b.is_finite());
        assert!(
            (0.0..=1.0).contains(&alpha_a),
            "alpha_a out of [0,1]: {alpha_a}"
        );
        assert!(
            (0.0..=1.0).contains(&alpha_b),
            "alpha_b out of [0,1]: {alpha_b}"
        );
        assert!(
            (alpha_a + alpha_b - 1.0).abs() < 1e-5,
            "edge-modulated attention must still sum to 1: {alpha_a}+{alpha_b}"
        );
        // The two edges have different attributes, so attention should not be
        // perfectly uniform.
        assert!(
            (alpha_a - alpha_b).abs() > 1e-4,
            "asymmetric edge features should break uniform attention"
        );
    }

    #[test]
    fn edge_feature_shape_mismatch_errors() {
        let g = ring_graph(4);
        let (layer, in_f, _out_f, nh, hd) = edge_layer();
        let n = g.n_nodes();
        let edge_dim = 2;
        let n_edges = g.n_edges();
        let x = vec![0.1_f32; n * in_f];
        let w = vec![0.1_f32; nh * hd * in_f];
        let aw = vec![0.1_f32; nh * 2 * hd];
        let we = vec![0.1_f32; nh * hd * edge_dim];
        let ea = vec![0.1_f32; nh * hd];
        // Wrong number of edge-feature rows.
        let bad_ef = vec![0.1_f32; (n_edges + 1) * edge_dim];
        let err = layer.forward_with_edges(&g, &x, &w, &aw, &bad_ef, &we, &ea);
        assert!(matches!(err, Err(GnnError::EdgeFeatureMismatch(..))));

        // Wrong edge attention vector length.
        let good_ef = vec![0.1_f32; n_edges * edge_dim];
        let bad_ea = vec![0.1_f32; nh * hd + 1];
        let err2 = layer.forward_with_edges(&g, &x, &w, &aw, &good_ef, &we, &bad_ea);
        assert!(matches!(err2, Err(GnnError::WeightShapeMismatch { .. })));
    }
}
