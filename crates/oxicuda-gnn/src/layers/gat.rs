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

        for h in 0..nh {
            let a_off = h * 2 * hd; // offset into attn_weight for head h
            let wx_off = h * n * hd;

            // Compute attention logits for all edges
            // e_ij = LeakyReLU(a_src^T Wx_i + a_dst^T Wx_j)
            // Collect per-node: for each node i, compute unnorm attentions to neighbors
            // Then softmax within each node's neighborhood
            let mut node_out = vec![0.0_f32; n * hd];

            for i in 0..n {
                let neighbors = graph.neighbors(i)?;
                if neighbors.is_empty() {
                    // No neighbors: output is 0
                    continue;
                }

                // Compute a_src^T * Wx_i (constant per source node)
                let mut a_src_dot: f32 = 0.0;
                for k in 0..hd {
                    a_src_dot += attn_weight[a_off + k] * wx[wx_off + i * hd + k];
                }

                // Compute edge scores
                let mut scores = Vec::with_capacity(neighbors.len());
                for &j in neighbors {
                    let mut a_dst_dot: f32 = 0.0;
                    for k in 0..hd {
                        a_dst_dot += attn_weight[a_off + hd + k] * wx[wx_off + j * hd + k];
                    }
                    let raw = a_src_dot + a_dst_dot;
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
                for (idx_j, (&j, &alpha)) in neighbors.iter().zip(alphas.iter()).enumerate() {
                    let _ = idx_j;
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
}
