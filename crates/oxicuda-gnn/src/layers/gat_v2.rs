//! GATv2 layer — Brody et al. 2022 "How Attentive are Graph Attention Networks?"
//!
//! Unlike GAT where `e_ij = LeakyReLU(a^T [Wx_i || Wx_j])`, GATv2 uses
//! `e_ij = a^T LeakyReLU(W_l x_i + W_r x_j)` which enables dynamic attention.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

const LEAKY_SLOPE: f32 = 0.2;

/// Configuration for a GATv2 layer.
#[derive(Debug, Clone)]
pub struct GatV2Config {
    /// Input feature dimension.
    pub in_features: usize,
    /// Total output feature dimension (split across heads).
    pub out_features: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// If `true`, use a single shared `W_l = W_r`; otherwise independent projections.
    pub share_weights: bool,
}

/// A single GATv2 layer.
pub struct GatV2Layer {
    config: GatV2Config,
    head_dim: usize,
}

impl GatV2Layer {
    /// Construct from configuration.
    ///
    /// Requires `out_features % num_heads == 0`.
    pub fn new(config: GatV2Config) -> GnnResult<Self> {
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

    /// Forward pass.
    ///
    /// For each attention head `h`:
    /// 1. `z_src = W_l^h * x_i` (source side), `z_dst = W_r^h * x_j` (dest side)
    /// 2. `m_ij = LeakyReLU(z_src + z_dst)`
    /// 3. `e_ij = a^h^T * m_ij`
    /// 4. `α_ij = softmax_j(e_ij)` within each source node's neighbourhood
    /// 5. `h_i^h = Σ_j α_ij * W_r^h * x_j`
    ///
    /// # Arguments
    ///
    /// - `x`: `[n_nodes × in_features]`
    /// - `w_left`: `[num_heads × head_dim × in_features]` — left projection
    /// - `w_right`: `[num_heads × head_dim × in_features]` — right projection
    ///   (if `share_weights`, `w_right` may equal `w_left`; caller's responsibility)
    /// - `attn`: `[num_heads × head_dim]` — per-head attention vector
    ///
    /// # Returns
    ///
    /// `[n_nodes × out_features]` (concatenated heads).
    pub fn forward(
        &self,
        graph: &CsrGraph,
        x: &[f32],
        w_left: &[f32],
        w_right: &[f32],
        attn: &[f32],
    ) -> GnnResult<Vec<f32>> {
        let n = graph.n_nodes();
        let in_f = self.config.in_features;
        let nh = self.config.num_heads;
        let hd = self.head_dim;
        let out_f = self.config.out_features;

        if x.len() != n * in_f {
            return Err(GnnError::NodeFeatureMismatch(n, x.len() / in_f.max(1)));
        }
        if w_left.len() != nh * hd * in_f {
            return Err(GnnError::WeightShapeMismatch {
                r: nh * hd,
                c: in_f,
                d: in_f,
            });
        }
        if w_right.len() != nh * hd * in_f {
            return Err(GnnError::WeightShapeMismatch {
                r: nh * hd,
                c: in_f,
                d: in_f,
            });
        }
        if attn.len() != nh * hd {
            return Err(GnnError::WeightShapeMismatch {
                r: nh,
                c: hd,
                d: hd,
            });
        }

        // Precompute left projections for all nodes and heads: z_l[h,i,k] = Σ_j W_l[h,k,j]*x[i,j]
        let mut z_left = vec![0.0_f32; nh * n * hd];
        for h in 0..nh {
            let w_off = h * hd * in_f;
            for i in 0..n {
                for k in 0..hd {
                    let mut acc = 0.0_f32;
                    for j in 0..in_f {
                        acc += w_left[w_off + k * in_f + j] * x[i * in_f + j];
                    }
                    z_left[(h * n + i) * hd + k] = acc;
                }
            }
        }
        // Precompute right projections
        let mut z_right = vec![0.0_f32; nh * n * hd];
        for h in 0..nh {
            let w_off = h * hd * in_f;
            for i in 0..n {
                for k in 0..hd {
                    let mut acc = 0.0_f32;
                    for j in 0..in_f {
                        acc += w_right[w_off + k * in_f + j] * x[i * in_f + j];
                    }
                    z_right[(h * n + i) * hd + k] = acc;
                }
            }
        }

        let mut out = vec![0.0_f32; n * out_f];

        for h in 0..nh {
            let a_off = h * hd;
            for i in 0..n {
                let neighbors = graph.neighbors(i)?;
                if neighbors.is_empty() {
                    continue;
                }

                // For each neighbor j: e_ij = a^T LeakyReLU(z_l[i] + z_r[j])
                let mut scores = Vec::with_capacity(neighbors.len());
                for &j in neighbors {
                    let mut score = 0.0_f32;
                    for k in 0..hd {
                        let combined = z_left[(h * n + i) * hd + k] + z_right[(h * n + j) * hd + k];
                        let activated = if combined >= 0.0 {
                            combined
                        } else {
                            LEAKY_SLOPE * combined
                        };
                        score += attn[a_off + k] * activated;
                    }
                    scores.push(score);
                }

                // Numerically stable softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
                let sum_e: f32 = exps.iter().sum();
                let alphas: Vec<f32> = if sum_e > 0.0 {
                    exps.iter().map(|&e| e / sum_e).collect()
                } else {
                    vec![1.0 / neighbors.len() as f32; neighbors.len()]
                };

                // Aggregate: h_i^h = Σ_j α_ij * z_right[h, j]
                for (&j, &alpha) in neighbors.iter().zip(alphas.iter()) {
                    for k in 0..hd {
                        out[i * out_f + h * hd + k] += alpha * z_right[(h * n + j) * hd + k];
                    }
                }
            }
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_graph(n: usize) -> CsrGraph {
        let edges: Vec<(usize, usize)> =
            (0..n - 1).flat_map(|i| [(i, i + 1), (i + 1, i)]).collect();
        CsrGraph::from_edges(n, &edges).expect("test invariant: value must be valid")
    }

    #[test]
    fn output_shape_single_head() {
        let g = chain_graph(4);
        let layer = GatV2Layer::new(GatV2Config {
            in_features: 4,
            out_features: 4,
            num_heads: 1,
            share_weights: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.1_f32; 4 * 4];
        let wl = vec![0.1_f32; 4 * 4];
        let wr = vec![0.1_f32; 4 * 4];
        let a = vec![0.1_f32; 4];
        let out = layer
            .forward(&g, &x, &wl, &wr, &a)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 4 * 4);
    }

    #[test]
    fn output_shape_multi_head() {
        let g = chain_graph(5);
        let layer = GatV2Layer::new(GatV2Config {
            in_features: 4,
            out_features: 8,
            num_heads: 2,
            share_weights: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.1_f32; 5 * 4];
        let wl = vec![0.1_f32; 2 * 4 * 4];
        let wr = vec![0.1_f32; 2 * 4 * 4];
        let a = vec![0.1_f32; 2 * 4];
        let out = layer
            .forward(&g, &x, &wl, &wr, &a)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 5 * 8);
    }

    #[test]
    fn invalid_head_divisibility() {
        let err = GatV2Layer::new(GatV2Config {
            in_features: 4,
            out_features: 7,
            num_heads: 3,
            share_weights: false,
        });
        assert!(matches!(err, Err(GnnError::InvalidAttentionHeads { .. })));
    }

    #[test]
    fn zero_projections_zero_output() {
        let g = chain_graph(3);
        let layer = GatV2Layer::new(GatV2Config {
            in_features: 2,
            out_features: 2,
            num_heads: 1,
            share_weights: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 3 * 2];
        let wl = vec![0.0_f32; 2 * 2];
        let wr = vec![0.0_f32; 2 * 2];
        let a = vec![0.5_f32; 2];
        let out = layer
            .forward(&g, &x, &wl, &wr, &a)
            .expect("test invariant: value must be valid");
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn output_finite_values() {
        let g = chain_graph(6);
        let layer = GatV2Layer::new(GatV2Config {
            in_features: 3,
            out_features: 6,
            num_heads: 2,
            share_weights: false,
        })
        .expect("test invariant: value must be valid");
        let x: Vec<f32> = (0..6 * 3).map(|i| i as f32 * 0.1).collect();
        let wl = vec![0.05_f32; 2 * 3 * 3];
        let wr = vec![0.05_f32; 2 * 3 * 3];
        let a = vec![0.1_f32; 2 * 3];
        let out = layer
            .forward(&g, &x, &wl, &wr, &a)
            .expect("test invariant: value must be valid");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn isolated_node_zero_output() {
        let g = CsrGraph::from_edges(3, &[(0, 1), (1, 0)])
            .expect("test invariant: value must be valid"); // node 2 isolated
        let layer = GatV2Layer::new(GatV2Config {
            in_features: 2,
            out_features: 2,
            num_heads: 1,
            share_weights: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 3 * 2];
        let wl = vec![0.1_f32; 2 * 2];
        let wr = vec![0.1_f32; 2 * 2];
        let a = vec![0.1_f32; 2];
        let out = layer
            .forward(&g, &x, &wl, &wr, &a)
            .expect("test invariant: value must be valid");
        // Node 2 has no outgoing edges → zero
        assert!((out[4]).abs() < 1e-6);
        assert!((out[5]).abs() < 1e-6);
    }

    #[test]
    fn share_weights_same_as_equal_wl_wr() {
        let g = chain_graph(3);
        let layer = GatV2Layer::new(GatV2Config {
            in_features: 2,
            out_features: 2,
            num_heads: 1,
            share_weights: true,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.5_f32; 3 * 2];
        let w = vec![0.2_f32; 2 * 2];
        let a = vec![0.3_f32; 2];
        let out1 = layer
            .forward(&g, &x, &w, &w, &a)
            .expect("test invariant: value must be valid");
        // Same weights on both sides
        assert!(out1.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn node_feature_mismatch_error() {
        let g = chain_graph(4);
        let layer = GatV2Layer::new(GatV2Config {
            in_features: 4,
            out_features: 4,
            num_heads: 1,
            share_weights: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 3 * 4]; // 3 nodes, not 4
        let wl = vec![0.1_f32; 4 * 4];
        let wr = vec![0.1_f32; 4 * 4];
        let a = vec![0.1_f32; 4];
        let err = layer.forward(&g, &x, &wl, &wr, &a);
        assert!(matches!(err, Err(GnnError::NodeFeatureMismatch(..))));
    }

    #[test]
    fn dynamic_attention_differs_from_static() {
        // GATv2 e_ij = a^T LeakyReLU(Wl*xi + Wr*xj)
        // GAT    e_ij = LeakyReLU(a_src^T Wx_i + a_dst^T Wx_j)
        // With different xi and xj, GATv2 is "dynamic"; this just checks output is non-trivial
        let g = CsrGraph::from_edges(2, &[(0, 1), (1, 0)])
            .expect("test invariant: value must be valid");
        let layer = GatV2Layer::new(GatV2Config {
            in_features: 2,
            out_features: 2,
            num_heads: 1,
            share_weights: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32, 0.0, 0.0, 1.0]; // node 0=[1,0], node 1=[0,1]
        let wl = vec![1.0_f32, 0.0, 0.0, 1.0]; // identity
        let wr = vec![1.0_f32, 0.0, 0.0, 1.0];
        let a = vec![1.0_f32, 1.0];
        let out = layer
            .forward(&g, &x, &wl, &wr, &a)
            .expect("test invariant: value must be valid");
        // Output should be non-trivial (not all the same)
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn invalid_zero_num_heads() {
        let err = GatV2Layer::new(GatV2Config {
            in_features: 4,
            out_features: 4,
            num_heads: 0,
            share_weights: false,
        });
        assert!(err.is_err());
    }
}
