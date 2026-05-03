//! Graph Isomorphism Network (GIN) layer — Xu et al. 2019.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

/// Configuration for a GIN layer.
#[derive(Debug, Clone)]
pub struct GinConfig {
    /// Input feature dimension.
    pub in_features: usize,
    /// Hidden dimension of the two-layer MLP.
    pub hidden_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// ε: self-loop weighting factor.
    pub epsilon: f32,
    /// If `true`, ε is treated as a learnable parameter (not used in this
    /// pure-Rust implementation, which simply uses the fixed value from config).
    pub train_epsilon: bool,
}

/// A single GIN layer.
pub struct GinLayer {
    config: GinConfig,
}

impl GinLayer {
    /// Construct from configuration.
    pub fn new(config: GinConfig) -> GnnResult<Self> {
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

    /// Forward pass.
    ///
    /// `h_v^k = MLP^k((1 + ε^k) * h_v^{k-1} + Σ_{u ∈ N(v)} h_u^{k-1})`
    ///
    /// The MLP is two linear layers separated by BatchNorm + ReLU:
    /// `Linear(in_f → hidden) → BatchNorm → ReLU → Linear(hidden → out) → BatchNorm`
    ///
    /// # Arguments
    ///
    /// - `graph`: CSR graph
    /// - `x`: `[n_nodes × in_features]`
    /// - `w1`: `[hidden_features × in_features]`
    /// - `b1`: `[hidden_features]`
    /// - `w2`: `[out_features × hidden_features]`
    /// - `b2`: `[out_features]`
    ///
    /// # Returns
    ///
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
        let eps = self.config.epsilon;

        if x.len() != n * in_f {
            return Err(GnnError::NodeFeatureMismatch(n, x.len() / in_f.max(1)));
        }
        if w1.len() != hid * in_f {
            return Err(GnnError::WeightShapeMismatch {
                r: hid,
                c: in_f,
                d: in_f,
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

        // Step 1: sum aggregation over neighbours
        let mut aggr = vec![0.0_f32; n * in_f];
        for i in 0..n {
            let nb = graph.neighbors(i)?;
            for &j in nb {
                for k in 0..in_f {
                    aggr[i * in_f + k] += x[j * in_f + k];
                }
            }
        }

        // Step 2: combine: combined[i] = (1 + eps) * x[i] + aggr[i]
        let mut combined = vec![0.0_f32; n * in_f];
        for i in 0..n {
            for k in 0..in_f {
                combined[i * in_f + k] = (1.0 + eps) * x[i * in_f + k] + aggr[i * in_f + k];
            }
        }

        // Step 3: MLP(combined)
        self.mlp(&combined, n, w1, b1, w2, b2)
    }

    /// Two-layer MLP with BatchNorm between layers.
    ///
    /// `Linear(in_f → hid) → BatchNorm → ReLU → Linear(hid → out) → BatchNorm`
    fn mlp(
        &self,
        x: &[f32],
        n_nodes: usize,
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
    ) -> GnnResult<Vec<f32>> {
        let in_f = self.config.in_features;
        let hid = self.config.hidden_features;
        let out_f = self.config.out_features;

        // Layer 1: [n × in_f] → [n × hid]
        let mut h1 = vec![0.0_f32; n_nodes * hid];
        for i in 0..n_nodes {
            for k in 0..hid {
                let mut acc = b1[k];
                for j in 0..in_f {
                    acc += w1[k * in_f + j] * x[i * in_f + j];
                }
                h1[i * hid + k] = acc;
            }
        }

        // BatchNorm 1
        let h1_bn = Self::batch_norm(&h1, n_nodes, hid);

        // ReLU
        let h1_act: Vec<f32> = h1_bn.iter().map(|&v| v.max(0.0)).collect();

        // Layer 2: [n × hid] → [n × out_f]
        let mut h2 = vec![0.0_f32; n_nodes * out_f];
        for i in 0..n_nodes {
            for k in 0..out_f {
                let mut acc = b2[k];
                for j in 0..hid {
                    acc += w2[k * hid + j] * h1_act[i * hid + j];
                }
                h2[i * out_f + k] = acc;
            }
        }

        // BatchNorm 2
        Ok(Self::batch_norm(&h2, n_nodes, out_f))
    }

    /// Per-feature BatchNorm (zero mean, unit variance) over the node batch.
    ///
    /// `x_hat[i,k] = (x[i,k] - μ_k) / (σ_k + ε)` where ε=1e-5.
    fn batch_norm(x: &[f32], n: usize, d: usize) -> Vec<f32> {
        if n == 0 || d == 0 {
            return x.to_vec();
        }
        let eps = 1e-5_f32;
        let mut out = x.to_vec();
        for k in 0..d {
            // Compute mean
            let mean: f32 = (0..n).map(|i| x[i * d + k]).sum::<f32>() / n as f32;
            // Compute variance
            let var: f32 = (0..n)
                .map(|i| {
                    let diff = x[i * d + k] - mean;
                    diff * diff
                })
                .sum::<f32>()
                / n as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..n {
                out[i * d + k] = (x[i * d + k] - mean) * inv_std;
            }
        }
        out
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
    fn output_shape_correct() {
        let g = ring_graph(5);
        let config = GinConfig {
            in_features: 4,
            hidden_features: 8,
            out_features: 3,
            epsilon: 0.0,
            train_epsilon: false,
        };
        let layer = GinLayer::new(config).expect("test invariant: value must be valid");
        let x = vec![0.1_f32; 5 * 4];
        let w1 = vec![0.1_f32; 8 * 4];
        let b1 = vec![0.0_f32; 8];
        let w2 = vec![0.1_f32; 3 * 8];
        let b2 = vec![0.0_f32; 3];
        let out = layer
            .forward(&g, &x, &w1, &b1, &w2, &b2)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 5 * 3);
    }

    #[test]
    fn epsilon_zero_is_pure_aggregation_plus_self() {
        // With epsilon=0, input to MLP is self + sum_neighbors
        // With epsilon=1, input to MLP is 2*self + sum_neighbors
        let g = CsrGraph::from_edges(2, &[(0, 1)]).expect("test invariant: value must be valid");
        let make_layer = |eps: f32| {
            GinLayer::new(GinConfig {
                in_features: 1,
                hidden_features: 1,
                out_features: 1,
                epsilon: eps,
                train_epsilon: false,
            })
            .expect("test invariant: value must be valid")
        };

        let x = vec![1.0_f32, 2.0]; // node 0 = 1, node 1 = 2
        let w1 = vec![1.0_f32]; // identity
        let b1 = vec![0.0_f32];
        let w2 = vec![1.0_f32];
        let b2 = vec![0.0_f32];

        let out_eps0 = make_layer(0.0)
            .forward(&g, &x, &w1, &b1, &w2, &b2)
            .expect("test invariant: value must be valid");
        let out_eps1 = make_layer(1.0)
            .forward(&g, &x, &w1, &b1, &w2, &b2)
            .expect("test invariant: value must be valid");

        // With eps=0: node 0 combined = 1*x[0] + aggr[0] = 1 + x[1] = 3
        // With eps=1: node 0 combined = 2*x[0] + aggr[0] = 2 + x[1] = 4
        // After batch norm and identity weights the difference is preserved in sign/order
        // Both outputs should differ
        let diff: f32 = out_eps0
            .iter()
            .zip(out_eps1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "eps=0 and eps=1 outputs must differ");
    }

    #[test]
    fn batch_norm_zero_mean() {
        // Batch norm output should have ~0 mean per feature
        let g = ring_graph(4);
        let config = GinConfig {
            in_features: 2,
            hidden_features: 4,
            out_features: 2,
            epsilon: 0.0,
            train_epsilon: false,
        };
        let layer = GinLayer::new(config).expect("test invariant: value must be valid");
        let x: Vec<f32> = (0..4 * 2).map(|i| i as f32).collect();
        let w1 = vec![0.1_f32; 4 * 2];
        let b1 = vec![0.0_f32; 4];
        let w2 = vec![0.1_f32; 2 * 4];
        let b2 = vec![0.0_f32; 2];
        let out = layer
            .forward(&g, &x, &w1, &b1, &w2, &b2)
            .expect("test invariant: value must be valid");
        // Final output is batch-normed, so mean over nodes should be ~0 per feature
        for k in 0..2 {
            let mean: f32 = (0..4).map(|i| out[i * 2 + k]).sum::<f32>() / 4.0;
            assert!(
                mean.abs() < 1e-4,
                "mean of feature {k} should be ~0, got {mean}"
            );
        }
    }

    #[test]
    fn output_finite_values() {
        let g = ring_graph(6);
        let config = GinConfig {
            in_features: 3,
            hidden_features: 6,
            out_features: 3,
            epsilon: 0.5,
            train_epsilon: true,
        };
        let layer = GinLayer::new(config).expect("test invariant: value must be valid");
        let x: Vec<f32> = (0..6 * 3).map(|i| i as f32 * 0.1).collect();
        let w1 = vec![0.05_f32; 6 * 3];
        let b1 = vec![0.0_f32; 6];
        let w2 = vec![0.05_f32; 3 * 6];
        let b2 = vec![0.0_f32; 3];
        let out = layer
            .forward(&g, &x, &w1, &b1, &w2, &b2)
            .expect("test invariant: value must be valid");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn invalid_zero_in_features() {
        let err = GinLayer::new(GinConfig {
            in_features: 0,
            hidden_features: 4,
            out_features: 4,
            epsilon: 0.0,
            train_epsilon: false,
        });
        assert!(err.is_err());
    }

    #[test]
    fn invalid_zero_hidden_features() {
        let err = GinLayer::new(GinConfig {
            in_features: 4,
            hidden_features: 0,
            out_features: 4,
            epsilon: 0.0,
            train_epsilon: false,
        });
        assert!(err.is_err());
    }

    #[test]
    fn feature_mismatch_error() {
        let g = ring_graph(4);
        let layer = GinLayer::new(GinConfig {
            in_features: 3,
            hidden_features: 6,
            out_features: 3,
            epsilon: 0.0,
            train_epsilon: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 3 * 3]; // only 3 nodes
        let w1 = vec![0.1_f32; 6 * 3];
        let b1 = vec![0.0_f32; 6];
        let w2 = vec![0.1_f32; 3 * 6];
        let b2 = vec![0.0_f32; 3];
        let err = layer.forward(&g, &x, &w1, &b1, &w2, &b2);
        assert!(matches!(err, Err(GnnError::NodeFeatureMismatch(..))));
    }

    #[test]
    fn negative_epsilon_affects_output() {
        let g = CsrGraph::from_edges(2, &[(0, 1)]).expect("test invariant: value must be valid");
        let layer_pos = GinLayer::new(GinConfig {
            in_features: 1,
            hidden_features: 2,
            out_features: 1,
            epsilon: 1.0,
            train_epsilon: false,
        })
        .expect("test invariant: value must be valid");
        let layer_neg = GinLayer::new(GinConfig {
            in_features: 1,
            hidden_features: 2,
            out_features: 1,
            epsilon: -0.5,
            train_epsilon: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32, 2.0];
        let w1 = vec![1.0_f32, 0.5];
        let b1 = vec![0.0_f32, 0.0];
        let w2 = vec![1.0_f32, 0.0];
        let b2 = vec![0.0_f32];
        let out_pos = layer_pos
            .forward(&g, &x, &w1, &b1, &w2, &b2)
            .expect("test invariant: value must be valid");
        let out_neg = layer_neg
            .forward(&g, &x, &w1, &b1, &w2, &b2)
            .expect("test invariant: value must be valid");
        let diff: f32 = out_pos
            .iter()
            .zip(out_neg.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn single_node_graph_works() {
        let g = CsrGraph::from_edges(1, &[(0, 0)]).expect("test invariant: value must be valid");
        let layer = GinLayer::new(GinConfig {
            in_features: 2,
            hidden_features: 4,
            out_features: 2,
            epsilon: 0.0,
            train_epsilon: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32, 2.0];
        let w1 = vec![0.1_f32; 4 * 2];
        let b1 = vec![0.0_f32; 4];
        let w2 = vec![0.1_f32; 2 * 4];
        let b2 = vec![0.0_f32; 2];
        let out = layer
            .forward(&g, &x, &w1, &b1, &w2, &b2)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 2);
    }
}
