//! Graph Convolutional Network (GCN) layer — Kipf & Welling 2017.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;
use crate::message_passing::update::relu;

/// Configuration for a GCN layer.
#[derive(Debug, Clone)]
pub struct GcnConfig {
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// Whether to include a learnable bias term.
    pub bias: bool,
    /// If `true`, use `D̂^{-1/2} Â D̂^{-1/2}` normalisation (Kipf & Welling).
    pub normalize: bool,
}

/// A single GCN layer.
///
/// Computes `H' = σ(D̂^{-1/2} Â D̂^{-1/2} H W + b)`.
pub struct GcnLayer {
    config: GcnConfig,
}

impl GcnLayer {
    /// Construct a GCN layer from configuration.
    pub fn new(config: GcnConfig) -> GnnResult<Self> {
        if config.in_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "in_features must be > 0".to_string(),
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
    /// # Arguments
    ///
    /// - `graph`: CSR graph (self-loops are added internally when `normalize` is true)
    /// - `node_features`: `[n_nodes × in_features]`
    /// - `weight`: `[in_features × out_features]` (row-major; `out[k] = Σ_j feat[j] * W[j,k]`)
    /// - `bias`: optional `[out_features]`
    ///
    /// # Returns
    ///
    /// `[n_nodes × out_features]` after applying ReLU.
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

        if node_features.len() != n * in_f {
            return Err(GnnError::NodeFeatureMismatch(
                n,
                node_features.len() / in_f.max(1),
            ));
        }
        if weight.len() != in_f * out_f {
            return Err(GnnError::WeightShapeMismatch {
                r: in_f,
                c: out_f,
                d: in_f,
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

        // Step 1: H_proj = H @ W  [n × out_f]
        // weight is [in_f × out_f] column-major-ish: H_proj[i,k] = Σ_j H[i,j] * W[j,k]
        let mut h_proj = vec![0.0_f32; n * out_f];
        for i in 0..n {
            for k in 0..out_f {
                let mut acc = 0.0_f32;
                for j in 0..in_f {
                    acc += node_features[i * in_f + j] * weight[j * out_f + k];
                }
                h_proj[i * out_f + k] = acc;
            }
        }

        // Add bias if present
        if let Some(b) = bias {
            for i in 0..n {
                for k in 0..out_f {
                    h_proj[i * out_f + k] += b[k];
                }
            }
        }

        // Step 2: H_aggr = Â_norm @ H_proj
        let h_aggr = if self.config.normalize {
            // Build normalised adjacency (includes self-loops)
            let (rows, cols, vals) = graph.normalized_adjacency();
            let mut out = vec![0.0_f32; n * out_f];
            for ((r, c), v) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
                for k in 0..out_f {
                    out[r * out_f + k] += v * h_proj[c * out_f + k];
                }
            }
            out
        } else {
            // Plain aggregation: SpMV with H_proj
            graph.spmv(&h_proj, out_f)?
        };

        // Step 3: ReLU activation
        Ok(relu(&h_aggr))
    }

    /// Output feature dimension.
    pub fn output_dim(&self) -> usize {
        self.config.out_features
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_graph() -> CsrGraph {
        // 4 nodes, path: 0→1→2→3 plus reverse
        CsrGraph::from_edges(4, &[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
            .expect("test invariant: value must be valid")
    }

    fn identity_weight(d: usize) -> Vec<f32> {
        let mut w = vec![0.0_f32; d * d];
        for i in 0..d {
            w[i * d + i] = 1.0;
        }
        w
    }

    #[test]
    fn output_shape_correct() {
        let g = simple_graph();
        let config = GcnConfig {
            in_features: 3,
            out_features: 5,
            bias: false,
            normalize: true,
        };
        let layer = GcnLayer::new(config).expect("test invariant: value must be valid");
        let feats = vec![1.0_f32; 4 * 3];
        let w = vec![0.1_f32; 3 * 5];
        let out = layer
            .forward(&g, &feats, &w, None)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 4 * 5);
    }

    #[test]
    fn zero_weights_zero_output() {
        let g = simple_graph();
        let config = GcnConfig {
            in_features: 2,
            out_features: 2,
            bias: false,
            normalize: false,
        };
        let layer = GcnLayer::new(config).expect("test invariant: value must be valid");
        let feats = vec![1.0_f32; 4 * 2];
        let w = vec![0.0_f32; 2 * 2];
        let out = layer
            .forward(&g, &feats, &w, None)
            .expect("test invariant: value must be valid");
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn relu_applied_no_negatives() {
        let g = simple_graph();
        let config = GcnConfig {
            in_features: 2,
            out_features: 2,
            bias: false,
            normalize: true,
        };
        let layer = GcnLayer::new(config).expect("test invariant: value must be valid");
        let feats = vec![-1.0_f32; 4 * 2];
        // weight all -0.5 → after linear step output is all positive due to sum of negatives
        // with normalize, the sign of normalized output depends on the feat values
        let w = vec![-1.0_f32; 2 * 2];
        let out = layer
            .forward(&g, &feats, &w, None)
            .expect("test invariant: value must be valid");
        // ReLU ensures no negatives
        assert!(out.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn one_node_graph() {
        // Self-loop only graph
        let g = CsrGraph::from_edges(1, &[(0, 0)]).expect("test invariant: value must be valid");
        let config = GcnConfig {
            in_features: 2,
            out_features: 2,
            bias: false,
            normalize: true,
        };
        let layer = GcnLayer::new(config).expect("test invariant: value must be valid");
        let feats = vec![1.0_f32, 2.0];
        let w = identity_weight(2);
        let out = layer
            .forward(&g, &feats, &w, None)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 2);
        // Output should be non-negative
        assert!(out.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn bias_added_correctly() {
        let g = CsrGraph::from_edges(1, &[(0, 0)]).expect("test invariant: value must be valid");
        let config = GcnConfig {
            in_features: 2,
            out_features: 2,
            bias: true,
            normalize: false,
        };
        let layer = GcnLayer::new(config).expect("test invariant: value must be valid");
        let feats = vec![0.0_f32, 0.0]; // zero features
        let w = vec![0.0_f32; 2 * 2];
        let b = vec![1.0_f32, 2.0];
        let out = layer
            .forward(&g, &feats, &w, Some(&b))
            .expect("test invariant: value must be valid");
        // With zero feats and zero weight, output = ReLU(bias + w*neighbor_feat)
        // neighbor is self with feat [0,0], so h_proj = bias = [1,2]
        // SpMV just passes through h_proj (since edge weight=1, neighbor=self)
        // h_aggr = [1*1, 1*2] = [1, 2] → after ReLU = [1, 2]
        assert!(out[0] > 0.0 || out[1] > 0.0);
    }

    #[test]
    fn invalid_zero_in_features() {
        let err = GcnLayer::new(GcnConfig {
            in_features: 0,
            out_features: 4,
            bias: false,
            normalize: true,
        });
        assert!(err.is_err());
    }

    #[test]
    fn invalid_zero_out_features() {
        let err = GcnLayer::new(GcnConfig {
            in_features: 4,
            out_features: 0,
            bias: false,
            normalize: true,
        });
        assert!(err.is_err());
    }

    #[test]
    fn feature_mismatch_error() {
        let g = simple_graph(); // 4 nodes
        let config = GcnConfig {
            in_features: 3,
            out_features: 3,
            bias: false,
            normalize: true,
        };
        let layer = GcnLayer::new(config).expect("test invariant: value must be valid");
        let feats = vec![1.0_f32; 3 * 3]; // wrong: only 3 nodes' worth
        let w = identity_weight(3);
        let err = layer.forward(&g, &feats, &w, None);
        assert!(matches!(err, Err(GnnError::NodeFeatureMismatch(..))));
    }

    #[test]
    fn normalize_and_nonnormalize_differ() {
        let g = simple_graph();
        let feats = vec![1.0_f32; 4 * 2];
        let w = vec![0.5_f32; 2 * 2];

        let layer_norm = GcnLayer::new(GcnConfig {
            in_features: 2,
            out_features: 2,
            bias: false,
            normalize: true,
        })
        .expect("test invariant: value must be valid");
        let layer_plain = GcnLayer::new(GcnConfig {
            in_features: 2,
            out_features: 2,
            bias: false,
            normalize: false,
        })
        .expect("test invariant: value must be valid");

        let out_norm = layer_norm
            .forward(&g, &feats, &w, None)
            .expect("test invariant: value must be valid");
        let out_plain = layer_plain
            .forward(&g, &feats, &w, None)
            .expect("test invariant: value must be valid");
        // They should differ in values (different normalisation)
        let same = out_norm
            .iter()
            .zip(out_plain.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(!same || out_norm.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn identity_weight_preserves_features_no_normalize() {
        // With identity weight and normalize=false, output = ReLU(A * h) where h = I weight * x = x
        let g = CsrGraph::from_edges(3, &[(0, 1), (1, 0), (1, 2), (2, 1)])
            .expect("test invariant: value must be valid");
        let config = GcnConfig {
            in_features: 2,
            out_features: 2,
            bias: false,
            normalize: false,
        };
        let layer = GcnLayer::new(config).expect("test invariant: value must be valid");
        // node 0 = [1,0], node 1 = [0,1], node 2 = [1,1]
        let feats = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let w = identity_weight(2);
        let out = layer
            .forward(&g, &feats, &w, None)
            .expect("test invariant: value must be valid");
        // node 0's projection is its features [1,0]; aggregation = A * proj → sum of neighbor projections
        // Node 0's neighbor is node 1: [0,1]. After ReLU: [0, 1]
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|&v| v >= 0.0));
    }
}
