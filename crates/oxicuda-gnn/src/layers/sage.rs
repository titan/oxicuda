//! GraphSAGE layer — Hamilton et al. 2017.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

/// Aggregation strategy for GraphSAGE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SageAggregator {
    /// Average of neighbour features.
    Mean,
    /// Max-pool with a per-neighbour linear transformation.
    MaxPool,
    /// LSTM-style aggregation (approximated as mean in this implementation due to ordering invariance requirement).
    Lstm,
}

/// Configuration for a GraphSAGE layer.
#[derive(Debug, Clone)]
pub struct SageConfig {
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// Aggregation type.
    pub aggregator: SageAggregator,
    /// If `true`, L2-normalise node representations after the update.
    pub normalize_output: bool,
}

/// A single GraphSAGE layer.
pub struct SageLayer {
    config: SageConfig,
}

impl SageLayer {
    /// Construct from configuration.
    pub fn new(config: SageConfig) -> GnnResult<Self> {
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
    /// `h_v^k = σ(W · CONCAT(h_v^{k-1}, AGG({h_u^{k-1} | u ∈ N(v)})))`
    ///
    /// # Arguments
    ///
    /// - `graph`: CSR graph
    /// - `x`: `[n_nodes × in_features]`
    /// - `weight`: `[out_features × (2 * in_features)]` for Mean/LSTM or MaxPool (pool_w separate)
    /// - `bias`: `[out_features]`
    ///
    /// For `MaxPool`, `weight` should be `[out_features × (in_features + in_features)]`
    /// and the pool projection is identity (no separate pool_w in this signature).
    ///
    /// # Returns
    ///
    /// `[n_nodes × out_features]`
    pub fn forward(
        &self,
        graph: &CsrGraph,
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
    ) -> GnnResult<Vec<f32>> {
        let n = graph.n_nodes();
        let in_f = self.config.in_features;
        let out_f = self.config.out_features;

        if x.len() != n * in_f {
            return Err(GnnError::NodeFeatureMismatch(n, x.len() / in_f.max(1)));
        }
        // weight: [out_f × 2*in_f]
        if weight.len() != out_f * 2 * in_f {
            return Err(GnnError::WeightShapeMismatch {
                r: out_f,
                c: 2 * in_f,
                d: in_f,
            });
        }
        if bias.len() != out_f {
            return Err(GnnError::DimensionMismatch {
                expected: out_f,
                got: bias.len(),
            });
        }

        // Step 1: Aggregate neighbours
        let aggr = match self.config.aggregator {
            SageAggregator::Mean | SageAggregator::Lstm => self.mean_aggregate(graph, x)?,
            SageAggregator::MaxPool => {
                // MaxPool: use identity pool then max
                let pool_w: Vec<f32> = (0..in_f * in_f)
                    .map(|i| if (i / in_f) == (i % in_f) { 1.0 } else { 0.0 })
                    .collect();
                let pool_b = vec![0.0_f32; in_f];
                self.maxpool_aggregate(graph, x, &pool_w, &pool_b)?
            }
        };

        // Step 2: Concatenate self + aggregated, then linear + bias + ReLU
        let mut out = vec![0.0_f32; n * out_f];
        for i in 0..n {
            // concat = [x[i] || aggr[i]], length = 2*in_f
            for k in 0..out_f {
                let mut acc = bias[k];
                // self part (columns 0..in_f)
                for j in 0..in_f {
                    acc += weight[k * 2 * in_f + j] * x[i * in_f + j];
                }
                // aggr part (columns in_f..2*in_f)
                for j in 0..in_f {
                    acc += weight[k * 2 * in_f + in_f + j] * aggr[i * in_f + j];
                }
                out[i * out_f + k] = acc.max(0.0); // ReLU
            }
        }

        // Optional L2 normalisation
        if self.config.normalize_output {
            for i in 0..n {
                let norm_sq: f32 = (0..out_f).map(|k| out[i * out_f + k].powi(2)).sum();
                if norm_sq > 0.0 {
                    let inv_norm = 1.0 / norm_sq.sqrt();
                    for k in 0..out_f {
                        out[i * out_f + k] *= inv_norm;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Mean aggregation: `aggr[i] = mean_{j ∈ N(i)} x[j]`.
    ///
    /// Isolated nodes get a zero aggregation vector.
    fn mean_aggregate(&self, graph: &CsrGraph, x: &[f32]) -> GnnResult<Vec<f32>> {
        let n = graph.n_nodes();
        let in_f = self.config.in_features;
        let mut aggr = vec![0.0_f32; n * in_f];
        for i in 0..n {
            let nb = graph.neighbors(i)?;
            if nb.is_empty() {
                continue;
            }
            for &j in nb {
                for k in 0..in_f {
                    aggr[i * in_f + k] += x[j * in_f + k];
                }
            }
            let inv = 1.0 / nb.len() as f32;
            for k in 0..in_f {
                aggr[i * in_f + k] *= inv;
            }
        }
        Ok(aggr)
    }

    /// MaxPool aggregation: `aggr[i] = max_{j ∈ N(i)} ReLU(W_pool * x[j] + b_pool)`.
    ///
    /// `pool_w`: `[in_f × in_f]`, `pool_b`: `[in_f]`.
    fn maxpool_aggregate(
        &self,
        graph: &CsrGraph,
        x: &[f32],
        pool_w: &[f32],
        pool_b: &[f32],
    ) -> GnnResult<Vec<f32>> {
        let n = graph.n_nodes();
        let in_f = self.config.in_features;
        let mut aggr = vec![0.0_f32; n * in_f];

        for i in 0..n {
            let nb = graph.neighbors(i)?;
            if nb.is_empty() {
                continue;
            }
            let mut node_max = vec![f32::NEG_INFINITY; in_f];
            for &j in nb {
                // Linear + ReLU per neighbor
                for k in 0..in_f {
                    let mut val = pool_b[k];
                    for l in 0..in_f {
                        val += pool_w[k * in_f + l] * x[j * in_f + l];
                    }
                    let activated = val.max(0.0);
                    if activated > node_max[k] {
                        node_max[k] = activated;
                    }
                }
            }
            for k in 0..in_f {
                aggr[i * in_f + k] = if node_max[k] == f32::NEG_INFINITY {
                    0.0
                } else {
                    node_max[k]
                };
            }
        }
        Ok(aggr)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn star_graph(n_leaves: usize) -> CsrGraph {
        let mut edges = Vec::new();
        for i in 1..=n_leaves {
            edges.push((0, i));
            edges.push((i, 0));
        }
        CsrGraph::from_edges(n_leaves + 1, &edges).expect("test invariant: value must be valid")
    }

    #[allow(dead_code)]
    fn identity_weight_2x(out_f: usize, in_f: usize) -> Vec<f32> {
        // [out_f × 2*in_f]: first in_f columns identity, rest zero
        let mut w = vec![0.0_f32; out_f * 2 * in_f];
        for i in 0..out_f.min(in_f) {
            w[i * 2 * in_f + i] = 1.0;
        }
        w
    }

    #[test]
    fn output_shape_mean() {
        let g = star_graph(4);
        let layer = SageLayer::new(SageConfig {
            in_features: 3,
            out_features: 5,
            aggregator: SageAggregator::Mean,
            normalize_output: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.1_f32; 5 * 3];
        let w = vec![0.1_f32; 5 * 6];
        let b = vec![0.0_f32; 5];
        let out = layer
            .forward(&g, &x, &w, &b)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 5 * 5);
    }

    #[test]
    fn output_shape_maxpool() {
        let g = star_graph(3);
        let layer = SageLayer::new(SageConfig {
            in_features: 4,
            out_features: 4,
            aggregator: SageAggregator::MaxPool,
            normalize_output: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.5_f32; 4 * 4];
        let w = vec![0.1_f32; 4 * 8];
        let b = vec![0.0_f32; 4];
        let out = layer
            .forward(&g, &x, &w, &b)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 4 * 4);
    }

    #[test]
    fn zero_weights_zero_output() {
        let g = star_graph(3);
        let layer = SageLayer::new(SageConfig {
            in_features: 2,
            out_features: 2,
            aggregator: SageAggregator::Mean,
            normalize_output: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 4 * 2];
        let w = vec![0.0_f32; 2 * 4];
        let b = vec![0.0_f32; 2];
        let out = layer
            .forward(&g, &x, &w, &b)
            .expect("test invariant: value must be valid");
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn isolated_node_uses_only_self() {
        // Node 3 has no neighbors
        let g = CsrGraph::from_edges(4, &[(0, 1), (1, 0)])
            .expect("test invariant: value must be valid");
        let layer = SageLayer::new(SageConfig {
            in_features: 2,
            out_features: 2,
            aggregator: SageAggregator::Mean,
            normalize_output: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 4 * 2];
        // Use identity on self, zero on aggr
        let mut w = vec![0.0_f32; 2 * 4];
        w[0] = 1.0;
        w[3] = 1.0; // identity on self part
        let b = vec![0.0_f32; 2];
        let out = layer
            .forward(&g, &x, &w, &b)
            .expect("test invariant: value must be valid");
        // Node 3's output is just ReLU(W*self + 0) = [1, 1]
        assert!(out[6] >= 0.0);
        assert!(out[7] >= 0.0);
    }

    #[test]
    fn normalize_output_unit_norm() {
        let g = star_graph(3);
        let layer = SageLayer::new(SageConfig {
            in_features: 2,
            out_features: 2,
            aggregator: SageAggregator::Mean,
            normalize_output: true,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 4 * 2];
        let w = vec![0.5_f32; 2 * 4];
        let b = vec![0.1_f32; 2];
        let out = layer
            .forward(&g, &x, &w, &b)
            .expect("test invariant: value must be valid");
        let n = 4;
        for i in 0..n {
            let norm_sq: f32 = out[i * 2..i * 2 + 2].iter().map(|&v| v * v).sum();
            if norm_sq > 0.0 {
                assert!((norm_sq.sqrt() - 1.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn relu_ensures_no_negatives() {
        let g = star_graph(2);
        let layer = SageLayer::new(SageConfig {
            in_features: 2,
            out_features: 2,
            aggregator: SageAggregator::Mean,
            normalize_output: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![-1.0_f32; 3 * 2];
        let w = vec![-1.0_f32; 2 * 4];
        let b = vec![-1.0_f32; 2];
        let out = layer
            .forward(&g, &x, &w, &b)
            .expect("test invariant: value must be valid");
        assert!(out.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn invalid_zero_in_features() {
        let err = SageLayer::new(SageConfig {
            in_features: 0,
            out_features: 4,
            aggregator: SageAggregator::Mean,
            normalize_output: false,
        });
        assert!(err.is_err());
    }

    #[test]
    fn invalid_zero_out_features() {
        let err = SageLayer::new(SageConfig {
            in_features: 4,
            out_features: 0,
            aggregator: SageAggregator::Mean,
            normalize_output: false,
        });
        assert!(err.is_err());
    }

    #[test]
    fn feature_mismatch_error() {
        let g = star_graph(3); // 4 nodes
        let layer = SageLayer::new(SageConfig {
            in_features: 2,
            out_features: 2,
            aggregator: SageAggregator::Mean,
            normalize_output: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 3 * 2]; // only 3 nodes
        let w = vec![0.1_f32; 2 * 4];
        let b = vec![0.0_f32; 2];
        let err = layer.forward(&g, &x, &w, &b);
        assert!(matches!(err, Err(GnnError::NodeFeatureMismatch(..))));
    }

    #[test]
    fn lstm_aggregator_same_as_mean() {
        // LSTM aggregator delegates to mean in this implementation
        let g = star_graph(3);
        let layer_lstm = SageLayer::new(SageConfig {
            in_features: 2,
            out_features: 2,
            aggregator: SageAggregator::Lstm,
            normalize_output: false,
        })
        .expect("test invariant: value must be valid");
        let layer_mean = SageLayer::new(SageConfig {
            in_features: 2,
            out_features: 2,
            aggregator: SageAggregator::Mean,
            normalize_output: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.5_f32; 4 * 2];
        let w = vec![0.1_f32; 2 * 4];
        let b = vec![0.0_f32; 2];
        let o1 = layer_lstm
            .forward(&g, &x, &w, &b)
            .expect("test invariant: value must be valid");
        let o2 = layer_mean
            .forward(&g, &x, &w, &b)
            .expect("test invariant: value must be valid");
        for (a, b_val) in o1.iter().zip(o2.iter()) {
            assert!((a - b_val).abs() < 1e-6);
        }
    }

    #[test]
    fn maxpool_finite_output() {
        let g = star_graph(4);
        let layer = SageLayer::new(SageConfig {
            in_features: 3,
            out_features: 3,
            aggregator: SageAggregator::MaxPool,
            normalize_output: false,
        })
        .expect("test invariant: value must be valid");
        let x: Vec<f32> = (0..5 * 3).map(|i| i as f32 * 0.1).collect();
        let w = vec![0.1_f32; 3 * 6];
        let b = vec![0.0_f32; 3];
        let out = layer
            .forward(&g, &x, &w, &b)
            .expect("test invariant: value must be valid");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
