//! Differentiable Pooling (DiffPool) — Ying et al. 2018.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

/// Configuration for DiffPool.
#[derive(Debug, Clone)]
pub struct DiffPoolConfig {
    /// Input node feature dimension.
    pub in_features: usize,
    /// Number of clusters (coarsened nodes).
    pub n_clusters: usize,
}

/// Result of a DiffPool forward pass.
#[derive(Debug, Clone)]
pub struct DiffPoolResult {
    /// Coarsened node features `[n_clusters × in_features]`.
    pub coarse_x: Vec<f32>,
    /// Coarsened (dense) adjacency matrix `[n_clusters × n_clusters]`.
    pub coarse_adj: Vec<f32>,
    /// Row-stochastic soft assignment matrix `[n × n_clusters]`.
    pub assignment: Vec<f32>,
    /// Link prediction loss `||A - S S^T||_F`.
    pub link_loss: f32,
    /// Entropy regularisation loss `(1/n) Σ_i H(S_i)`.
    pub entropy_loss: f32,
}

/// Differentiable Pooling layer.
pub struct DiffPool {
    config: DiffPoolConfig,
}

impl DiffPool {
    /// Construct from configuration.
    pub fn new(config: DiffPoolConfig) -> GnnResult<Self> {
        if config.in_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "in_features must be > 0".to_string(),
            ));
        }
        if config.n_clusters == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "n_clusters must be > 0".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Forward pass.
    ///
    /// # Algorithm
    ///
    /// 1. `S = softmax(s_logits)` row-wise where `s_logits[n × k]`
    /// 2. `X' = S^T X  [k × d]`
    /// 3. `A' = S^T A S  [k × k]`
    /// 4. Compute auxiliary losses
    ///
    /// # Arguments
    ///
    /// - `graph`: CSR graph
    /// - `x`: `[n × in_features]`
    /// - `s_logits`: `[n × n_clusters]` unnormalized cluster assignment logits
    ///
    /// # Returns
    ///
    /// [`DiffPoolResult`]
    pub fn forward(
        &self,
        graph: &CsrGraph,
        x: &[f32],
        s_logits: &[f32],
    ) -> GnnResult<DiffPoolResult> {
        let n = graph.n_nodes();
        let d = self.config.in_features;
        let k = self.config.n_clusters;

        if x.len() != n * d {
            return Err(GnnError::NodeFeatureMismatch(n, x.len() / d.max(1)));
        }
        if s_logits.len() != n * k {
            return Err(GnnError::DimensionMismatch {
                expected: n * k,
                got: s_logits.len(),
            });
        }

        // Step 1: Row-wise softmax of s_logits → S [n × k]
        let mut assignment = vec![0.0_f32; n * k];
        for i in 0..n {
            let row_start = i * k;
            let row = &s_logits[row_start..row_start + k];
            let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - max_v).exp()).collect();
            let sum_e: f32 = exps.iter().sum();
            let inv = if sum_e > 0.0 {
                1.0 / sum_e
            } else {
                1.0 / k as f32
            };
            for j in 0..k {
                assignment[row_start + j] = exps[j] * inv;
            }
        }

        // Step 2: X' = S^T X  [k × d]
        let mut coarse_x = vec![0.0_f32; k * d];
        for j in 0..k {
            for i in 0..n {
                let s_ij = assignment[i * k + j];
                for feat in 0..d {
                    coarse_x[j * d + feat] += s_ij * x[i * d + feat];
                }
            }
        }

        // Step 3: Build dense adjacency A [n × n] from CSR, then A' = S^T A S [k × k]
        // To avoid O(n^2) memory we compute S^T A S in two steps:
        // temp [k × n] = S^T A  (each entry: temp[j, col] = Σ_i S[i,j] * A[i, col])
        // A' [k × k] = temp S   (each entry: A'[j1, j2] = Σ_col temp[j1, col] * S[col, j2])

        // Compute temp [k × n]: iterate over CSR edges
        let mut temp = vec![0.0_f32; k * n];
        for i in 0..n {
            let start = graph.row_ptr()[i];
            let end = graph.row_ptr()[i + 1];
            for e in start..end {
                let col = graph.col_idx()[e];
                let w = graph.edge_weight()[e];
                for j in 0..k {
                    temp[j * n + col] += assignment[i * k + j] * w;
                }
            }
        }

        // Compute A' [k × k] = temp @ S
        let mut coarse_adj = vec![0.0_f32; k * k];
        for j1 in 0..k {
            for j2 in 0..k {
                let mut acc = 0.0_f32;
                for col in 0..n {
                    acc += temp[j1 * n + col] * assignment[col * k + j2];
                }
                coarse_adj[j1 * k + j2] = acc;
            }
        }

        // Step 4: Link prediction loss ||A - S S^T||_F²  (sum of squared diff)
        // S S^T [n × n]: entry (i, i2) = Σ_j S[i,j] * S[i2,j]
        // We compute this loss without materialising the full n×n matrix by iterating CSR edges
        let lp_loss = Self::link_prediction_loss(graph, &assignment, n, k);
        let ent_loss = Self::entropy_loss(&assignment, n, k);

        Ok(DiffPoolResult {
            coarse_x,
            coarse_adj,
            assignment,
            link_loss: lp_loss,
            entropy_loss: ent_loss,
        })
    }

    /// Link prediction loss: `||A - S S^T||_F`.
    ///
    /// Computed as `sqrt(Σ_{ij} (A[i,j] - (S S^T)[i,j])²)`.
    pub fn link_prediction_loss(a: &CsrGraph, s: &[f32], n: usize, k: usize) -> f32 {
        // Iterate over all (i, j) pairs in A and compute (A[i,j] - SST[i,j])²
        // For efficiency, only compute over edges and diagonal (rest of A is 0)
        let mut sq_sum = 0.0_f32;

        // Build set of (i, j) pairs that are edges
        let mut edge_set = std::collections::HashSet::new();
        for i in 0..n {
            for e in a.row_ptr()[i]..a.row_ptr()[i + 1] {
                let j = a.col_idx()[e];
                let w = a.edge_weight()[e];
                // SST[i,j] = Σ_l S[i,l] * S[j,l]
                let sst_ij: f32 = (0..k).map(|l| s[i * k + l] * s[j * k + l]).sum();
                sq_sum += (w - sst_ij).powi(2);
                edge_set.insert((i, j));
            }
        }
        // For all non-edge (i,j), A[i,j]=0 so contribution = SST[i,j]²
        // To avoid O(n²), approximate using Frobenius of SST: ||SST||_F² - Σ_{edges} SST[i,j]²
        // Then total ≈ sq_sum + ||SST||_F² - Σ_{edges} SST[i,j]²
        // Compute ||SST||_F² = tr(SST * SST) = ||S^T S||_F²  [k×k matrix]
        let mut sts = vec![0.0_f32; k * k]; // S^T S
        for i in 0..n {
            for l1 in 0..k {
                for l2 in 0..k {
                    sts[l1 * k + l2] += s[i * k + l1] * s[i * k + l2];
                }
            }
        }
        let frob_sst_sq: f32 = (0..k)
            .map(|l1| (0..k).map(|l2| sts[l1 * k + l2].powi(2)).sum::<f32>())
            .sum();

        // Subtract edge SST contributions (already counted)
        let edge_sst_sq: f32 = edge_set
            .iter()
            .map(|&(i, j)| {
                let sst_ij: f32 = (0..k).map(|l| s[i * k + l] * s[j * k + l]).sum();
                sst_ij.powi(2)
            })
            .sum();

        let total_sq = sq_sum + frob_sst_sq - edge_sst_sq;
        total_sq.max(0.0).sqrt()
    }

    /// Entropy regularisation loss: `(1/n) Σ_i H(S_i)`.
    ///
    /// Penalises soft (uncertain) assignments by encouraging each row to be close to a one-hot vector.
    pub fn entropy_loss(s: &[f32], n: usize, k: usize) -> f32 {
        if n == 0 || k == 0 {
            return 0.0;
        }
        let eps = 1e-10_f32;
        let mut total_entropy = 0.0_f32;
        for i in 0..n {
            let mut h = 0.0_f32;
            for j in 0..k {
                let p = s[i * k + j].max(eps);
                h -= p * p.ln();
            }
            total_entropy += h;
        }
        total_entropy / n as f32
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_graph(n: usize) -> CsrGraph {
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| [(i, (i + 1) % n), ((i + 1) % n, i)])
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        CsrGraph::from_edges(n, &edges).expect("test invariant: value must be valid")
    }

    #[test]
    fn coarse_x_shape() {
        let g = ring_graph(6);
        let d = 4;
        let k = 2;
        let config = DiffPoolConfig {
            in_features: d,
            n_clusters: k,
        };
        let dp = DiffPool::new(config).expect("test invariant: value must be valid");
        let x = vec![0.1_f32; 6 * d];
        let logits = vec![0.5_f32; 6 * k];
        let res = dp
            .forward(&g, &x, &logits)
            .expect("test invariant: value must be valid");
        assert_eq!(res.coarse_x.len(), k * d);
    }

    #[test]
    fn coarse_adj_shape() {
        let g = ring_graph(4);
        let k = 2;
        let config = DiffPoolConfig {
            in_features: 3,
            n_clusters: k,
        };
        let dp = DiffPool::new(config).expect("test invariant: value must be valid");
        let x = vec![0.1_f32; 4 * 3];
        let logits = vec![0.5_f32; 4 * k];
        let res = dp
            .forward(&g, &x, &logits)
            .expect("test invariant: value must be valid");
        assert_eq!(res.coarse_adj.len(), k * k);
    }

    #[test]
    fn assignment_rows_sum_to_one() {
        let g = ring_graph(5);
        let k = 3;
        let config = DiffPoolConfig {
            in_features: 2,
            n_clusters: k,
        };
        let dp = DiffPool::new(config).expect("test invariant: value must be valid");
        let x = vec![0.1_f32; 5 * 2];
        let logits: Vec<f32> = (0..5 * k).map(|i| i as f32 * 0.1).collect();
        let res = dp
            .forward(&g, &x, &logits)
            .expect("test invariant: value must be valid");
        for i in 0..5 {
            let row_sum: f32 = res.assignment[i * k..(i + 1) * k].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "row {i} sums to {row_sum}, not 1"
            );
        }
    }

    #[test]
    fn entropy_loss_non_negative() {
        let g = ring_graph(4);
        let k = 3;
        let config = DiffPoolConfig {
            in_features: 2,
            n_clusters: k,
        };
        let dp = DiffPool::new(config).expect("test invariant: value must be valid");
        let x = vec![0.5_f32; 4 * 2];
        let logits = vec![1.0_f32; 4 * k];
        let res = dp
            .forward(&g, &x, &logits)
            .expect("test invariant: value must be valid");
        assert!(res.entropy_loss >= 0.0);
    }

    #[test]
    fn entropy_loss_one_hot_assignment_zero() {
        // With very large positive logits for one cluster and very negative for others,
        // the assignment approaches one-hot → entropy ≈ 0
        let g = ring_graph(3);
        let k = 2;
        let config = DiffPoolConfig {
            in_features: 2,
            n_clusters: k,
        };
        let dp = DiffPool::new(config).expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 3 * 2];
        // All nodes assigned strongly to cluster 0
        let mut logits = vec![-100.0_f32; 3 * k];
        for i in 0..3 {
            logits[i * k] = 100.0; // cluster 0 gets very large logit
        }
        let res = dp
            .forward(&g, &x, &logits)
            .expect("test invariant: value must be valid");
        assert!(
            res.entropy_loss < 0.01,
            "entropy should be near 0, got {}",
            res.entropy_loss
        );
    }

    #[test]
    fn link_loss_non_negative() {
        let g = ring_graph(4);
        let k = 2;
        let config = DiffPoolConfig {
            in_features: 2,
            n_clusters: k,
        };
        let dp = DiffPool::new(config).expect("test invariant: value must be valid");
        let x = vec![0.1_f32; 4 * 2];
        let logits = vec![0.5_f32; 4 * k];
        let res = dp
            .forward(&g, &x, &logits)
            .expect("test invariant: value must be valid");
        assert!(res.link_loss >= 0.0);
    }

    #[test]
    fn uniform_features_coarse_x_finite() {
        let g = ring_graph(6);
        let k = 3;
        let config = DiffPoolConfig {
            in_features: 4,
            n_clusters: k,
        };
        let dp = DiffPool::new(config).expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 6 * 4];
        let logits = vec![0.0_f32; 6 * k]; // uniform logits → uniform assignment
        let res = dp
            .forward(&g, &x, &logits)
            .expect("test invariant: value must be valid");
        assert!(res.coarse_x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn coarse_adj_non_negative() {
        let g = ring_graph(4);
        let k = 2;
        let config = DiffPoolConfig {
            in_features: 2,
            n_clusters: k,
        };
        let dp = DiffPool::new(config).expect("test invariant: value must be valid");
        let x = vec![0.5_f32; 4 * 2];
        let logits = vec![0.5_f32; 4 * k];
        let res = dp
            .forward(&g, &x, &logits)
            .expect("test invariant: value must be valid");
        assert!(res.coarse_adj.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn invalid_zero_in_features() {
        let err = DiffPool::new(DiffPoolConfig {
            in_features: 0,
            n_clusters: 3,
        });
        assert!(err.is_err());
    }

    #[test]
    fn invalid_zero_clusters() {
        let err = DiffPool::new(DiffPoolConfig {
            in_features: 4,
            n_clusters: 0,
        });
        assert!(err.is_err());
    }

    #[test]
    fn node_feature_mismatch_error() {
        let g = ring_graph(4);
        let config = DiffPoolConfig {
            in_features: 3,
            n_clusters: 2,
        };
        let dp = DiffPool::new(config).expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 3 * 3]; // 3 nodes instead of 4
        let logits = vec![0.5_f32; 4 * 2];
        let err = dp.forward(&g, &x, &logits);
        assert!(matches!(err, Err(GnnError::NodeFeatureMismatch(..))));
    }
}
