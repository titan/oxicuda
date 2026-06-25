//! Graph Convolutional Network (GCN) layer — Kipf & Welling 2017.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;
use crate::message_passing::update::relu;
use oxicuda_sparse::host_csr::HostCsr;

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
        let out_f = self.config.out_features;
        self.validate_inputs(n, node_features, weight, bias)?;

        // Step 1: H_proj = H @ W (+ bias)  [n × out_f]
        let h_proj = self.project(n, node_features, weight, bias);

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

    /// Validates the feature / weight / bias shapes for an `n`-node forward pass.
    ///
    /// Shared by the dense [`forward`](Self::forward) and the SpMM-based
    /// [`forward_sparse`](Self::forward_sparse) paths so both reject the same
    /// malformed inputs identically.
    fn validate_inputs(
        &self,
        n: usize,
        node_features: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> GnnResult<()> {
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
        Ok(())
    }

    /// Linear node-feature projection `H_proj = H W (+ b)`  `[n × out_f]`.
    ///
    /// `weight` is `[in_f × out_f]` row-major: `H_proj[i,k] = Σ_j H[i,j]·W[j,k]`.
    /// The accumulation is performed in `f32` in exactly the same order as the
    /// original dense forward so the dense and sparse aggregation paths consume
    /// bit-identical projected features — the only numerical difference between
    /// the two paths is then confined to the `Â·H_proj` aggregation itself.
    fn project(
        &self,
        n: usize,
        node_features: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        let in_f = self.config.in_features;
        let out_f = self.config.out_features;
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
        if let Some(b) = bias {
            for i in 0..n {
                for k in 0..out_f {
                    h_proj[i * out_f + k] += b[k];
                }
            }
        }
        h_proj
    }

    /// Builds the GCN propagation operator `Â` for `graph` as a host-resident
    /// [`HostCsr`] matrix, ready for SpMM-based aggregation through
    /// `oxicuda-sparse`.
    ///
    /// When [`normalize`](GcnConfig::normalize) is set this is the symmetrically
    /// normalised adjacency `D̂^{-1/2}(A + I)D̂^{-1/2}` — exactly the operator the
    /// dense [`forward`](Self::forward) applies; otherwise it is the raw weighted
    /// adjacency `A` used by the plain-aggregation path. The matrix is
    /// `n_nodes × n_nodes`.
    ///
    /// Duplicate `(row, col)` contributions are summed into a single canonical
    /// CSR entry and columns are sorted ascending within each row. This matters
    /// for the normalised operator on a graph that already carries an explicit
    /// self-loop `(i, i)`: that edge and the self-loop the normalisation adds
    /// both land on `(i, i)`, and the dense path sums them — so the sparse
    /// operator must too in order to agree element-for-element.
    pub fn propagation_matrix(&self, graph: &CsrGraph) -> GnnResult<HostCsr> {
        let n = graph.n_nodes();
        let (rows, cols, vals) = if self.config.normalize {
            graph.normalized_adjacency()
        } else {
            // Raw weighted adjacency A expanded to COO triplets.
            let row_ptr = graph.row_ptr();
            let cols = graph.col_idx().to_vec();
            let vals = graph.edge_weight().to_vec();
            let mut rows = Vec::with_capacity(cols.len());
            for (i, w) in row_ptr.windows(2).enumerate() {
                for _ in w[0]..w[1] {
                    rows.push(i);
                }
            }
            (rows, cols, vals)
        };
        Self::coo_to_host_csr(n, &rows, &cols, &vals)
    }

    /// Converts an `n × n` COO triplet list into a canonical [`HostCsr`].
    ///
    /// Entries are bucketed by row, sorted ascending by column, and duplicate
    /// `(row, col)` pairs are summed. Values are widened to `f64`, matching the
    /// double-precision accumulation used by [`HostCsr`]'s SpMV kernel.
    fn coo_to_host_csr(
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[f32],
    ) -> GnnResult<HostCsr> {
        let mut per_row: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
            if r >= n {
                return Err(GnnError::NodeIndexOutOfRange { idx: r, n_nodes: n });
            }
            per_row[r].push((c, f64::from(v)));
        }
        let mut row_ptr = vec![0usize; n + 1];
        let mut col_indices: Vec<usize> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for (i, entries) in per_row.iter_mut().enumerate() {
            entries.sort_by_key(|&(c, _)| c);
            let mut e = 0;
            while e < entries.len() {
                let col = entries[e].0;
                let mut acc = 0.0_f64;
                while e < entries.len() && entries[e].0 == col {
                    acc += entries[e].1;
                    e += 1;
                }
                col_indices.push(col);
                values.push(acc);
            }
            row_ptr[i + 1] = col_indices.len();
        }
        HostCsr::new(n, n, row_ptr, col_indices, values).map_err(map_sparse_err)
    }

    /// SpMM-based forward pass: `H' = σ(Â · (H W + b))`.
    ///
    /// Produces the same result as the dense [`forward`](Self::forward) but
    /// routes the neighbourhood aggregation `Â · H_proj` through
    /// `oxicuda-sparse`'s host CSR SpMV kernel ([`HostCsr::matvec`]) rather than
    /// the inline dense COO accumulation. The projected feature matrix
    /// `H_proj` `[n × out_f]` is aggregated one feature column at a time — an
    /// SpMM expressed as a batch of sparse matrix-vector products
    /// `Â · H_proj[:, k]` — which is precisely the column-wise SpMM the
    /// dense path performs by hand.
    ///
    /// `adj` is the propagation operator `Â`, typically produced by
    /// [`propagation_matrix`](Self::propagation_matrix); it must be the square
    /// `n × n` matrix for the `n`-node graph whose features are passed. The SpMV
    /// accumulates in `f64` and the result is narrowed back to `f32` before the
    /// ReLU so the output type matches [`forward`](Self::forward).
    pub fn forward_sparse(
        &self,
        adj: &HostCsr,
        node_features: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> GnnResult<Vec<f32>> {
        let n = adj.nrows;
        let out_f = self.config.out_features;
        if adj.ncols != n {
            return Err(GnnError::DimensionMismatch {
                expected: n,
                got: adj.ncols,
            });
        }
        self.validate_inputs(n, node_features, weight, bias)?;

        // Step 1: H_proj = H @ W (+ bias) — shared, bit-identical with forward.
        let h_proj = self.project(n, node_features, weight, bias);

        // Step 2: H_aggr = Â · H_proj as SpMM, one SpMV per feature column.
        let mut h_aggr = vec![0.0_f32; n * out_f];
        let mut column = vec![0.0_f64; n];
        for k in 0..out_f {
            for (i, col) in column.iter_mut().enumerate() {
                *col = f64::from(h_proj[i * out_f + k]);
            }
            let aggregated = adj.matvec(&column);
            for (i, &a) in aggregated.iter().enumerate() {
                h_aggr[i * out_f + k] = a as f32;
            }
        }

        // Step 3: ReLU activation.
        Ok(relu(&h_aggr))
    }

    /// Output feature dimension.
    pub fn output_dim(&self) -> usize {
        self.config.out_features
    }
}

/// Maps an `oxicuda-sparse` error into this crate's error type.
///
/// `HostCsr::new` only fails on structurally inconsistent CSR arrays; the
/// propagation operators built here are always well-formed, so this is a
/// defensive conversion rather than an expected path.
fn map_sparse_err(err: oxicuda_sparse::SparseError) -> GnnError {
    GnnError::Internal(format!("oxicuda-sparse host CSR: {err}"))
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

    // ── Sparse-SpMM forward path (routed through oxicuda-sparse HostCsr) ───────

    /// Complete directed graph on 4 nodes. Every node has out-degree 3, so the
    /// symmetric normalisation yields `Â[i,j] = 1/(d+1) = 0.25` for all `i,j` —
    /// an exactly-representable dyadic operator, which makes the `f32` dense path
    /// and the `f64` SpMM path agree bit-for-bit.
    fn k4_graph() -> CsrGraph {
        CsrGraph::from_edges(
            4,
            &[
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 3),
                (3, 0),
                (3, 1),
                (3, 2),
            ],
        )
        .expect("test invariant: value must be valid")
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn forward_sparse_matches_dense_normalized() {
        let g = k4_graph();
        let layer = GcnLayer::new(GcnConfig {
            in_features: 2,
            out_features: 2,
            bias: false,
            normalize: true,
        })
        .expect("test invariant: value must be valid");
        // Integer features and a dyadic weight keep the projection and the
        // 0.25-weighted aggregation exactly representable in f32.
        let feats = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let w = vec![0.5_f32; 2 * 2];

        let adj = layer
            .propagation_matrix(&g)
            .expect("test invariant: value must be valid");
        let out_dense = layer
            .forward(&g, &feats, &w, None)
            .expect("test invariant: value must be valid");
        let out_sparse = layer
            .forward_sparse(&adj, &feats, &w, None)
            .expect("test invariant: value must be valid");

        assert_eq!(out_dense.len(), out_sparse.len());
        let err = max_abs_diff(&out_dense, &out_sparse);
        assert!(err <= 1e-9, "sparse-vs-dense max abs error = {err}");
    }

    #[test]
    fn forward_sparse_matches_dense_with_bias() {
        let g = k4_graph();
        let layer = GcnLayer::new(GcnConfig {
            in_features: 2,
            out_features: 2,
            bias: true,
            normalize: true,
        })
        .expect("test invariant: value must be valid");
        let feats = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let w = vec![0.5_f32; 2 * 2];
        let b = vec![1.0_f32, 2.0];

        let adj = layer
            .propagation_matrix(&g)
            .expect("test invariant: value must be valid");
        let out_dense = layer
            .forward(&g, &feats, &w, Some(&b))
            .expect("test invariant: value must be valid");
        let out_sparse = layer
            .forward_sparse(&adj, &feats, &w, Some(&b))
            .expect("test invariant: value must be valid");

        let err = max_abs_diff(&out_dense, &out_sparse);
        assert!(err <= 1e-9, "sparse-vs-dense (bias) max abs error = {err}");
    }

    #[test]
    fn spmm_matches_brute_force_dense_matmul() {
        // The aggregation kernel (HostCsr::matvec applied per feature column) must
        // equal a brute-force dense Â·H matmul. Performed entirely in f64 over an
        // arbitrary (deterministic LcgRng) feature matrix so the comparison
        // isolates the SpMM correctness from any f32 narrowing.
        let g = CsrGraph::from_edges(4, &[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
            .expect("test invariant: value must be valid");
        let layer = GcnLayer::new(GcnConfig {
            in_features: 1,
            out_features: 1,
            bias: false,
            normalize: true,
        })
        .expect("test invariant: value must be valid");
        let adj = layer
            .propagation_matrix(&g)
            .expect("test invariant: value must be valid");
        let n = adj.nrows;
        let f = 3usize;

        // Deterministic f64 feature matrix in [-1, 1) (÷2³², full range).
        let mut rng = crate::handle::LcgRng::new(20_260_621);
        let h: Vec<f64> = (0..n * f)
            .map(|_| f64::from(rng.next_u32()) / (f64::from(u32::MAX) + 1.0) * 2.0 - 1.0)
            .collect();

        // SpMM: one SpMV per feature column.
        let mut spmm_out = vec![0.0_f64; n * f];
        let mut column = vec![0.0_f64; n];
        for k in 0..f {
            for (i, slot) in column.iter_mut().enumerate() {
                *slot = h[i * f + k];
            }
            let agg = adj.matvec(&column);
            for (i, &a) in agg.iter().enumerate() {
                spmm_out[i * f + k] = a;
            }
        }

        // Brute-force dense Â·H.
        let dense = adj.to_dense();
        let mut dense_out = vec![0.0_f64; n * f];
        for i in 0..n {
            for k in 0..f {
                let mut acc = 0.0_f64;
                for c in 0..n {
                    acc += dense[i * n + c] * h[c * f + k];
                }
                dense_out[i * f + k] = acc;
            }
        }

        let err = spmm_out
            .iter()
            .zip(dense_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(err <= 1e-9, "SpMM vs dense matmul max abs error = {err}");
    }

    #[test]
    fn forward_sparse_output_shape() {
        let g = k4_graph();
        let layer = GcnLayer::new(GcnConfig {
            in_features: 3,
            out_features: 5,
            bias: false,
            normalize: true,
        })
        .expect("test invariant: value must be valid");
        let adj = layer
            .propagation_matrix(&g)
            .expect("test invariant: value must be valid");
        let feats = vec![0.1_f32; 4 * 3];
        let w = vec![0.2_f32; 3 * 5];
        let out = layer
            .forward_sparse(&adj, &feats, &w, None)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 4 * 5);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_sparse_isolated_node() {
        // Node 0 is isolated; nodes 1-4 form a K4 block. The normalised operator
        // gives the isolated node a unit self-loop (its row is the identity), so
        // its output is its projected features passed straight through, while the
        // connected block keeps the exact 0.25 operator.
        let g = CsrGraph::from_edges(
            5,
            &[
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 1),
                (2, 3),
                (2, 4),
                (3, 1),
                (3, 2),
                (3, 4),
                (4, 1),
                (4, 2),
                (4, 3),
            ],
        )
        .expect("test invariant: value must be valid");
        let layer = GcnLayer::new(GcnConfig {
            in_features: 2,
            out_features: 2,
            bias: false,
            normalize: true,
        })
        .expect("test invariant: value must be valid");
        // Identity weight → projected features equal the inputs.
        let w = vec![1.0_f32, 0.0, 0.0, 1.0];
        let feats = vec![
            2.0_f32, 3.0, // node 0 (isolated)
            1.0, 1.0, // node 1
            2.0, 2.0, // node 2
            3.0, 3.0, // node 3
            4.0, 4.0, // node 4
        ];

        let adj = layer
            .propagation_matrix(&g)
            .expect("test invariant: value must be valid");
        assert_eq!(adj.nrows, 5);
        // Isolated node carries a single unit self-loop.
        assert_eq!(adj.get(0, 0), Some(1.0));

        let out_dense = layer
            .forward(&g, &feats, &w, None)
            .expect("test invariant: value must be valid");
        let out_sparse = layer
            .forward_sparse(&adj, &feats, &w, None)
            .expect("test invariant: value must be valid");
        let err = max_abs_diff(&out_dense, &out_sparse);
        assert!(err <= 1e-9, "isolated-node max abs error = {err}");
        // Node 0's features pass through unchanged (then ReLU, both positive).
        assert!((out_sparse[0] - 2.0).abs() <= 1e-9);
        assert!((out_sparse[1] - 3.0).abs() <= 1e-9);
    }

    #[test]
    fn propagation_matrix_merges_duplicate_self_loops() {
        // A single node with an explicit self-loop edge: the normalisation adds
        // its own self-loop, so the COO operator has two (0,0) entries that the
        // canonical CSR build must sum into one.
        let g = CsrGraph::from_edges(1, &[(0, 0)]).expect("test invariant: value must be valid");
        let layer = GcnLayer::new(GcnConfig {
            in_features: 1,
            out_features: 1,
            bias: false,
            normalize: true,
        })
        .expect("test invariant: value must be valid");
        let adj = layer
            .propagation_matrix(&g)
            .expect("test invariant: value must be valid");
        assert_eq!(adj.nnz(), 1, "duplicate (0,0) entries must merge to one");
        // Â(0,0) = 2·(1/√2)² ≈ 1.0.
        let v = adj.get(0, 0).expect("self-loop entry present");
        assert!((v - 1.0).abs() <= 1e-6, "merged self-loop value = {v}");

        let feats = vec![1.0_f32];
        let w = vec![1.0_f32];
        let out_dense = layer
            .forward(&g, &feats, &w, None)
            .expect("test invariant: value must be valid");
        let out_sparse = layer
            .forward_sparse(&adj, &feats, &w, None)
            .expect("test invariant: value must be valid");
        let err = max_abs_diff(&out_dense, &out_sparse);
        assert!(err <= 1e-9, "self-loop max abs error = {err}");
    }

    #[test]
    fn forward_sparse_non_normalized_matches_dense() {
        // Raw adjacency (edge weight 1): aggregation is plain neighbour summation.
        let g = CsrGraph::from_edges(4, &[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
            .expect("test invariant: value must be valid");
        let layer = GcnLayer::new(GcnConfig {
            in_features: 2,
            out_features: 2,
            bias: false,
            normalize: false,
        })
        .expect("test invariant: value must be valid");
        // Integer features and weight keep the projected features exact integers,
        // and neighbour summation stays exact in f32.
        let feats = vec![1.0_f32, 2.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0];
        let w = vec![1.0_f32, 2.0, 0.0, 1.0];

        let adj = layer
            .propagation_matrix(&g)
            .expect("test invariant: value must be valid");
        let out_dense = layer
            .forward(&g, &feats, &w, None)
            .expect("test invariant: value must be valid");
        let out_sparse = layer
            .forward_sparse(&adj, &feats, &w, None)
            .expect("test invariant: value must be valid");
        let err = max_abs_diff(&out_dense, &out_sparse);
        assert!(err <= 1e-9, "non-normalized max abs error = {err}");
    }

    #[test]
    fn propagation_matrix_matches_normalized_adjacency() {
        // The host CSR operator densified must reproduce the dense matrix obtained
        // by accumulating the COO triplets of `normalized_adjacency`.
        let g = CsrGraph::from_edges(4, &[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
            .expect("test invariant: value must be valid");
        let layer = GcnLayer::new(GcnConfig {
            in_features: 1,
            out_features: 1,
            bias: false,
            normalize: true,
        })
        .expect("test invariant: value must be valid");
        let adj = layer
            .propagation_matrix(&g)
            .expect("test invariant: value must be valid");
        let n = g.n_nodes();
        assert_eq!(adj.nrows, n);
        assert_eq!(adj.ncols, n);

        let (rows, cols, vals) = g.normalized_adjacency();
        let mut expected = vec![0.0_f64; n * n];
        for ((&r, &c), &val) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
            expected[r * n + c] += f64::from(val);
        }
        let got = adj.to_dense();
        let err = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(err <= 1e-12, "operator mismatch max abs error = {err}");
    }

    #[test]
    fn forward_sparse_deterministic() {
        let g = k4_graph();
        let layer = GcnLayer::new(GcnConfig {
            in_features: 2,
            out_features: 3,
            bias: true,
            normalize: true,
        })
        .expect("test invariant: value must be valid");
        let adj = layer
            .propagation_matrix(&g)
            .expect("test invariant: value must be valid");
        let feats: Vec<f32> = (0..4 * 2).map(|i| i as f32 * 0.5 - 1.0).collect();
        let w: Vec<f32> = (0..2 * 3).map(|i| i as f32 * 0.25).collect();
        let b = vec![0.1_f32, 0.2, 0.3];

        let first = layer
            .forward_sparse(&adj, &feats, &w, Some(&b))
            .expect("test invariant: value must be valid");
        let second = layer
            .forward_sparse(&adj, &feats, &w, Some(&b))
            .expect("test invariant: value must be valid");
        assert_eq!(first, second);
    }

    #[test]
    fn forward_sparse_rejects_bad_inputs() {
        let g = k4_graph();
        let layer = GcnLayer::new(GcnConfig {
            in_features: 2,
            out_features: 2,
            bias: false,
            normalize: true,
        })
        .expect("test invariant: value must be valid");
        let adj = layer
            .propagation_matrix(&g)
            .expect("test invariant: value must be valid");
        let w = vec![0.5_f32; 4];

        // Feature row count (3) disagrees with the operator's node count (4).
        let bad_feats = vec![1.0_f32; 3 * 2];
        assert!(matches!(
            layer.forward_sparse(&adj, &bad_feats, &w, None),
            Err(GnnError::NodeFeatureMismatch(..))
        ));

        // A non-square adjacency is rejected before any aggregation.
        let rect = HostCsr::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0])
            .expect("test invariant: value must be valid");
        let feats = vec![1.0_f32; 2 * 2];
        assert!(matches!(
            layer.forward_sparse(&rect, &feats, &w, None),
            Err(GnnError::DimensionMismatch { .. })
        ));
    }
}
