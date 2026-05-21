//! Relational Graph Convolutional Network (R-GCN) layer.
//!
//! Reference: Schlichtkrull, Kipf, Bloem, van den Berg, Titov & Welling,
//! *"Modeling Relational Data with Graph Convolutional Networks"*, ESWC 2018.
//!
//! R-GCN generalises the GCN to multi-relational (knowledge) graphs by
//! learning a separate transformation per relation type and aggregating
//! incoming messages along each relation independently:
//!
//! ```text
//! h_i^{l+1} = ReLU( W_0 h_i^l + Σ_r Σ_{j ∈ N_i^r} (1 / c_{i,r}) W_r h_j^l )
//! ```
//!
//! where `N_i^r` is the set of neighbours of node `i` under relation `r`,
//! `c_{i,r} = |N_i^r|` is the per-relation in-degree normaliser, and `W_0`
//! is an optional self-loop transformation.
//!
//! To keep the parameter count tractable for a large number of relations,
//! every relation weight is expressed through **basis decomposition**:
//!
//! ```text
//! W_r = Σ_{b=1}^{B} a_{rb} V_b
//! ```
//!
//! so that only `B` shared basis matrices `V_b` (each `out_dim × in_dim`) and
//! the per-relation coefficients `a_{rb}` are learned.
//!
//! # CSR edge-direction convention
//!
//! Each relation is supplied as one [`CsrGraph`]. Following the convention of
//! [`crate::layers::gcn`] — where the forward pass aggregates the neighbours
//! that the CSR yields for a node — **the neighbour list `CsrGraph::neighbors(i)`
//! is treated as the set of source nodes `j` whose features flow *into* node
//! `i` under that relation** (i.e. each stored CSR entry on row `i` with column
//! `j` denotes an incoming edge `j → i`). Aggregation for node `i` therefore
//! sums over `CsrGraph::neighbors(i)` and normalises by that node's neighbour
//! count `c_{i,r}`. Nodes with no incoming edges in a relation (`c_{i,r} = 0`)
//! contribute nothing from that relation.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;
use crate::handle::LcgRng;

/// Configuration for an [`RgcnLayer`].
#[derive(Debug, Clone)]
pub struct RgcnConfig {
    /// Input feature dimension.
    pub in_dim: usize,
    /// Output feature dimension.
    pub out_dim: usize,
    /// Number of relation types.
    pub n_relations: usize,
    /// Number of shared basis matrices used in the decomposition.
    pub n_bases: usize,
    /// Whether to include the self-loop transformation `W_0`.
    pub self_loop: bool,
}

/// A single relational graph convolution layer with basis decomposition.
///
/// Stores the shared basis matrices, the per-relation mixing coefficients and
/// (optionally) the self-loop weight. All matrices are kept row-major.
pub struct RgcnLayer {
    /// Layer configuration.
    config: RgcnConfig,
    /// Basis matrices: `n_bases` blocks, each `out_dim * in_dim` row-major.
    bases: Vec<f32>,
    /// Mixing coefficients: `n_relations * n_bases` row-major (`coeffs[r][b]`).
    coeffs: Vec<f32>,
    /// Self-loop transformation `W_0`: `out_dim * in_dim` row-major.
    ///
    /// Empty when `config.self_loop` is `false`.
    w_self: Vec<f32>,
}

impl RgcnLayer {
    /// Construct an R-GCN layer, initialising parameters from `rng`.
    ///
    /// Basis matrices and the self-loop weight are drawn from a Glorot-style
    /// normal distribution scaled by `sqrt(1 / in_dim)`; relation coefficients
    /// are drawn from a unit normal scaled by `sqrt(1 / n_bases)` so that the
    /// effective relation weight keeps a comparable scale.
    ///
    /// # Errors
    ///
    /// Returns [`GnnError::InvalidLayerConfig`] if `in_dim`, `out_dim`,
    /// `n_relations` or `n_bases` is zero.
    pub fn new(config: RgcnConfig, rng: &mut LcgRng) -> GnnResult<Self> {
        if config.in_dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "R-GCN: in_dim must be > 0".to_string(),
            ));
        }
        if config.out_dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "R-GCN: out_dim must be > 0".to_string(),
            ));
        }
        if config.n_relations == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "R-GCN: n_relations must be > 0".to_string(),
            ));
        }
        if config.n_bases == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "R-GCN: n_bases must be > 0".to_string(),
            ));
        }

        let in_dim = config.in_dim;
        let out_dim = config.out_dim;
        let n_relations = config.n_relations;
        let n_bases = config.n_bases;

        // Glorot-ish scale: σ = sqrt(1 / fan_in).
        let weight_scale = (1.0_f32 / in_dim as f32).sqrt();
        let coeff_scale = (1.0_f32 / n_bases as f32).sqrt();

        let bases = sample_normal(n_bases * out_dim * in_dim, weight_scale, rng);
        let coeffs = sample_normal(n_relations * n_bases, coeff_scale, rng);
        let w_self = if config.self_loop {
            sample_normal(out_dim * in_dim, weight_scale, rng)
        } else {
            Vec::new()
        };

        Ok(Self {
            config,
            bases,
            coeffs,
            w_self,
        })
    }

    /// Reconstruct the relation weight `W_r = Σ_b a_{rb} V_b`.
    ///
    /// Returns an `out_dim × in_dim` row-major matrix.
    ///
    /// # Errors
    ///
    /// Returns [`GnnError::NodeIndexOutOfRange`] if `r >= n_relations`.
    pub fn relation_weight(&self, r: usize) -> GnnResult<Vec<f32>> {
        if r >= self.config.n_relations {
            return Err(GnnError::NodeIndexOutOfRange {
                idx: r,
                n_nodes: self.config.n_relations,
            });
        }
        let out_dim = self.config.out_dim;
        let in_dim = self.config.in_dim;
        let n_bases = self.config.n_bases;
        let block = out_dim * in_dim;

        let mut w = vec![0.0_f32; block];
        for b in 0..n_bases {
            let a = self.coeffs[r * n_bases + b];
            let base = &self.bases[b * block..(b + 1) * block];
            for (w_elem, &v_elem) in w.iter_mut().zip(base.iter()) {
                *w_elem += a * v_elem;
            }
        }
        Ok(w)
    }

    /// Forward pass over one [`CsrGraph`] per relation.
    ///
    /// # Arguments
    ///
    /// - `relation_graphs`: exactly `n_relations` graphs; `relation_graphs[r]`
    ///   holds the edges of relation `r`. All graphs must share the same node
    ///   count. See the module documentation for the CSR direction convention.
    /// - `h`: node feature matrix `[n_nodes × in_dim]` row-major.
    ///
    /// # Returns
    ///
    /// `[n_nodes × out_dim]` row-major after applying ReLU element-wise.
    ///
    /// # Errors
    ///
    /// Returns an error if `relation_graphs.len() != n_relations`, if the
    /// graphs disagree on `n_nodes`, or if `h.len() != n_nodes * in_dim`.
    pub fn forward(&self, relation_graphs: &[CsrGraph], h: &[f32]) -> GnnResult<Vec<f32>> {
        let n_relations = self.config.n_relations;
        let in_dim = self.config.in_dim;
        let out_dim = self.config.out_dim;

        if relation_graphs.len() != n_relations {
            return Err(GnnError::DimensionMismatch {
                expected: n_relations,
                got: relation_graphs.len(),
            });
        }

        let n_nodes = match relation_graphs.first() {
            Some(g) => g.n_nodes(),
            // n_relations >= 1 was validated in `new`, so this branch is
            // unreachable in practice; return a defensive error instead of
            // panicking.
            None => {
                return Err(GnnError::InvalidLayerConfig(
                    "R-GCN: at least one relation graph required".to_string(),
                ));
            }
        };
        for g in relation_graphs {
            if g.n_nodes() != n_nodes {
                return Err(GnnError::DimensionMismatch {
                    expected: n_nodes,
                    got: g.n_nodes(),
                });
            }
        }

        if h.len() != n_nodes * in_dim {
            return Err(GnnError::NodeFeatureMismatch(
                n_nodes,
                h.len() / in_dim.max(1),
            ));
        }

        let mut out = vec![0.0_f32; n_nodes * out_dim];

        // Self-loop term: out_i += W_0 · h_i.
        if self.config.self_loop {
            for i in 0..n_nodes {
                let h_i = &h[i * in_dim..(i + 1) * in_dim];
                let out_i = &mut out[i * out_dim..(i + 1) * out_dim];
                mat_vec_accumulate(&self.w_self, h_i, out_i, out_dim, in_dim);
            }
        }

        // Per-relation aggregation: out_i += (1 / c_{i,r}) · Σ_j W_r · h_j.
        for (r, graph) in relation_graphs.iter().enumerate() {
            let w_r = self.relation_weight(r)?;
            for i in 0..n_nodes {
                let neighbors = graph.neighbors(i)?;
                let count = neighbors.len();
                if count == 0 {
                    continue;
                }
                let norm = 1.0_f32 / count as f32;
                // Sum neighbour features first, then a single matrix-vector
                // product per node — algebraically identical to transforming
                // each neighbour and is markedly cheaper.
                let mut neighbor_sum = vec![0.0_f32; in_dim];
                for &j in neighbors {
                    let h_j = &h[j * in_dim..(j + 1) * in_dim];
                    for (acc, &val) in neighbor_sum.iter_mut().zip(h_j.iter()) {
                        *acc += val;
                    }
                }
                for elem in &mut neighbor_sum {
                    *elem *= norm;
                }
                let out_i = &mut out[i * out_dim..(i + 1) * out_dim];
                mat_vec_accumulate(&w_r, &neighbor_sum, out_i, out_dim, in_dim);
            }
        }

        // ReLU activation.
        for v in &mut out {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        Ok(out)
    }

    /// Total number of learnable parameters.
    ///
    /// `n_bases · out_dim · in_dim + n_relations · n_bases + (self_loop ? out_dim · in_dim : 0)`.
    pub fn n_params(&self) -> usize {
        let block = self.config.out_dim * self.config.in_dim;
        let mut total = self.config.n_bases * block + self.config.n_relations * self.config.n_bases;
        if self.config.self_loop {
            total += block;
        }
        total
    }

    /// Output feature dimension.
    #[inline]
    pub fn output_dim(&self) -> usize {
        self.config.out_dim
    }
}

/// Accumulate `out += W · x` where `W` is `rows × cols` row-major and `x` has
/// length `cols`. `out` has length `rows`.
#[inline]
fn mat_vec_accumulate(w: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for (k, out_k) in out.iter_mut().enumerate().take(rows) {
        let row = &w[k * cols..(k + 1) * cols];
        let mut acc = 0.0_f32;
        for (&w_elem, &x_elem) in row.iter().zip(x.iter()) {
            acc += w_elem * x_elem;
        }
        *out_k += acc;
    }
}

/// Draw `n` samples from `N(0, scale²)` using the RNG's normal generator.
fn sample_normal(n: usize, scale: f32, rng: &mut LcgRng) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    while out.len() + 1 < n {
        let (a, b) = rng.next_normal_pair();
        out.push(a * scale);
        out.push(b * scale);
    }
    if out.len() < n {
        let (a, _) = rng.next_normal_pair();
        out.push(a * scale);
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer(
        in_dim: usize,
        out_dim: usize,
        n_relations: usize,
        n_bases: usize,
        self_loop: bool,
        seed: u64,
    ) -> RgcnLayer {
        let mut rng = LcgRng::new(seed);
        RgcnLayer::new(
            RgcnConfig {
                in_dim,
                out_dim,
                n_relations,
                n_bases,
                self_loop,
            },
            &mut rng,
        )
        .expect("test invariant: layer must construct")
    }

    fn empty_relation(n_nodes: usize) -> CsrGraph {
        // A graph with the right node count but no edges.
        CsrGraph::new(n_nodes, vec![0usize; n_nodes + 1], vec![])
            .expect("test invariant: graph must construct")
    }

    #[test]
    fn relation_weight_shape() {
        let layer = make_layer(3, 5, 4, 2, true, 1);
        let w = layer
            .relation_weight(2)
            .expect("test invariant: relation must exist");
        assert_eq!(w.len(), 5 * 3);
    }

    #[test]
    fn single_basis_relation_weight_is_coeff_times_basis() {
        // n_bases = 1 → W_r == coeffs[r][0] · bases[0].
        let layer = make_layer(2, 3, 2, 1, false, 7);
        for r in 0..2 {
            let w = layer
                .relation_weight(r)
                .expect("test invariant: relation must exist");
            let a = layer.coeffs[r]; // r * 1 + 0
            for (idx, &w_elem) in w.iter().enumerate() {
                let expected = a * layer.bases[idx];
                assert!(
                    (w_elem - expected).abs() < 1e-6,
                    "mismatch at r={r}, idx={idx}: {w_elem} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn forward_output_shape() {
        let layer = make_layer(3, 4, 2, 2, true, 11);
        let g0 = CsrGraph::from_edges(3, &[(0, 1), (1, 2)])
            .expect("test invariant: graph must construct");
        let g1 = CsrGraph::from_edges(3, &[(2, 0)]).expect("test invariant: graph must construct");
        let h = vec![0.5_f32; 3 * 3];
        let out = layer
            .forward(&[g0, g1], &h)
            .expect("test invariant: forward must succeed");
        assert_eq!(out.len(), 3 * 4);
    }

    #[test]
    fn forward_output_non_negative() {
        let layer = make_layer(3, 4, 2, 2, true, 13);
        let g0 = CsrGraph::from_edges(3, &[(0, 1), (1, 2), (2, 0)])
            .expect("test invariant: graph must construct");
        let g1 = CsrGraph::from_edges(3, &[(0, 2), (1, 0)])
            .expect("test invariant: graph must construct");
        // Negative features to exercise the ReLU.
        let h: Vec<f32> = (0..9).map(|i| -(i as f32) - 1.0).collect();
        let out = layer
            .forward(&[g0, g1], &h)
            .expect("test invariant: forward must succeed");
        assert!(out.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn isolated_node_self_loop_equals_relu_w_self_h() {
        // Node 2 has no incoming edges in either relation.
        let layer = make_layer(2, 2, 2, 2, true, 17);
        let g0 = CsrGraph::from_edges(3, &[(0, 1)]).expect("test invariant: graph must construct");
        let g1 = CsrGraph::from_edges(3, &[(1, 0)]).expect("test invariant: graph must construct");
        let h = vec![0.3_f32, -0.7, 0.1, 0.2, -0.5, 0.9];
        let out = layer
            .forward(&[g0, g1], &h)
            .expect("test invariant: forward must succeed");
        // out_2 should equal ReLU(W_0 · h_2).
        let h2 = &h[4..6];
        let mut expected = vec![0.0_f32; 2];
        mat_vec_accumulate(&layer.w_self, h2, &mut expected, 2, 2);
        for e in &mut expected {
            *e = e.max(0.0);
        }
        for k in 0..2 {
            assert!(
                (out[4 + k] - expected[k]).abs() < 1e-6,
                "node 2 mismatch at {k}: {} vs {}",
                out[4 + k],
                expected[k]
            );
        }
    }

    #[test]
    fn no_self_loop_isolated_node_is_zero() {
        // self_loop = false and node has no edges → all-zero row.
        let layer = make_layer(2, 3, 2, 2, false, 19);
        let g0 = empty_relation(2);
        let g1 = CsrGraph::from_edges(2, &[(0, 1)]).expect("test invariant: graph must construct");
        let h = vec![1.0_f32, -2.0, 3.0, -4.0];
        let out = layer
            .forward(&[g0, g1], &h)
            .expect("test invariant: forward must succeed");
        // Node 0 has no incoming edges in either relation (g1 edge is 0→1,
        // which under the convention is an incoming edge to node 1).
        for (k, &v) in out[0..3].iter().enumerate() {
            assert!(v.abs() < 1e-7, "node 0 not zero at {k}: {v}");
        }
    }

    #[test]
    fn single_relation_normalized_message_pass() {
        // One relation, no self loop, n_bases = 1 so W_r is exact.
        let layer = make_layer(2, 2, 1, 1, false, 23);
        // Node 0 receives from nodes 1 and 2 (two incoming edges).
        let g0 = CsrGraph::from_edges(3, &[(0, 1), (0, 2)])
            .expect("test invariant: graph must construct");
        let h = vec![0.0_f32, 0.0, 1.0, 2.0, 3.0, 4.0];
        let out = layer
            .forward(std::slice::from_ref(&g0), &h)
            .expect("test invariant: forward must succeed");
        // Expected: ReLU( W_r · mean(h_1, h_2) ).
        let w = layer
            .relation_weight(0)
            .expect("test invariant: relation must exist");
        let mean = [(1.0 + 3.0) / 2.0, (2.0 + 4.0) / 2.0];
        let mut expected = vec![0.0_f32; 2];
        mat_vec_accumulate(&w, &mean, &mut expected, 2, 2);
        for e in &mut expected {
            *e = e.max(0.0);
        }
        for k in 0..2 {
            assert!(
                (out[k] - expected[k]).abs() < 1e-5,
                "node 0 mismatch at {k}: {} vs {}",
                out[k],
                expected[k]
            );
        }
    }

    #[test]
    fn two_relations_contribute_additively() {
        // No self loop, n_bases small but use raw relation weights for the
        // expectation. Node 0 has exactly one incoming edge in rel0 (from 1)
        // and one in rel1 (from 2).
        let layer = make_layer(2, 2, 2, 2, false, 29);
        let g0 = CsrGraph::from_edges(3, &[(0, 1)]).expect("test invariant: graph must construct");
        let g1 = CsrGraph::from_edges(3, &[(0, 2)]).expect("test invariant: graph must construct");
        let h = vec![5.0_f32, 6.0, 1.0, 2.0, 3.0, 4.0];
        let out = layer
            .forward(&[g0, g1], &h)
            .expect("test invariant: forward must succeed");
        let w0 = layer
            .relation_weight(0)
            .expect("test invariant: relation must exist");
        let w1 = layer
            .relation_weight(1)
            .expect("test invariant: relation must exist");
        // c = 1 for both relations: pre-activation = W_0·h_1 + W_1·h_2.
        let h1 = [1.0_f32, 2.0];
        let h2 = [3.0_f32, 4.0];
        let mut expected = vec![0.0_f32; 2];
        mat_vec_accumulate(&w0, &h1, &mut expected, 2, 2);
        mat_vec_accumulate(&w1, &h2, &mut expected, 2, 2);
        for e in &mut expected {
            *e = e.max(0.0);
        }
        for k in 0..2 {
            assert!(
                (out[k] - expected[k]).abs() < 1e-5,
                "additive mismatch at {k}: {} vs {}",
                out[k],
                expected[k]
            );
        }
    }

    #[test]
    fn normalization_two_identical_neighbors_average() {
        // Node 0 has two incoming edges in rel0, both from node 1 → average
        // collapses to h_1 (each weighted 1/2, summed twice).
        let layer = make_layer(2, 2, 1, 1, false, 31);
        let g0 = CsrGraph::from_edges(2, &[(0, 1), (0, 1)])
            .expect("test invariant: graph must construct");
        let h = vec![0.0_f32, 0.0, 7.0, -3.0];
        let out = layer
            .forward(std::slice::from_ref(&g0), &h)
            .expect("test invariant: forward must succeed");
        let w = layer
            .relation_weight(0)
            .expect("test invariant: relation must exist");
        // mean of two copies of h_1 = h_1.
        let h1 = [7.0_f32, -3.0];
        let mut expected = vec![0.0_f32; 2];
        mat_vec_accumulate(&w, &h1, &mut expected, 2, 2);
        for e in &mut expected {
            *e = e.max(0.0);
        }
        for k in 0..2 {
            assert!(
                (out[k] - expected[k]).abs() < 1e-5,
                "average mismatch at {k}: {} vs {}",
                out[k],
                expected[k]
            );
        }
    }

    #[test]
    fn n_params_with_self_loop() {
        let layer = make_layer(3, 5, 4, 2, true, 37);
        // n_bases*out*in + n_relations*n_bases + out*in
        let expected = 2 * 5 * 3 + 4 * 2 + 5 * 3;
        assert_eq!(layer.n_params(), expected);
    }

    #[test]
    fn n_params_without_self_loop() {
        let layer = make_layer(3, 5, 4, 2, false, 41);
        let expected = 2 * 5 * 3 + 4 * 2;
        assert_eq!(layer.n_params(), expected);
    }

    #[test]
    fn err_in_dim_zero() {
        let mut rng = LcgRng::new(1);
        let res = RgcnLayer::new(
            RgcnConfig {
                in_dim: 0,
                out_dim: 4,
                n_relations: 2,
                n_bases: 2,
                self_loop: true,
            },
            &mut rng,
        );
        assert!(matches!(res, Err(GnnError::InvalidLayerConfig(_))));
    }

    #[test]
    fn err_out_dim_zero() {
        let mut rng = LcgRng::new(1);
        let res = RgcnLayer::new(
            RgcnConfig {
                in_dim: 4,
                out_dim: 0,
                n_relations: 2,
                n_bases: 2,
                self_loop: true,
            },
            &mut rng,
        );
        assert!(matches!(res, Err(GnnError::InvalidLayerConfig(_))));
    }

    #[test]
    fn err_n_relations_zero() {
        let mut rng = LcgRng::new(1);
        let res = RgcnLayer::new(
            RgcnConfig {
                in_dim: 4,
                out_dim: 4,
                n_relations: 0,
                n_bases: 2,
                self_loop: true,
            },
            &mut rng,
        );
        assert!(matches!(res, Err(GnnError::InvalidLayerConfig(_))));
    }

    #[test]
    fn err_n_bases_zero() {
        let mut rng = LcgRng::new(1);
        let res = RgcnLayer::new(
            RgcnConfig {
                in_dim: 4,
                out_dim: 4,
                n_relations: 2,
                n_bases: 0,
                self_loop: true,
            },
            &mut rng,
        );
        assert!(matches!(res, Err(GnnError::InvalidLayerConfig(_))));
    }

    #[test]
    fn err_wrong_number_of_relation_graphs() {
        let layer = make_layer(2, 2, 2, 2, true, 43);
        let g0 = CsrGraph::from_edges(3, &[(0, 1)]).expect("test invariant: graph must construct");
        let h = vec![0.0_f32; 3 * 2];
        // Only one graph but n_relations = 2.
        let res = layer.forward(std::slice::from_ref(&g0), &h);
        assert!(matches!(res, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_mismatched_n_nodes_across_relations() {
        let layer = make_layer(2, 2, 2, 2, true, 47);
        let g0 = CsrGraph::from_edges(3, &[(0, 1)]).expect("test invariant: graph must construct");
        let g1 = CsrGraph::from_edges(4, &[(0, 1)]).expect("test invariant: graph must construct");
        let h = vec![0.0_f32; 3 * 2];
        let res = layer.forward(&[g0, g1], &h);
        assert!(matches!(res, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_h_wrong_length() {
        let layer = make_layer(2, 2, 2, 2, true, 53);
        let g0 = CsrGraph::from_edges(3, &[(0, 1)]).expect("test invariant: graph must construct");
        let g1 = CsrGraph::from_edges(3, &[(1, 0)]).expect("test invariant: graph must construct");
        let h = vec![0.0_f32; 3 * 3]; // in_dim is 2, not 3
        let res = layer.forward(&[g0, g1], &h);
        assert!(matches!(res, Err(GnnError::NodeFeatureMismatch(..))));
    }

    #[test]
    fn err_relation_weight_out_of_range() {
        let layer = make_layer(2, 2, 2, 2, true, 59);
        let res = layer.relation_weight(5);
        assert!(matches!(res, Err(GnnError::NodeIndexOutOfRange { .. })));
    }

    #[test]
    fn deterministic_given_seed() {
        let layer_a = make_layer(3, 4, 3, 2, true, 1234);
        let layer_b = make_layer(3, 4, 3, 2, true, 1234);
        let g0 = CsrGraph::from_edges(3, &[(0, 1), (1, 2)])
            .expect("test invariant: graph must construct");
        let g1 = CsrGraph::from_edges(3, &[(2, 0)]).expect("test invariant: graph must construct");
        let g2 = CsrGraph::from_edges(3, &[(0, 2), (1, 0)])
            .expect("test invariant: graph must construct");
        let h: Vec<f32> = (0..9).map(|i| i as f32 * 0.1).collect();
        let out_a = layer_a
            .forward(&[g0.clone(), g1.clone(), g2.clone()], &h)
            .expect("test invariant: forward must succeed");
        let out_b = layer_b
            .forward(&[g0, g1, g2], &h)
            .expect("test invariant: forward must succeed");
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn forward_finite_output() {
        let layer = make_layer(4, 6, 3, 3, true, 61);
        let g0 = CsrGraph::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)])
            .expect("test invariant: graph must construct");
        let g1 = CsrGraph::from_edges(4, &[(0, 2), (1, 3)])
            .expect("test invariant: graph must construct");
        let g2 = CsrGraph::from_edges(4, &[(2, 0), (3, 1)])
            .expect("test invariant: graph must construct");
        let h: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.3).collect();
        let out = layer
            .forward(&[g0, g1, g2], &h)
            .expect("test invariant: forward must succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
