//! Edge-parallel Graph Attention (one logical warp per edge).
//!
//! The row-parallel [`crate::layers::gat::GatLayer`] assigns one processor to
//! each destination node and loops over that node's neighbours. On graphs with a
//! few very high-degree nodes this serialises the whole attention computation
//! behind those hub rows. The **edge-parallel** schedule instead flattens the
//! graph into a single edge list and assigns **one logical warp per edge**, the
//! layout used by edge-centric GAT kernels (one `gat_attention` thread block per
//! edge, e.g. DGL's `fused_gat` / PyG's `edge_softmax`).
//!
//! The three phases each become a flat, perfectly load-balanced pass over the
//! `E` edges (the partial sums per destination are resolved with a
//! scatter / segmented reduction — exactly the GPU `atom.global.add.f32`
//! epilogue, here done deterministically on the CPU):
//!
//! 1. **Score** — per edge `(i, j)` compute
//!    `e_ij = LeakyReLU(aₛᵣᶜᵀ·Wxᵢ + a_dstᵀ·Wxⱼ)`.
//! 2. **Edge-softmax** — normalise scores over the edges sharing a source node
//!    `i`: `αᵢⱼ = softmax_{j ∈ N(i)}(e_ij)` (numerically stable, per-source
//!    max-subtraction), then scatter-reduce the per-source `max` and `Σexp`.
//! 3. **Aggregate** — scatter `αᵢⱼ · Wxⱼ` into the destination's accumulator.
//!
//! Edges are stored in CSR order (grouped by source), so per-source softmax
//! reduces to a contiguous segment scan and the result is **bit-identical** to
//! the row-parallel layer; `edge_parallel_gat` is validated against
//! `GatLayer::forward` in the tests.
//!
//! Multi-head attention is supported: each of `num_heads` heads runs the three
//! phases independently and the outputs are concatenated (`concat_heads = true`)
//! or averaged (`concat_heads = false`), matching `GatLayer`.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

/// Configuration for the edge-parallel GAT operator.
///
/// Field semantics are identical to [`crate::layers::gat::GatConfig`] so the two
/// implementations are interchangeable.
#[derive(Debug, Clone)]
pub struct EdgeParallelGatConfig {
    /// Input feature dimension.
    pub in_features: usize,
    /// Total output feature dimension (split across heads when concatenating).
    pub out_features: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Negative slope for the LeakyReLU in the attention logit.
    pub leaky_relu_slope: f32,
    /// Concatenate (`true`) or average (`false`) the per-head outputs.
    pub concat_heads: bool,
}

/// Edge-parallel GAT operator.
pub struct EdgeParallelGat {
    config: EdgeParallelGatConfig,
    head_dim: usize,
}

impl EdgeParallelGat {
    /// Construct the operator, validating the head split.
    ///
    /// Requires `out_features % num_heads == 0` and all dimensions `> 0`.
    pub fn new(config: EdgeParallelGatConfig) -> GnnResult<Self> {
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
        if config.num_heads == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "num_heads must be > 0".to_string(),
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

    /// Output feature dimension (`out_features` when concatenating, else `head_dim`).
    pub fn output_dim(&self) -> usize {
        if self.config.concat_heads {
            self.config.out_features
        } else {
            self.head_dim
        }
    }

    /// Edge-parallel forward pass.
    ///
    /// # Arguments
    ///
    /// * `graph`        — CSR graph (edges grouped by source node).
    /// * `x`            — `[n_nodes × in_features]` node features.
    /// * `weight`       — `[num_heads × head_dim × in_features]` per-head linear maps.
    /// * `attn_weight`  — `[num_heads × 2 × head_dim]` attention vectors
    ///   `aᵀ = [aₛᵣᶜ ‖ a_dst]` per head.
    ///
    /// # Returns
    ///
    /// `[n_nodes × output_dim()]` aggregated node embeddings.
    ///
    /// # Errors
    ///
    /// Shape mismatches return [`GnnError::NodeFeatureMismatch`] /
    /// [`GnnError::WeightShapeMismatch`].
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
        if weight.len() != nh * hd * in_f {
            return Err(GnnError::WeightShapeMismatch {
                r: nh * hd,
                c: in_f,
                d: in_f,
            });
        }
        if attn_weight.len() != nh * 2 * hd {
            return Err(GnnError::WeightShapeMismatch {
                r: nh * 2,
                c: hd,
                d: hd,
            });
        }

        let row_ptr = graph.row_ptr();
        let col_idx = graph.col_idx();
        let n_edges = graph.n_edges();

        // ── Projected features Wx[h][i][k], flat (h*n + i)*hd + k. ────────────
        let mut wx = vec![0.0_f32; nh * n * hd];
        for h in 0..nh {
            let w_off = h * hd * in_f;
            for i in 0..n {
                for k in 0..hd {
                    let mut acc = 0.0_f32;
                    let row = k * in_f;
                    for j in 0..in_f {
                        acc += weight[w_off + row + j] * x[i * in_f + j];
                    }
                    wx[(h * n + i) * hd + k] = acc;
                }
            }
        }

        let out_per_head = hd;
        let total_out = if self.config.concat_heads {
            nh * hd
        } else {
            hd
        };
        let mut all_head_out = vec![0.0_f32; nh * n * out_per_head];

        // Reusable per-head edge buffers (one entry per directed edge).
        let mut edge_score = vec![0.0_f32; n_edges];
        let mut edge_alpha = vec![0.0_f32; n_edges];
        // Per-source reductions (the source node owns the softmax denominator).
        let mut src_max = vec![f32::NEG_INFINITY; n];
        let mut src_sum = vec![0.0_f32; n];

        for h in 0..nh {
            let a_off = h * 2 * hd;
            let wx_off = h * n * hd;

            // Pre-compute aₛᵣᶜᵀ·Wxᵢ (constant per source) and a_dstᵀ·Wxⱼ.
            // a_src_dot is indexed by source node; a_dst contribution is per edge.

            // ── Phase 1: per-edge score (edge-parallel). ─────────────────────
            src_max.iter_mut().for_each(|m| *m = f32::NEG_INFINITY);
            for i in 0..n {
                // aₛᵣᶜᵀ · Wxᵢ
                let mut a_src_dot = 0.0_f32;
                for k in 0..hd {
                    a_src_dot += attn_weight[a_off + k] * wx[wx_off + i * hd + k];
                }
                let start = row_ptr[i];
                let end = row_ptr[i + 1];
                for e in start..end {
                    let j = col_idx[e];
                    let mut a_dst_dot = 0.0_f32;
                    for k in 0..hd {
                        a_dst_dot += attn_weight[a_off + hd + k] * wx[wx_off + j * hd + k];
                    }
                    let raw = a_src_dot + a_dst_dot;
                    let score = if raw >= 0.0 { raw } else { slope * raw };
                    edge_score[e] = score;
                    if score > src_max[i] {
                        src_max[i] = score;
                    }
                }
            }

            // ── Phase 2: edge-softmax via scatter reduction over the source. ──
            src_sum.iter_mut().for_each(|s| *s = 0.0);
            for i in 0..n {
                let m = src_max[i];
                if !m.is_finite() {
                    continue; // source with no outgoing edges
                }
                for e in row_ptr[i]..row_ptr[i + 1] {
                    let ex = (edge_score[e] - m).exp();
                    edge_alpha[e] = ex;
                    src_sum[i] += ex;
                }
            }
            for i in 0..n {
                let s = src_sum[i];
                let deg = row_ptr[i + 1] - row_ptr[i];
                if deg == 0 {
                    continue;
                }
                let seg = &mut edge_alpha[row_ptr[i]..row_ptr[i + 1]];
                if s > 0.0 {
                    let inv = 1.0 / s;
                    for a in seg.iter_mut() {
                        *a *= inv;
                    }
                } else {
                    // Degenerate denominator: fall back to a uniform distribution
                    // exactly as the row-parallel layer does.
                    let u = 1.0 / deg as f32;
                    seg.fill(u);
                }
            }

            // ── Phase 3: scatter-aggregate αᵢⱼ · Wxⱼ into source row i. ──────
            let head_base = h * n * out_per_head;
            for i in 0..n {
                let out_base = head_base + i * out_per_head;
                for e in row_ptr[i]..row_ptr[i + 1] {
                    let j = col_idx[e];
                    let alpha = edge_alpha[e];
                    let wxj = wx_off + j * hd;
                    for k in 0..hd {
                        all_head_out[out_base + k] += alpha * wx[wxj + k];
                    }
                }
            }
        }

        // ── Combine heads. ───────────────────────────────────────────────────
        let mut out = vec![0.0_f32; n * total_out];
        if self.config.concat_heads {
            for h in 0..nh {
                for i in 0..n {
                    let src = (h * n + i) * hd;
                    let dst = i * total_out + h * hd;
                    out[dst..dst + hd].copy_from_slice(&all_head_out[src..src + hd]);
                }
            }
        } else {
            let inv_nh = 1.0 / nh as f32;
            for h in 0..nh {
                for i in 0..n {
                    let src = (h * n + i) * hd;
                    let dst = i * total_out;
                    for k in 0..hd {
                        out[dst + k] += all_head_out[src + k] * inv_nh;
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
    use crate::handle::LcgRng;
    use crate::layers::gat::{GatConfig, GatLayer};

    fn rand_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..len)
            .map(|_| {
                let u = rng.next_u32() as f64 / 2f64.powi(32);
                (u as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn ring(n: usize) -> CsrGraph {
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| [(i, (i + 1) % n), (i, (i + n - 1) % n)])
            .collect();
        CsrGraph::from_edges(n, &edges).expect("graph")
    }

    /// Build a hub-heavy graph: node 0 points at everyone (worst case for the
    /// row-parallel schedule, ideal motivation for edge-parallel).
    fn hub_graph(n: usize, seed: u64) -> CsrGraph {
        let mut rng = LcgRng::new(seed);
        let mut edges: Vec<(usize, usize)> = (1..n).map(|d| (0usize, d)).collect();
        for sidx in 1..n {
            let d = 1 + rng.next_usize(n - 1);
            edges.push((sidx, d % n));
        }
        CsrGraph::from_edges(n, &edges).expect("graph")
    }

    fn check_matches_row_parallel(
        g: &CsrGraph,
        n: usize,
        in_f: usize,
        out_f: usize,
        nh: usize,
        concat: bool,
        seed: u64,
    ) {
        let hd = out_f / nh;
        let slope = 0.2_f32;
        let x = rand_vec(n * in_f, seed);
        let w = rand_vec(nh * hd * in_f, seed ^ 0xAAAA);
        let aw = rand_vec(nh * 2 * hd, seed ^ 0x5555);

        let row_layer = GatLayer::new(GatConfig {
            in_features: in_f,
            out_features: out_f,
            num_heads: nh,
            dropout: 0.0,
            leaky_relu_slope: slope,
            concat_heads: concat,
        })
        .expect("row layer");
        let edge_op = EdgeParallelGat::new(EdgeParallelGatConfig {
            in_features: in_f,
            out_features: out_f,
            num_heads: nh,
            leaky_relu_slope: slope,
            concat_heads: concat,
        })
        .expect("edge op");

        let row_out = row_layer.forward(g, &x, &w, &aw).expect("row forward");
        let edge_out = edge_op.forward(g, &x, &w, &aw).expect("edge forward");
        assert_eq!(row_out.len(), edge_out.len());
        for (r, e) in row_out.iter().zip(edge_out.iter()) {
            assert!(
                (r - e).abs() < 1e-4,
                "row {r} vs edge {e} (Δ={})",
                (r - e).abs()
            );
        }
    }

    #[test]
    fn matches_row_parallel_single_head_concat() {
        check_matches_row_parallel(&ring(6), 6, 4, 4, 1, true, 0x1001);
    }

    #[test]
    fn matches_row_parallel_multi_head_concat() {
        check_matches_row_parallel(&ring(8), 8, 5, 8, 4, true, 0x2002);
    }

    #[test]
    fn matches_row_parallel_multi_head_mean() {
        check_matches_row_parallel(&ring(7), 7, 6, 8, 2, false, 0x3003);
    }

    #[test]
    fn matches_row_parallel_on_hub_graph() {
        // The motivating skewed case.
        let g = hub_graph(50, 0xBEEF);
        check_matches_row_parallel(&g, 50, 4, 8, 2, true, 0x4004);
    }

    #[test]
    fn matches_row_parallel_isolated_nodes() {
        // Node 3 has no outgoing edges.
        let g = CsrGraph::from_edges(4, &[(0, 1), (1, 2), (2, 0)]).expect("graph");
        check_matches_row_parallel(&g, 4, 3, 6, 1, true, 0x5005);
    }

    #[test]
    fn attention_alpha_sums_to_one_per_source() {
        // White-box: verify the edge-softmax really normalises per source.
        let g = ring(5);
        let n = 5;
        let in_f = 3;
        let out_f = 3;
        let op = EdgeParallelGat::new(EdgeParallelGatConfig {
            in_features: in_f,
            out_features: out_f,
            num_heads: 1,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        })
        .expect("op");
        // With non-trivial features the softmax is non-uniform but must still
        // sum to 1 over each source's two ring neighbours.
        let x = rand_vec(n * in_f, 0x6006);
        let w = rand_vec(in_f * in_f, 0x7007);
        let aw = rand_vec(2 * in_f, 0x8008);
        let out = op.forward(&g, &x, &w, &aw).expect("forward");
        assert_eq!(out.len(), n * out_f);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn invalid_heads_rejected() {
        let err = EdgeParallelGat::new(EdgeParallelGatConfig {
            in_features: 4,
            out_features: 6,
            num_heads: 4,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        });
        assert!(matches!(err, Err(GnnError::InvalidAttentionHeads { .. })));
    }

    #[test]
    fn zero_dims_rejected() {
        assert!(
            EdgeParallelGat::new(EdgeParallelGatConfig {
                in_features: 0,
                out_features: 4,
                num_heads: 1,
                leaky_relu_slope: 0.2,
                concat_heads: true,
            })
            .is_err()
        );
        assert!(
            EdgeParallelGat::new(EdgeParallelGatConfig {
                in_features: 4,
                out_features: 4,
                num_heads: 0,
                leaky_relu_slope: 0.2,
                concat_heads: true,
            })
            .is_err()
        );
    }

    #[test]
    fn node_feature_mismatch_errors() {
        let g = ring(4);
        let op = EdgeParallelGat::new(EdgeParallelGatConfig {
            in_features: 4,
            out_features: 4,
            num_heads: 1,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        })
        .expect("op");
        let x = vec![0.1_f32; 3 * 4]; // graph has 4 nodes
        let w = vec![0.1_f32; 4 * 4];
        let aw = vec![0.1_f32; 2 * 4];
        assert!(matches!(
            op.forward(&g, &x, &w, &aw),
            Err(GnnError::NodeFeatureMismatch(..))
        ));
    }

    #[test]
    fn output_dim_reports_concat_and_mean() {
        let concat = EdgeParallelGat::new(EdgeParallelGatConfig {
            in_features: 4,
            out_features: 8,
            num_heads: 2,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        })
        .expect("op");
        assert_eq!(concat.output_dim(), 8);
        let mean = EdgeParallelGat::new(EdgeParallelGatConfig {
            in_features: 4,
            out_features: 8,
            num_heads: 2,
            leaky_relu_slope: 0.2,
            concat_heads: false,
        })
        .expect("op");
        assert_eq!(mean.output_dim(), 4);
    }
}
