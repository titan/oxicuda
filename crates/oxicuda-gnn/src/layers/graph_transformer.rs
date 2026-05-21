//! Graph Transformer layer — Dwivedi & Bresson 2021 + Graphormer (Ying et al. 2021).
//!
//! Multi-head scaled dot-product attention over graph nodes with optional
//! edge-feature bias and Graphormer structural bias (BFS-distance based).
//!
//! # References
//!
//! - Dwivedi & Bresson (2021) "A Generalization of Transformers to Graphs"
//! - Ying et al. (2021) "Do Transformers Really Perform Bad for Graph Representation?"

use crate::error::{GnnError, GnnResult};
use crate::handle::LcgRng;
use std::collections::VecDeque;

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for a Graph Transformer layer.
#[derive(Debug, Clone)]
pub struct GraphTransformerConfig {
    /// Input feature dimension per node.
    pub in_features: usize,
    /// Total output feature dimension (must be divisible by `n_heads`).
    pub out_features: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Edge feature dimension. Set to 0 for no edge features.
    pub edge_features: usize,
    /// Dropout rate applied to attention weights (0.0 = no dropout).
    pub dropout_rate: f32,
    /// Whether to add learned bias terms to Q/K/V/O projections.
    pub use_bias: bool,
    /// Whether to add Graphormer BFS-distance structural bias.
    pub use_graphormer_bias: bool,
    /// Maximum BFS distance for the Graphormer bias table (e.g., 8).
    pub max_distance: usize,
}

// ─── Weights ──────────────────────────────────────────────────────────────────

/// Learned parameters for a `GraphTransformerLayer`.
///
/// All weight matrices are stored in row-major order.
#[derive(Debug, Clone)]
pub struct GraphTransformerWeights {
    /// Query projection: `[n_heads × head_dim × in_features]`.
    pub w_q: Vec<f32>,
    /// Key projection: `[n_heads × head_dim × in_features]`.
    pub w_k: Vec<f32>,
    /// Value projection: `[n_heads × head_dim × in_features]`.
    pub w_v: Vec<f32>,
    /// Output projection: `[out_features × out_features]`.
    pub w_o: Vec<f32>,
    /// Edge feature projection per head: `[n_heads × edge_features]`;
    /// empty when `edge_features == 0`.
    pub w_e: Vec<f32>,
    /// Bias for Q projection: `[n_heads × head_dim]`; zeros when `!use_bias`.
    pub b_q: Vec<f32>,
    /// Bias for K projection: `[n_heads × head_dim]`; zeros when `!use_bias`.
    pub b_k: Vec<f32>,
    /// Bias for V projection: `[n_heads × head_dim]`; zeros when `!use_bias`.
    pub b_v: Vec<f32>,
    /// Bias for output projection: `[out_features]`; zeros when `!use_bias`.
    pub b_o: Vec<f32>,
    /// Graphormer bias table: `[(max_distance+1) × n_heads]`;
    /// empty when `!use_graphormer_bias`.
    pub graphormer_bias: Vec<f32>,
    /// Layer norm gain γ: `[out_features]`.
    pub ln_weight: Vec<f32>,
    /// Layer norm bias β: `[out_features]`.
    pub ln_bias: Vec<f32>,
}

// ─── Layer ────────────────────────────────────────────────────────────────────

/// Graph Transformer layer with optional Graphormer structural bias.
pub struct GraphTransformerLayer {
    /// Layer configuration.
    pub cfg: GraphTransformerConfig,
    /// Learned weights.
    pub weights: GraphTransformerWeights,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Row-wise softmax in-place over a `[rows × cols]` flat matrix.
fn softmax_rows(mat: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut mat[r * cols..(r + 1) * cols];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv_sum;
            }
        }
    }
}

/// Layer normalization for a row vector of length `d`.
/// γ, β ∈ ℝ^d; ε = 1e-5.
fn layer_norm(v: &[f32], gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    let d = v.len();
    let mean: f32 = v.iter().sum::<f32>() / d as f32;
    let var: f32 = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / d as f32;
    let inv_std = 1.0 / (var + 1e-5_f32).sqrt();
    (0..d)
        .map(|k| (v[k] - mean) * inv_std * gamma[k] + beta[k])
        .collect()
}

impl GraphTransformerLayer {
    /// Construct a new layer with Kaiming-uniform initialised weights.
    ///
    /// # Errors
    ///
    /// Returns [`GnnError::InvalidLayerConfig`] if:
    /// - `in_features == 0`
    /// - `out_features == 0`
    /// - `n_heads == 0`
    /// - `out_features % n_heads != 0`
    pub fn new(cfg: GraphTransformerConfig, rng: &mut LcgRng) -> GnnResult<Self> {
        if cfg.in_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "GraphTransformer: in_features must be > 0".to_string(),
            ));
        }
        if cfg.out_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "GraphTransformer: out_features must be > 0".to_string(),
            ));
        }
        if cfg.n_heads == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "GraphTransformer: n_heads must be > 0".to_string(),
            ));
        }
        if cfg.out_features % cfg.n_heads != 0 {
            return Err(GnnError::InvalidAttentionHeads {
                dim: cfg.out_features,
                heads: cfg.n_heads,
            });
        }

        let head_dim = cfg.out_features / cfg.n_heads;
        let in_f = cfg.in_features;
        let out_f = cfg.out_features;
        let nh = cfg.n_heads;

        // Kaiming uniform: range = ±√(6/fan_in).
        let qkv_fan = in_f;
        let qkv_bound = (6.0_f32 / qkv_fan as f32).sqrt();
        let o_fan = out_f;
        let o_bound = (6.0_f32 / o_fan as f32).sqrt();

        let kaiming_vec = |n: usize, bound: f32, r: &mut LcgRng| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    let u = r.next_f32();
                    (2.0 * u - 1.0) * bound
                })
                .collect()
        };

        let w_q = kaiming_vec(nh * head_dim * in_f, qkv_bound, rng);
        let w_k = kaiming_vec(nh * head_dim * in_f, qkv_bound, rng);
        let w_v = kaiming_vec(nh * head_dim * in_f, qkv_bound, rng);
        let w_o = kaiming_vec(out_f * out_f, o_bound, rng);

        let w_e = if cfg.edge_features > 0 {
            let e_fan = cfg.edge_features;
            let e_bound = (6.0_f32 / e_fan as f32).sqrt();
            kaiming_vec(nh * cfg.edge_features, e_bound, rng)
        } else {
            Vec::new()
        };

        let b_q = vec![0.0_f32; nh * head_dim];
        let b_k = vec![0.0_f32; nh * head_dim];
        let b_v = vec![0.0_f32; nh * head_dim];
        let b_o = vec![0.0_f32; out_f];

        let graphormer_bias = if cfg.use_graphormer_bias {
            vec![0.0_f32; (cfg.max_distance + 1) * nh]
        } else {
            Vec::new()
        };

        let ln_weight = vec![1.0_f32; out_f];
        let ln_bias = vec![0.0_f32; out_f];

        let weights = GraphTransformerWeights {
            w_q,
            w_k,
            w_v,
            w_o,
            w_e,
            b_q,
            b_k,
            b_v,
            b_o,
            graphormer_bias,
            ln_weight,
            ln_bias,
        };

        Ok(Self { cfg, weights })
    }

    // ─── Attention ────────────────────────────────────────────────────────────

    /// Scaled dot-product attention for a single head.
    ///
    /// Computes `softmax( Q K^T / √d_k + bias ) V`.
    ///
    /// # Arguments
    ///
    /// - `q`, `k`, `v`: `[seq_len × head_dim]` arrays.
    /// - `bias`: `[seq_len × seq_len]` additive bias (may be all zeros).
    /// - `seq_len`: number of query/key/value positions.
    /// - `head_dim`: dimension of each position.
    ///
    /// # Returns
    ///
    /// `[seq_len × head_dim]` output.
    pub fn attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        bias: &[f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        // Compute A = Q K^T / √d_k + bias  [seq_len × seq_len]
        let mut attn = vec![0.0_f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let dot: f32 = (0..head_dim)
                    .map(|k_idx| q[i * head_dim + k_idx] * k[j * head_dim + k_idx])
                    .sum();
                attn[i * seq_len + j] = dot * scale + bias[i * seq_len + j];
            }
        }

        // Row-wise softmax
        softmax_rows(&mut attn, seq_len, seq_len);

        // Output = A V  [seq_len × head_dim]
        let mut out = vec![0.0_f32; seq_len * head_dim];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let a_ij = attn[i * seq_len + j];
                for d in 0..head_dim {
                    out[i * head_dim + d] += a_ij * v[j * head_dim + d];
                }
            }
        }
        out
    }

    // ─── BFS distances ────────────────────────────────────────────────────────

    /// Compute BFS shortest-path distances from every node to every other node.
    ///
    /// Returns a `[n_nodes × n_nodes]` distance matrix (row-major).
    /// Nodes unreachable from the source get distance `max_distance + 1`
    /// (sentinel for "infinity") where `max_distance` is passed at call-time
    /// but if this is called statically we use `usize::MAX / 2` as sentinel.
    /// In practice callers should use the returned value and clamp to their
    /// `max_distance` when indexing the graphormer_bias table.
    ///
    /// Self-distance is 0.
    pub fn compute_bfs_distances(
        row_ptr: &[usize],
        col_idx: &[usize],
        n_nodes: usize,
    ) -> Vec<usize> {
        let sentinel = n_nodes + 1;
        let mut dist = vec![sentinel; n_nodes * n_nodes];

        for src in 0..n_nodes {
            dist[src * n_nodes + src] = 0;
            let mut queue = VecDeque::new();
            queue.push_back(src);

            while let Some(u) = queue.pop_front() {
                let d_u = dist[src * n_nodes + u];
                let start = row_ptr[u];
                let end = row_ptr[u + 1];
                for &nb in &col_idx[start..end] {
                    if dist[src * n_nodes + nb] == sentinel {
                        dist[src * n_nodes + nb] = d_u + 1;
                        queue.push_back(nb);
                    }
                }
            }
        }
        dist
    }

    // ─── Forward ──────────────────────────────────────────────────────────────

    /// Forward pass: multi-head graph attention with optional edge and Graphormer bias.
    ///
    /// # Arguments
    ///
    /// - `node_features`: `[n_nodes × in_features]` row-major.
    /// - `n_nodes`: number of nodes.
    /// - `row_ptr`: CSR row pointer array `[n_nodes + 1]`.
    /// - `col_idx`: CSR column index array `[n_edges]`.
    /// - `edge_features`: `[n_edges × edge_features]`; empty if `edge_features == 0`.
    /// - `distances`: `[n_nodes × n_nodes]` BFS distances; empty if `!use_graphormer_bias`.
    /// - `rng`: random source for dropout.
    ///
    /// # Returns
    ///
    /// `[n_nodes × out_features]`.
    pub fn forward(
        &self,
        node_features: &[f32],
        n_nodes: usize,
        row_ptr: &[usize],
        col_idx: &[usize],
        edge_features: &[f32],
        distances: &[usize],
        rng: &mut LcgRng,
    ) -> GnnResult<Vec<f32>> {
        let in_f = self.cfg.in_features;
        let out_f = self.cfg.out_features;
        let nh = self.cfg.n_heads;
        let head_dim = out_f / nh;
        let ef = self.cfg.edge_features;
        let max_dist = self.cfg.max_distance;

        if n_nodes == 0 {
            return Err(GnnError::EmptyGraph);
        }
        if node_features.len() != n_nodes * in_f {
            return Err(GnnError::NodeFeatureMismatch(
                n_nodes,
                node_features.len() / in_f.max(1),
            ));
        }

        let n_edges = col_idx.len();
        if ef > 0 && !edge_features.is_empty() && edge_features.len() != n_edges * ef {
            return Err(GnnError::EdgeFeatureMismatch(
                n_edges,
                edge_features.len() / ef.max(1),
            ));
        }
        if self.cfg.use_graphormer_bias
            && !distances.is_empty()
            && distances.len() != n_nodes * n_nodes
        {
            return Err(GnnError::DimensionMismatch {
                expected: n_nodes * n_nodes,
                got: distances.len(),
            });
        }

        // Project Q, K, V for all heads.
        // Layout: qkv_h[h][i][d] at index (h * n_nodes + i) * head_dim + d.
        let mut q_all = vec![0.0_f32; nh * n_nodes * head_dim];
        let mut k_all = vec![0.0_f32; nh * n_nodes * head_dim];
        let mut v_all = vec![0.0_f32; nh * n_nodes * head_dim];

        for h in 0..nh {
            let wq_off = h * head_dim * in_f;
            let wk_off = h * head_dim * in_f;
            let wv_off = h * head_dim * in_f;
            let bq_off = h * head_dim;
            let bk_off = h * head_dim;
            let bv_off = h * head_dim;

            for i in 0..n_nodes {
                for d in 0..head_dim {
                    let mut qval = if self.cfg.use_bias {
                        self.weights.b_q[bq_off + d]
                    } else {
                        0.0_f32
                    };
                    let mut kval = if self.cfg.use_bias {
                        self.weights.b_k[bk_off + d]
                    } else {
                        0.0_f32
                    };
                    let mut vval = if self.cfg.use_bias {
                        self.weights.b_v[bv_off + d]
                    } else {
                        0.0_f32
                    };
                    for f_idx in 0..in_f {
                        let x_if = node_features[i * in_f + f_idx];
                        qval += self.weights.w_q[wq_off + d * in_f + f_idx] * x_if;
                        kval += self.weights.w_k[wk_off + d * in_f + f_idx] * x_if;
                        vval += self.weights.w_v[wv_off + d * in_f + f_idx] * x_if;
                    }
                    q_all[(h * n_nodes + i) * head_dim + d] = qval;
                    k_all[(h * n_nodes + i) * head_dim + d] = kval;
                    v_all[(h * n_nodes + i) * head_dim + d] = vval;
                }
            }
        }

        // Compute per-head outputs.
        // We accumulate into a head-indexed buffer: head_out[h][i][d].
        let mut head_out = vec![0.0_f32; nh * n_nodes * head_dim];

        for h in 0..nh {
            let q_slice = &q_all[h * n_nodes * head_dim..(h + 1) * n_nodes * head_dim];
            let k_slice = &k_all[h * n_nodes * head_dim..(h + 1) * n_nodes * head_dim];
            let v_slice = &v_all[h * n_nodes * head_dim..(h + 1) * n_nodes * head_dim];

            // Build additive bias matrix B ∈ ℝ^{n_nodes × n_nodes}.
            let mut bias_mat = vec![0.0_f32; n_nodes * n_nodes];

            // Edge-feature bias: for each edge (i→j), B[i,j] += W_e_h · edge_feat_{e}.
            if ef > 0 && !edge_features.is_empty() && !self.weights.w_e.is_empty() {
                let we_off = h * ef;
                let mut edge_idx = 0usize;
                for i in 0..n_nodes {
                    let start = row_ptr[i];
                    let end = row_ptr[i + 1];
                    for &j in &col_idx[start..end] {
                        let ef_offset = edge_idx * ef;
                        let dot: f32 = self.weights.w_e[we_off..we_off + ef]
                            .iter()
                            .zip(edge_features[ef_offset..ef_offset + ef].iter())
                            .map(|(&w, &v)| w * v)
                            .sum();
                        bias_mat[i * n_nodes + j] += dot;
                        edge_idx += 1;
                    }
                }
            }

            // Graphormer bias: B[i,j] += graphormer_bias[min(dist(i,j), max_dist) * n_heads + h].
            if self.cfg.use_graphormer_bias
                && !distances.is_empty()
                && !self.weights.graphormer_bias.is_empty()
            {
                for i in 0..n_nodes {
                    for j in 0..n_nodes {
                        let raw_d = distances[i * n_nodes + j];
                        let clamped = raw_d.min(max_dist);
                        bias_mat[i * n_nodes + j] += self.weights.graphormer_bias[clamped * nh + h];
                    }
                }
            }

            // Scaled dot-product attention (no dropout path uses rng).
            let o_h = Self::attention(q_slice, k_slice, v_slice, &bias_mat, n_nodes, head_dim);

            // Apply dropout to attention output (not attention weights) at training time.
            if self.cfg.dropout_rate > 0.0 {
                let keep_prob = 1.0 - self.cfg.dropout_rate;
                let scale = if keep_prob > 0.0 {
                    1.0 / keep_prob
                } else {
                    0.0
                };
                for (slot, &val) in head_out[h * n_nodes * head_dim..(h + 1) * n_nodes * head_dim]
                    .iter_mut()
                    .zip(o_h.iter())
                {
                    let keep = rng.next_f32() >= self.cfg.dropout_rate;
                    *slot = if keep { val * scale } else { 0.0 };
                }
            } else {
                head_out[h * n_nodes * head_dim..(h + 1) * n_nodes * head_dim]
                    .copy_from_slice(&o_h);
            }
        }

        // Concatenate heads: out_concat[i][h*head_dim + d] = head_out[h][i][d].
        let mut concat = vec![0.0_f32; n_nodes * out_f];
        for h in 0..nh {
            for i in 0..n_nodes {
                for d in 0..head_dim {
                    concat[i * out_f + h * head_dim + d] =
                        head_out[(h * n_nodes + i) * head_dim + d];
                }
            }
        }

        // Output projection: H = concat W_o^T + b_o.
        let mut projected = vec![0.0_f32; n_nodes * out_f];
        for i in 0..n_nodes {
            for d in 0..out_f {
                let mut acc = if self.cfg.use_bias {
                    self.weights.b_o[d]
                } else {
                    0.0_f32
                };
                for k in 0..out_f {
                    acc += concat[i * out_f + k] * self.weights.w_o[d * out_f + k];
                }
                projected[i * out_f + d] = acc;
            }
        }

        // Layer norm + optional residual (only when in_features == out_features).
        let mut output = vec![0.0_f32; n_nodes * out_f];
        let use_residual = in_f == out_f;

        for i in 0..n_nodes {
            let row: Vec<f32> = if use_residual {
                // Residual: projected + original input (reinterpreted as out_f).
                (0..out_f)
                    .map(|d| projected[i * out_f + d] + node_features[i * in_f + d])
                    .collect()
            } else {
                projected[i * out_f..(i + 1) * out_f].to_vec()
            };
            let normed = layer_norm(&row, &self.weights.ln_weight, &self.weights.ln_bias);
            output[i * out_f..(i + 1) * out_f].copy_from_slice(&normed);
        }

        Ok(output)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_graph_star() -> (Vec<usize>, Vec<usize>, usize) {
        // Star graph: center = 0, leaves = 1,2,3.
        // Edges: 0→1, 0→2, 0→3, 1→0, 2→0, 3→0.
        let n = 4;
        let row_ptr = vec![0, 3, 4, 5, 6];
        let col_idx = vec![1, 2, 3, 0, 0, 0];
        (row_ptr, col_idx, n)
    }

    fn chain_graph(len: usize) -> (Vec<usize>, Vec<usize>) {
        // 0-1-2-..-(len-1), bidirectional.
        let mut row_ptr = vec![0usize; len + 1];
        let mut col_idx = Vec::new();
        for i in 0..len {
            let mut nb_count = 0;
            if i > 0 {
                col_idx.push(i - 1);
                nb_count += 1;
            }
            if i + 1 < len {
                col_idx.push(i + 1);
                nb_count += 1;
            }
            row_ptr[i + 1] = row_ptr[i] + nb_count;
        }
        (row_ptr, col_idx)
    }

    fn make_layer_basic() -> GraphTransformerLayer {
        let cfg = GraphTransformerConfig {
            in_features: 4,
            out_features: 4,
            n_heads: 2,
            edge_features: 0,
            dropout_rate: 0.0,
            use_bias: true,
            use_graphormer_bias: false,
            max_distance: 8,
        };
        let mut rng = LcgRng::new(42);
        GraphTransformerLayer::new(cfg, &mut rng).expect("test invariant: layer must construct")
    }

    // ─── Shape & finiteness ───────────────────────────────────────────────────

    #[test]
    fn output_shape_basic() {
        let (row_ptr, col_idx, n_nodes) = small_graph_star();
        let layer = make_layer_basic();
        let feats = vec![0.1_f32; n_nodes * 4];
        let mut rng = LcgRng::new(1);
        let out = layer
            .forward(&feats, n_nodes, &row_ptr, &col_idx, &[], &[], &mut rng)
            .expect("test invariant: forward must succeed");
        assert_eq!(out.len(), n_nodes * 4);
    }

    #[test]
    fn output_finite() {
        let (row_ptr, col_idx, n_nodes) = small_graph_star();
        let layer = make_layer_basic();
        let feats: Vec<f32> = (0..n_nodes * 4).map(|i| i as f32 * 0.01).collect();
        let mut rng = LcgRng::new(2);
        let out = layer
            .forward(&feats, n_nodes, &row_ptr, &col_idx, &[], &[], &mut rng)
            .expect("test invariant: forward must succeed");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all outputs must be finite"
        );
    }

    // ─── Head counts ─────────────────────────────────────────────────────────

    #[test]
    fn n_heads_1() {
        let cfg = GraphTransformerConfig {
            in_features: 4,
            out_features: 4,
            n_heads: 1,
            edge_features: 0,
            dropout_rate: 0.0,
            use_bias: true,
            use_graphormer_bias: false,
            max_distance: 4,
        };
        let mut rng = LcgRng::new(10);
        let layer =
            GraphTransformerLayer::new(cfg, &mut rng).expect("test invariant: must construct");
        let (row_ptr, col_idx, n_nodes) = small_graph_star();
        let feats = vec![0.5_f32; n_nodes * 4];
        let out = layer
            .forward(&feats, n_nodes, &row_ptr, &col_idx, &[], &[], &mut rng)
            .expect("test invariant: must succeed");
        assert_eq!(out.len(), n_nodes * 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn n_heads_4() {
        let cfg = GraphTransformerConfig {
            in_features: 8,
            out_features: 8,
            n_heads: 4,
            edge_features: 0,
            dropout_rate: 0.0,
            use_bias: false,
            use_graphormer_bias: false,
            max_distance: 4,
        };
        let mut rng = LcgRng::new(11);
        let layer =
            GraphTransformerLayer::new(cfg, &mut rng).expect("test invariant: must construct");
        let n_nodes = 5;
        let row_ptr = vec![0, 1, 2, 3, 4, 5];
        let col_idx = vec![1, 2, 3, 4, 0];
        let feats = vec![0.2_f32; n_nodes * 8];
        let out = layer
            .forward(&feats, n_nodes, &row_ptr, &col_idx, &[], &[], &mut rng)
            .expect("test invariant: must succeed");
        assert_eq!(out.len(), n_nodes * 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ─── Edge features ────────────────────────────────────────────────────────

    #[test]
    fn no_edge_features() {
        let cfg = GraphTransformerConfig {
            in_features: 4,
            out_features: 4,
            n_heads: 2,
            edge_features: 0,
            dropout_rate: 0.0,
            use_bias: true,
            use_graphormer_bias: false,
            max_distance: 4,
        };
        let mut rng = LcgRng::new(20);
        let layer =
            GraphTransformerLayer::new(cfg, &mut rng).expect("test invariant: must construct");
        let (row_ptr, col_idx, n_nodes) = small_graph_star();
        let feats = vec![0.3_f32; n_nodes * 4];
        let out = layer
            .forward(&feats, n_nodes, &row_ptr, &col_idx, &[], &[], &mut rng)
            .expect("test invariant: must succeed");
        assert_eq!(out.len(), n_nodes * 4);
    }

    #[test]
    fn with_edge_features() {
        let cfg = GraphTransformerConfig {
            in_features: 4,
            out_features: 4,
            n_heads: 2,
            edge_features: 4,
            dropout_rate: 0.0,
            use_bias: true,
            use_graphormer_bias: false,
            max_distance: 4,
        };
        let mut rng = LcgRng::new(21);
        let layer =
            GraphTransformerLayer::new(cfg, &mut rng).expect("test invariant: must construct");
        let (row_ptr, col_idx, n_nodes) = small_graph_star();
        let n_edges = col_idx.len();
        let feats = vec![0.1_f32; n_nodes * 4];
        let ef = vec![0.5_f32; n_edges * 4];
        let out = layer
            .forward(&feats, n_nodes, &row_ptr, &col_idx, &ef, &[], &mut rng)
            .expect("test invariant: must succeed");
        assert_eq!(out.len(), n_nodes * 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ─── Graphormer bias ──────────────────────────────────────────────────────

    #[test]
    fn with_graphormer_bias() {
        let n_nodes = 4;
        let cfg = GraphTransformerConfig {
            in_features: 4,
            out_features: 4,
            n_heads: 2,
            edge_features: 0,
            dropout_rate: 0.0,
            use_bias: false,
            use_graphormer_bias: true,
            max_distance: 4,
        };
        let mut rng = LcgRng::new(30);
        let layer =
            GraphTransformerLayer::new(cfg, &mut rng).expect("test invariant: must construct");
        let (row_ptr, col_idx, _) = small_graph_star();
        let feats = vec![0.1_f32; n_nodes * 4];
        let dists = GraphTransformerLayer::compute_bfs_distances(&row_ptr, &col_idx, n_nodes);
        let out = layer
            .forward(&feats, n_nodes, &row_ptr, &col_idx, &[], &dists, &mut rng)
            .expect("test invariant: must succeed");
        assert_eq!(out.len(), n_nodes * 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ─── BFS distances ────────────────────────────────────────────────────────

    #[test]
    fn bfs_distances_star_graph() {
        let (row_ptr, col_idx, n_nodes) = small_graph_star();
        let dists = GraphTransformerLayer::compute_bfs_distances(&row_ptr, &col_idx, n_nodes);
        // Self-distance
        for i in 0..n_nodes {
            assert_eq!(dists[i * n_nodes + i], 0, "self-distance must be 0");
        }
        // Center (0) to leaf (1,2,3): distance 1
        assert_eq!(dists[1], 1);
        assert_eq!(dists[2], 1);
        assert_eq!(dists[3], 1);
        // Leaf to center: distance 1
        assert_eq!(dists[n_nodes], 1);
        // Leaf (1) to leaf (2): must pass through center, distance 2
        assert_eq!(dists[n_nodes + 2], 2);
        assert_eq!(dists[n_nodes + 3], 2);
    }

    #[test]
    fn bfs_distances_chain() {
        let n = 4;
        let (row_ptr, col_idx) = chain_graph(n);
        let dists = GraphTransformerLayer::compute_bfs_distances(&row_ptr, &col_idx, n);
        // 0-1-2-3: dist(0,3)=3, dist(0,2)=2, dist(1,3)=2.
        assert_eq!(dists[3], 3, "chain: dist(0,3) should be 3");
        assert_eq!(dists[2], 2, "chain: dist(0,2) should be 2");
        assert_eq!(dists[n + 3], 2, "chain: dist(1,3) should be 2");
        assert_eq!(dists[3 * n], 3, "chain: dist(3,0) should be 3");
    }

    #[test]
    fn bfs_distances_disconnected() {
        // 0-1 and isolated node 2.
        let n = 3;
        let row_ptr = vec![0, 1, 2, 2];
        let col_idx = vec![1, 0];
        let dists = GraphTransformerLayer::compute_bfs_distances(&row_ptr, &col_idx, n);
        let sentinel = n + 1;
        // Node 2 is disconnected: dist(0,2) = sentinel.
        assert_eq!(
            dists[2], sentinel,
            "disconnected node should have sentinel distance"
        );
        assert_eq!(dists[2 * n], sentinel, "disconnected node to others");
        // Self-distance 0.
        assert_eq!(dists[2 * n + 2], 0);
    }

    // ─── Attention function ──────────────────────────────────────────────────

    #[test]
    fn attention_output_shape() {
        let seq_len = 4;
        let head_dim = 8;
        let q = vec![0.1_f32; seq_len * head_dim];
        let k = vec![0.1_f32; seq_len * head_dim];
        let v = vec![0.2_f32; seq_len * head_dim];
        let bias = vec![0.0_f32; seq_len * seq_len];
        let out = GraphTransformerLayer::attention(&q, &k, &v, &bias, seq_len, head_dim);
        assert_eq!(
            out.len(),
            seq_len * head_dim,
            "attention output shape mismatch"
        );
    }

    #[test]
    fn attention_softmax_rows_sum_to_1() {
        // Verify that for uniform Q/K, each softmax row sums to 1.
        let seq_len = 5;
        let head_dim = 4;
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((seq_len * head_dim - i) as f32) * 0.05)
            .collect();
        let v = vec![1.0_f32; seq_len * head_dim];
        let bias = vec![0.0_f32; seq_len * seq_len];

        // We re-compute the attention weights to check softmax sum.
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let mut attn = vec![0.0_f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let dot: f32 = (0..head_dim)
                    .map(|d| q[i * head_dim + d] * k[j * head_dim + d])
                    .sum();
                attn[i * seq_len + j] = dot * scale;
            }
        }
        softmax_rows(&mut attn, seq_len, seq_len);
        for i in 0..seq_len {
            let row_sum: f32 = attn[i * seq_len..(i + 1) * seq_len].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "softmax row {i} sum={row_sum:.6} != 1.0"
            );
        }

        // Full attention output should also be finite.
        let out = GraphTransformerLayer::attention(&q, &k, &v, &bias, seq_len, head_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ─── Error cases ─────────────────────────────────────────────────────────

    #[test]
    fn err_features_not_divisible() {
        let cfg = GraphTransformerConfig {
            in_features: 4,
            out_features: 6,
            n_heads: 4,
            edge_features: 0,
            dropout_rate: 0.0,
            use_bias: true,
            use_graphormer_bias: false,
            max_distance: 4,
        };
        let mut rng = LcgRng::new(99);
        let result = GraphTransformerLayer::new(cfg, &mut rng);
        assert!(
            matches!(result, Err(GnnError::InvalidAttentionHeads { .. })),
            "expected InvalidAttentionHeads error"
        );
    }

    #[test]
    fn err_n_nodes_zero() {
        let layer = make_layer_basic();
        let mut rng = LcgRng::new(5);
        let err = layer.forward(&[], 0, &[0], &[], &[], &[], &mut rng);
        assert!(matches!(err, Err(GnnError::EmptyGraph)));
    }

    #[test]
    fn err_in_features_zero() {
        let cfg = GraphTransformerConfig {
            in_features: 0,
            out_features: 4,
            n_heads: 2,
            edge_features: 0,
            dropout_rate: 0.0,
            use_bias: true,
            use_graphormer_bias: false,
            max_distance: 4,
        };
        let mut rng = LcgRng::new(6);
        let result = GraphTransformerLayer::new(cfg, &mut rng);
        assert!(result.is_err(), "expected error for in_features=0");
    }

    // ─── Residual connection ──────────────────────────────────────────────────

    #[test]
    fn residual_only_when_dims_match() {
        // When in_features != out_features: no residual, but no crash.
        let cfg = GraphTransformerConfig {
            in_features: 4,
            out_features: 8,
            n_heads: 4,
            edge_features: 0,
            dropout_rate: 0.0,
            use_bias: false,
            use_graphormer_bias: false,
            max_distance: 4,
        };
        let mut rng = LcgRng::new(50);
        let layer =
            GraphTransformerLayer::new(cfg, &mut rng).expect("test invariant: must construct");
        let (row_ptr, col_idx, n_nodes) = small_graph_star();
        let feats = vec![1.0_f32; n_nodes * 4];
        let out = layer
            .forward(&feats, n_nodes, &row_ptr, &col_idx, &[], &[], &mut rng)
            .expect("test invariant: forward must succeed");
        assert_eq!(
            out.len(),
            n_nodes * 8,
            "output should be n_nodes × out_features"
        );
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all outputs must be finite"
        );
    }

    // ─── Additional coverage ─────────────────────────────────────────────────

    #[test]
    fn forward_with_dropout() {
        let cfg = GraphTransformerConfig {
            in_features: 4,
            out_features: 4,
            n_heads: 2,
            edge_features: 0,
            dropout_rate: 0.5,
            use_bias: true,
            use_graphormer_bias: false,
            max_distance: 4,
        };
        let mut rng = LcgRng::new(77);
        let layer =
            GraphTransformerLayer::new(cfg, &mut rng).expect("test invariant: must construct");
        let (row_ptr, col_idx, n_nodes) = small_graph_star();
        let feats = vec![0.5_f32; n_nodes * 4];
        let out = layer
            .forward(&feats, n_nodes, &row_ptr, &col_idx, &[], &[], &mut rng)
            .expect("test invariant: must succeed");
        assert_eq!(out.len(), n_nodes * 4);
        // Output may contain zeros due to dropout but must be finite.
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bfs_self_distance_always_zero() {
        let n = 5;
        let (row_ptr, col_idx) = chain_graph(n);
        let dists = GraphTransformerLayer::compute_bfs_distances(&row_ptr, &col_idx, n);
        for i in 0..n {
            assert_eq!(dists[i * n + i], 0, "self-distance of node {i} must be 0");
        }
    }
}
