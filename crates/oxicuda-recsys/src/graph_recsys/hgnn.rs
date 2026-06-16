//! HGNN — Heterogeneous Graph Neural Network for multi-relation recommendation.
//!
//! Reference: inspired by HetGNN (Zhang et al., KDD 2019) and HAN (Wang et al., WWW 2019).
//!
//! Architecture:
//!   Maintains one embedding table per node-type. Each propagation layer
//!   aggregates incoming messages across all relation types:
//!     1. For each incoming edge (src_type, src, rel, dst_type, dst):
//!        `msg = ReLU(W_rel[r] · h_src)`
//!     2. For each destination node: new_h = h_dst + mean(all incoming msgs)
//!     3. Normalize each embedding to unit norm after the layer.
//!   The procedure is repeated for `n_layers` rounds.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// In-place ReLU.
#[inline]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// Matrix-vector product `W · v` where `W` is `d × d` (row-major) and `v` has
/// length `d`.
fn matvec(w: &[f32], v: &[f32], d: usize) -> Vec<f32> {
    (0..d)
        .map(|r| {
            w[r * d..(r + 1) * d]
                .iter()
                .zip(v.iter())
                .map(|(&wij, &vj)| wij * vj)
                .sum::<f32>()
        })
        .collect()
}

/// Normalize a slice to unit L2 norm in-place. Does nothing if the norm is
/// below `1e-10` (all-zero / near-zero vector).
fn normalize_inplace(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// A typed, directed edge in the heterogeneous interaction graph.
#[derive(Debug, Clone)]
pub struct HeteroEdge {
    /// Source node index (local to `src_type`'s embedding table).
    pub src: usize,
    /// Type index of the source node.
    pub src_type: usize,
    /// Destination node index (local to `dst_type`'s embedding table).
    pub dst: usize,
    /// Type index of the destination node.
    pub dst_type: usize,
    /// Relation index; selects the projection matrix `W_rel[rel]`.
    pub rel: usize,
}

/// Hyper-parameters for [`Hgnn`].
#[derive(Debug, Clone)]
pub struct HgnnConfig {
    /// Number of distinct node types (must be >= 1).
    pub n_node_types: usize,
    /// Number of distinct relation types (must be >= 1).
    pub n_relations: usize,
    /// Embedding dimensionality (must be >= 1).
    pub embed_dim: usize,
    /// Number of message-passing layers (must be >= 1).
    pub n_layers: usize,
}

/// Heterogeneous Graph Neural Network for recommendation.
///
/// Embedding tables are indexed per node-type; type 0 is treated as *users*
/// and type 1 as *items* for the `score` convenience method.
pub struct Hgnn {
    /// Per-node-type embedding tables: `embeddings[t]` has length
    /// `n_per_type[t] * embed_dim`.
    embeddings: Vec<Vec<f32>>,
    /// Number of nodes for each node type.
    n_per_type: Vec<usize>,
    /// Per-relation projection matrices: `w_rel[r]` has length `embed_dim²`
    /// (row-major `embed_dim × embed_dim`).
    w_rel: Vec<Vec<f32>>,
    /// Model configuration.
    cfg: HgnnConfig,
}

impl Hgnn {
    /// Construct a new `Hgnn` with Kaiming-uniform initialisation.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidConfig`] when any config field is zero.
    /// - [`RecsysError::InvalidConfig`] when `n_per_type.len() != cfg.n_node_types`.
    /// - [`RecsysError::InvalidConfig`] when any `n_per_type[t] == 0`.
    pub fn new(n_per_type: &[usize], cfg: HgnnConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if cfg.n_node_types == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "n_node_types must be >= 1".into(),
            });
        }
        if cfg.n_relations == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "n_relations must be >= 1".into(),
            });
        }
        if cfg.embed_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
        }
        if cfg.n_layers == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "n_layers must be >= 1".into(),
            });
        }
        if n_per_type.len() != cfg.n_node_types {
            return Err(RecsysError::InvalidConfig {
                msg: format!(
                    "n_per_type has {} entries but n_node_types={}",
                    n_per_type.len(),
                    cfg.n_node_types
                ),
            });
        }
        for (t, &n) in n_per_type.iter().enumerate() {
            if n == 0 {
                return Err(RecsysError::InvalidConfig {
                    msg: format!("node type {t} has 0 nodes"),
                });
            }
        }

        let d = cfg.embed_dim;
        let emb_scale = (1.0 / d as f32).sqrt();
        let w_scale = (2.0 / d as f32).sqrt();

        let embeddings: Vec<Vec<f32>> = n_per_type
            .iter()
            .map(|&n| (0..n * d).map(|_| rng.next_normal() * emb_scale).collect())
            .collect();

        let w_rel: Vec<Vec<f32>> = (0..cfg.n_relations)
            .map(|_| (0..d * d).map(|_| rng.next_normal() * w_scale).collect())
            .collect();

        Ok(Self {
            embeddings,
            n_per_type: n_per_type.to_vec(),
            w_rel,
            cfg,
        })
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Core propagation
    // ──────────────────────────────────────────────────────────────────────────

    /// Run `cfg.n_layers` rounds of heterogeneous message passing.
    ///
    /// Each layer:
    ///   1. For each destination node, aggregate `ReLU(W_rel[r] · h_src)` over
    ///      all incoming edges, then take the mean.
    ///   2. Add the original embedding (residual connection):
    ///      `h_dst_new = mean(msgs) + h_dst`.
    ///   3. Normalize each updated embedding to unit L2 norm.
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInteraction`] when `edges` is empty.
    /// - [`RecsysError::InvalidConfig`] when any edge references an out-of-bounds
    ///   type or node index.
    pub fn propagate(&mut self, edges: &[HeteroEdge]) -> RecsysResult<()> {
        if edges.is_empty() {
            return Err(RecsysError::EmptyInteraction);
        }
        // Validate all edges upfront.
        for (idx, e) in edges.iter().enumerate() {
            self.validate_edge(e, idx)?;
        }

        let d = self.cfg.embed_dim;

        for _layer in 0..self.cfg.n_layers {
            // Temporary buffers: accumulated messages and counts per node-type.
            let mut acc: Vec<Vec<f32>> = self
                .n_per_type
                .iter()
                .map(|&n| vec![0.0_f32; n * d])
                .collect();
            let mut cnt: Vec<Vec<u32>> = self.n_per_type.iter().map(|&n| vec![0u32; n]).collect();

            // Accumulate messages from all incoming edges.
            for e in edges {
                let src_emb = &self.embeddings[e.src_type][e.src * d..(e.src + 1) * d].to_vec();
                let w = &self.w_rel[e.rel];
                let projected = matvec(w, src_emb, d);
                // Apply ReLU.
                let msg: Vec<f32> = projected.iter().map(|&v| relu(v)).collect();

                let dst_acc = &mut acc[e.dst_type][e.dst * d..(e.dst + 1) * d];
                for (a, &m) in dst_acc.iter_mut().zip(msg.iter()) {
                    *a += m;
                }
                cnt[e.dst_type][e.dst] += 1;
            }

            // Build new embeddings: residual + mean-pooled messages, then
            // normalize to unit norm.
            let mut new_embeddings: Vec<Vec<f32>> = self
                .embeddings
                .iter()
                .zip(self.n_per_type.iter())
                .map(|(emb, &n)| {
                    let mut v = vec![0.0_f32; n * d];
                    // Copy current embeddings as residual.
                    v.copy_from_slice(emb);
                    v
                })
                .collect();

            for (t, n) in self.n_per_type.iter().enumerate() {
                for node in 0..*n {
                    let c = cnt[t][node];
                    if c == 0 {
                        // No incoming edges → keep original embedding (already
                        // copied) and normalize.
                        let row = &mut new_embeddings[t][node * d..(node + 1) * d];
                        normalize_inplace(row);
                        continue;
                    }
                    let inv_c = 1.0 / c as f32;
                    let acc_row = &acc[t][node * d..(node + 1) * d];
                    let new_row = &mut new_embeddings[t][node * d..(node + 1) * d];
                    // Residual: new_row already holds h_dst.
                    for (nr, &ar) in new_row.iter_mut().zip(acc_row.iter()) {
                        *nr += ar * inv_c;
                    }
                    normalize_inplace(new_row);
                }
            }

            self.embeddings = new_embeddings;
        }

        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Scoring
    // ──────────────────────────────────────────────────────────────────────────

    /// Inner-product score between a *user* (node-type 0, index `user_id`) and
    /// an *item* (node-type 1, index `item_id`).
    ///
    /// # Errors
    /// - [`RecsysError::InvalidConfig`] when fewer than 2 node types exist.
    /// - [`RecsysError::UnknownUser`] / [`RecsysError::UnknownItem`] on
    ///   out-of-bounds indices.
    pub fn score(&self, user_id: usize, item_id: usize) -> RecsysResult<f32> {
        if self.cfg.n_node_types < 2 {
            return Err(RecsysError::InvalidConfig {
                msg: "score() requires at least 2 node types (type-0 = users, type-1 = items)"
                    .into(),
            });
        }
        if user_id >= self.n_per_type[0] {
            return Err(RecsysError::UnknownUser { id: user_id });
        }
        if item_id >= self.n_per_type[1] {
            return Err(RecsysError::UnknownItem { id: item_id });
        }
        let d = self.cfg.embed_dim;
        let user_vec = &self.embeddings[0][user_id * d..(user_id + 1) * d];
        let item_vec = &self.embeddings[1][item_id * d..(item_id + 1) * d];
        let dot: f32 = user_vec
            .iter()
            .zip(item_vec.iter())
            .map(|(&u, &v)| u * v)
            .sum();
        Ok(dot)
    }

    /// Get the embedding slice for node `node_id` of `node_type`.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidConfig`] when `node_type >= n_node_types`.
    /// - [`RecsysError::ItemOutOfBounds`] when `node_id >= n_per_type[node_type]`.
    pub fn embed(&self, node_type: usize, node_id: usize) -> RecsysResult<&[f32]> {
        if node_type >= self.cfg.n_node_types {
            return Err(RecsysError::InvalidConfig {
                msg: format!(
                    "node_type {node_type} >= n_node_types {}",
                    self.cfg.n_node_types
                ),
            });
        }
        let n = self.n_per_type[node_type];
        if node_id >= n {
            return Err(RecsysError::ItemOutOfBounds { idx: node_id, n });
        }
        let d = self.cfg.embed_dim;
        Ok(&self.embeddings[node_type][node_id * d..(node_id + 1) * d])
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ──────────────────────────────────────────────────────────────────────────

    fn validate_edge(&self, e: &HeteroEdge, edge_idx: usize) -> RecsysResult<()> {
        if e.src_type >= self.cfg.n_node_types {
            return Err(RecsysError::InvalidConfig {
                msg: format!(
                    "edge[{edge_idx}].src_type={} >= n_node_types={}",
                    e.src_type, self.cfg.n_node_types
                ),
            });
        }
        if e.dst_type >= self.cfg.n_node_types {
            return Err(RecsysError::InvalidConfig {
                msg: format!(
                    "edge[{edge_idx}].dst_type={} >= n_node_types={}",
                    e.dst_type, self.cfg.n_node_types
                ),
            });
        }
        if e.rel >= self.cfg.n_relations {
            return Err(RecsysError::InvalidConfig {
                msg: format!(
                    "edge[{edge_idx}].rel={} >= n_relations={}",
                    e.rel, self.cfg.n_relations
                ),
            });
        }
        let src_n = self.n_per_type[e.src_type];
        if e.src >= src_n {
            return Err(RecsysError::ItemOutOfBounds {
                idx: e.src,
                n: src_n,
            });
        }
        let dst_n = self.n_per_type[e.dst_type];
        if e.dst >= dst_n {
            return Err(RecsysError::ItemOutOfBounds {
                idx: e.dst,
                n: dst_n,
            });
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn small_model(rng: &mut LcgRng) -> Hgnn {
        let cfg = HgnnConfig {
            n_node_types: 2,
            n_relations: 2,
            embed_dim: 4,
            n_layers: 2,
        };
        Hgnn::new(&[4, 6], cfg, rng).expect("small_model construction should succeed")
    }

    fn simple_edges() -> Vec<HeteroEdge> {
        vec![
            HeteroEdge {
                src: 0,
                src_type: 0,
                dst: 0,
                dst_type: 1,
                rel: 0,
            },
            HeteroEdge {
                src: 0,
                src_type: 0,
                dst: 1,
                dst_type: 1,
                rel: 1,
            },
            HeteroEdge {
                src: 1,
                src_type: 0,
                dst: 2,
                dst_type: 1,
                rel: 0,
            },
            HeteroEdge {
                src: 2,
                src_type: 0,
                dst: 0,
                dst_type: 1,
                rel: 1,
            },
        ]
    }

    #[test]
    fn construction_succeeds() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert_eq!(model.n_per_type[0], 4);
        assert_eq!(model.n_per_type[1], 6);
    }

    #[test]
    fn err_embed_dim_zero() {
        let mut rng = make_rng();
        let cfg = HgnnConfig {
            n_node_types: 2,
            n_relations: 1,
            embed_dim: 0,
            n_layers: 1,
        };
        assert!(matches!(
            Hgnn::new(&[2, 3], cfg, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn err_n_relations_zero() {
        let mut rng = make_rng();
        let cfg = HgnnConfig {
            n_node_types: 2,
            n_relations: 0,
            embed_dim: 4,
            n_layers: 1,
        };
        assert!(matches!(
            Hgnn::new(&[2, 3], cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn err_n_layers_zero() {
        let mut rng = make_rng();
        let cfg = HgnnConfig {
            n_node_types: 2,
            n_relations: 2,
            embed_dim: 4,
            n_layers: 0,
        };
        assert!(matches!(
            Hgnn::new(&[2, 3], cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn err_mismatch_n_per_type_length() {
        let mut rng = make_rng();
        let cfg = HgnnConfig {
            n_node_types: 3,
            n_relations: 1,
            embed_dim: 4,
            n_layers: 1,
        };
        // Providing only 2 entries when 3 types are declared.
        assert!(matches!(
            Hgnn::new(&[2, 3], cfg, &mut rng),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn propagate_ok_and_score_finite() {
        let mut rng = make_rng();
        let mut model = small_model(&mut rng);
        model
            .propagate(&simple_edges())
            .expect("propagate should succeed");
        let s = model.score(0, 0).expect("score should succeed");
        assert!(s.is_finite(), "score must be finite, got {s}");
    }

    #[test]
    fn embeddings_unit_norm_after_propagation() {
        let mut rng = make_rng();
        let mut model = small_model(&mut rng);
        model
            .propagate(&simple_edges())
            .expect("propagate should succeed");
        for t in 0..model.cfg.n_node_types {
            for node in 0..model.n_per_type[t] {
                let emb = model.embed(t, node).expect("embed should succeed");
                let norm: f32 = emb.iter().map(|&x| x * x).sum::<f32>().sqrt();
                assert!(
                    (norm - 1.0).abs() < 1e-5,
                    "type={t} node={node} norm={norm:.6} should be ~1.0"
                );
            }
        }
    }

    #[test]
    fn err_propagate_empty_edges() {
        let mut rng = make_rng();
        let mut model = small_model(&mut rng);
        assert!(matches!(
            model.propagate(&[]),
            Err(RecsysError::EmptyInteraction)
        ));
    }

    #[test]
    fn err_score_unknown_user() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert!(matches!(
            model.score(999, 0),
            Err(RecsysError::UnknownUser { .. })
        ));
    }

    #[test]
    fn err_score_unknown_item() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert!(matches!(
            model.score(0, 999),
            Err(RecsysError::UnknownItem { .. })
        ));
    }

    #[test]
    fn err_embed_type_out_of_bounds() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert!(matches!(
            model.embed(999, 0),
            Err(RecsysError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn err_embed_node_out_of_bounds() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert!(matches!(
            model.embed(0, 999),
            Err(RecsysError::ItemOutOfBounds { .. })
        ));
    }

    #[test]
    fn embed_returns_correct_length() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        let emb = model.embed(0, 0).expect("embed should succeed");
        assert_eq!(emb.len(), model.cfg.embed_dim);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut rng_a = LcgRng::new(7);
        let mut rng_b = LcgRng::new(7);
        let mut model_a = small_model(&mut rng_a);
        let mut model_b = small_model(&mut rng_b);
        model_a
            .propagate(&simple_edges())
            .expect("value should be present");
        model_b
            .propagate(&simple_edges())
            .expect("value should be present");
        let s_a = model_a.score(1, 2).expect("score should succeed");
        let s_b = model_b.score(1, 2).expect("score should succeed");
        assert!(
            (s_a - s_b).abs() < 1e-6,
            "same seed must yield same score ({s_a} vs {s_b})"
        );
    }

    #[test]
    fn three_node_type_model() {
        let mut rng = make_rng();
        let cfg = HgnnConfig {
            n_node_types: 3,
            n_relations: 3,
            embed_dim: 8,
            n_layers: 1,
        };
        let model = Hgnn::new(&[5, 10, 3], cfg, &mut rng).expect("3-type model should succeed");
        assert_eq!(model.n_per_type.len(), 3);
        let emb = model.embed(2, 0).expect("type-2 embed should succeed");
        assert_eq!(emb.len(), 8);
    }
}
