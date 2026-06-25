//! Heterogeneous message-passing dispatch over a [`HeteroGraph`].
//!
//! Where [`crate::layers::rgcn::RgcnLayer`] consumes a *flat* `&[CsrGraph]` (one
//! anonymous CSR per relation), real heterogeneous graphs carry **named node
//! types** and **typed relations** `(src_type, rel, dst_type)`. `HeteroConv`
//! dispatches a relation-specific linear message map over every stored relation
//! of a [`HeteroGraph`] and reduces the per-relation messages into one output
//! embedding *per destination node type* — the pattern of PyG's `HeteroConv`
//! wrapping R-GCN-style relation convolutions (Schlichtkrull et al. 2018).
//!
//! For each relation `r = (s, rel, d)` the adjacency is the [`CsrGraph`] stored
//! by [`HeteroGraph`], which is **source-indexed**: `A_r.neighbors(j)` lists the
//! destination nodes that source `j` sends to (edges `(src, dst)`). Messages are
//! therefore *scattered* from sources to destinations and the destination
//! in-degree `c_{i,r}` (number of incoming edges of relation `r`) is counted on
//! the fly:
//!
//! ```text
//!   m_i^r = (1 / c_{i,r}) · Σ_{j → i in A_r}  H_s[j] · W_r        (relation message)
//!   c_{i,r} = max(1, indeg_r(i))                                  (mean normaliser)
//! ```
//!
//! and the destination-type output stacks a self transform with the sum over all
//! relations that point *into* that type:
//!
//! ```text
//!   H'_d[i] = H_d[i] · W_self^d  +  Σ_{r : dst(r)=d}  m_i^r
//! ```
//!
//! Each relation owns an `in_dim(src) × out_dim(dst)` weight matrix; each
//! destination type owns a `in_dim(d) × out_dim(d)` self-loop matrix. Output
//! features per type all share the configured `out_features`.

use std::collections::HashMap;

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;
use crate::graph::heterogeneous::HeteroGraph;

/// Configuration for a [`HeteroConv`] layer.
#[derive(Debug, Clone)]
pub struct HeteroConvConfig {
    /// Per-node-type input feature dimension, keyed by type name.
    pub in_features: HashMap<String, usize>,
    /// Shared output feature dimension for every node type.
    pub out_features: usize,
    /// Apply mean (in-degree) normalisation to relation messages when `true`;
    /// otherwise messages are plain neighbour sums.
    pub mean_aggregation: bool,
    /// Add a per-destination-type self-loop transform `H_d · W_self^d`.
    pub add_self_loops: bool,
}

impl HeteroConvConfig {
    /// Validate the configuration: non-empty type map and positive dimensions.
    pub fn validate(&self) -> GnnResult<()> {
        if self.out_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "out_features must be > 0".to_string(),
            ));
        }
        if self.in_features.is_empty() {
            return Err(GnnError::InvalidLayerConfig(
                "in_features must declare at least one node type".to_string(),
            ));
        }
        for (name, &dim) in &self.in_features {
            if dim == 0 {
                return Err(GnnError::InvalidLayerConfig(format!(
                    "in_features for type '{name}' must be > 0"
                )));
            }
        }
        Ok(())
    }
}

/// Heterogeneous convolution dispatcher.
pub struct HeteroConv {
    config: HeteroConvConfig,
}

/// Per-type weight matrices supplied to [`HeteroConv::forward`].
///
/// All matrices are row-major and flattened.
#[derive(Debug, Clone)]
pub struct HeteroConvWeights {
    /// Relation weight `W_r`, keyed by the `(src_type, rel, dst_type)` triple;
    /// shape `[in_dim(src) × out_features]`.
    pub relation_weight: HashMap<(String, String, String), Vec<f32>>,
    /// Self-loop weight `W_self^d`, keyed by destination type name;
    /// shape `[in_dim(d) × out_features]`. Only consulted when
    /// `add_self_loops` is set.
    pub self_weight: HashMap<String, Vec<f32>>,
}

impl HeteroConv {
    /// Construct a dispatcher from a validated configuration.
    pub fn new(config: HeteroConvConfig) -> GnnResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Input dimension for a node type, or an error if it was not declared.
    fn in_dim(&self, node_type: &str) -> GnnResult<usize> {
        self.config
            .in_features
            .get(node_type)
            .copied()
            .ok_or_else(|| {
                GnnError::InvalidLayerConfig(format!(
                    "node type '{node_type}' missing from in_features"
                ))
            })
    }

    /// Dispatch heterogeneous message passing.
    ///
    /// # Arguments
    ///
    /// * `graph`    — the heterogeneous graph (typed relations).
    /// * `features` — per-node-type feature matrices, keyed by type name; each
    ///   `[n_nodes(type) × in_dim(type)]`, row-major.
    /// * `weights`  — per-relation and per-self-loop weight matrices.
    ///
    /// # Returns
    ///
    /// A map from destination-type name to its `[n_nodes(type) × out_features]`
    /// output embedding. Only types that are a destination of at least one
    /// relation (or have a self-loop) appear in the output.
    ///
    /// # Errors
    ///
    /// * [`GnnError::NodeFeatureMismatch`] if a feature matrix has the wrong size.
    /// * [`GnnError::WeightShapeMismatch`] if a weight matrix has the wrong size.
    /// * [`GnnError::InvalidLayerConfig`] if a relation references an undeclared
    ///   type or a required weight matrix is missing.
    pub fn forward(
        &self,
        graph: &HeteroGraph,
        features: &HashMap<String, Vec<f32>>,
        weights: &HeteroConvWeights,
    ) -> GnnResult<HashMap<String, Vec<f32>>> {
        let out_f = self.config.out_features;

        // Validate every supplied feature matrix up-front.
        for node_type in graph.node_types() {
            let n = graph.n_nodes(node_type)?;
            let dim = self.in_dim(node_type)?;
            if let Some(feat) = features.get(node_type) {
                if feat.len() != n * dim {
                    return Err(GnnError::NodeFeatureMismatch(n, feat.len() / dim.max(1)));
                }
            }
        }

        let mut out: HashMap<String, Vec<f32>> = HashMap::new();

        // ── Self-loop transforms per destination type. ───────────────────────
        if self.config.add_self_loops {
            for node_type in graph.node_types() {
                let n = graph.n_nodes(node_type)?;
                let dim = self.in_dim(node_type)?;
                let feat = match features.get(node_type) {
                    Some(f) => f,
                    None => continue, // no features ⇒ nothing to self-transform
                };
                let w = weights.self_weight.get(node_type).ok_or_else(|| {
                    GnnError::InvalidLayerConfig(format!(
                        "missing self weight for destination type '{node_type}'"
                    ))
                })?;
                if w.len() != dim * out_f {
                    return Err(GnnError::WeightShapeMismatch {
                        r: dim,
                        c: out_f,
                        d: dim,
                    });
                }
                let acc = out
                    .entry(node_type.clone())
                    .or_insert_with(|| vec![0.0; n * out_f]);
                dense_matmul_accumulate(feat, w, n, dim, out_f, acc);
            }
        }

        // ── Per-relation message dispatch. ───────────────────────────────────
        for (src_type, rel, dst_type) in graph.edge_types() {
            let adjacency = graph.adjacency(src_type, dst_type)?;
            let src_feat = match features.get(src_type) {
                Some(f) => f,
                None => continue, // no source features ⇒ no message to send
            };
            let src_dim = self.in_dim(src_type)?;
            let n_dst = graph.n_nodes(dst_type)?;

            let triple = (src_type.clone(), rel.clone(), dst_type.clone());
            let w = weights.relation_weight.get(&triple).ok_or_else(|| {
                GnnError::InvalidLayerConfig(format!(
                    "missing relation weight for ({src_type}, {rel}, {dst_type})"
                ))
            })?;
            if w.len() != src_dim * out_f {
                return Err(GnnError::WeightShapeMismatch {
                    r: src_dim,
                    c: out_f,
                    d: src_dim,
                });
            }

            // First project the *source* features once (W_r is shared across all
            // destinations), then scatter the projected vectors to destinations
            // — the cheaper `(H_s W_r)` ordering versus per-edge matmul. The CSR
            // is source-indexed, so the number of source rows is the source-type
            // node count.
            let n_src = adjacency.n_nodes();
            let mut projected = vec![0.0_f32; n_src * out_f];
            dense_matmul_accumulate(src_feat, w, n_src, src_dim, out_f, &mut projected);

            let acc = out
                .entry(dst_type.clone())
                .or_insert_with(|| vec![0.0; n_dst * out_f]);
            scatter_relation(
                adjacency,
                &projected,
                n_dst,
                out_f,
                self.config.mean_aggregation,
                acc,
            )?;
        }

        Ok(out)
    }
}

/// `acc[i,:] += X[i,:] · W` for a dense `[n × in_dim]` input and `[in_dim × out]`
/// weight, all row-major. Accumulates so callers can fuse self-loop + relations.
fn dense_matmul_accumulate(
    x: &[f32],
    w: &[f32],
    n: usize,
    in_dim: usize,
    out: usize,
    acc: &mut [f32],
) {
    for i in 0..n {
        let x_row = i * in_dim;
        let a_row = i * out;
        for j in 0..in_dim {
            let xij = x[x_row + j];
            if xij == 0.0 {
                continue;
            }
            let w_row = j * out;
            for (k, a) in acc[a_row..a_row + out].iter_mut().enumerate() {
                *a += xij * w[w_row + k];
            }
        }
    }
}

/// Scatter projected source vectors into destination accumulators.
///
/// The CSR is source-indexed: `adjacency.neighbors(j)` yields the destination
/// nodes reached from source `j`. With `mean = true`, each destination divides
/// by its incoming-edge count (`max(1, indeg)`), the R-GCN per-relation
/// normaliser; the in-degrees are counted in a first pass since the storage is
/// source-major.
fn scatter_relation(
    adjacency: &CsrGraph,
    projected: &[f32],
    n_dst: usize,
    out: usize,
    mean: bool,
    acc: &mut [f32],
) -> GnnResult<()> {
    let n_src = adjacency.n_nodes();

    // Pass 1: destination in-degree counts (only needed for mean normalisation).
    let indeg = if mean {
        let mut counts = vec![0usize; n_dst];
        for j in 0..n_src {
            for &d in adjacency.neighbors(j)? {
                if d >= n_dst {
                    return Err(GnnError::NodeIndexOutOfRange {
                        idx: d,
                        n_nodes: n_dst,
                    });
                }
                counts[d] += 1;
            }
        }
        Some(counts)
    } else {
        None
    };

    // Pass 2: scatter source → destination.
    for j in 0..n_src {
        let p_row = j * out;
        for &d in adjacency.neighbors(j)? {
            if d >= n_dst {
                return Err(GnnError::NodeIndexOutOfRange {
                    idx: d,
                    n_nodes: n_dst,
                });
            }
            let scale = match &indeg {
                Some(counts) => 1.0 / counts[d].max(1) as f32,
                None => 1.0,
            };
            let a_row = d * out;
            for (k, a) in acc[a_row..a_row + out].iter_mut().enumerate() {
                *a += scale * projected[p_row + k];
            }
        }
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rand_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..len)
            .map(|_| {
                let u = rng.next_u32() as f64 / 2f64.powi(32);
                (u as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// paper(5) -cites-> paper(5), author(3) -writes-> paper(5).
    ///
    /// Note: `HeteroGraph` stores each relation as a source-indexed CSR built
    /// with `from_edges(n_src, ..)`, which validates destination indices against
    /// `n_src`. The `writes` relation therefore keeps `paper`-destinations `< 3`
    /// (the `author` count); `cites` is `paper→paper` so all five are reachable.
    fn citation_graph() -> HeteroGraph {
        let mut g = HeteroGraph::new();
        g.add_node_type("paper", 5);
        g.add_node_type("author", 3);
        g.add_edge_type("paper", "cites", "paper", &[(0, 1), (1, 2), (2, 3), (4, 3)])
            .expect("edge");
        g.add_edge_type(
            "author",
            "writes",
            "paper",
            &[(0, 0), (1, 1), (2, 2), (0, 1)],
        )
        .expect("edge");
        g
    }

    fn make_config(out: usize, mean: bool, self_loops: bool) -> HeteroConvConfig {
        let mut in_features = HashMap::new();
        in_features.insert("paper".to_string(), 4);
        in_features.insert("author".to_string(), 3);
        HeteroConvConfig {
            in_features,
            out_features: out,
            mean_aggregation: mean,
            add_self_loops: self_loops,
        }
    }

    fn make_weights(g: &HeteroGraph, out: usize, seed: u64) -> HeteroConvWeights {
        let mut relation_weight = HashMap::new();
        for (s, r, d) in g.edge_types() {
            let in_dim = if s == "paper" { 4 } else { 3 };
            relation_weight.insert(
                (s.clone(), r.clone(), d.clone()),
                rand_vec(in_dim * out, seed ^ (s.len() as u64) ^ (r.len() as u64)),
            );
        }
        let mut self_weight = HashMap::new();
        self_weight.insert("paper".to_string(), rand_vec(4 * out, seed ^ 0x11));
        self_weight.insert("author".to_string(), rand_vec(3 * out, seed ^ 0x22));
        HeteroConvWeights {
            relation_weight,
            self_weight,
        }
    }

    fn make_features(seed: u64) -> HashMap<String, Vec<f32>> {
        let mut f = HashMap::new();
        f.insert("paper".to_string(), rand_vec(5 * 4, seed));
        f.insert("author".to_string(), rand_vec(3 * 3, seed ^ 0xABCD));
        f
    }

    #[test]
    fn output_shape_paper_only_destination() {
        let g = citation_graph();
        let cfg = make_config(6, true, false);
        let conv = HeteroConv::new(cfg).expect("conv");
        let feats = make_features(0x1234);
        let w = make_weights(&g, 6, 0x5678);
        let out = conv.forward(&g, &feats, &w).expect("forward");
        // Only "paper" is a destination of any relation.
        assert!(out.contains_key("paper"));
        assert!(!out.contains_key("author"));
        assert_eq!(out["paper"].len(), 5 * 6);
        assert!(out["paper"].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn self_loops_add_author_output() {
        let g = citation_graph();
        let cfg = make_config(6, true, true);
        let conv = HeteroConv::new(cfg).expect("conv");
        let feats = make_features(0x1111);
        let w = make_weights(&g, 6, 0x2222);
        let out = conv.forward(&g, &feats, &w).expect("forward");
        // With self-loops, author (a source-only type) now has its own output.
        assert!(out.contains_key("author"));
        assert_eq!(out["author"].len(), 3 * 6);
    }

    #[test]
    fn mean_vs_sum_differ() {
        let g = citation_graph();
        let feats = make_features(0x3333);
        let w = make_weights(&g, 5, 0x4444);
        let mean_out = HeteroConv::new(make_config(5, true, false))
            .expect("conv")
            .forward(&g, &feats, &w)
            .expect("forward");
        let sum_out = HeteroConv::new(make_config(5, false, false))
            .expect("conv")
            .forward(&g, &feats, &w)
            .expect("forward");
        // paper node 3 has 2 incoming "cites" edges ⇒ mean halves vs sum there.
        let diff: f32 = mean_out["paper"]
            .iter()
            .zip(sum_out["paper"].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-3, "mean and sum aggregation should differ");
    }

    #[test]
    fn matches_manual_single_relation_mean() {
        // Hand-verifiable: one relation, identity-ish weight, mean aggregation.
        let mut g = HeteroGraph::new();
        g.add_node_type("a", 2);
        g.add_node_type("b", 2);
        // a0->b0, a1->b0 (b0 has in-degree 2), a0->b1 (b1 in-degree 1).
        g.add_edge_type("a", "r", "b", &[(0, 0), (1, 0), (0, 1)])
            .expect("edge");
        let mut in_features = HashMap::new();
        in_features.insert("a".to_string(), 2);
        in_features.insert("b".to_string(), 2);
        let cfg = HeteroConvConfig {
            in_features,
            out_features: 2,
            mean_aggregation: true,
            add_self_loops: false,
        };
        let conv = HeteroConv::new(cfg).expect("conv");
        // a features: a0=[1,0], a1=[0,1]. Identity weight ⇒ projected = features.
        let mut feats = HashMap::new();
        feats.insert("a".to_string(), vec![1.0, 0.0, 0.0, 1.0]);
        feats.insert("b".to_string(), vec![0.0, 0.0, 0.0, 0.0]);
        let mut relation_weight = HashMap::new();
        relation_weight.insert(
            ("a".to_string(), "r".to_string(), "b".to_string()),
            vec![1.0, 0.0, 0.0, 1.0], // 2×2 identity
        );
        let w = HeteroConvWeights {
            relation_weight,
            self_weight: HashMap::new(),
        };
        let out = conv.forward(&g, &feats, &w).expect("forward");
        let b = &out["b"];
        // b0 = mean(a0,a1) = ([1,0]+[0,1])/2 = [0.5, 0.5].
        assert!((b[0] - 0.5).abs() < 1e-6);
        assert!((b[1] - 0.5).abs() < 1e-6);
        // b1 = mean(a0) = [1, 0].
        assert!((b[2] - 1.0).abs() < 1e-6);
        assert!((b[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn missing_relation_weight_errors() {
        let g = citation_graph();
        let cfg = make_config(4, true, false);
        let conv = HeteroConv::new(cfg).expect("conv");
        let feats = make_features(0x9999);
        let w = HeteroConvWeights {
            relation_weight: HashMap::new(), // none supplied
            self_weight: HashMap::new(),
        };
        assert!(matches!(
            conv.forward(&g, &feats, &w),
            Err(GnnError::InvalidLayerConfig(_))
        ));
    }

    #[test]
    fn wrong_relation_weight_shape_errors() {
        let g = citation_graph();
        let cfg = make_config(4, true, false);
        let conv = HeteroConv::new(cfg).expect("conv");
        let feats = make_features(0x7777);
        let mut relation_weight = HashMap::new();
        for (s, r, d) in g.edge_types() {
            relation_weight.insert((s.clone(), r.clone(), d.clone()), vec![0.0; 3]); // wrong
        }
        let w = HeteroConvWeights {
            relation_weight,
            self_weight: HashMap::new(),
        };
        assert!(matches!(
            conv.forward(&g, &feats, &w),
            Err(GnnError::WeightShapeMismatch { .. })
        ));
    }

    #[test]
    fn feature_size_mismatch_errors() {
        let g = citation_graph();
        let cfg = make_config(4, true, false);
        let conv = HeteroConv::new(cfg).expect("conv");
        let mut feats = make_features(0x5151);
        feats.insert("paper".to_string(), vec![0.0; 3]); // too short for 5×4
        let w = make_weights(&g, 4, 0x6262);
        assert!(matches!(
            conv.forward(&g, &feats, &w),
            Err(GnnError::NodeFeatureMismatch(..))
        ));
    }

    #[test]
    fn config_validation_rejects_bad_dims() {
        let mut in_features = HashMap::new();
        in_features.insert("x".to_string(), 0usize);
        let cfg = HeteroConvConfig {
            in_features,
            out_features: 4,
            mean_aggregation: true,
            add_self_loops: false,
        };
        assert!(cfg.validate().is_err());

        let cfg2 = HeteroConvConfig {
            in_features: HashMap::new(),
            out_features: 4,
            mean_aggregation: true,
            add_self_loops: false,
        };
        assert!(cfg2.validate().is_err());

        let mut in_features3 = HashMap::new();
        in_features3.insert("x".to_string(), 4usize);
        let cfg3 = HeteroConvConfig {
            in_features: in_features3,
            out_features: 0,
            mean_aggregation: true,
            add_self_loops: false,
        };
        assert!(cfg3.validate().is_err());
    }

    #[test]
    fn isolated_destination_node_stays_self_only() {
        // paper node 4 receives no "cites"/"writes" edge ⇒ with self-loops its
        // output equals exactly its self transform.
        let g = citation_graph();
        let cfg = make_config(3, false, true);
        let conv = HeteroConv::new(cfg).expect("conv");
        let feats = make_features(0xC0DE);
        let w = make_weights(&g, 3, 0xFACE);
        let out = conv.forward(&g, &feats, &w).expect("forward");
        // Compute paper4 self transform directly.
        let paper_feat = &feats["paper"];
        let self_w = &w.self_weight["paper"];
        let mut expected = [0.0_f32; 3];
        for j in 0..4 {
            let x = paper_feat[4 * 4 + j];
            for (k, e) in expected.iter_mut().enumerate() {
                *e += x * self_w[j * 3 + k];
            }
        }
        let got = &out["paper"][4 * 3..5 * 3];
        // node 4 is a source of a "cites" edge (4->3) but not a destination, so
        // nothing was aggregated into row 4: output == self transform.
        for (g_, e_) in got.iter().zip(expected.iter()) {
            assert!((g_ - e_).abs() < 1e-5, "{g_} vs {e_}");
        }
    }
}
