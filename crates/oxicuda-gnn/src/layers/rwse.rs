//! RWSE — Random-Walk Structural Encoding (Dwivedi et al., "Graph Neural
//! Networks with Learnable Structural and Positional Representations",
//! ICLR 2022; also the random-walk PE of GraphGPS, Rampášek 2022).
//!
//! RWSE summarises the local structural role of every node by the diagonal of
//! the powers of the random-walk transition operator `M = D⁻¹ A` (row-stochastic
//! one-step walk on the graph). For walk length `K` the encoding of node `i` is
//! the vector of **return probabilities**:
//!
//! ```text
//!   p_i = [ (M¹)_{ii}, (M²)_{ii}, …, (M^K)_{ii} ] ∈ ℝ^K
//! ```
//!
//! i.e. the probability that a random walker starting at `i` is back at `i`
//! after `k` steps. These return probabilities encode rich local topology
//! (cycles, degrees, triangle counts) and are invariant to node relabelling in
//! the sense that permuting the node indices permutes the encoding rows. RWSE
//! is computed once **offline** and concatenated to the input node features.
//!
//! Notes:
//! * The operator uses the supplied (possibly weighted) adjacency directly:
//!   `M[i,j] = w_{ij} / Σ_k w_{ik}`. Self-loop edges are honoured if present in
//!   the graph.
//! * An isolated node (out-degree `0`) has an all-zero transition row, so its
//!   return probabilities are `0` for every `k ≥ 1`.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

// ─── Free function ────────────────────────────────────────────────────────────

/// Compute the random-walk structural encoding `[n_nodes × walk_length]`.
///
/// Entry `enc[i * walk_length + (k − 1)] = (Mᵏ)_{ii}` is the `k`-step return
/// probability of node `i` under `M = D⁻¹ A`.
///
/// The powers are formed by repeated sparse-times-dense propagation of the
/// `n × n` walk matrix, reading its diagonal after every step. This is the usual
/// offline RWSE precomputation; it costs `O(walk_length · nnz · n)` time and
/// `O(n²)` scratch memory, which is appropriate for the moderate graphs handled
/// on the CPU path.
///
/// # Arguments
///
/// * `graph` — CSR graph supplying the (weighted) adjacency.
/// * `walk_length` — number of random-walk steps `K` (`≥ 1`).
///
/// # Returns
///
/// `[n_nodes × walk_length]` row-major return-probability matrix.
///
/// # Errors
///
/// * [`GnnError::InvalidLayerConfig`] if `walk_length == 0`.
/// * [`GnnError::NonFiniteOutput`] if any computed value is non-finite.
pub fn random_walk_se(graph: &CsrGraph, walk_length: usize) -> GnnResult<Vec<f32>> {
    if walk_length == 0 {
        return Err(GnnError::InvalidLayerConfig(
            "RWSE: walk_length must be >= 1".to_string(),
        ));
    }
    let n = graph.n_nodes();

    // Per-node inverse out-degree (sum of outgoing edge weights).
    let mut inv_deg = vec![0.0_f32; n];
    for (i, slot) in inv_deg.iter_mut().enumerate() {
        let w = graph.edge_weights(i)?;
        let deg: f32 = w.iter().sum();
        *slot = if deg > 0.0 { 1.0 / deg } else { 0.0 };
    }

    // walk = M (one step), stored dense [n × n], row-major walk[i*n + j].
    let mut walk = vec![0.0_f32; n * n];
    for i in 0..n {
        let nbrs = graph.neighbors(i)?;
        let wts = graph.edge_weights(i)?;
        let inv = inv_deg[i];
        for (idx, &j) in nbrs.iter().enumerate() {
            walk[i * n + j] += wts[idx] * inv;
        }
    }

    let mut enc = vec![0.0_f32; n * walk_length];
    for i in 0..n {
        enc[i * walk_length] = walk[i * n + i];
    }

    // Iterate walk ← M · walk, recording the diagonal at each step.
    let mut next = vec![0.0_f32; n * n];
    for k in 1..walk_length {
        for slot in next.iter_mut() {
            *slot = 0.0;
        }
        for i in 0..n {
            let nbrs = graph.neighbors(i)?;
            let wts = graph.edge_weights(i)?;
            let inv = inv_deg[i];
            for (idx, &j) in nbrs.iter().enumerate() {
                let m_ij = wts[idx] * inv;
                if m_ij != 0.0 {
                    for c in 0..n {
                        next[i * n + c] += m_ij * walk[j * n + c];
                    }
                }
            }
        }
        std::mem::swap(&mut walk, &mut next);
        for i in 0..n {
            enc[i * walk_length + k] = walk[i * n + i];
        }
    }

    if enc.iter().any(|v| !v.is_finite()) {
        return Err(GnnError::NonFiniteOutput("random_walk_se"));
    }
    Ok(enc)
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for an [`RwseEncoder`].
#[derive(Debug, Clone, Copy)]
pub struct RwseConfig {
    /// Number of random-walk steps `K` (`≥ 1`); the encoding dimension.
    pub walk_length: usize,
}

// ─── Encoder ──────────────────────────────────────────────────────────────────

/// Random-walk structural-encoding helper.
///
/// Produces a `[n × walk_length]` positional/structural encoding and can
/// concatenate it onto existing node features.
pub struct RwseEncoder {
    config: RwseConfig,
}

impl RwseEncoder {
    /// Construct an encoder from configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GnnError::InvalidLayerConfig`] if `walk_length == 0`.
    pub fn new(config: RwseConfig) -> GnnResult<Self> {
        if config.walk_length == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "RWSE: walk_length must be >= 1".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Dimension of the produced encoding (`walk_length`).
    #[inline]
    pub fn encoding_dim(&self) -> usize {
        self.config.walk_length
    }

    /// Compute the `[n × walk_length]` return-probability encoding.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`random_walk_se`].
    pub fn encode(&self, graph: &CsrGraph) -> GnnResult<Vec<f32>> {
        random_walk_se(graph, self.config.walk_length)
    }

    /// Concatenate the RWSE encoding onto node features.
    ///
    /// Given node features `x` (`[n × feat_dim]`), returns
    /// `[n × (feat_dim + walk_length)]` where each node row is
    /// `[ x_i ‖ rwse_i ]`.
    ///
    /// # Errors
    ///
    /// * [`GnnError::NodeFeatureMismatch`] if `x.len() != n * feat_dim`.
    /// * Propagates errors from [`random_walk_se`].
    pub fn augment(&self, graph: &CsrGraph, x: &[f32], feat_dim: usize) -> GnnResult<Vec<f32>> {
        let n = graph.n_nodes();
        if x.len() != n * feat_dim {
            return Err(GnnError::NodeFeatureMismatch(n, x.len() / feat_dim.max(1)));
        }
        let enc = self.encode(graph)?;
        let k = self.config.walk_length;
        let out_dim = feat_dim + k;
        let mut out = vec![0.0_f32; n * out_dim];
        for i in 0..n {
            for d in 0..feat_dim {
                out[i * out_dim + d] = x[i * feat_dim + d];
            }
            for d in 0..k {
                out[i * out_dim + feat_dim + d] = enc[i * k + d];
            }
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cycle() -> CsrGraph {
        CsrGraph::from_edges(2, &[(0, 1), (1, 0)]).expect("test invariant: value must be valid")
    }

    fn triangle() -> CsrGraph {
        CsrGraph::from_edges(3, &[(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)])
            .expect("test invariant: value must be valid")
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn new_valid() {
        let enc = RwseEncoder::new(RwseConfig { walk_length: 4 })
            .expect("test invariant: value must be valid");
        assert_eq!(enc.encoding_dim(), 4);
    }

    #[test]
    fn new_invalid_zero_walk() {
        assert!(RwseEncoder::new(RwseConfig { walk_length: 0 }).is_err());
    }

    #[test]
    fn free_fn_zero_walk_errors() {
        let g = two_cycle();
        assert!(random_walk_se(&g, 0).is_err());
    }

    // ── Self-loop node returns probability 1 at every step ───────────────────

    #[test]
    fn self_loop_returns_one() {
        let g = CsrGraph::from_edges(1, &[(0, 0)]).expect("test invariant: value must be valid");
        let enc = random_walk_se(&g, 4).expect("test invariant: value must be valid");
        assert_eq!(enc.len(), 4);
        for (k, &p) in enc.iter().enumerate() {
            assert!((p - 1.0).abs() < 1e-6, "step {k}: p={p}");
        }
    }

    // ── 2-cycle: return prob alternates 0, 1, 0, 1 ───────────────────────────

    #[test]
    fn two_cycle_alternates() {
        let g = two_cycle();
        let k = 4;
        let enc = random_walk_se(&g, k).expect("test invariant: value must be valid");
        // Node 0: M¹_ii = 0, M²_ii = 1, M³_ii = 0, M⁴_ii = 1.
        let expected = [0.0_f32, 1.0, 0.0, 1.0];
        for node in 0..2 {
            for (step, &e) in expected.iter().enumerate() {
                let p = enc[node * k + step];
                assert!(
                    (p - e).abs() < 1e-6,
                    "node {node} step {step}: got {p}, want {e}"
                );
            }
        }
    }

    // ── Triangle: analytic return probabilities (1 + 2(−1/2)^k)/3 ────────────

    #[test]
    fn triangle_analytic_return_probs() {
        let g = triangle();
        let k = 3;
        let enc = random_walk_se(&g, k).expect("test invariant: value must be valid");
        // (Mᵏ)_ii = (1 + 2·(−1/2)^k) / 3 for K₃.
        let expected = [0.0_f32, 0.5, 0.25];
        for node in 0..3 {
            for (step, &e) in expected.iter().enumerate() {
                let p = enc[node * k + step];
                assert!(
                    (p - e).abs() < 1e-5,
                    "node {node} step {step}: got {p}, want {e}"
                );
            }
        }
    }

    // ── Return probabilities are valid probabilities in [0, 1] ───────────────

    #[test]
    fn return_probs_in_unit_interval() {
        let g = triangle();
        let enc = random_walk_se(&g, 6).expect("test invariant: value must be valid");
        for &p in &enc {
            assert!((-1e-6..=1.0 + 1e-6).contains(&p), "p out of range: {p}");
        }
    }

    // ── First-step return prob is zero for a simple loop-free graph ──────────

    #[test]
    fn first_step_zero_without_self_loops() {
        let g = triangle();
        let enc = random_walk_se(&g, 3).expect("test invariant: value must be valid");
        for node in 0..3 {
            assert!((enc[node * 3]).abs() < 1e-6, "node {node} step 1 not zero");
        }
    }

    // ── Permutation: relabelling nodes permutes the encoding rows ────────────

    #[test]
    fn permutation_permutes_rows() {
        // Path 0–1–2 (degrees 1,2,1) and its relabelling 2–0–1 via perm.
        let g = CsrGraph::from_edges(3, &[(0, 1), (1, 0), (1, 2), (2, 1)])
            .expect("test invariant: value must be valid");
        let k = 4;
        let enc = random_walk_se(&g, k).expect("test invariant: value must be valid");

        // Relabel: new node a = old perm[a]. perm = [2,1,0] (reverse the path).
        let perm = [2usize, 1, 0];
        let mut edges = Vec::new();
        for old in 0..3 {
            for &j in g.neighbors(old).expect("nb") {
                // inverse permutation: old node `old` becomes new index `inv[old]`.
                let inv = |x: usize| perm.iter().position(|&p| p == x).expect("inv");
                edges.push((inv(old), inv(j)));
            }
        }
        let g2 = CsrGraph::from_edges(3, &edges).expect("test invariant: value must be valid");
        let enc2 = random_walk_se(&g2, k).expect("test invariant: value must be valid");

        // enc2 row `a` must equal enc row `perm[a]`.
        for a in 0..3 {
            for step in 0..k {
                let got = enc2[a * k + step];
                let want = enc[perm[a] * k + step];
                assert!(
                    (got - want).abs() < 1e-5,
                    "row {a} step {step}: {got} vs {want}"
                );
            }
        }
    }

    // ── Isolated node has zero return probability ────────────────────────────

    #[test]
    fn isolated_node_zero() {
        // Node 0 isolated; nodes 1,2 form a 2-cycle.
        let g = CsrGraph::from_edges(3, &[(1, 2), (2, 1)])
            .expect("test invariant: value must be valid");
        let k = 3;
        let enc = random_walk_se(&g, k).expect("test invariant: value must be valid");
        // Node 0 (isolated) occupies the first `k` entries of the encoding.
        for (step, &p) in enc.iter().take(k).enumerate() {
            assert!(p.abs() < 1e-6, "isolated node step {step} nonzero");
        }
    }

    // ── Encoder encode matches the free function ─────────────────────────────

    #[test]
    fn encoder_encode_matches_free_fn() {
        let g = triangle();
        let enc = RwseEncoder::new(RwseConfig { walk_length: 5 })
            .expect("test invariant: value must be valid");
        let a = enc.encode(&g).expect("test invariant: value must be valid");
        let b = random_walk_se(&g, 5).expect("test invariant: value must be valid");
        assert_eq!(a, b);
    }

    // ── Augment concatenates features and encoding ───────────────────────────

    #[test]
    fn augment_concatenates() {
        let g = two_cycle();
        let feat_dim = 2;
        let walk_length = 3;
        let enc = RwseEncoder::new(RwseConfig { walk_length })
            .expect("test invariant: value must be valid");
        let x = vec![10.0_f32, 20.0, 30.0, 40.0]; // 2 nodes × 2 feats
        let out = enc
            .augment(&g, &x, feat_dim)
            .expect("test invariant: value must be valid");
        let out_dim = feat_dim + walk_length;
        assert_eq!(out.len(), 2 * out_dim);
        // Original features preserved in the prefix of each row.
        assert!((out[0] - 10.0).abs() < 1e-6);
        assert!((out[1] - 20.0).abs() < 1e-6);
        assert!((out[out_dim] - 30.0).abs() < 1e-6);
        assert!((out[out_dim + 1] - 40.0).abs() < 1e-6);
        // RWSE suffix for node 0: [0, 1, 0] (2-cycle).
        assert!((out[feat_dim]).abs() < 1e-6);
        assert!((out[feat_dim + 1] - 1.0).abs() < 1e-6);
        assert!((out[feat_dim + 2]).abs() < 1e-6);
    }

    #[test]
    fn augment_feature_mismatch_errors() {
        let g = two_cycle();
        let enc = RwseEncoder::new(RwseConfig { walk_length: 2 })
            .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 3]; // wrong: 2 nodes × ? does not divide evenly to 2
        let err = enc.augment(&g, &x, 2);
        assert!(matches!(err, Err(GnnError::NodeFeatureMismatch(..))));
    }

    // ── Weighted graph: row normalisation respects edge weights ──────────────

    #[test]
    fn weighted_graph_normalised() {
        // Node 0 -> 1 (w=3), 0 -> 2 (w=1); back edges weight 1.
        let g =
            CsrGraph::from_edges_weighted(3, &[(0, 1, 3.0), (0, 2, 1.0), (1, 0, 1.0), (2, 0, 1.0)])
                .expect("test invariant: value must be valid");
        let enc = random_walk_se(&g, 2).expect("test invariant: value must be valid");
        // M[0,1] = 3/4, M[0,2] = 1/4, M[1,0] = 1, M[2,0] = 1.
        // (M²)_00 = M[0,1]·M[1,0] + M[0,2]·M[2,0] = 3/4 + 1/4 = 1.
        // Node 0 row occupies enc[0..2]: step 1 then step 2.
        assert!((enc[0]).abs() < 1e-6, "step1 should be 0");
        assert!((enc[1] - 1.0).abs() < 1e-6, "step2 should be 1");
    }
}
