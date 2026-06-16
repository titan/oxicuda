//! Graph hyperdimensional encoding (Poduval, Alimohamadi, Zakeri, Imani et al., DAC 2022 —
//! "GrapHD: Graph-Based Hyperdimensional Memorization and Retrieval").
//!
//! A graph `G = (V, E)` is embedded into a single binary hypervector by assigning each vertex
//! an i.i.d. random `{±1}^D` codeword and representing each edge as the *binding* of its two
//! endpoint codewords. The graph-level memory is the *bundle* (majority superposition) of all
//! edge hypervectors:
//!
//! ```text
//! H_G = ⨁_{(u,v) ∈ E}  edge(u, v) ,        edge(u, v) = code(u) ⊗ code(v) .
//! ```
//!
//! Because binary binding (`⊗`, element-wise sign product) is its own inverse, the set of
//! neighbours of a vertex `u` can be probed by *unbinding* `code(u)` from the graph memory and
//! cleaning up the result against the vertex codebook: `H_G ⊗ code(u)` correlates with
//! `code(v)` for every edge `(u, v)`, so each neighbour appears as a similarity peak. This
//! gives O(1) (dimension-bounded) connectivity queries and graph-level similarity directly
//! comparable via Hamming / cosine distance.
//!
//! For **directed** graphs the source endpoint is cyclically permuted before binding,
//! `edge(u → v) = ρ(code(u)) ⊗ code(v)`, so that `edge(u → v) ≠ edge(v → u)` and direction is
//! recoverable. All hypervectors use the crate-standard binary `Vec<i8>` representation.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::ops::binding::{binary_bind, binary_unbind};
use crate::ops::bundling::bundle_binary;
use crate::ops::permutation::{cyclic_shift, cyclic_shift_right};
use crate::vector::binary::{binary_dot, random_binary};

/// Graph HD encoder with a fixed vertex codebook.
///
/// Construct with [`new`](GraphHdEncoder::new) (random vertex codewords), add edges, then
/// call [`graph_hv`](GraphHdEncoder::graph_hv) for the graph-level embedding or
/// [`neighbors`](GraphHdEncoder::neighbors) to probe connectivity.
pub struct GraphHdEncoder {
    /// Hypervector dimension.
    dim: usize,
    /// Number of vertices.
    n_vertices: usize,
    /// If `true`, edges are direction-sensitive (source endpoint is permuted).
    directed: bool,
    /// Per-vertex random codewords (`n_vertices` entries, each length `dim`).
    vertex_hvs: Vec<Vec<i8>>,
    /// Accumulated edge list as `(u, v)` pairs.
    edges: Vec<(usize, usize)>,
}

impl GraphHdEncoder {
    /// Create a new graph encoder with random vertex codewords.
    ///
    /// - `n_vertices`: number of vertices (must be ≥ 1).
    /// - `dim`: hypervector dimension (must be ≥ 1).
    /// - `directed`: whether edges are directed.
    /// - `rng`: random number generator for vertex codewords.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `dim == 0`.
    /// - [`HdcError::EmptyInput`] if `n_vertices == 0`.
    pub fn new(n_vertices: usize, dim: usize, directed: bool, rng: &mut LcgRng) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if n_vertices == 0 {
            return Err(HdcError::EmptyInput);
        }
        let mut vertex_hvs = Vec::with_capacity(n_vertices);
        for _ in 0..n_vertices {
            vertex_hvs.push(random_binary(dim, rng)?);
        }
        Ok(Self {
            dim,
            n_vertices,
            directed,
            vertex_hvs,
            edges: Vec::new(),
        })
    }

    /// Encode a single edge `(u, v)` as a binary hypervector.
    ///
    /// Undirected: `code(u) ⊗ code(v)`. Directed: `ρ(code(u)) ⊗ code(v)` (source permuted).
    ///
    /// # Errors
    ///
    /// - [`HdcError::FeatureIndexOutOfRange`] if `u` or `v` is `>= n_vertices`.
    pub fn edge_hv(&self, u: usize, v: usize) -> HdcResult<Vec<i8>> {
        if u >= self.n_vertices {
            return Err(HdcError::FeatureIndexOutOfRange {
                feat: u,
                max: self.n_vertices,
            });
        }
        if v >= self.n_vertices {
            return Err(HdcError::FeatureIndexOutOfRange {
                feat: v,
                max: self.n_vertices,
            });
        }
        if self.directed {
            let permuted_src = cyclic_shift(&self.vertex_hvs[u], 1)?;
            binary_bind(&permuted_src, &self.vertex_hvs[v])
        } else {
            binary_bind(&self.vertex_hvs[u], &self.vertex_hvs[v])
        }
    }

    /// Add an edge `(u, v)` to the graph.
    ///
    /// # Errors
    ///
    /// - [`HdcError::FeatureIndexOutOfRange`] if either endpoint is out of range.
    pub fn add_edge(&mut self, u: usize, v: usize) -> HdcResult<()> {
        if u >= self.n_vertices || v >= self.n_vertices {
            let bad = if u >= self.n_vertices { u } else { v };
            return Err(HdcError::FeatureIndexOutOfRange {
                feat: bad,
                max: self.n_vertices,
            });
        }
        self.edges.push((u, v));
        Ok(())
    }

    /// Compute the graph-level hypervector: bundle of all edge HVs.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if no edges have been added.
    pub fn graph_hv(&self, rng: &mut LcgRng) -> HdcResult<Vec<i8>> {
        if self.edges.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        let mut edge_hvs = Vec::with_capacity(self.edges.len());
        for &(u, v) in &self.edges {
            edge_hvs.push(self.edge_hv(u, v)?);
        }
        bundle_binary(&edge_hvs, rng)
    }

    /// Probe the neighbours of vertex `u` by unbinding from the graph memory and ranking
    /// every vertex codeword by its correlation with the unbound residual.
    ///
    /// Returns `(vertex_id, dot_score)` pairs sorted by descending score. Genuine neighbours
    /// appear with markedly higher scores than non-neighbours. For directed graphs this probes
    /// out-edges `u → v` (the residual is de-permuted before cleanup).
    ///
    /// # Errors
    ///
    /// - [`HdcError::FeatureIndexOutOfRange`] if `u >= n_vertices`.
    /// - [`HdcError::DimensionMismatch`] if `graph_hv` length does not match `dim`.
    pub fn neighbors(&self, u: usize, graph_hv: &[i8]) -> HdcResult<Vec<(usize, i64)>> {
        if u >= self.n_vertices {
            return Err(HdcError::FeatureIndexOutOfRange {
                feat: u,
                max: self.n_vertices,
            });
        }
        if graph_hv.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: graph_hv.len(),
            });
        }
        // Unbind code(u) from the graph memory.
        let residual = if self.directed {
            // edge(u → v) = ρ(code(u)) ⊗ code(v) ⇒ residual = graph ⊗ ρ(code(u)).
            let permuted_src = cyclic_shift(&self.vertex_hvs[u], 1)?;
            binary_unbind(graph_hv, &permuted_src)?
        } else {
            binary_unbind(graph_hv, &self.vertex_hvs[u])?
        };
        let mut scored: Vec<(usize, i64)> = Vec::with_capacity(self.n_vertices);
        for (vid, hv) in self.vertex_hvs.iter().enumerate() {
            scored.push((vid, binary_dot(&residual, hv)?));
        }
        scored.sort_by_key(|&(_, dot)| std::cmp::Reverse(dot));
        Ok(scored)
    }

    /// Return the top-`k` most likely neighbours of vertex `u` (vertex ids only).
    ///
    /// # Errors
    ///
    /// Same as [`neighbors`](GraphHdEncoder::neighbors).
    pub fn top_neighbors(&self, u: usize, graph_hv: &[i8], k: usize) -> HdcResult<Vec<usize>> {
        let ranked = self.neighbors(u, graph_hv)?;
        Ok(ranked.into_iter().take(k).map(|(vid, _)| vid).collect())
    }

    /// Recover the source vertex of a directed edge given the destination, by de-permuting.
    /// Only meaningful for directed encoders. Returns the best-matching source vertex id.
    ///
    /// # Errors
    ///
    /// - [`HdcError::FeatureIndexOutOfRange`] if `v >= n_vertices`.
    /// - [`HdcError::DimensionMismatch`] if `graph_hv` length is wrong.
    pub fn directed_source(&self, v: usize, graph_hv: &[i8]) -> HdcResult<usize> {
        if v >= self.n_vertices {
            return Err(HdcError::FeatureIndexOutOfRange {
                feat: v,
                max: self.n_vertices,
            });
        }
        if graph_hv.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: graph_hv.len(),
            });
        }
        // residual = graph ⊗ code(v) ≈ Σ ρ(code(src)); de-permute then clean up.
        let unbound = binary_unbind(graph_hv, &self.vertex_hvs[v])?;
        let depermuted = cyclic_shift_right(&unbound, 1)?;
        let mut best = 0usize;
        let mut best_dot = i64::MIN;
        for (vid, hv) in self.vertex_hvs.iter().enumerate() {
            let dot = binary_dot(&depermuted, hv)?;
            if dot > best_dot {
                best_dot = dot;
                best = vid;
            }
        }
        Ok(best)
    }

    /// Number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.n_vertices
    }

    /// Number of edges added so far.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Whether the encoder is directed.
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// Hypervector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::hamming::hamming_frac;
    use crate::handle::LcgRng;

    fn build_undirected(
        n: usize,
        dim: usize,
        edges: &[(usize, usize)],
        seed: u64,
    ) -> (GraphHdEncoder, Vec<i8>) {
        let mut rng = LcgRng::new(seed);
        let mut enc = GraphHdEncoder::new(n, dim, false, &mut rng).expect("new");
        for &(u, v) in edges {
            enc.add_edge(u, v).expect("add_edge");
        }
        let mut brng = LcgRng::new(seed ^ 0xABCD);
        let hv = enc.graph_hv(&mut brng).expect("graph_hv");
        (enc, hv)
    }

    #[test]
    fn new_rejects_bad_args() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            GraphHdEncoder::new(0, 256, false, &mut rng),
            Err(HdcError::EmptyInput)
        ));
        assert!(matches!(
            GraphHdEncoder::new(4, 0, false, &mut rng),
            Err(HdcError::ZeroDimension)
        ));
    }

    #[test]
    fn edge_hv_undirected_symmetric() {
        let mut rng = LcgRng::new(2);
        let enc = GraphHdEncoder::new(5, 512, false, &mut rng).expect("new");
        let uv = enc.edge_hv(1, 3).expect("uv");
        let vu = enc.edge_hv(3, 1).expect("vu");
        // Undirected binding is commutative ⇒ identical.
        assert_eq!(uv, vu);
    }

    #[test]
    fn edge_hv_directed_asymmetric() {
        let mut rng = LcgRng::new(3);
        let enc = GraphHdEncoder::new(5, 512, true, &mut rng).expect("new");
        let uv = enc.edge_hv(1, 3).expect("uv");
        let vu = enc.edge_hv(3, 1).expect("vu");
        let dist = hamming_frac(&uv, &vu).expect("hamming");
        assert!(dist > 0.3, "directed edges should differ: dist={dist}");
    }

    #[test]
    fn edge_hv_out_of_range_errors() {
        let mut rng = LcgRng::new(4);
        let enc = GraphHdEncoder::new(3, 256, false, &mut rng).expect("new");
        assert!(matches!(
            enc.edge_hv(0, 9),
            Err(HdcError::FeatureIndexOutOfRange { .. })
        ));
        assert!(matches!(
            enc.edge_hv(9, 0),
            Err(HdcError::FeatureIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn add_edge_out_of_range_errors() {
        let mut rng = LcgRng::new(5);
        let mut enc = GraphHdEncoder::new(3, 256, false, &mut rng).expect("new");
        assert!(matches!(
            enc.add_edge(0, 9),
            Err(HdcError::FeatureIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn graph_hv_empty_errors() {
        let mut rng = LcgRng::new(6);
        let enc = GraphHdEncoder::new(3, 256, false, &mut rng).expect("new");
        let mut brng = LcgRng::new(7);
        assert!(matches!(enc.graph_hv(&mut brng), Err(HdcError::EmptyInput)));
    }

    #[test]
    fn graph_hv_shape_and_validity() {
        let (_, hv) = build_undirected(5, 1024, &[(0, 1), (1, 2), (2, 3)], 10);
        assert_eq!(hv.len(), 1024);
        assert!(hv.iter().all(|&v| v == 1 || v == -1));
    }

    #[test]
    fn neighbors_recovers_connected_vertices() {
        // Star graph centred at vertex 0: edges 0-1, 0-2, 0-3.
        let dim = 4000;
        let (enc, hv) = build_undirected(6, dim, &[(0, 1), (0, 2), (0, 3)], 20);
        let top = enc.top_neighbors(0, &hv, 3).expect("top");
        // 1, 2, 3 should be the top-3 neighbours (order independent).
        let mut found = top.clone();
        found.sort_unstable();
        assert_eq!(found, vec![1, 2, 3], "got {top:?}");
    }

    #[test]
    fn neighbors_separates_connected_from_unconnected() {
        // Path graph: 0-1-2. Vertex 4 and 5 are isolated.
        let dim = 5000;
        let (enc, hv) = build_undirected(6, dim, &[(0, 1), (1, 2)], 21);
        let ranked = enc.neighbors(1, &hv).expect("neighbors");
        // Build a score map.
        let score = |vid: usize| {
            ranked
                .iter()
                .find(|&&(id, _)| id == vid)
                .map(|&(_, s)| s)
                .expect("score")
        };
        let s0 = score(0);
        let s2 = score(2);
        let s4 = score(4);
        // Genuine neighbours (0, 2) of vertex 1 should outscore the isolated vertex 4.
        assert!(
            s0 > s4,
            "neighbour 0 ({s0}) should outscore isolated 4 ({s4})"
        );
        assert!(
            s2 > s4,
            "neighbour 2 ({s2}) should outscore isolated 4 ({s4})"
        );
    }

    #[test]
    fn neighbors_out_of_range_errors() {
        let (enc, hv) = build_undirected(4, 512, &[(0, 1)], 22);
        assert!(matches!(
            enc.neighbors(9, &hv),
            Err(HdcError::FeatureIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn neighbors_wrong_graph_dim_errors() {
        let (enc, _) = build_undirected(4, 512, &[(0, 1)], 23);
        let bad = vec![1i8; 256];
        assert!(matches!(
            enc.neighbors(0, &bad),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn directed_source_recovers_origin() {
        // Single directed edge 2 → 4 should let us recover source 2 from destination 4.
        let dim = 5000;
        let mut rng = LcgRng::new(30);
        let mut enc = GraphHdEncoder::new(6, dim, true, &mut rng).expect("new");
        enc.add_edge(2, 4).expect("add");
        let mut brng = LcgRng::new(31);
        let hv = enc.graph_hv(&mut brng).expect("graph_hv");
        let src = enc.directed_source(4, &hv).expect("source");
        assert_eq!(src, 2, "recovered wrong source: {src}");
    }

    #[test]
    fn directed_neighbors_recover_out_edges() {
        // Directed star out of vertex 0: 0→1, 0→2.
        let dim = 5000;
        let mut rng = LcgRng::new(32);
        let mut enc = GraphHdEncoder::new(6, dim, true, &mut rng).expect("new");
        enc.add_edge(0, 1).expect("add");
        enc.add_edge(0, 2).expect("add");
        let mut brng = LcgRng::new(33);
        let hv = enc.graph_hv(&mut brng).expect("graph_hv");
        let top = enc.top_neighbors(0, &hv, 2).expect("top");
        let mut found = top.clone();
        found.sort_unstable();
        assert_eq!(found, vec![1, 2], "got {top:?}");
    }

    #[test]
    fn accessors_report_state() {
        let mut rng = LcgRng::new(40);
        let mut enc = GraphHdEncoder::new(7, 1024, true, &mut rng).expect("new");
        assert_eq!(enc.n_vertices(), 7);
        assert_eq!(enc.dim(), 1024);
        assert!(enc.is_directed());
        assert_eq!(enc.n_edges(), 0);
        enc.add_edge(0, 1).expect("add");
        enc.add_edge(1, 2).expect("add");
        assert_eq!(enc.n_edges(), 2);
    }

    #[test]
    fn different_graphs_produce_different_hvs() {
        let dim = 2000;
        let (_, hv_a) = build_undirected(6, dim, &[(0, 1), (1, 2)], 50);
        let (_, hv_b) = build_undirected(6, dim, &[(3, 4), (4, 5)], 50);
        let dist = hamming_frac(&hv_a, &hv_b).expect("hamming");
        assert!(dist > 0.3, "distinct graphs too similar: dist={dist}");
    }
}
