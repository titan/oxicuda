//! Heterogeneous graph: multiple node types and edge relation types.

use std::collections::HashMap;

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

/// Multi-type node and edge graph.
///
/// Node types are keyed by name and have individual node counts.
/// Edge types are triples `(src_type, rel_type, dst_type)` and are stored
/// as CSR subgraphs indexed by `(src_type, dst_type)` key.
#[derive(Debug, Clone)]
pub struct HeteroGraph {
    node_types: Vec<String>,
    node_counts: HashMap<String, usize>,
    edge_types: Vec<(String, String, String)>,
    adjacency: HashMap<(String, String), CsrGraph>,
}

impl HeteroGraph {
    /// Create an empty heterogeneous graph.
    pub fn new() -> Self {
        Self {
            node_types: Vec::new(),
            node_counts: HashMap::new(),
            edge_types: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Register a node type with the given count.
    ///
    /// If the type already exists its count is updated.
    pub fn add_node_type(&mut self, type_name: impl Into<String>, count: usize) {
        let name: String = type_name.into();
        if !self.node_types.contains(&name) {
            self.node_types.push(name.clone());
        }
        self.node_counts.insert(name, count);
    }

    /// Add an edge type `(src_type, rel, dst_type)` with the given edge list.
    ///
    /// Both `src_type` and `dst_type` must have been registered via
    /// [`Self::add_node_type`].  Node indices in `edges` are local to their type.
    pub fn add_edge_type(
        &mut self,
        src_type: &str,
        rel: impl Into<String>,
        dst_type: &str,
        edges: &[(usize, usize)],
    ) -> GnnResult<()> {
        let n_src = self
            .node_counts
            .get(src_type)
            .copied()
            .ok_or_else(|| GnnError::Internal(format!("unknown src node type '{src_type}'")))?;
        let n_dst = self
            .node_counts
            .get(dst_type)
            .copied()
            .ok_or_else(|| GnnError::Internal(format!("unknown dst node type '{dst_type}'")))?;

        // Validate indices against source count
        for &(s, _) in edges {
            if s >= n_src {
                return Err(GnnError::NodeIndexOutOfRange {
                    idx: s,
                    n_nodes: n_src,
                });
            }
        }
        // Validate dst indices against destination count
        for &(_, d) in edges {
            if d >= n_dst {
                return Err(GnnError::NodeIndexOutOfRange {
                    idx: d,
                    n_nodes: n_dst,
                });
            }
        }

        let rel_str: String = rel.into();
        let triple = (src_type.to_string(), rel_str, dst_type.to_string());
        if !self.edge_types.contains(&triple) {
            self.edge_types.push(triple);
        }

        let csr = CsrGraph::from_edges(n_src, edges)?;
        let key = (src_type.to_string(), dst_type.to_string());
        self.adjacency.insert(key, csr);
        Ok(())
    }

    /// Get the node count for the given type.
    pub fn n_nodes(&self, node_type: &str) -> GnnResult<usize> {
        self.node_counts
            .get(node_type)
            .copied()
            .ok_or_else(|| GnnError::Internal(format!("unknown node type '{node_type}'")))
    }

    /// Get the edge count between two node types.
    pub fn n_edges(&self, src_type: &str, dst_type: &str) -> GnnResult<usize> {
        let key = (src_type.to_string(), dst_type.to_string());
        self.adjacency
            .get(&key)
            .map(|g| g.n_edges())
            .ok_or_else(|| GnnError::NoEdgeType {
                src: src_type.to_string(),
                dst: dst_type.to_string(),
            })
    }

    /// Get the CSR adjacency matrix between two node types.
    pub fn adjacency(&self, src_type: &str, dst_type: &str) -> GnnResult<&CsrGraph> {
        let key = (src_type.to_string(), dst_type.to_string());
        self.adjacency
            .get(&key)
            .ok_or_else(|| GnnError::NoEdgeType {
                src: src_type.to_string(),
                dst: dst_type.to_string(),
            })
    }

    /// Registered node type names.
    pub fn node_types(&self) -> &[String] {
        &self.node_types
    }

    /// Registered edge type triples.
    pub fn edge_types(&self) -> &[(String, String, String)] {
        &self.edge_types
    }
}

impl Default for HeteroGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn citation_graph() -> HeteroGraph {
        let mut g = HeteroGraph::new();
        g.add_node_type("paper", 5);
        g.add_node_type("author", 3);
        g.add_edge_type("paper", "cites", "paper", &[(0, 1), (1, 2), (2, 3)])
            .expect("test invariant: value must be valid");
        g.add_edge_type("author", "writes", "paper", &[(0, 0), (1, 1), (2, 2)])
            .expect("test invariant: value must be valid");
        g
    }

    #[test]
    fn node_type_registration() {
        let mut g = HeteroGraph::new();
        g.add_node_type("user", 100);
        g.add_node_type("item", 200);
        assert_eq!(
            g.n_nodes("user")
                .expect("test invariant: value must be valid"),
            100
        );
        assert_eq!(
            g.n_nodes("item")
                .expect("test invariant: value must be valid"),
            200
        );
    }

    #[test]
    fn unknown_node_type_returns_error() {
        let g = HeteroGraph::new();
        assert!(g.n_nodes("ghost").is_err());
    }

    #[test]
    fn edge_type_registration_and_count() {
        let g = citation_graph();
        assert_eq!(
            g.n_edges("paper", "paper")
                .expect("test invariant: value must be valid"),
            3
        );
        assert_eq!(
            g.n_edges("author", "paper")
                .expect("test invariant: value must be valid"),
            3
        );
    }

    #[test]
    fn no_edge_type_error() {
        let g = citation_graph();
        let err = g.n_edges("paper", "author");
        assert!(matches!(err, Err(GnnError::NoEdgeType { .. })));
    }

    #[test]
    fn adjacency_retrieval() {
        let g = citation_graph();
        let csr = g
            .adjacency("paper", "paper")
            .expect("test invariant: value must be valid");
        assert_eq!(csr.n_nodes(), 5);
        assert_eq!(csr.n_edges(), 3);
    }

    #[test]
    fn adjacency_missing_error() {
        let g = citation_graph();
        assert!(g.adjacency("author", "author").is_err());
    }

    #[test]
    fn node_types_returns_registered() {
        let g = citation_graph();
        let types = g.node_types();
        assert!(types.contains(&"paper".to_string()));
        assert!(types.contains(&"author".to_string()));
    }

    #[test]
    fn edge_types_returns_triples() {
        let g = citation_graph();
        let et = g.edge_types();
        assert_eq!(et.len(), 2);
        let has_cites = et.iter().any(|(s, r, _)| s == "paper" && r == "cites");
        assert!(has_cites);
    }

    #[test]
    fn out_of_range_src_edge_error() {
        let mut g = HeteroGraph::new();
        g.add_node_type("a", 2);
        g.add_node_type("b", 2);
        let err = g.add_edge_type("a", "knows", "b", &[(5, 0)]);
        assert!(matches!(err, Err(GnnError::NodeIndexOutOfRange { .. })));
    }

    #[test]
    fn update_node_type_count() {
        let mut g = HeteroGraph::new();
        g.add_node_type("x", 10);
        g.add_node_type("x", 20); // update
        assert_eq!(
            g.n_nodes("x").expect("test invariant: value must be valid"),
            20
        );
    }

    #[test]
    fn default_impl() {
        let g = HeteroGraph::default();
        assert!(g.node_types().is_empty());
        assert!(g.edge_types().is_empty());
    }
}
