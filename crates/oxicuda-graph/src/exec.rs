//! Instantiated executable graph — a CPU-side model of CUDA graph
//! instantiation and update.
//!
//! On real hardware, a [`ComputeGraph`] is *instantiated* into an executable
//! graph (`cudaGraphExec_t`) once, and then launched many times. Between
//! launches the cheap thing to do is to *update* node parameters in place
//! (`cudaGraphExecKernelNodeSetParams`, `cudaGraphExecUpdate`) rather than
//! re-instantiate from scratch — but an in-place update is only legal when the
//! new topology matches the instantiated one exactly.
//!
//! This module models that lifecycle purely on the CPU:
//!
//! * [`ExecGraph::instantiate`] validates a [`ComputeGraph`] is a DAG and
//!   snapshots its topology (nodes + edges) and node parameters.
//! * [`ExecGraph::update_node`] applies a single-node parameter update,
//!   validating the parameter shape (e.g. a kernel node may only be updated
//!   with kernel parameters, and a memcpy's size may not change).
//! * [`ExecGraph::update`] applies a whole-graph update against a freshly
//!   built [`ComputeGraph`], requiring identical topology and only differing
//!   parameters — exactly the `cudaGraphExecUpdate` contract.
//! * [`ExecGraph::diff`] computes a structural / parameter diff against
//!   another [`ComputeGraph`], reporting whether an in-place update is
//!   possible at all.
//! * [`ExecGraph::clone`] preserves the full topology of the executable graph.
//!
//! No GPU is required; this is pure data-structure and validation work.

use std::collections::BTreeSet;

use crate::error::{GraphError, GraphResult};
use crate::graph::ComputeGraph;
use crate::node::{KernelConfig, NodeId, NodeKind};

// ---------------------------------------------------------------------------
// NodeParamUpdate
// ---------------------------------------------------------------------------

/// A parameter-only update to a single instantiated node.
///
/// Each variant matches the [`NodeKind`] it updates; applying an update whose
/// variant does not match the target node's kind is rejected. Parameters that
/// affect topology or resource sizing (buffer counts, transfer sizes) are
/// intentionally *not* updatable here — those require re-instantiation.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeParamUpdate {
    /// New launch configuration for a kernel node. The function name may
    /// change (CUDA permits swapping the kernel function on update) but the
    /// node must already be a [`NodeKind::KernelLaunch`].
    Kernel {
        /// Optional new function name (`None` keeps the existing one).
        function_name: Option<String>,
        /// New grid/block/shared-memory configuration.
        config: KernelConfig,
    },
    /// New fill value for a memset node. The size is fixed at instantiation
    /// and may not change.
    Memset {
        /// New byte fill value.
        value: u8,
    },
}

impl NodeParamUpdate {
    /// Returns a short tag describing which kind this update targets.
    #[must_use]
    pub fn target_tag(&self) -> &'static str {
        match self {
            Self::Kernel { .. } => "kernel",
            Self::Memset { .. } => "memset",
        }
    }
}

// ---------------------------------------------------------------------------
// ExecGraph
// ---------------------------------------------------------------------------

/// An instantiated executable graph: a validated topology snapshot plus the
/// per-node parameters that an in-place update may modify.
///
/// `ExecGraph` is cheap to clone and clones preserve topology exactly.
#[derive(Debug, Clone)]
pub struct ExecGraph {
    /// Snapshot of every node's operation kind (indexed by `NodeId.0`).
    kinds: Vec<NodeKind>,
    /// Canonicalised edge set `(from, to)` — used for topology equality.
    edges: BTreeSet<(u32, u32)>,
    /// Cached topological execution order computed at instantiation.
    execution_order: Vec<NodeId>,
}

impl ExecGraph {
    /// Instantiates `graph` into an executable graph.
    ///
    /// Validates the graph is a non-empty DAG (by computing a topological
    /// order) and snapshots its topology and node parameters.
    ///
    /// # Errors
    ///
    /// * [`GraphError::EmptyGraph`] if the graph has no nodes.
    /// * Propagates any error from [`ComputeGraph::topological_order`].
    pub fn instantiate(graph: &ComputeGraph) -> GraphResult<Self> {
        let execution_order = graph.topological_order()?;
        let kinds: Vec<NodeKind> = graph.nodes().iter().map(|n| n.kind.clone()).collect();
        let edges: BTreeSet<(u32, u32)> =
            graph.edges().into_iter().map(|(a, b)| (a.0, b.0)).collect();
        Ok(Self {
            kinds,
            edges,
            execution_order,
        })
    }

    /// Returns the number of nodes in the executable graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.kinds.len()
    }

    /// Returns the number of dependency edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns the cached topological execution order.
    #[must_use]
    pub fn execution_order(&self) -> &[NodeId] {
        &self.execution_order
    }

    /// Returns the operation kind of a node.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::NodeNotFound`] if `id` is out of range.
    pub fn node_kind(&self, id: NodeId) -> GraphResult<&NodeKind> {
        self.kinds
            .get(id.0 as usize)
            .ok_or(GraphError::NodeNotFound(id))
    }

    /// Returns `true` if the dependency `from → to` is present.
    #[must_use]
    pub fn has_edge(&self, from: NodeId, to: NodeId) -> bool {
        self.edges.contains(&(from.0, to.0))
    }

    /// Applies a single-node parameter update in place.
    ///
    /// Validates that the update variant matches the target node's kind and
    /// that no topology-affecting parameter would change.
    ///
    /// # Errors
    ///
    /// * [`GraphError::NodeNotFound`] if `id` is out of range.
    /// * [`GraphError::ValidationFailed`] if the update variant does not match
    ///   the node's kind.
    pub fn update_node(&mut self, id: NodeId, update: NodeParamUpdate) -> GraphResult<()> {
        let kind = self
            .kinds
            .get_mut(id.0 as usize)
            .ok_or(GraphError::NodeNotFound(id))?;
        match (kind, update) {
            (
                NodeKind::KernelLaunch {
                    function_name,
                    config,
                    ..
                },
                NodeParamUpdate::Kernel {
                    function_name: new_name,
                    config: new_config,
                },
            ) => {
                if let Some(name) = new_name {
                    *function_name = name;
                }
                *config = new_config;
                Ok(())
            }
            (NodeKind::Memset { value, .. }, NodeParamUpdate::Memset { value: new_value }) => {
                *value = new_value;
                Ok(())
            }
            (kind, update) => Err(GraphError::ValidationFailed(format!(
                "node {id} is a '{}' node but update targets '{}'",
                kind.tag(),
                update.target_tag()
            ))),
        }
    }

    /// Applies a whole-graph update against a freshly built [`ComputeGraph`].
    ///
    /// This mirrors `cudaGraphExecUpdate`: the new graph must have *identical
    /// topology* (same node count, same node kinds in the structural sense,
    /// same edge set). Only updatable parameters (kernel config / function
    /// name, memset value) may differ; differing topology or non-updatable
    /// parameters (e.g. a memcpy size) make the update impossible and the
    /// caller must re-instantiate.
    ///
    /// On success, all node parameters are refreshed from `new_graph` and the
    /// cached execution order is preserved (topology is unchanged).
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::ValidationFailed`] describing the first
    /// incompatibility found.
    pub fn update(&mut self, new_graph: &ComputeGraph) -> GraphResult<()> {
        let diff = self.diff(new_graph)?;
        if !diff.is_updatable() {
            return Err(GraphError::ValidationFailed(diff.reject_reason()));
        }
        // Topology matches and every change is a parameter change: refresh
        // node kinds wholesale. Edges and execution order are unchanged.
        self.kinds = new_graph.nodes().iter().map(|n| n.kind.clone()).collect();
        Ok(())
    }

    /// Computes a structural / parameter diff against `other`.
    ///
    /// Reports whether an in-place [`update`](Self::update) is possible and,
    /// if so, which nodes changed parameters.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::EmptyGraph`] if `other` is empty (it could not
    /// have been instantiated).
    pub fn diff(&self, other: &ComputeGraph) -> GraphResult<ExecGraphDiff> {
        if other.is_empty() {
            return Err(GraphError::EmptyGraph);
        }
        // Node-count mismatch → structural change, not updatable.
        if other.node_count() != self.kinds.len() {
            return Ok(ExecGraphDiff {
                node_count_changed: true,
                topology_changed: true,
                non_updatable_nodes: Vec::new(),
                changed_params: Vec::new(),
            });
        }
        // Edge-set mismatch → topology change.
        let other_edges: BTreeSet<(u32, u32)> =
            other.edges().into_iter().map(|(a, b)| (a.0, b.0)).collect();
        let topology_changed = other_edges != self.edges;

        let mut non_updatable_nodes = Vec::new();
        let mut changed_params = Vec::new();
        for (i, new_node) in other.nodes().iter().enumerate() {
            let old = &self.kinds[i];
            let new = &new_node.kind;
            match classify_change(old, new) {
                NodeChange::Same => {}
                NodeChange::Param => changed_params.push(NodeId(i as u32)),
                NodeChange::NonUpdatable => non_updatable_nodes.push(NodeId(i as u32)),
            }
        }

        Ok(ExecGraphDiff {
            node_count_changed: false,
            topology_changed,
            non_updatable_nodes,
            changed_params,
        })
    }
}

// ---------------------------------------------------------------------------
// Change classification
// ---------------------------------------------------------------------------

enum NodeChange {
    /// Identical parameters.
    Same,
    /// Only updatable parameters differ.
    Param,
    /// A structural / non-updatable parameter differs (kind change, memcpy
    /// size change, buffer-count change, …).
    NonUpdatable,
}

/// Classifies how `new` differs from `old` for update purposes.
fn classify_change(old: &NodeKind, new: &NodeKind) -> NodeChange {
    match (old, new) {
        (
            NodeKind::KernelLaunch {
                function_name: of,
                config: oc,
                fusible: ob,
            },
            NodeKind::KernelLaunch {
                function_name: nf,
                config: nc,
                fusible: nb,
            },
        ) => {
            if ob != nb {
                // The fusible flag participates in fusion topology decisions;
                // treat a change as non-updatable.
                NodeChange::NonUpdatable
            } else if of == nf && oc == nc {
                NodeChange::Same
            } else {
                NodeChange::Param
            }
        }
        (
            NodeKind::Memset {
                size_bytes: os,
                value: ov,
            },
            NodeKind::Memset {
                size_bytes: ns,
                value: nv,
            },
        ) => {
            if os != ns {
                NodeChange::NonUpdatable
            } else if ov == nv {
                NodeChange::Same
            } else {
                NodeChange::Param
            }
        }
        // Memcpy: only identical params are "Same"; any change is structural
        // (transfer size / direction are not in-place updatable in this model).
        (
            NodeKind::Memcpy {
                dir: od,
                size_bytes: os,
            },
            NodeKind::Memcpy {
                dir: nd,
                size_bytes: ns,
            },
        ) => {
            if od == nd && os == ns {
                NodeChange::Same
            } else {
                NodeChange::NonUpdatable
            }
        }
        // Parameter-free kinds: equal iff the same variant.
        (a, b) if a == b => NodeChange::Same,
        // Any kind mismatch (e.g. Kernel vs Memcpy) is structural.
        _ => NodeChange::NonUpdatable,
    }
}

// ---------------------------------------------------------------------------
// ExecGraphDiff
// ---------------------------------------------------------------------------

/// The result of diffing an [`ExecGraph`] against a [`ComputeGraph`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecGraphDiff {
    /// The two graphs have a different number of nodes.
    pub node_count_changed: bool,
    /// The dependency edge sets differ.
    pub topology_changed: bool,
    /// Nodes whose change is structural / non-updatable (kind change, memcpy
    /// size change, fusible-flag change, …).
    pub non_updatable_nodes: Vec<NodeId>,
    /// Nodes whose updatable parameters differ (these *can* be updated).
    pub changed_params: Vec<NodeId>,
}

impl ExecGraphDiff {
    /// Returns `true` if the two graphs are structurally and parametrically
    /// identical.
    #[must_use]
    pub fn is_identical(&self) -> bool {
        !self.node_count_changed
            && !self.topology_changed
            && self.non_updatable_nodes.is_empty()
            && self.changed_params.is_empty()
    }

    /// Returns `true` if an in-place [`ExecGraph::update`] is possible: the
    /// topology matches and every difference is an updatable parameter.
    #[must_use]
    pub fn is_updatable(&self) -> bool {
        !self.node_count_changed && !self.topology_changed && self.non_updatable_nodes.is_empty()
    }

    /// Returns a human-readable reason the update was rejected.
    ///
    /// Only meaningful when [`is_updatable`](Self::is_updatable) is `false`.
    #[must_use]
    pub fn reject_reason(&self) -> String {
        if self.node_count_changed {
            "node count differs — topology must match for in-place update".to_owned()
        } else if self.topology_changed {
            "dependency edges differ — topology must match for in-place update".to_owned()
        } else if !self.non_updatable_nodes.is_empty() {
            format!(
                "{} node(s) changed in a non-updatable way (kind / size / fusibility)",
                self.non_updatable_nodes.len()
            )
        } else {
            "update is possible".to_owned()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::node::MemcpyDir;

    fn diamond() -> (ComputeGraph, [NodeId; 4]) {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = b.add_kernel("a", 4, 256, 0).finish();
        let l = b.add_kernel("l", 4, 256, 0).finish();
        let r = b.add_kernel("r", 4, 256, 0).finish();
        let d = b.add_kernel("d", 4, 256, 0).finish();
        b.dep(a, l).dep(a, r).dep(l, d).dep(r, d);
        (b.build().expect("diamond builds"), [a, l, r, d])
    }

    #[test]
    fn instantiate_empty_errors() {
        let g = ComputeGraph::new();
        assert!(matches!(
            ExecGraph::instantiate(&g),
            Err(GraphError::EmptyGraph)
        ));
    }

    #[test]
    fn instantiate_snapshots_topology() {
        let (g, _) = diamond();
        let ex = ExecGraph::instantiate(&g).expect("instantiate");
        assert_eq!(ex.node_count(), 4);
        assert_eq!(ex.edge_count(), 4);
        assert_eq!(ex.execution_order().len(), 4);
    }

    #[test]
    fn execution_order_respects_dependencies() {
        let (g, [a, l, r, d]) = diamond();
        let ex = ExecGraph::instantiate(&g).expect("instantiate");
        let order = ex.execution_order();
        let pos = |n: NodeId| order.iter().position(|&x| x == n).expect("present");
        assert!(pos(a) < pos(l));
        assert!(pos(a) < pos(r));
        assert!(pos(l) < pos(d));
        assert!(pos(r) < pos(d));
    }

    #[test]
    fn clone_preserves_topology() {
        let (g, [a, _l, _r, d]) = diamond();
        let ex = ExecGraph::instantiate(&g).expect("instantiate");
        let cloned = ex.clone();
        assert_eq!(cloned.node_count(), ex.node_count());
        assert_eq!(cloned.edge_count(), ex.edge_count());
        assert!(cloned.has_edge(a, NodeId(1)));
        assert_eq!(cloned.execution_order(), ex.execution_order());
        // d is a sink: no outgoing edge.
        assert!(!cloned.has_edge(d, a));
    }

    #[test]
    fn update_node_kernel_config() {
        let (g, [a, ..]) = diamond();
        let mut ex = ExecGraph::instantiate(&g).expect("instantiate");
        ex.update_node(
            a,
            NodeParamUpdate::Kernel {
                function_name: Some("a_v2".into()),
                config: KernelConfig::linear(8, 128, 512),
            },
        )
        .expect("kernel update");
        let k = ex.node_kind(a).expect("node a");
        match k {
            NodeKind::KernelLaunch {
                function_name,
                config,
                ..
            } => {
                assert_eq!(function_name, "a_v2");
                assert_eq!(config.grid, (8, 1, 1));
                assert_eq!(config.shared_mem_bytes, 512);
            }
            _ => panic!("expected kernel"),
        }
    }

    #[test]
    fn update_node_wrong_kind_rejected() {
        // A memset node updated with kernel params must fail validation.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let z = b.add_memset("zero", 4096, 0);
        let g = b.build().expect("builds");
        let mut ex = ExecGraph::instantiate(&g).expect("instantiate");
        let res = ex.update_node(
            z,
            NodeParamUpdate::Kernel {
                function_name: None,
                config: KernelConfig::linear(1, 1, 0),
            },
        );
        assert!(matches!(res, Err(GraphError::ValidationFailed(_))));
    }

    #[test]
    fn update_node_out_of_range() {
        let (g, _) = diamond();
        let mut ex = ExecGraph::instantiate(&g).expect("instantiate");
        let res = ex.update_node(NodeId(99), NodeParamUpdate::Memset { value: 1 });
        assert!(matches!(res, Err(GraphError::NodeNotFound(_))));
    }

    #[test]
    fn whole_graph_update_param_only_succeeds() {
        let (g0, _) = diamond();
        let mut ex = ExecGraph::instantiate(&g0).expect("instantiate");
        // Build the same topology but with a different kernel config for `a`.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = b.add_kernel("a", 16, 64, 0).finish(); // changed config
        let l = b.add_kernel("l", 4, 256, 0).finish();
        let r = b.add_kernel("r", 4, 256, 0).finish();
        let d = b.add_kernel("d", 4, 256, 0).finish();
        b.dep(a, l).dep(a, r).dep(l, d).dep(r, d);
        let g1 = b.build().expect("builds");
        ex.update(&g1).expect("param-only update should succeed");
        // Verify the new config is reflected.
        match ex.node_kind(a).expect("a") {
            NodeKind::KernelLaunch { config, .. } => assert_eq!(config.grid, (16, 1, 1)),
            _ => panic!("kernel"),
        }
    }

    #[test]
    fn whole_graph_update_topology_change_rejected() {
        let (g0, _) = diamond();
        let mut ex = ExecGraph::instantiate(&g0).expect("instantiate");
        // Same node count but a different edge set (drop one edge).
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = b.add_kernel("a", 4, 256, 0).finish();
        let l = b.add_kernel("l", 4, 256, 0).finish();
        let r = b.add_kernel("r", 4, 256, 0).finish();
        let d = b.add_kernel("d", 4, 256, 0).finish();
        b.dep(a, l).dep(a, r).dep(l, d); // r→d missing
        let g1 = b.build().expect("builds");
        let res = ex.update(&g1);
        assert!(matches!(res, Err(GraphError::ValidationFailed(_))));
    }

    #[test]
    fn whole_graph_update_node_count_change_rejected() {
        let (g0, _) = diamond();
        let mut ex = ExecGraph::instantiate(&g0).expect("instantiate");
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = b.add_kernel("a", 4, 256, 0).finish();
        let l = b.add_kernel("l", 4, 256, 0).finish();
        b.dep(a, l);
        let g1 = b.build().expect("builds");
        let res = ex.update(&g1);
        assert!(matches!(res, Err(GraphError::ValidationFailed(_))));
    }

    #[test]
    fn diff_identical_graph() {
        let (g, _) = diamond();
        let ex = ExecGraph::instantiate(&g).expect("instantiate");
        let d = ex.diff(&g).expect("diff");
        assert!(d.is_identical());
        assert!(d.is_updatable());
        assert!(d.changed_params.is_empty());
    }

    #[test]
    fn diff_reports_changed_params() {
        let (g0, [a, ..]) = diamond();
        let ex = ExecGraph::instantiate(&g0).expect("instantiate");
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let na = b.add_kernel("a", 99, 1, 0).finish(); // changed config
        let l = b.add_kernel("l", 4, 256, 0).finish();
        let r = b.add_kernel("r", 4, 256, 0).finish();
        let dd = b.add_kernel("d", 4, 256, 0).finish();
        b.dep(na, l).dep(na, r).dep(l, dd).dep(r, dd);
        let g1 = b.build().expect("builds");
        let diff = ex.diff(&g1).expect("diff");
        assert!(diff.is_updatable());
        assert!(!diff.is_identical());
        assert_eq!(diff.changed_params, vec![a]);
    }

    #[test]
    fn diff_memcpy_size_change_non_updatable() {
        let mut b0 = GraphBuilder::new().with_auto_infer_edges(false);
        let up0 = b0.add_memcpy("up", MemcpyDir::HostToDevice, 1024);
        let k0 = b0.add_kernel("k", 1, 32, 0).finish();
        b0.dep(up0, k0);
        let g0 = b0.build().expect("builds");
        let ex = ExecGraph::instantiate(&g0).expect("instantiate");

        let mut b1 = GraphBuilder::new().with_auto_infer_edges(false);
        let up1 = b1.add_memcpy("up", MemcpyDir::HostToDevice, 2048); // size changed
        let k1 = b1.add_kernel("k", 1, 32, 0).finish();
        b1.dep(up1, k1);
        let g1 = b1.build().expect("builds");

        let diff = ex.diff(&g1).expect("diff");
        assert!(!diff.is_updatable());
        assert_eq!(diff.non_updatable_nodes, vec![up0]);
        assert!(matches!(ex.diff(&g1).map(|d| d.is_updatable()), Ok(false)));
    }

    #[test]
    fn diff_kind_change_non_updatable() {
        let mut b0 = GraphBuilder::new().with_auto_infer_edges(false);
        let n0 = b0.add_kernel("k", 1, 32, 0).finish();
        let g0 = b0.build().expect("builds");
        let ex = ExecGraph::instantiate(&g0).expect("instantiate");

        let mut b1 = GraphBuilder::new().with_auto_infer_edges(false);
        let _n1 = b1.add_memset("z", 4096, 0); // kind changed kernel→memset
        let g1 = b1.build().expect("builds");

        let diff = ex.diff(&g1).expect("diff");
        assert!(!diff.is_updatable());
        assert_eq!(diff.non_updatable_nodes, vec![n0]);
    }

    #[test]
    fn diff_empty_other_errors() {
        let (g, _) = diamond();
        let ex = ExecGraph::instantiate(&g).expect("instantiate");
        let empty = ComputeGraph::new();
        assert!(matches!(ex.diff(&empty), Err(GraphError::EmptyGraph)));
    }

    #[test]
    fn param_update_target_tag() {
        assert_eq!(NodeParamUpdate::Memset { value: 1 }.target_tag(), "memset");
        assert_eq!(
            NodeParamUpdate::Kernel {
                function_name: None,
                config: KernelConfig::linear(1, 1, 0)
            }
            .target_tag(),
            "kernel"
        );
    }

    #[test]
    fn stress_large_graph_instantiates_and_diffs() {
        // CPU-side portion of the "10K-node graph builds, compiles, instantiates"
        // stress target. (On-device capture/launch remains GPU-gated.)
        const N: usize = 10_000;
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        // A wide-then-deep structure: a binary-tree-ish chain so the topology
        // has real edges (each node depends on the one two slots back).
        let ids: Vec<NodeId> = (0..N)
            .map(|i| b.add_kernel(&format!("k{i}"), 1, 32, 0).finish())
            .collect();
        for i in 2..N {
            b.dep(ids[i - 1], ids[i]);
            b.dep(ids[i - 2], ids[i]);
        }
        let g = b.build().expect("large graph builds");
        let ex = ExecGraph::instantiate(&g).expect("large graph instantiates");
        assert_eq!(ex.node_count(), N);
        assert_eq!(ex.execution_order().len(), N);
        // An identical graph diffs as updatable with no changes.
        let diff = ex.diff(&g).expect("self-diff");
        assert!(diff.is_identical());
    }

    #[test]
    fn update_memset_value() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let z = b.add_memset("z", 4096, 0x00);
        let g = b.build().expect("builds");
        let mut ex = ExecGraph::instantiate(&g).expect("instantiate");
        ex.update_node(z, NodeParamUpdate::Memset { value: 0xff })
            .expect("memset update");
        match ex.node_kind(z).expect("z") {
            NodeKind::Memset { value, .. } => assert_eq!(*value, 0xff),
            _ => panic!("memset"),
        }
    }
}
