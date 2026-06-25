//! Host-side HIP graph (`hipGraph_t`) model: nodes, dependency edges,
//! instantiation, and replay ordering.
//!
//! Mirrors the `hipGraphCreate` / `hipGraphAddKernelNode` /
//! `hipGraphInstantiate` / `hipGraphLaunch` pipeline as pure data structures.
//! A [`HipGraph`] is a DAG of compute/copy/empty nodes with explicit
//! dependencies; [`HipGraph::instantiate`] checks the DAG is acyclic and
//! produces an [`ExecutableGraph`] carrying a valid topological execution
//! order, exactly as the runtime would — but entirely on CPU.

use crate::error::{RocmError, RocmResult};
use crate::stream::MemcpyKind;
use std::collections::HashMap;

// ─── GraphNode ──────────────────────────────────────────────────────────────

/// The kind of work a graph node performs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    /// A kernel-launch node (`hipGraphAddKernelNode`).
    Kernel {
        /// Kernel entry-point name.
        name: String,
    },
    /// A memory-copy node (`hipGraphAddMemcpyNode`).
    Memcpy {
        /// Copy direction.
        kind: MemcpyKind,
        /// Bytes transferred.
        bytes: u64,
    },
    /// A memory-set node (`hipGraphAddMemsetNode`).
    Memset {
        /// Bytes written.
        bytes: u64,
        /// Fill value.
        value: u8,
    },
    /// An empty synchronisation barrier node (`hipGraphAddEmptyNode`).
    Empty,
}

/// A node in a [`HipGraph`].
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Stable node id assigned at insertion time.
    pub id: u64,
    /// The work this node performs.
    pub kind: NodeKind,
}

// ─── HipGraph ───────────────────────────────────────────────────────────────

/// A directed acyclic graph of GPU work, built incrementally.
#[derive(Debug, Clone, Default)]
pub struct HipGraph {
    nodes: Vec<GraphNode>,
    /// Edges as `(predecessor_id, successor_id)`.
    edges: Vec<(u64, u64)>,
    next_id: u64,
}

impl HipGraph {
    /// Create an empty graph (`hipGraphCreate`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node with the given kind, returning its id.
    fn add_node(&mut self, kind: NodeKind) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(GraphNode { id, kind });
        id
    }

    /// Add a kernel-launch node.
    pub fn add_kernel(&mut self, name: impl Into<String>) -> u64 {
        self.add_node(NodeKind::Kernel { name: name.into() })
    }

    /// Add a memory-copy node.
    pub fn add_memcpy(&mut self, kind: MemcpyKind, bytes: u64) -> u64 {
        self.add_node(NodeKind::Memcpy { kind, bytes })
    }

    /// Add a memory-set node.
    pub fn add_memset(&mut self, bytes: u64, value: u8) -> u64 {
        self.add_node(NodeKind::Memset { bytes, value })
    }

    /// Add an empty barrier node.
    pub fn add_empty(&mut self) -> u64 {
        self.add_node(NodeKind::Empty)
    }

    /// Add a dependency edge `from → to`, meaning `to` may not start until
    /// `from` completes (`hipGraphAddDependencies`).
    ///
    /// # Errors
    ///
    /// Returns [`RocmError::InvalidArgument`] if either id is unknown or the
    /// edge is a self-loop.
    pub fn add_dependency(&mut self, from: u64, to: u64) -> RocmResult<()> {
        if from == to {
            return Err(RocmError::InvalidArgument(
                "graph node cannot depend on itself".into(),
            ));
        }
        if !self.nodes.iter().any(|n| n.id == from) {
            return Err(RocmError::InvalidArgument(format!(
                "unknown predecessor node {from}"
            )));
        }
        if !self.nodes.iter().any(|n| n.id == to) {
            return Err(RocmError::InvalidArgument(format!(
                "unknown successor node {to}"
            )));
        }
        if !self.edges.contains(&(from, to)) {
            self.edges.push((from, to));
        }
        Ok(())
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of dependency edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Root nodes (no incoming edges) — the entry points of the graph.
    pub fn roots(&self) -> Vec<u64> {
        self.nodes
            .iter()
            .filter(|n| !self.edges.iter().any(|(_, to)| *to == n.id))
            .map(|n| n.id)
            .collect()
    }

    /// Instantiate the graph into an executable form (`hipGraphInstantiate`),
    /// validating that it is acyclic and computing a topological launch order.
    ///
    /// # Errors
    ///
    /// Returns [`RocmError::DeviceError`] if the graph contains a cycle.
    pub fn instantiate(&self) -> RocmResult<ExecutableGraph> {
        let order = self.topological_order()?;
        Ok(ExecutableGraph {
            order,
            nodes: self.nodes.clone(),
        })
    }

    /// Compute a Kahn topological ordering of node ids.
    fn topological_order(&self) -> RocmResult<Vec<u64>> {
        let mut indegree: HashMap<u64, usize> = self.nodes.iter().map(|n| (n.id, 0usize)).collect();
        for (_, to) in &self.edges {
            if let Some(d) = indegree.get_mut(to) {
                *d += 1;
            }
        }
        // Seed the queue with indegree-0 nodes, preserving insertion order.
        let mut queue: Vec<u64> = self
            .nodes
            .iter()
            .filter(|n| indegree.get(&n.id).copied().unwrap_or(0) == 0)
            .map(|n| n.id)
            .collect();

        let mut order: Vec<u64> = Vec::with_capacity(self.nodes.len());
        let mut head = 0usize;
        while head < queue.len() {
            let id = queue[head];
            head += 1;
            order.push(id);
            // Relax outgoing edges in insertion order.
            for (from, to) in &self.edges {
                if *from == id {
                    if let Some(d) = indegree.get_mut(to) {
                        *d -= 1;
                        if *d == 0 {
                            queue.push(*to);
                        }
                    }
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(RocmError::DeviceError(
                "hip graph contains a cycle; cannot instantiate".into(),
            ));
        }
        Ok(order)
    }
}

// ─── ExecutableGraph ────────────────────────────────────────────────────────

/// An instantiated, launch-ready graph (`hipGraphExec_t`).
#[derive(Debug, Clone)]
pub struct ExecutableGraph {
    /// Topologically-ordered node ids.
    order: Vec<u64>,
    /// The nodes, by value, for parameter-update lookups.
    nodes: Vec<GraphNode>,
}

impl ExecutableGraph {
    /// The valid launch order of node ids.
    pub fn launch_order(&self) -> &[u64] {
        &self.order
    }

    /// Number of nodes in the executable graph.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// `true` if the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The ordered sequence of kernel names that would be launched, in
    /// execution order (memory/empty nodes are skipped).
    pub fn kernel_sequence(&self) -> Vec<String> {
        let by_id: HashMap<u64, &GraphNode> = self.nodes.iter().map(|n| (n.id, n)).collect();
        self.order
            .iter()
            .filter_map(|id| match by_id.get(id).map(|n| &n.kind) {
                Some(NodeKind::Kernel { name }) => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    /// Update a kernel node's launched function name in place
    /// (`hipGraphExecKernelNodeSetParams` equivalent for the function target).
    ///
    /// # Errors
    ///
    /// Returns [`RocmError::InvalidArgument`] if `id` is unknown or is not a
    /// kernel node.
    pub fn update_kernel_name(&mut self, id: u64, name: impl Into<String>) -> RocmResult<()> {
        let node = self
            .nodes
            .iter_mut()
            .find(|n| n.id == id)
            .ok_or_else(|| RocmError::InvalidArgument(format!("unknown node {id}")))?;
        match &mut node.kind {
            NodeKind::Kernel { name: n } => {
                *n = name.into();
                Ok(())
            }
            _ => Err(RocmError::InvalidArgument(format!(
                "node {id} is not a kernel node"
            ))),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_linear_chain_and_instantiate() {
        let mut g = HipGraph::new();
        let a = g.add_memcpy(MemcpyKind::HostToDevice, 1024);
        let b = g.add_kernel("gemm_f32");
        let c = g.add_memcpy(MemcpyKind::DeviceToHost, 1024);
        g.add_dependency(a, b).expect("edge a->b");
        g.add_dependency(b, c).expect("edge b->c");

        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
        assert_eq!(g.roots(), vec![a]);

        let exec = g.instantiate().expect("instantiate");
        assert_eq!(exec.launch_order(), &[a, b, c]);
        assert_eq!(exec.kernel_sequence(), vec!["gemm_f32".to_string()]);
    }

    #[test]
    fn diamond_dependency_orders_correctly() {
        // a → {b, c} → d
        let mut g = HipGraph::new();
        let a = g.add_kernel("split");
        let b = g.add_kernel("left");
        let c = g.add_kernel("right");
        let d = g.add_kernel("merge");
        g.add_dependency(a, b).unwrap();
        g.add_dependency(a, c).unwrap();
        g.add_dependency(b, d).unwrap();
        g.add_dependency(c, d).unwrap();

        let exec = g.instantiate().expect("instantiate");
        let order = exec.launch_order();
        // a before b, c; b, c before d.
        let pos = |id: u64| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(a) < pos(b));
        assert!(pos(a) < pos(c));
        assert!(pos(b) < pos(d));
        assert!(pos(c) < pos(d));
        assert_eq!(exec.len(), 4);
    }

    #[test]
    fn cycle_detection_fails_instantiate() {
        let mut g = HipGraph::new();
        let a = g.add_kernel("a");
        let b = g.add_kernel("b");
        g.add_dependency(a, b).unwrap();
        g.add_dependency(b, a).unwrap();
        let err = g.instantiate().unwrap_err();
        assert!(matches!(err, RocmError::DeviceError(_)));
    }

    #[test]
    fn self_dependency_rejected() {
        let mut g = HipGraph::new();
        let a = g.add_kernel("a");
        let err = g.add_dependency(a, a).unwrap_err();
        assert!(matches!(err, RocmError::InvalidArgument(_)));
    }

    #[test]
    fn unknown_node_dependency_rejected() {
        let mut g = HipGraph::new();
        let a = g.add_kernel("a");
        assert!(g.add_dependency(a, 999).is_err());
        assert!(g.add_dependency(999, a).is_err());
    }

    #[test]
    fn empty_graph_instantiates_empty() {
        let g = HipGraph::new();
        let exec = g.instantiate().expect("empty instantiate");
        assert!(exec.is_empty());
        assert!(exec.kernel_sequence().is_empty());
    }

    #[test]
    fn memset_and_empty_nodes_skipped_in_kernel_sequence() {
        let mut g = HipGraph::new();
        let z = g.add_memset(2048, 0);
        let k = g.add_kernel("relu");
        let e = g.add_empty();
        g.add_dependency(z, k).unwrap();
        g.add_dependency(k, e).unwrap();
        let exec = g.instantiate().expect("instantiate");
        assert_eq!(exec.kernel_sequence(), vec!["relu".to_string()]);
        assert_eq!(exec.len(), 3);
    }

    #[test]
    fn exec_kernel_param_update() {
        let mut g = HipGraph::new();
        let k = g.add_kernel("gemm_f32");
        let mut exec = g.instantiate().expect("instantiate");
        exec.update_kernel_name(k, "gemm_f16").expect("update");
        assert_eq!(exec.kernel_sequence(), vec!["gemm_f16".to_string()]);
    }

    #[test]
    fn exec_update_rejects_non_kernel_node() {
        let mut g = HipGraph::new();
        let m = g.add_memcpy(MemcpyKind::DeviceToDevice, 64);
        let mut exec = g.instantiate().expect("instantiate");
        let err = exec.update_kernel_name(m, "x").unwrap_err();
        assert!(matches!(err, RocmError::InvalidArgument(_)));
    }

    #[test]
    fn duplicate_dependency_is_idempotent() {
        let mut g = HipGraph::new();
        let a = g.add_kernel("a");
        let b = g.add_kernel("b");
        g.add_dependency(a, b).unwrap();
        g.add_dependency(a, b).unwrap();
        assert_eq!(g.edge_count(), 1);
    }
}
