//! CPU model of CUDA Runtime graph capture and construction
//! (`cudaStreamBeginCapture` / `cudaStreamEndCapture`, the `cudaGraphAdd*Node`
//! family, `cudaGraphInstantiate` / clone / update).
//!
//! This is a GPU-free model of the *data structures and ordering* the runtime
//! builds when capturing or constructing a graph.  It implements the
//! stream-capture state machine (idle → active → ended), records operations as
//! dependency-linked nodes, computes a topological execution order, and models
//! instantiate / clone / update — everything except actually running the work
//! on a device.
//!
//! It works at the runtime surface in terms of [`CudaStream`] and produces a
//! [`CudaGraph`] whose [`CudaGraph::topological_sort`] is a valid topological
//! order.  (The sibling `oxicuda-driver` crate has a lower-level `Graph`; this
//! models the cudart `cudaGraph_t` surface and its capture semantics, which the
//! driver layer does not expose.)

use crate::error::{CudaRtError, CudaRtResult};
use crate::event::EventFlags;
use crate::stream::CudaStream;

// ─── Node types ──────────────────────────────────────────────────────────────

/// The kind of operation a graph node represents (mirrors `cudaGraphNodeType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphNodeKind {
    /// A kernel launch node.
    Kernel,
    /// A memory copy node.
    Memcpy,
    /// A memset node.
    Memset,
    /// A host-callback node.
    Host,
    /// A child-graph node.
    ChildGraph,
    /// An empty (dependency-only) node.
    Empty,
    /// An event-record node.
    EventRecord,
    /// An event-wait node.
    EventWait,
}

/// A single node in a [`CudaGraph`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphNode {
    kind: GraphNodeKind,
    /// Bytes moved (memcpy / memset) — 0 for other node kinds. Part of the
    /// node's parameters, mutable via [`CudaGraphExec::update_node_bytes`].
    bytes: usize,
    /// A human label (e.g. kernel name) for inspection.
    label: String,
}

impl GraphNode {
    /// The kind of this node.
    #[must_use]
    pub fn kind(&self) -> GraphNodeKind {
        self.kind
    }

    /// Bytes associated with this node (memcpy / memset payload).
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The node's label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

// ─── CudaGraph ───────────────────────────────────────────────────────────────

/// A CPU model of a `cudaGraph_t`: nodes plus directed dependency edges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CudaGraph {
    nodes: Vec<GraphNode>,
    /// Directed edges `(from, to)`: `from` must execute before `to`.
    edges: Vec<(usize, usize)>,
}

impl CudaGraph {
    /// Create an empty graph (`cudaGraphCreate`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn push_node(&mut self, kind: GraphNodeKind, bytes: usize, label: String) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(GraphNode { kind, bytes, label });
        idx
    }

    /// Add a kernel node (`cudaGraphAddKernelNode`). Returns the node index.
    pub fn add_kernel_node(&mut self, name: &str, deps: &[usize]) -> CudaRtResult<usize> {
        self.validate_deps(deps)?;
        let idx = self.push_node(GraphNodeKind::Kernel, 0, name.to_string());
        self.link_deps(idx, deps);
        Ok(idx)
    }

    /// Add a memcpy node (`cudaGraphAddMemcpyNode`). Returns the node index.
    pub fn add_memcpy_node(&mut self, bytes: usize, deps: &[usize]) -> CudaRtResult<usize> {
        self.validate_deps(deps)?;
        let idx = self.push_node(GraphNodeKind::Memcpy, bytes, "memcpy".to_string());
        self.link_deps(idx, deps);
        Ok(idx)
    }

    /// Add a memset node (`cudaGraphAddMemsetNode`). Returns the node index.
    pub fn add_memset_node(&mut self, bytes: usize, deps: &[usize]) -> CudaRtResult<usize> {
        self.validate_deps(deps)?;
        let idx = self.push_node(GraphNodeKind::Memset, bytes, "memset".to_string());
        self.link_deps(idx, deps);
        Ok(idx)
    }

    /// Add an empty (dependency-only) node (`cudaGraphAddEmptyNode`).
    pub fn add_empty_node(&mut self, deps: &[usize]) -> CudaRtResult<usize> {
        self.validate_deps(deps)?;
        let idx = self.push_node(GraphNodeKind::Empty, 0, "empty".to_string());
        self.link_deps(idx, deps);
        Ok(idx)
    }

    /// Add a child-graph node embedding `child` (`cudaGraphAddChildGraphNode`).
    pub fn add_child_graph_node(
        &mut self,
        child: &CudaGraph,
        deps: &[usize],
    ) -> CudaRtResult<usize> {
        self.validate_deps(deps)?;
        // The child must itself be a valid DAG.
        child.topological_sort()?;
        let idx = self.push_node(
            GraphNodeKind::ChildGraph,
            child.total_bytes(),
            format!("child[{}]", child.node_count()),
        );
        self.link_deps(idx, deps);
        Ok(idx)
    }

    fn validate_deps(&self, deps: &[usize]) -> CudaRtResult<()> {
        for &d in deps {
            if d >= self.nodes.len() {
                return Err(CudaRtError::InvalidValue);
            }
        }
        Ok(())
    }

    fn link_deps(&mut self, node: usize, deps: &[usize]) {
        for &d in deps {
            self.edges.push((d, node));
        }
    }

    /// Add a dependency edge `from → to` (`cudaGraphAddDependencies`).
    ///
    /// # Errors
    ///
    /// [`CudaRtError::InvalidValue`] if either index is out of range, or if the
    /// edge would create a self-loop.
    pub fn add_dependency(&mut self, from: usize, to: usize) -> CudaRtResult<()> {
        if from >= self.nodes.len() || to >= self.nodes.len() || from == to {
            return Err(CudaRtError::InvalidValue);
        }
        self.edges.push((from, to));
        Ok(())
    }

    /// Number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of dependency edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// All nodes in insertion order.
    #[must_use]
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// All dependency edges as `(from, to)` pairs.
    #[must_use]
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// Total bytes across memcpy / memset nodes (and embedded child graphs).
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.nodes.iter().map(|n| n.bytes).sum()
    }

    /// Compute a topological execution order via Kahn's algorithm.
    ///
    /// # Errors
    ///
    /// [`CudaRtError::InvalidValue`] if the graph contains a cycle (not a DAG).
    pub fn topological_sort(&self) -> CudaRtResult<Vec<usize>> {
        let n = self.nodes.len();
        let mut indegree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(from, to) in &self.edges {
            adj[from].push(to);
            indegree[to] += 1;
        }
        // Seed the queue with indegree-zero nodes in ascending index order for a
        // deterministic result.
        let mut queue: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        let mut head = 0;
        while head < queue.len() {
            let node = queue[head];
            head += 1;
            order.push(node);
            for &succ in &adj[node] {
                indegree[succ] -= 1;
                if indegree[succ] == 0 {
                    queue.push(succ);
                }
            }
        }
        if order.len() != n {
            return Err(CudaRtError::InvalidValue);
        }
        Ok(order)
    }

    /// Instantiate the graph into an executable (`cudaGraphInstantiate`).
    ///
    /// Validates the graph is a DAG and precomputes the execution order.
    ///
    /// # Errors
    ///
    /// [`CudaRtError::InvalidValue`] if the graph is cyclic.
    pub fn instantiate(&self) -> CudaRtResult<CudaGraphExec> {
        let order = self.topological_sort()?;
        Ok(CudaGraphExec {
            graph: self.clone(),
            order,
        })
    }

    /// Clone the graph (`cudaGraphClone`) — a deep copy of nodes and edges.
    #[must_use]
    pub fn clone_graph(&self) -> CudaGraph {
        self.clone()
    }
}

// ─── CudaGraphExec ───────────────────────────────────────────────────────────

/// An instantiated, launch-ready graph (`cudaGraphExec_t`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaGraphExec {
    graph: CudaGraph,
    order: Vec<usize>,
}

impl CudaGraphExec {
    /// The pre-computed topological execution order.
    #[must_use]
    pub fn execution_order(&self) -> &[usize] {
        &self.order
    }

    /// The underlying graph snapshot.
    #[must_use]
    pub fn graph(&self) -> &CudaGraph {
        &self.graph
    }

    /// Node count of the instantiated graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Apply a whole-graph update (`cudaGraphExecUpdate`).
    ///
    /// The runtime permits a fast in-place update only when the *topology* of
    /// the new graph matches the instantiated one (same node count, same node
    /// kinds in topological order, same edge count).  Only node *parameters*
    /// (e.g. memcpy byte counts) may differ.  A topology change requires a full
    /// re-instantiate and is reported as
    /// [`CudaRtError::GraphExecUpdateFailure`].
    ///
    /// # Errors
    ///
    /// [`CudaRtError::GraphExecUpdateFailure`] if the topology differs.
    pub fn update(&mut self, new_graph: &CudaGraph) -> CudaRtResult<()> {
        if new_graph.node_count() != self.graph.node_count()
            || new_graph.edge_count() != self.graph.edge_count()
        {
            return Err(CudaRtError::GraphExecUpdateFailure);
        }
        let new_order = new_graph
            .topological_sort()
            .map_err(|_| CudaRtError::GraphExecUpdateFailure)?;
        // Node kinds must match position-by-position in topological order.
        for (&old_idx, &new_idx) in self.order.iter().zip(new_order.iter()) {
            if self.graph.nodes[old_idx].kind != new_graph.nodes[new_idx].kind {
                return Err(CudaRtError::GraphExecUpdateFailure);
            }
        }
        self.graph = new_graph.clone();
        self.order = new_order;
        Ok(())
    }

    /// Update the byte count of a single node in place (a parameter-only update).
    ///
    /// # Errors
    ///
    /// [`CudaRtError::InvalidValue`] if `node` is out of range.
    pub fn update_node_bytes(&mut self, node: usize, bytes: usize) -> CudaRtResult<()> {
        let n = self
            .graph
            .nodes
            .get_mut(node)
            .ok_or(CudaRtError::InvalidValue)?;
        n.bytes = bytes;
        Ok(())
    }
}

// ─── Stream capture state machine ────────────────────────────────────────────

/// Capture status of a stream (mirrors `cudaStreamCaptureStatus`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CaptureStatus {
    /// The stream is not capturing.
    None,
    /// The stream is actively capturing operations.
    Active,
    /// The capture has been invalidated by an illegal operation.
    Invalidated,
}

/// Mode of a stream capture (mirrors `cudaStreamCaptureMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CaptureMode {
    /// Global: capture is unsafe-operation-aware across all threads.
    Global,
    /// Thread-local: only this thread's potentially-unsafe calls are intercepted.
    ThreadLocal,
    /// Relaxed: no interception of potentially-unsafe calls.
    Relaxed,
}

/// A CPU model of a stream capture in progress.
///
/// Begin a capture with [`StreamCapture::begin`], record operations (each call
/// appends a node chained after the previous one, reproducing the in-order
/// semantics of a single stream), then [`StreamCapture::end`] to obtain the
/// captured [`CudaGraph`].
#[derive(Debug)]
pub struct StreamCapture {
    stream: CudaStream,
    mode: CaptureMode,
    status: CaptureStatus,
    graph: CudaGraph,
    /// Index of the most recently recorded node, to chain the next one after it.
    last_node: Option<usize>,
    /// Per-stream event flags recorded for this capture. Events recorded into a
    /// captured stream carry these flags (e.g. [`EventFlags::DISABLE_TIMING`]),
    /// which the runtime tracks per capture so `cudaStreamGetCaptureInfo` can
    /// report them alongside the status.
    event_flags: EventFlags,
}

impl StreamCapture {
    /// Begin capturing `stream` (`cudaStreamBeginCapture`).
    ///
    /// Uses [`EventFlags::DEFAULT`] for the per-stream event flags; use
    /// [`Self::begin_with_flags`] to record a specific set.
    ///
    /// # Errors
    ///
    /// [`CudaRtError::StreamCaptureUnsupported`] if `stream` is the legacy
    /// default stream — CUDA forbids capturing the legacy default stream.
    pub fn begin(stream: CudaStream, mode: CaptureMode) -> CudaRtResult<Self> {
        Self::begin_with_flags(stream, mode, EventFlags::DEFAULT)
    }

    /// Begin capturing `stream`, recording the per-stream `event_flags`
    /// (`cudaStreamBeginCapture` with explicit capture-event flags).
    ///
    /// # Errors
    ///
    /// [`CudaRtError::StreamCaptureUnsupported`] if `stream` is the legacy
    /// default stream.
    pub fn begin_with_flags(
        stream: CudaStream,
        mode: CaptureMode,
        event_flags: EventFlags,
    ) -> CudaRtResult<Self> {
        if stream.is_default() {
            return Err(CudaRtError::StreamCaptureUnsupported);
        }
        Ok(Self {
            stream,
            mode,
            status: CaptureStatus::Active,
            graph: CudaGraph::new(),
            last_node: None,
            event_flags,
        })
    }

    /// The stream being captured.
    #[must_use]
    pub fn stream(&self) -> CudaStream {
        self.stream
    }

    /// The capture mode.
    #[must_use]
    pub fn mode(&self) -> CaptureMode {
        self.mode
    }

    /// The per-stream event flags recorded for this capture.
    #[must_use]
    pub fn event_flags(&self) -> EventFlags {
        self.event_flags
    }

    /// Current capture status (`cudaStreamGetCaptureInfo`).
    #[must_use]
    pub fn status(&self) -> CaptureStatus {
        self.status
    }

    /// Current capture info, mirroring `cudaStreamGetCaptureInfo`: the capture
    /// [`CaptureStatus`] together with the per-stream [`EventFlags`] recorded for
    /// this capture.
    ///
    /// Returns the status and, when the stream is actively capturing, the
    /// recorded event flags; when not capturing ([`CaptureStatus::None`]) the
    /// flags carry [`EventFlags::DEFAULT`] since no capture-event state applies.
    #[must_use]
    pub fn capture_info(&self) -> (CaptureStatus, EventFlags) {
        match self.status {
            CaptureStatus::None => (CaptureStatus::None, EventFlags::DEFAULT),
            other => (other, self.event_flags),
        }
    }

    /// `true` while the capture is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.status == CaptureStatus::Active
    }

    /// Number of operations recorded so far.
    #[must_use]
    pub fn recorded_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Record an operation on the captured stream, chaining it after the
    /// previous one (single-stream in-order capture).
    fn record(&mut self, push: impl FnOnce(&mut CudaGraph, &[usize]) -> CudaRtResult<usize>) {
        if self.status != CaptureStatus::Active {
            return;
        }
        // Chain the new node after the previously recorded one (if any),
        // reproducing a single stream's in-order execution semantics.
        let deps: Vec<usize> = self.last_node.into_iter().collect();
        if let Ok(idx) = push(&mut self.graph, &deps) {
            self.last_node = Some(idx);
        }
    }

    /// Record a kernel launch into the capture.
    pub fn record_kernel(&mut self, name: &str) {
        self.record(|g, deps| g.add_kernel_node(name, deps));
    }

    /// Record a memcpy into the capture.
    pub fn record_memcpy(&mut self, bytes: usize) {
        self.record(|g, deps| g.add_memcpy_node(bytes, deps));
    }

    /// Record a memset into the capture.
    pub fn record_memset(&mut self, bytes: usize) {
        self.record(|g, deps| g.add_memset_node(bytes, deps));
    }

    /// Invalidate the capture (models an illegal operation during capture, e.g.
    /// a synchronizing call). After this, [`Self::end`] fails.
    pub fn invalidate(&mut self) {
        self.status = CaptureStatus::Invalidated;
    }

    /// End the capture and return the recorded graph (`cudaStreamEndCapture`),
    /// consuming the capture handle.
    ///
    /// # Errors
    ///
    /// - [`CudaRtError::StreamCaptureInvalidated`] if the capture was invalidated.
    /// - [`CudaRtError::StreamCaptureUnmatched`] if the capture was already ended.
    pub fn end(mut self) -> CudaRtResult<CudaGraph> {
        self.end_in_place()
    }

    /// End the capture in place, transitioning the status back to
    /// [`CaptureStatus::None`] and returning the recorded graph, while leaving
    /// the (now-idle) capture handle observable via [`Self::capture_info`].
    ///
    /// This is the non-consuming form backing [`Self::end`]; it lets a caller
    /// query `cudaStreamGetCaptureInfo`-equivalent state after ending.
    ///
    /// # Errors
    ///
    /// - [`CudaRtError::StreamCaptureInvalidated`] if the capture was invalidated.
    /// - [`CudaRtError::StreamCaptureUnmatched`] if the capture was already ended.
    pub fn end_in_place(&mut self) -> CudaRtResult<CudaGraph> {
        match self.status {
            CaptureStatus::Active => {
                self.status = CaptureStatus::None;
                Ok(std::mem::take(&mut self.graph))
            }
            CaptureStatus::Invalidated => Err(CudaRtError::StreamCaptureInvalidated),
            CaptureStatus::None => Err(CudaRtError::StreamCaptureUnmatched),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_driver::ffi::CUstream;

    fn stream(token: usize) -> CudaStream {
        // SAFETY: handle is used only as an opaque token; never dereferenced.
        unsafe { CudaStream::from_raw(CUstream(token as *mut std::ffi::c_void)) }
    }

    #[test]
    fn empty_graph_topo_is_empty() {
        let g = CudaGraph::new();
        assert_eq!(
            g.topological_sort().expect("empty dag"),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn linear_chain_topological_order() {
        let mut g = CudaGraph::new();
        let a = g.add_memcpy_node(4096, &[]).expect("a");
        let b = g.add_kernel_node("k", &[a]).expect("b");
        let c = g.add_memcpy_node(4096, &[b]).expect("c");
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
        // The only valid order of a→b→c.
        assert_eq!(g.topological_sort().expect("dag"), vec![a, b, c]);
    }

    #[test]
    fn diamond_topology_respects_dependencies() {
        // a → {b, c} → d
        let mut g = CudaGraph::new();
        let a = g.add_empty_node(&[]).expect("a");
        let b = g.add_kernel_node("b", &[a]).expect("b");
        let c = g.add_kernel_node("c", &[a]).expect("c");
        let d = g.add_empty_node(&[b, c]).expect("d");
        let order = g.topological_sort().expect("dag");
        let pos = |x: usize| order.iter().position(|&i| i == x).expect("present");
        assert!(pos(a) < pos(b));
        assert!(pos(a) < pos(c));
        assert!(pos(b) < pos(d));
        assert!(pos(c) < pos(d));
    }

    #[test]
    fn cycle_is_rejected() {
        let mut g = CudaGraph::new();
        let a = g.add_empty_node(&[]).expect("a");
        let b = g.add_empty_node(&[a]).expect("b");
        // Close the cycle b → a.
        g.add_dependency(b, a).expect("edge");
        assert_eq!(g.topological_sort(), Err(CudaRtError::InvalidValue));
        assert_eq!(g.instantiate().err(), Some(CudaRtError::InvalidValue));
    }

    #[test]
    fn dependency_on_unknown_node_rejected() {
        let mut g = CudaGraph::new();
        assert_eq!(g.add_kernel_node("k", &[5]), Err(CudaRtError::InvalidValue));
        assert_eq!(g.add_dependency(0, 0), Err(CudaRtError::InvalidValue));
    }

    #[test]
    fn instantiate_precomputes_order() {
        let mut g = CudaGraph::new();
        let a = g.add_memset_node(256, &[]).expect("a");
        let b = g.add_kernel_node("k", &[a]).expect("b");
        let exec = g.instantiate().expect("exec");
        assert_eq!(exec.execution_order(), &[a, b]);
        assert_eq!(exec.node_count(), 2);
    }

    #[test]
    fn clone_is_independent_deep_copy() {
        let mut g = CudaGraph::new();
        g.add_kernel_node("k", &[]).expect("k");
        let mut c = g.clone_graph();
        c.add_kernel_node("k2", &[]).expect("k2");
        // Mutating the clone must not affect the original.
        assert_eq!(g.node_count(), 1);
        assert_eq!(c.node_count(), 2);
    }

    #[test]
    fn child_graph_node_embeds_bytes_and_validates() {
        let mut child = CudaGraph::new();
        child.add_memcpy_node(1024, &[]).expect("child memcpy");
        let mut parent = CudaGraph::new();
        let n = parent
            .add_child_graph_node(&child, &[])
            .expect("child node");
        assert_eq!(parent.nodes()[n].kind(), GraphNodeKind::ChildGraph);
        assert_eq!(parent.nodes()[n].bytes(), 1024);
    }

    #[test]
    fn exec_update_accepts_same_topology() {
        let mut g = CudaGraph::new();
        let a = g.add_memcpy_node(4096, &[]).expect("a");
        let b = g.add_kernel_node("k", &[a]).expect("b");
        let _ = b;
        let mut exec = g.instantiate().expect("exec");
        // A topologically identical graph with a different memcpy size.
        let mut g2 = CudaGraph::new();
        let a2 = g2.add_memcpy_node(8192, &[]).expect("a2");
        g2.add_kernel_node("k", &[a2]).expect("b2");
        assert!(exec.update(&g2).is_ok());
        assert_eq!(exec.graph().total_bytes(), 8192);
    }

    #[test]
    fn exec_update_rejects_topology_change() {
        let mut g = CudaGraph::new();
        let a = g.add_memcpy_node(4096, &[]).expect("a");
        g.add_kernel_node("k", &[a]).expect("b");
        let mut exec = g.instantiate().expect("exec");
        // Different node count → must fail.
        let mut g2 = CudaGraph::new();
        g2.add_memcpy_node(4096, &[]).expect("a2");
        assert_eq!(exec.update(&g2), Err(CudaRtError::GraphExecUpdateFailure));
    }

    #[test]
    fn exec_update_rejects_kind_swap() {
        // Same node + edge count but a kernel becomes a memset → topology differs.
        let mut g = CudaGraph::new();
        let a = g.add_memcpy_node(4096, &[]).expect("a");
        g.add_kernel_node("k", &[a]).expect("b");
        let mut exec = g.instantiate().expect("exec");
        let mut g2 = CudaGraph::new();
        let a2 = g2.add_memcpy_node(4096, &[]).expect("a2");
        g2.add_memset_node(256, &[a2]).expect("b2");
        assert_eq!(exec.update(&g2), Err(CudaRtError::GraphExecUpdateFailure));
    }

    #[test]
    fn capture_default_stream_is_unsupported() {
        assert_eq!(
            StreamCapture::begin(CudaStream::DEFAULT, CaptureMode::Global).err(),
            Some(CudaRtError::StreamCaptureUnsupported)
        );
    }

    #[test]
    fn capture_records_in_order_chain() {
        let mut cap = StreamCapture::begin(stream(1), CaptureMode::ThreadLocal).expect("begin");
        assert!(cap.is_active());
        assert_eq!(cap.status(), CaptureStatus::Active);
        cap.record_memcpy(4096);
        cap.record_kernel("k");
        cap.record_memcpy(4096);
        assert_eq!(cap.recorded_count(), 3);
        let g = cap.end().expect("end");
        // The captured graph is a 3-node linear chain in record order.
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
        assert_eq!(g.topological_sort().expect("dag"), vec![0, 1, 2]);
        assert_eq!(g.nodes()[0].kind(), GraphNodeKind::Memcpy);
        assert_eq!(g.nodes()[1].kind(), GraphNodeKind::Kernel);
    }

    #[test]
    fn capture_info_reports_status_and_flags() {
        // begin-capture with explicit event flags → status Active, flags recorded.
        let flags = EventFlags::DISABLE_TIMING;
        let mut cap =
            StreamCapture::begin_with_flags(stream(7), CaptureMode::Global, flags).expect("begin");
        let (status, got_flags) = cap.capture_info();
        assert_eq!(status, CaptureStatus::Active);
        assert_eq!(got_flags, flags);
        assert_eq!(cap.event_flags(), flags);

        cap.record_kernel("k");

        // end-capture → status returns to None (idle); flags reset to DEFAULT.
        let g = cap.end_in_place().expect("end");
        assert_eq!(g.node_count(), 1);
        let (status_after, flags_after) = cap.capture_info();
        assert_eq!(status_after, CaptureStatus::None);
        assert_eq!(flags_after, EventFlags::DEFAULT);
        // A second end after returning to idle is unmatched.
        assert_eq!(
            cap.end_in_place().err(),
            Some(CudaRtError::StreamCaptureUnmatched)
        );
    }

    #[test]
    fn capture_default_flags_when_unspecified() {
        // begin() (no explicit flags) records DEFAULT event flags.
        let cap = StreamCapture::begin(stream(8), CaptureMode::Global).expect("begin");
        assert_eq!(cap.event_flags(), EventFlags::DEFAULT);
        let (status, flags) = cap.capture_info();
        assert_eq!(status, CaptureStatus::Active);
        assert_eq!(flags, EventFlags::DEFAULT);
    }

    #[test]
    fn capture_info_active_after_invalidate() {
        // Invalidated captures still surface their recorded flags with the
        // Invalidated status (not reset to DEFAULT — only None resets).
        let flags = EventFlags::BLOCKING_SYNC;
        let mut cap =
            StreamCapture::begin_with_flags(stream(9), CaptureMode::Global, flags).expect("begin");
        cap.invalidate();
        let (status, got) = cap.capture_info();
        assert_eq!(status, CaptureStatus::Invalidated);
        assert_eq!(got, flags);
    }

    #[test]
    fn invalidated_capture_cannot_end() {
        let mut cap = StreamCapture::begin(stream(2), CaptureMode::Global).expect("begin");
        cap.record_kernel("k");
        cap.invalidate();
        assert_eq!(cap.status(), CaptureStatus::Invalidated);
        assert_eq!(cap.end().err(), Some(CudaRtError::StreamCaptureInvalidated));
    }

    #[test]
    fn captured_graph_instantiates_and_launch_order_matches() {
        let mut cap = StreamCapture::begin(stream(3), CaptureMode::Global).expect("begin");
        cap.record_memset(256);
        cap.record_kernel("k");
        let g = cap.end().expect("end");
        let exec = g.instantiate().expect("exec");
        assert_eq!(exec.execution_order(), &[0, 1]);
    }
}
