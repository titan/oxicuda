//! Stream-capture modeling — a CPU-side model of CUDA stream capture.
//!
//! CUDA lets a program record a sequence of asynchronous operations issued
//! into a stream and turn them into a graph via
//! `cudaStreamBeginCapture` / `cudaStreamEndCapture`. During capture, every
//! operation enqueued on the stream becomes a graph node, and inter-stream
//! dependencies introduced through events (`cudaEventRecord` /
//! `cudaStreamWaitEvent`) become graph edges — this is how a captured graph
//! reconstructs the fork / join parallelism of multi-stream code.
//!
//! This module provides a *pure-CPU* model of that mechanism. No GPU is
//! required: it is a state machine that tracks per-stream capture status,
//! records the operations issued into each captured stream, threads
//! dependencies through events, and finally emits a [`ComputeGraph`]. The
//! resulting graph can then be analysed, optimised, and (on real hardware)
//! lowered to a driver graph exactly like a hand-built graph.
//!
//! # State machine
//!
//! ```text
//!                 begin_capture(s)
//!     None  ───────────────────────────▶  Active
//!      ▲                                     │
//!      │  end_capture(s) -> ComputeGraph     │  record_*/wait_event
//!      └─────────────────────────────────────┘
//!                                            │
//!               (illegal op, e.g. join cycle)│
//!                                            ▼
//!                                       Invalidated
//! ```
//!
//! A captured stream that is forced into the `Invalidated` state (for
//! example, because a join would have introduced a cycle, or because an
//! operation was issued on a stream that was never begun) can no longer be
//! ended successfully: [`StreamCapture::end_capture`] returns an error and
//! the partially-built graph is discarded.

use std::collections::HashMap;

use crate::error::{GraphError, GraphResult};
use crate::graph::ComputeGraph;
use crate::node::{GraphNode, MemcpyDir, NodeId, NodeKind, StreamId};

// ---------------------------------------------------------------------------
// CaptureStatus
// ---------------------------------------------------------------------------

/// Capture status of a single stream, mirroring `cudaStreamCaptureStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureStatus {
    /// The stream is not currently capturing.
    None,
    /// The stream is actively capturing operations into the graph.
    Active,
    /// Capture was invalidated by an illegal operation; the graph is unusable.
    Invalidated,
}

impl std::fmt::Display for CaptureStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Active => write!(f, "active"),
            Self::Invalidated => write!(f, "invalidated"),
        }
    }
}

// ---------------------------------------------------------------------------
// CaptureEvent
// ---------------------------------------------------------------------------

/// Opaque identifier for an event used to thread cross-stream dependencies.
///
/// Mirrors a `cudaEvent_t` that is recorded on one stream and waited on by
/// another to reconstruct fork / join parallelism in a captured graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CaptureEvent(pub u32);

impl std::fmt::Display for CaptureEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "E{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// StreamCapture
// ---------------------------------------------------------------------------

/// A CPU-side model of one or more concurrently-capturing CUDA streams.
///
/// Operations are recorded into a single underlying [`ComputeGraph`]; the
/// capture session tracks, per stream, the most recently recorded node so it
/// can wire in the implicit sequential dependency between successive ops on
/// the same stream. Events bridge dependencies across streams.
///
/// # Example
///
/// ```
/// use oxicuda_graph::capture::{StreamCapture, CaptureEvent, CaptureStatus};
/// use oxicuda_graph::node::{MemcpyDir, StreamId};
///
/// let mut cap = StreamCapture::new();
/// let main = StreamId(0);
/// let side = StreamId(1);
///
/// cap.begin_capture(main).unwrap();
/// cap.begin_capture(side).unwrap();
/// // origin op on the main stream
/// let up = cap.record_memcpy(main, "upload", MemcpyDir::HostToDevice, 4096).unwrap();
///
/// // fork: side stream waits for an event recorded on the main stream
/// let ev = CaptureEvent(0);
/// cap.record_event(main, ev).unwrap();
/// cap.wait_event(side, ev).unwrap();
/// let k = cap.record_kernel(side, "relu", 4, 256, 0).unwrap();
///
/// // join back into the main stream
/// let ev2 = CaptureEvent(1);
/// cap.record_event(side, ev2).unwrap();
/// cap.wait_event(main, ev2).unwrap();
/// let dn = cap.record_memcpy(main, "download", MemcpyDir::DeviceToHost, 4096).unwrap();
///
/// assert_eq!(cap.status(main), CaptureStatus::Active);
/// let graph = cap.end_capture(main).unwrap();
/// // up → k (via fork) and k → dn (via join) must hold.
/// assert!(graph.is_reachable(up, k));
/// assert!(graph.is_reachable(k, dn));
/// ```
#[derive(Debug, Default)]
pub struct StreamCapture {
    /// The graph being assembled.
    graph: ComputeGraph,
    /// Per-stream capture status.
    status: HashMap<StreamId, CaptureStatus>,
    /// Per-stream most-recently-recorded node (for sequential ordering).
    last_on_stream: HashMap<StreamId, NodeId>,
    /// Event → node that last "recorded" it (the node a later wait depends on).
    event_source: HashMap<CaptureEvent, NodeId>,
    /// Cross-stream waits not yet attached to a node (stream → source nodes).
    ///
    /// When a stream waits on an event while it has no head node, the
    /// dependency is parked here and attached to the next op recorded on that
    /// stream.
    pending_waits: HashMap<StreamId, Vec<NodeId>>,
    /// Whether any stream entered the Invalidated state.
    poisoned: bool,
}

impl StreamCapture {
    /// Creates an empty capture session with no streams capturing.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the capture status of `stream`.
    ///
    /// A stream that was never begun reports [`CaptureStatus::None`].
    #[must_use]
    pub fn status(&self, stream: StreamId) -> CaptureStatus {
        self.status
            .get(&stream)
            .copied()
            .unwrap_or(CaptureStatus::None)
    }

    /// Returns `true` if any stream in this session is actively capturing.
    #[must_use]
    pub fn is_capturing(&self) -> bool {
        self.status.values().any(|s| *s == CaptureStatus::Active)
    }

    /// Begins capture on `stream`.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::Internal`] if `stream` is already capturing.
    pub fn begin_capture(&mut self, stream: StreamId) -> GraphResult<()> {
        match self.status(stream) {
            CaptureStatus::Active => Err(GraphError::Internal(format!(
                "stream {stream} is already capturing"
            ))),
            _ => {
                self.status.insert(stream, CaptureStatus::Active);
                Ok(())
            }
        }
    }

    /// Records a kernel-launch operation on a capturing stream.
    ///
    /// Returns the [`NodeId`] of the created node.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::Internal`] if `stream` is not actively capturing.
    pub fn record_kernel(
        &mut self,
        stream: StreamId,
        function_name: &str,
        num_blocks: u32,
        threads_per_block: u32,
        shared_mem: u32,
    ) -> GraphResult<NodeId> {
        let kind = NodeKind::KernelLaunch {
            function_name: function_name.to_owned(),
            config: crate::node::KernelConfig::linear(num_blocks, threads_per_block, shared_mem),
            fusible: false,
        };
        self.record_op(stream, kind, function_name)
    }

    /// Records a memcpy operation on a capturing stream.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::Internal`] if `stream` is not actively capturing.
    pub fn record_memcpy(
        &mut self,
        stream: StreamId,
        name: &str,
        dir: MemcpyDir,
        size_bytes: usize,
    ) -> GraphResult<NodeId> {
        self.record_op(stream, NodeKind::Memcpy { dir, size_bytes }, name)
    }

    /// Records a memset operation on a capturing stream.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::Internal`] if `stream` is not actively capturing.
    pub fn record_memset(
        &mut self,
        stream: StreamId,
        name: &str,
        size_bytes: usize,
        value: u8,
    ) -> GraphResult<NodeId> {
        self.record_op(stream, NodeKind::Memset { size_bytes, value }, name)
    }

    /// Records a host-callback operation on a capturing stream.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::Internal`] if `stream` is not actively capturing.
    pub fn record_host_callback(&mut self, stream: StreamId, label: &str) -> GraphResult<NodeId> {
        self.record_op(
            stream,
            NodeKind::HostCallback {
                label: label.to_owned(),
            },
            label,
        )
    }

    /// Internal: appends a node, wiring the sequential edge from the previous
    /// op on the same stream.
    fn record_op(&mut self, stream: StreamId, kind: NodeKind, name: &str) -> GraphResult<NodeId> {
        self.ensure_active(stream)?;
        let node = GraphNode::new(NodeId(0), kind)
            .with_name(name)
            .with_stream(stream);
        let id = self.graph.add_node(node);
        // Cross-stream waits parked before this op was issued attach first, so
        // the join dependency precedes the sequential predecessor.
        self.drain_pending_waits(stream, id)?;
        // Sequential ordering within the stream.
        if let Some(&prev) = self.last_on_stream.get(&stream) {
            // Within one stream the order is linear, so this never cycles.
            self.graph.add_edge(prev, id)?;
        }
        self.last_on_stream.insert(stream, id);
        Ok(id)
    }

    /// Records `event` on `stream`, capturing the current head node so that a
    /// later [`wait_event`](Self::wait_event) can depend on it.
    ///
    /// This is the "fork" half of fork / join capture. Recording an event on
    /// a stream that has issued no operations is a no-op source (the event
    /// carries no dependency).
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::Internal`] if `stream` is not actively capturing.
    pub fn record_event(&mut self, stream: StreamId, event: CaptureEvent) -> GraphResult<()> {
        self.ensure_active(stream)?;
        if let Some(&head) = self.last_on_stream.get(&stream) {
            self.event_source.insert(event, head);
        }
        Ok(())
    }

    /// Makes `stream` wait on `event`, threading the cross-stream dependency.
    ///
    /// This is the "join" half of fork / join capture: every operation
    /// recorded on `stream` *after* this wait depends on the node that
    /// recorded `event`. In CUDA semantics a stream-wait gates only subsequent
    /// work, so the dependency is parked and attached to the next op recorded
    /// on `stream` (it never re-orders operations already issued on the
    /// stream).
    ///
    /// A cycle is only detectable once the dependency is materialised against
    /// the next recorded op; see [`record_kernel`](Self::record_kernel) and
    /// friends.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::Internal`] if `stream` is not actively capturing.
    pub fn wait_event(&mut self, stream: StreamId, event: CaptureEvent) -> GraphResult<()> {
        self.ensure_active(stream)?;
        let Some(&source) = self.event_source.get(&event) else {
            // Waiting on an event that was never recorded carries no
            // dependency (mirrors CUDA's "completed event" semantics).
            return Ok(());
        };
        // Park the dependency so the *next* op recorded on this stream inherits
        // it. A stream-wait gates only subsequent work, never ops already
        // issued, so we deliberately do not touch the current head.
        self.pending_waits.entry(stream).or_default().push(source);
        Ok(())
    }

    /// Ends capture on `stream`, returning the assembled [`ComputeGraph`].
    ///
    /// Consumes the entire capture session: a graph is captured once.
    ///
    /// # Errors
    ///
    /// * [`GraphError::Internal`] if `stream` was not actively capturing.
    /// * [`GraphError::ValidationFailed`] (via [`GraphError::Internal`]) if any
    ///   stream was invalidated during capture.
    pub fn end_capture(mut self, stream: StreamId) -> GraphResult<ComputeGraph> {
        match self.status(stream) {
            CaptureStatus::Active => {}
            CaptureStatus::Invalidated => {
                return Err(GraphError::Internal(
                    "capture was invalidated by an illegal operation".to_owned(),
                ));
            }
            CaptureStatus::None => {
                return Err(GraphError::Internal(format!(
                    "stream {stream} is not capturing"
                )));
            }
        }
        if self.poisoned {
            return Err(GraphError::Internal(
                "capture was invalidated on another stream in the session".to_owned(),
            ));
        }
        self.status.insert(stream, CaptureStatus::None);
        Ok(std::mem::take(&mut self.graph))
    }

    /// Returns the number of nodes recorded so far.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Forcibly invalidates capture on `stream`, modelling an illegal
    /// operation that CUDA would reject (e.g. a disallowed cross-stream
    /// synchronisation, or unsafe memory operation issued during capture).
    ///
    /// An invalidated stream can no longer record ops or be ended successfully;
    /// the whole session is poisoned to mirror CUDA's behaviour where one
    /// stream's invalidation aborts the capture.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::Internal`] if `stream` is not actively capturing.
    pub fn invalidate(&mut self, stream: StreamId) -> GraphResult<()> {
        self.ensure_active(stream)?;
        self.status.insert(stream, CaptureStatus::Invalidated);
        self.poisoned = true;
        Ok(())
    }

    /// Internal: validates the stream is actively capturing, also draining any
    /// pending cross-stream waits into the next op.
    fn ensure_active(&self, stream: StreamId) -> GraphResult<()> {
        match self.status(stream) {
            CaptureStatus::Active => Ok(()),
            CaptureStatus::Invalidated => Err(GraphError::Internal(format!(
                "stream {stream} capture is invalidated"
            ))),
            CaptureStatus::None => Err(GraphError::Internal(format!(
                "stream {stream} is not capturing"
            ))),
        }
    }

    /// Drains the pending cross-stream waits for `stream`, wiring `source → id`
    /// for the freshly created node `id`. Any wait that would form a cycle
    /// invalidates the stream and poisons the session.
    fn drain_pending_waits(&mut self, stream: StreamId, id: NodeId) -> GraphResult<()> {
        if let Some(sources) = self.pending_waits.remove(&stream) {
            for source in sources {
                if source != id {
                    if let Err(e) = self.graph.add_edge(source, id) {
                        self.status.insert(stream, CaptureStatus::Invalidated);
                        self.poisoned = true;
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{MemcpyDir, StreamId};

    fn s(n: u32) -> StreamId {
        StreamId(n)
    }

    #[test]
    fn fresh_stream_is_not_capturing() {
        let cap = StreamCapture::new();
        assert_eq!(cap.status(s(0)), CaptureStatus::None);
        assert!(!cap.is_capturing());
    }

    #[test]
    fn begin_sets_active() {
        let mut cap = StreamCapture::new();
        cap.begin_capture(s(0)).expect("begin capture");
        assert_eq!(cap.status(s(0)), CaptureStatus::Active);
        assert!(cap.is_capturing());
    }

    #[test]
    fn double_begin_rejected() {
        let mut cap = StreamCapture::new();
        cap.begin_capture(s(0)).expect("begin capture");
        assert!(cap.begin_capture(s(0)).is_err());
    }

    #[test]
    fn record_on_uncaptured_stream_rejected() {
        let mut cap = StreamCapture::new();
        assert!(cap.record_kernel(s(0), "k", 1, 32, 0).is_err());
    }

    #[test]
    fn sequential_ops_are_chained() {
        let mut cap = StreamCapture::new();
        cap.begin_capture(s(0)).expect("begin");
        let a = cap.record_kernel(s(0), "a", 1, 32, 0).expect("a");
        let b = cap.record_kernel(s(0), "b", 1, 32, 0).expect("b");
        let c = cap.record_kernel(s(0), "c", 1, 32, 0).expect("c");
        let g = cap.end_capture(s(0)).expect("end");
        // Linear chain a→b→c.
        assert!(g.is_reachable(a, b));
        assert!(g.is_reachable(b, c));
        assert!(g.is_reachable(a, c));
        assert!(!g.is_reachable(c, a));
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn fork_join_reconstructs_dependencies() {
        let mut cap = StreamCapture::new();
        let main = s(0);
        let side = s(1);
        cap.begin_capture(main).expect("begin main");
        let up = cap
            .record_memcpy(main, "up", MemcpyDir::HostToDevice, 1024)
            .expect("up");
        // fork main → side
        cap.begin_capture(side).expect("begin side");
        let ev = CaptureEvent(0);
        cap.record_event(main, ev).expect("record ev");
        cap.wait_event(side, ev).expect("wait ev");
        let k = cap.record_kernel(side, "k", 1, 32, 0).expect("k");
        // join side → main
        let ev2 = CaptureEvent(1);
        cap.record_event(side, ev2).expect("record ev2");
        cap.wait_event(main, ev2).expect("wait ev2");
        let dn = cap
            .record_memcpy(main, "dn", MemcpyDir::DeviceToHost, 1024)
            .expect("dn");
        let g = cap.end_capture(main).expect("end");
        assert!(g.is_reachable(up, k), "fork dependency up→k missing");
        assert!(g.is_reachable(k, dn), "join dependency k→dn missing");
    }

    #[test]
    fn wait_on_unrecorded_event_is_noop() {
        let mut cap = StreamCapture::new();
        cap.begin_capture(s(0)).expect("begin");
        // No event recorded yet.
        assert!(cap.wait_event(s(0), CaptureEvent(9)).is_ok());
        let g = cap.end_capture(s(0)).expect("end");
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn end_without_begin_errs() {
        let cap = StreamCapture::new();
        assert!(cap.end_capture(s(0)).is_err());
    }

    #[test]
    fn pending_wait_applies_to_next_op() {
        // side waits before issuing any op; the dependency must attach to the
        // first op recorded afterwards.
        let mut cap = StreamCapture::new();
        let main = s(0);
        let side = s(1);
        cap.begin_capture(main).expect("begin main");
        cap.begin_capture(side).expect("begin side");
        let a = cap.record_kernel(main, "a", 1, 32, 0).expect("a");
        let ev = CaptureEvent(0);
        cap.record_event(main, ev).expect("rec");
        cap.wait_event(side, ev).expect("wait"); // side has no head yet
        let b = cap.record_kernel(side, "b", 1, 32, 0).expect("b");
        let g = cap.end_capture(main).expect("end");
        assert!(g.is_reachable(a, b), "pending wait a→b not applied");
    }

    #[test]
    fn status_display() {
        assert_eq!(CaptureStatus::None.to_string(), "none");
        assert_eq!(CaptureStatus::Active.to_string(), "active");
        assert_eq!(CaptureStatus::Invalidated.to_string(), "invalidated");
    }

    #[test]
    fn event_display() {
        assert_eq!(CaptureEvent(3).to_string(), "E3");
    }

    #[test]
    fn explicit_invalidation_aborts_capture() {
        let mut cap = StreamCapture::new();
        let main = s(0);
        cap.begin_capture(main).expect("begin main");
        cap.record_kernel(main, "a", 1, 32, 0).expect("a");
        cap.invalidate(main).expect("invalidate");
        assert_eq!(cap.status(main), CaptureStatus::Invalidated);
        // Further ops are rejected.
        assert!(cap.record_kernel(main, "b", 1, 32, 0).is_err());
        // The capture cannot be ended successfully.
        assert!(
            cap.end_capture(main).is_err(),
            "invalidated capture cannot end"
        );
    }

    #[test]
    fn invalidation_poisons_other_streams() {
        // Invalidating one stream poisons the whole session: a second,
        // still-active stream also fails to end.
        let mut cap = StreamCapture::new();
        let main = s(0);
        let side = s(1);
        cap.begin_capture(main).expect("begin main");
        cap.begin_capture(side).expect("begin side");
        cap.record_kernel(side, "x", 1, 32, 0).expect("x");
        cap.invalidate(main).expect("invalidate main");
        assert_eq!(cap.status(side), CaptureStatus::Active);
        assert!(
            cap.end_capture(side).is_err(),
            "poisoned session cannot end"
        );
    }

    #[test]
    fn node_count_tracks_recorded_ops() {
        let mut cap = StreamCapture::new();
        cap.begin_capture(s(0)).expect("begin");
        cap.record_kernel(s(0), "a", 1, 32, 0).expect("a");
        cap.record_memset(s(0), "z", 64, 0).expect("z");
        assert_eq!(cap.node_count(), 2);
    }
}
