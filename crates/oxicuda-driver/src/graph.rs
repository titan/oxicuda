//! CUDA Graph API for recording and replaying sequences of GPU operations.
//!
//! CUDA Graphs allow capturing a sequence of operations (kernel launches,
//! memory copies, memsets) into a graph data structure that can be
//! instantiated and launched repeatedly with minimal CPU overhead.
//!
//! # Architecture
//!
//! This module exposes a Rust-side graph representation that records
//! operations as nodes with explicit dependency edges. [`Graph::instantiate`]
//! translates that representation into the native CUDA Graph API
//! (`cuGraphCreate` + `cuGraphInstantiate`) whenever a CUDA driver is
//! available, and [`GraphExec::launch`] issues a real `cuGraphLaunch`. On macOS
//! (or any host without a driver) the graph is still built and validated
//! CPU-side, and launching reports [`CudaError::NotInitialized`].
//!
//! # Node lowering (important)
//!
//! A [`GraphNode`] stores only an operation *specification* — a kernel name,
//! copy direction/size, or memset size/value — and carries **no** resolved
//! `CUfunction` or device pointers. Kernel / memcpy / memset nodes therefore
//! cannot yet be lowered to their real `cuGraphAddKernelNode` /
//! `cuGraphAddMemcpyNode` / `cuGraphAddMemsetNode` form: they are added as
//! `cuGraphAddEmptyNode` barriers. The instantiated graph faithfully
//! reproduces the node count and dependency **topology**, but a `cuGraphLaunch`
//! performs none of those operations' work (`instantiate` logs a
//! `tracing::warn!` when any such node is lowered). Only genuine
//! [`GraphNode::Empty`] barriers are represented exactly.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_driver::graph::{Graph, GraphNode, MemcpyDirection};
//! let mut graph = Graph::new();
//!
//! let n0 = graph.add_memcpy_node(MemcpyDirection::HostToDevice, 4096);
//! let n1 = graph.add_kernel_node(
//!     "vector_add",
//!     (4, 1, 1),
//!     (256, 1, 1),
//!     0,
//! );
//! let n2 = graph.add_memcpy_node(MemcpyDirection::DeviceToHost, 4096);
//!
//! graph.add_dependency(n0, n1).ok();
//! graph.add_dependency(n1, n2).ok();
//!
//! assert_eq!(graph.node_count(), 3);
//! assert_eq!(graph.dependency_count(), 2);
//! ```

use crate::error::{CudaError, CudaResult};
use crate::stream::Stream;

// ---------------------------------------------------------------------------
// GraphNode — individual operation in a graph
// ---------------------------------------------------------------------------

/// Direction of a memory copy operation within a graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemcpyDirection {
    /// Host to device transfer.
    HostToDevice,
    /// Device to host transfer.
    DeviceToHost,
    /// Device to device transfer.
    DeviceToDevice,
}

impl std::fmt::Display for MemcpyDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "HtoD"),
            Self::DeviceToHost => write!(f, "DtoH"),
            Self::DeviceToDevice => write!(f, "DtoD"),
        }
    }
}

/// A single operation node within a [`Graph`].
///
/// Each variant represents a different type of GPU operation that can
/// be recorded into a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphNode {
    /// A kernel launch with grid/block configuration.
    KernelLaunch {
        /// Name of the kernel function.
        function_name: String,
        /// Grid dimensions `(x, y, z)`.
        grid: (u32, u32, u32),
        /// Block dimensions `(x, y, z)`.
        block: (u32, u32, u32),
        /// Dynamic shared memory in bytes.
        shared_mem: u32,
    },
    /// A memory copy operation.
    Memcpy {
        /// Direction of the copy.
        direction: MemcpyDirection,
        /// Size of the transfer in bytes.
        size: usize,
    },
    /// A memset operation (fill device memory with a byte value).
    Memset {
        /// Number of bytes to set.
        size: usize,
        /// Byte value to fill with.
        value: u8,
    },
    /// An empty/no-op node used as a synchronisation barrier.
    Empty,
}

impl std::fmt::Display for GraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KernelLaunch {
                function_name,
                grid,
                block,
                shared_mem,
            } => write!(
                f,
                "Kernel({}, grid=({},{},{}), block=({},{},{}), smem={})",
                function_name, grid.0, grid.1, grid.2, block.0, block.1, block.2, shared_mem,
            ),
            Self::Memcpy { direction, size } => {
                write!(f, "Memcpy({direction}, {size} bytes)")
            }
            Self::Memset { size, value } => {
                write!(f, "Memset({size} bytes, value=0x{value:02x})")
            }
            Self::Empty => write!(f, "Empty"),
        }
    }
}

// ---------------------------------------------------------------------------
// Graph — collection of nodes with dependency edges
// ---------------------------------------------------------------------------

/// A CUDA graph representing a DAG of GPU operations.
///
/// Nodes represent individual operations (kernel launches, memory copies,
/// memsets, or empty barriers). Dependencies are directed edges that
/// enforce execution ordering between nodes.
///
/// The graph can be instantiated into a [`GraphExec`] for repeated
/// low-overhead execution.
#[derive(Debug, Clone)]
pub struct Graph {
    nodes: Vec<GraphNode>,
    dependencies: Vec<(usize, usize)>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Creates a new empty graph with no nodes or dependencies.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            dependencies: Vec::new(),
        }
    }

    /// Adds a kernel launch node to the graph.
    ///
    /// Returns the index of the newly created node, which can be used
    /// to establish dependencies via [`add_dependency`](Self::add_dependency).
    ///
    /// # Parameters
    ///
    /// * `function_name` - Name of the kernel function.
    /// * `grid` - Grid dimensions `(x, y, z)`.
    /// * `block` - Block dimensions `(x, y, z)`.
    /// * `shared_mem` - Dynamic shared memory in bytes.
    pub fn add_kernel_node(
        &mut self,
        function_name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
    ) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(GraphNode::KernelLaunch {
            function_name: function_name.to_owned(),
            grid,
            block,
            shared_mem,
        });
        idx
    }

    /// Adds a memory copy node to the graph.
    ///
    /// Returns the index of the newly created node.
    ///
    /// # Parameters
    ///
    /// * `direction` - Direction of the memory copy.
    /// * `size` - Size of the transfer in bytes.
    pub fn add_memcpy_node(&mut self, direction: MemcpyDirection, size: usize) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(GraphNode::Memcpy { direction, size });
        idx
    }

    /// Adds a memset node to the graph.
    ///
    /// Returns the index of the newly created node.
    ///
    /// # Parameters
    ///
    /// * `size` - Number of bytes to set.
    /// * `value` - Byte value to fill with.
    pub fn add_memset_node(&mut self, size: usize, value: u8) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(GraphNode::Memset { size, value });
        idx
    }

    /// Adds an empty (no-op) node to the graph.
    ///
    /// Empty nodes are useful as synchronisation barriers — they have
    /// no work of their own but can serve as join points for multiple
    /// dependency chains.
    ///
    /// Returns the index of the newly created node.
    pub fn add_empty_node(&mut self) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(GraphNode::Empty);
        idx
    }

    /// Adds a dependency edge from node `from` to node `to`.
    ///
    /// This means `to` will not begin execution until `from` has
    /// completed. Both indices must refer to existing nodes.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if either index is out of bounds
    /// or if `from == to` (self-dependency).
    pub fn add_dependency(&mut self, from: usize, to: usize) -> CudaResult<()> {
        if from >= self.nodes.len() || to >= self.nodes.len() {
            return Err(CudaError::InvalidValue);
        }
        if from == to {
            return Err(CudaError::InvalidValue);
        }
        self.dependencies.push((from, to));
        Ok(())
    }

    /// Returns the total number of nodes in the graph.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the total number of dependency edges in the graph.
    #[inline]
    pub fn dependency_count(&self) -> usize {
        self.dependencies.len()
    }

    /// Returns a slice of all nodes in insertion order.
    #[inline]
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Returns a slice of all dependency edges as `(from, to)` pairs.
    #[inline]
    pub fn dependencies(&self) -> &[(usize, usize)] {
        &self.dependencies
    }

    /// Returns the node at the given index, or `None` if out of bounds.
    pub fn get_node(&self, index: usize) -> Option<&GraphNode> {
        self.nodes.get(index)
    }

    /// Performs a topological sort of the graph nodes.
    ///
    /// Returns the node indices in an order that respects all
    /// dependency edges, or an error if the graph contains a cycle.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if the graph contains a
    /// dependency cycle.
    pub fn topological_sort(&self) -> CudaResult<Vec<usize>> {
        let n = self.nodes.len();
        let mut in_degree = vec![0u32; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for &(from, to) in &self.dependencies {
            adj[from].push(to);
            in_degree[to] = in_degree[to].saturating_add(1);
        }

        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut result = Vec::with_capacity(n);

        while let Some(node) = queue.pop() {
            result.push(node);
            for &next in &adj[node] {
                in_degree[next] = in_degree[next].saturating_sub(1);
                if in_degree[next] == 0 {
                    queue.push(next);
                }
            }
        }

        if result.len() != n {
            return Err(CudaError::InvalidValue);
        }

        Ok(result)
    }

    /// Instantiates the graph into an executable form.
    ///
    /// The returned [`GraphExec`] can be launched on a stream with minimal
    /// CPU overhead.  The graph is always validated (topological sort)
    /// during instantiation.
    ///
    /// When a CUDA driver is available, a genuine `CUgraph` is built
    /// (`cuGraphCreate` + per-node `cuGraphAdd*Node` with the dependency DAG
    /// wired through real `CUgraphNode` edges) and finalised into a
    /// `CUgraphExec` via `cuGraphInstantiate`; [`GraphExec::launch`] then
    /// issues a real `cuGraphLaunch`.  Without a driver (macOS, or a host
    /// with no GPU) the `GraphExec` is CPU-side only and `launch` reports
    /// [`CudaError::NotInitialized`].
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the graph contains a dependency
    ///   cycle.
    /// * Any [`CudaError`] mapped from a failing `cuGraph*` driver call
    ///   when a driver is present (e.g. [`CudaError::OutOfMemory`]).
    pub fn instantiate(&self) -> CudaResult<GraphExec> {
        // Validate the graph is a DAG by performing a topological sort.
        // This must succeed regardless of driver availability.
        let execution_order = self.topological_sort()?;

        // Attempt a real driver-backed instantiation.  Fall back to a
        // CPU-side-only GraphExec for environmental reasons — no driver, no
        // GPU, no current CUDA context, or a driver predating the graph API
        // — since none of those indicate a malformed graph.  A genuine
        // graph-construction failure (e.g. OutOfMemory, InvalidValue) is a
        // real error and propagates to the caller.
        let (raw_graph, raw_exec) = match self.build_driver_graph() {
            Ok(handles) => handles,
            Err(
                CudaError::NotInitialized
                | CudaError::NotSupported
                | CudaError::InvalidContext
                | CudaError::NoDevice
                | CudaError::InvalidDevice
                | CudaError::Deinitialized,
            ) => (None, None),
            Err(other) => return Err(other),
        };

        Ok(GraphExec {
            graph: self.clone(),
            execution_order,
            raw_graph,
            raw_exec,
            owner: crate::context::current_ctx_owner(),
        })
    }

    /// Build a real CUDA driver graph from this in-memory representation.
    ///
    /// Returns `(Some(CUgraph), Some(CUgraphExec))` on success.  Returns
    /// [`CudaError::NotInitialized`] when no driver is loaded and
    /// [`CudaError::NotSupported`] when the loaded driver predates the CUDA
    /// Graph API; [`Graph::instantiate`] turns both (and other environmental
    /// errors) into a CPU-side-only `GraphExec`.  Any other error is a
    /// genuine driver failure.
    ///
    /// Each in-memory [`GraphNode`] is translated to a real driver node and
    /// the dependency edges are reproduced exactly.  Nodes are created in
    /// topological order so that, when `cuGraphAddEmptyNode` is given a
    /// node's dependency list, every referenced `CUgraphNode` already
    /// exists — regardless of the order edges were added to the in-memory
    /// graph.  Because [`GraphNode`] stores only an operation specification
    /// (no resolved `CUfunction` or device pointers), every node is added
    /// via `cuGraphAddEmptyNode`; the resulting driver graph preserves the
    /// node count and dependency topology and executes as a DAG of
    /// synchronisation barriers.
    fn build_driver_graph(
        &self,
    ) -> CudaResult<(Option<crate::ffi::CUgraph>, Option<crate::ffi::CUgraphExec>)> {
        use crate::ffi::{CUgraph, CUgraphExec, CUgraphNode};

        let api = crate::loader::try_driver()?;

        // Resolve every required graph entry point; a pre-10.0 driver lacks
        // them and yields a clean NotSupported fallback.
        let create = api.cu_graph_create.ok_or(CudaError::NotSupported)?;
        let add_empty = api.cu_graph_add_empty_node.ok_or(CudaError::NotSupported)?;
        let destroy = api.cu_graph_destroy.ok_or(CudaError::NotSupported)?;

        // A topological order of the in-memory nodes — guaranteed acyclic
        // because `instantiate` runs `topological_sort` first.
        let order = self.topological_sort()?;

        // HONESTY: `GraphNode` carries only an operation *specification*
        // (kernel name, copy direction/size, memset size/value) — no resolved
        // `CUfunction` or device pointers — so kernel / memcpy / memset nodes
        // cannot yet be lowered to their real `cuGraphAdd*Node` form. They are
        // added as `cuGraphAddEmptyNode` barriers, so a subsequent
        // `cuGraphLaunch` reproduces the DAG *topology* but performs **none** of
        // their work. Warn loudly so a successful `launch()` is never mistaken
        // for the nodes' operations having executed.
        let lowered_nodes = self
            .nodes
            .iter()
            .filter(|n| !matches!(n, GraphNode::Empty))
            .count();
        if lowered_nodes > 0 {
            tracing::warn!(
                lowered_nodes,
                "instantiating CUDA graph: {lowered_nodes} kernel/memcpy/memset node(s) \
                 are lowered to empty barriers (cuGraphAddEmptyNode) and will NOT perform \
                 their operation; only the dependency topology executes",
            );
        }

        // 1. Create an empty CUgraph.
        let mut raw_graph = CUgraph::default();
        // SAFETY: `create` was just resolved from the driver; `raw_graph` is
        // a valid out-pointer and flags=0 is the only documented value.
        crate::error::check(unsafe { create(&mut raw_graph, 0) })?;

        // From here on, any failure must destroy `raw_graph` before
        // returning so the driver-side object does not leak.
        let build = || -> CudaResult<CUgraphExec> {
            // 2. Add one real driver node per in-memory node, in topological
            //    order, wiring the incoming dependency edges as we go.
            //    `driver_nodes[idx]` holds the driver handle for in-memory
            //    node `idx` once it has been created.
            let mut driver_nodes: Vec<Option<CUgraphNode>> = vec![None; self.nodes.len()];
            for &node_idx in &order {
                // Collect the driver handles of every node this node depends
                // on — edges `(from, to)` with `to == node_idx`.  In a valid
                // topological order every `from` precedes `node_idx`, so each
                // handle is already present.
                let mut deps: Vec<CUgraphNode> = Vec::new();
                for &(from, to) in &self.dependencies {
                    if to == node_idx {
                        let handle = driver_nodes
                            .get(from)
                            .copied()
                            .flatten()
                            .ok_or(CudaError::InvalidValue)?;
                        deps.push(handle);
                    }
                }

                let dep_ptr = if deps.is_empty() {
                    std::ptr::null()
                } else {
                    deps.as_ptr()
                };

                let mut driver_node = CUgraphNode::default();
                // SAFETY: `add_empty` was resolved from the driver;
                // `driver_node` is a valid out-pointer, `raw_graph` is the
                // live graph created above, and `dep_ptr`/`deps.len()`
                // describe a valid (possibly empty) dependency slice whose
                // handles were all produced by earlier iterations.
                crate::error::check(unsafe {
                    add_empty(&mut driver_node, raw_graph, dep_ptr, deps.len())
                })?;
                driver_nodes[node_idx] = Some(driver_node);
            }

            // 3. Instantiate the populated graph into an executable form.
            self.instantiate_driver_graph(api, raw_graph)
        };

        match build() {
            Ok(raw_exec) => Ok((Some(raw_graph), Some(raw_exec))),
            Err(e) => {
                // SAFETY: `destroy` was resolved from the driver and
                // `raw_graph` is the live handle created above.
                let rc = unsafe { destroy(raw_graph) };
                if rc != 0 {
                    tracing::warn!(
                        cuda_error = rc,
                        "cuGraphDestroy failed while unwinding a failed instantiation"
                    );
                }
                Err(e)
            }
        }
    }

    /// Finalise a populated `CUgraph` into an executable `CUgraphExec`.
    ///
    /// Prefers `cuGraphInstantiateWithFlags` (CUDA 11.4+) and falls back to
    /// the legacy `cuGraphInstantiate_v2` signature.
    fn instantiate_driver_graph(
        &self,
        api: &crate::loader::DriverApi,
        raw_graph: crate::ffi::CUgraph,
    ) -> CudaResult<crate::ffi::CUgraphExec> {
        use crate::ffi::CUgraphExec;

        let mut raw_exec = CUgraphExec::default();

        if let Some(instantiate_flags) = api.cu_graph_instantiate_with_flags {
            // SAFETY: `instantiate_flags` was resolved from the driver;
            // `raw_exec` is a valid out-pointer, `raw_graph` is a live
            // populated graph, and flags=0 requests default instantiation.
            crate::error::check(unsafe { instantiate_flags(&mut raw_exec, raw_graph, 0) })?;
            return Ok(raw_exec);
        }

        let instantiate = api.cu_graph_instantiate.ok_or(CudaError::NotSupported)?;
        // SAFETY: `instantiate` was resolved from the driver; `raw_exec` is a
        // valid out-pointer, `raw_graph` is a live populated graph, and
        // passing null error-node / log-buffer pointers with a zero buffer
        // size is the documented "no diagnostics" configuration.
        crate::error::check(unsafe {
            instantiate(
                &mut raw_exec,
                raw_graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        })?;
        Ok(raw_exec)
    }
}

impl std::fmt::Display for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Graph({} nodes, {} deps)",
            self.nodes.len(),
            self.dependencies.len()
        )
    }
}

// ---------------------------------------------------------------------------
// GraphExec — instantiated executable graph
// ---------------------------------------------------------------------------

/// An instantiated, executable graph.
///
/// Created by [`Graph::instantiate`], a `GraphExec` holds a snapshot of the
/// graph and a pre-computed execution order.
///
/// # Driver backing
///
/// When a CUDA driver is available, `instantiate` builds a genuine
/// `CUgraph` (`cuGraphCreate` + one `cuGraphAdd*Node` per in-memory node,
/// with the dependency DAG wired through real `CUgraphNode` edges) and
/// finalises it into a `CUgraphExec` via `cuGraphInstantiate`.  In that
/// case [`launch`](Self::launch) issues a real `cuGraphLaunch`.
///
/// The in-memory [`GraphNode`] representation stores only an operation
/// *specification* (kernel name, copy direction/size, memset size/value) —
/// it carries no resolved `CUfunction` or device pointers.  Every node is
/// therefore translated to a real `cuGraphAddEmptyNode`: the resulting
/// driver graph reproduces the node count and dependency topology exactly
/// and executes on the GPU as a DAG of synchronisation barriers.  The
/// per-node dispatch in `Graph::build_driver_graph` is structured so that
/// kernel / memcpy / memset nodes that gain concrete device operands can be
/// promoted to `cuGraphAddKernelNode` / `cuGraphAddMemcpyNode` /
/// `cuGraphAddMemsetNode` without further restructuring.
///
/// On macOS (or any host without a CUDA driver), no driver handles are
/// created; the graph is still validated (topological sort) and
/// [`launch`](Self::launch) returns [`CudaError::NotInitialized`].
pub struct GraphExec {
    graph: Graph,
    execution_order: Vec<usize>,
    /// Real `CUgraph` handle, when a driver backed instantiation.
    raw_graph: Option<crate::ffi::CUgraph>,
    /// Real `CUgraphExec` handle, when a driver backed instantiation.
    raw_exec: Option<crate::ffi::CUgraphExec>,
    /// The context that owned the driver handles at instantiation, used to skip
    /// the driver destroys if that context was torn down first (avoids a
    /// use-after-free). `None` when no tracked context was current — see
    /// [`crate::context::current_ctx_owner`].
    owner: crate::context::CtxOwner,
}

impl GraphExec {
    /// Launches the executable graph on the given stream.
    ///
    /// When this `GraphExec` is backed by a real `CUgraphExec`, this issues
    /// `cuGraphLaunch(hGraphExec, hStream)`, submitting the entire graph to
    /// the stream with minimal CPU overhead.  Otherwise it surfaces the
    /// driver-load error.
    ///
    /// # Errors
    ///
    /// * [`CudaError::NotInitialized`] if the CUDA driver is not available
    ///   (e.g. on macOS, or a host without an NVIDIA GPU).
    /// * Any [`CudaError`] mapped from `cuGraphLaunch`.
    pub fn launch(&self, stream: &Stream) -> CudaResult<()> {
        let api = crate::loader::try_driver()?;

        // A driver is present.  If instantiation produced a real executable
        // graph, submit it; otherwise the driver lacks the graph API.
        let raw_exec = self.raw_exec.ok_or(CudaError::NotSupported)?;
        let launch = api.cu_graph_launch.ok_or(CudaError::NotSupported)?;

        // SAFETY: `launch` was just resolved from the driver; `raw_exec` is a
        // live `CUgraphExec` produced by `cuGraphInstantiate` and kept alive
        // by `self`, and `stream.raw()` is a valid `CUstream`.
        crate::error::check(unsafe { launch(raw_exec, stream.raw()) })
    }

    /// Returns a reference to the underlying graph.
    #[inline]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Returns the pre-computed execution order (topological sort).
    #[inline]
    pub fn execution_order(&self) -> &[usize] {
        &self.execution_order
    }

    /// Returns the total number of nodes that would be executed.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns `true` if this `GraphExec` is backed by a real, live
    /// `CUgraphExec` driver handle (as opposed to a CPU-side-only graph).
    #[inline]
    pub fn is_driver_backed(&self) -> bool {
        self.raw_exec.is_some()
    }
}

impl std::fmt::Debug for GraphExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphExec")
            .field("graph", &self.graph)
            .field("execution_order", &self.execution_order)
            .field("driver_backed", &self.is_driver_backed())
            .finish()
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        // Hold the registry lock across the destroys, and skip them entirely if
        // the owning context was already torn down (its `cuCtxDestroy` already
        // freed these graph objects — destroying them again would be a
        // use-after-free).
        let map = crate::context::lock_live_ctxs();
        if !crate::context::owner_is_live(&map, self.owner) {
            return;
        }
        // Release driver handles in reverse construction order: the
        // executable graph first, then the source graph.
        if let Ok(api) = crate::loader::try_driver() {
            if let (Some(exec), Some(destroy)) = (self.raw_exec, api.cu_graph_exec_destroy) {
                // SAFETY: `destroy` was resolved from the driver and `exec`
                // is a live handle produced by `cuGraphInstantiate`.
                let rc = unsafe { destroy(exec) };
                if rc != 0 {
                    tracing::warn!(cuda_error = rc, "cuGraphExecDestroy failed during drop");
                }
            }
            if let (Some(graph), Some(destroy)) = (self.raw_graph, api.cu_graph_destroy) {
                // SAFETY: `destroy` was resolved from the driver and `graph`
                // is a live handle produced by `cuGraphCreate`.
                let rc = unsafe { destroy(graph) };
                if rc != 0 {
                    tracing::warn!(cuda_error = rc, "cuGraphDestroy failed during drop");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StreamCapture — capture operations into a graph
// ---------------------------------------------------------------------------

/// Records GPU operations submitted to a stream into a [`Graph`].
///
/// Stream capture intercepts operations that would normally be submitted
/// to a CUDA stream and instead records them as graph nodes. The captured
/// operations can then be replayed efficiently via [`GraphExec`].
///
/// # Usage
///
/// ```rust,no_run
/// # use oxicuda_driver::graph::{StreamCapture, MemcpyDirection};
/// # use oxicuda_driver::stream::Stream;
/// # use std::sync::Arc;
/// # use oxicuda_driver::context::Context;
/// # fn main() -> oxicuda_driver::CudaResult<()> {
/// # let ctx: Arc<Context> = unimplemented!();
/// # let stream = Stream::new(&ctx)?;
/// let mut capture = StreamCapture::begin(&stream)?;
///
/// capture.record_kernel("my_kernel", (4, 1, 1), (256, 1, 1), 0);
/// capture.record_memcpy(MemcpyDirection::DeviceToHost, 1024);
///
/// let graph = capture.end()?;
/// assert_eq!(graph.node_count(), 2);
/// # Ok(())
/// # }
/// ```
pub struct StreamCapture {
    nodes: Vec<GraphNode>,
    /// Whether capture is still active (not yet ended).
    active: bool,
}

impl StreamCapture {
    /// Begins capturing operations on the given stream.
    ///
    /// On a real CUDA system, this would call
    /// `cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL)`.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotInitialized`] if the CUDA driver is not
    /// available.
    pub fn begin(_stream: &Stream) -> CudaResult<Self> {
        // Validate that the driver is available.
        let _api = crate::loader::try_driver()?;
        Ok(Self {
            nodes: Vec::new(),
            active: true,
        })
    }

    /// Records a kernel launch operation in the capture.
    ///
    /// # Parameters
    ///
    /// * `function_name` - Name of the kernel function.
    /// * `grid` - Grid dimensions `(x, y, z)`.
    /// * `block` - Block dimensions `(x, y, z)`.
    /// * `shared_mem` - Dynamic shared memory in bytes.
    pub fn record_kernel(
        &mut self,
        function_name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
    ) {
        if self.active {
            self.nodes.push(GraphNode::KernelLaunch {
                function_name: function_name.to_owned(),
                grid,
                block,
                shared_mem,
            });
        }
    }

    /// Records a memory copy operation in the capture.
    ///
    /// # Parameters
    ///
    /// * `direction` - Direction of the memory copy.
    /// * `size` - Size of the transfer in bytes.
    pub fn record_memcpy(&mut self, direction: MemcpyDirection, size: usize) {
        if self.active {
            self.nodes.push(GraphNode::Memcpy { direction, size });
        }
    }

    /// Records a memset operation in the capture.
    ///
    /// # Parameters
    ///
    /// * `size` - Number of bytes to set.
    /// * `value` - Byte value to fill with.
    pub fn record_memset(&mut self, size: usize, value: u8) {
        if self.active {
            self.nodes.push(GraphNode::Memset { size, value });
        }
    }

    /// Returns the number of operations recorded so far.
    #[inline]
    pub fn recorded_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns whether the capture is still active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Ends the capture and returns the resulting [`Graph`].
    ///
    /// On a real CUDA system, this would call `cuStreamEndCapture`
    /// and return the captured graph handle.
    ///
    /// The captured nodes are connected in a linear chain (each node
    /// depends on the previous one) to preserve the order in which
    /// operations were recorded.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::StreamCaptureUnmatched`] if the capture
    /// was already ended.
    pub fn end(mut self) -> CudaResult<Graph> {
        if !self.active {
            return Err(CudaError::StreamCaptureUnmatched);
        }
        self.active = false;

        let mut graph = Graph::new();
        let mut prev_idx: Option<usize> = None;

        for node in self.nodes.drain(..) {
            let idx = graph.nodes.len();
            graph.nodes.push(node);

            // Chain each node after the previous to maintain order.
            if let Some(prev) = prev_idx {
                graph.dependencies.push((prev, idx));
            }
            prev_idx = Some(idx);
        }

        Ok(graph)
    }
}

// ---------------------------------------------------------------------------
// StreamGraphCapture — real driver-backed stream capture
// ---------------------------------------------------------------------------

/// Driver-backed CUDA stream capture.
///
/// Where [`StreamCapture`] is a CPU-side recorder (it only logs operation
/// *specifications*), `StreamGraphCapture` drives the real
/// `cuStreamBeginCapture_v2` / `cuStreamEndCapture` API. The caller begins
/// capture on a live [`Stream`], submits ordinary GPU work to that stream
/// (kernel launches, async memset/memcpy) which the driver records instead of
/// executing, and [`end`](Self::end) finalises the captured `CUgraph` into a
/// launchable [`GraphExec`]. Launching that exec replays the captured work.
///
/// Requires a driver with stream-capture support (CUDA 10.0+); otherwise
/// [`begin`](Self::begin) returns [`CudaError::NotSupported`].
pub struct StreamGraphCapture<'s> {
    stream: &'s Stream,
    active: bool,
}

impl<'s> StreamGraphCapture<'s> {
    /// Begins driver-backed capture on `stream` with the given capture mode
    /// (e.g. [`CU_STREAM_CAPTURE_MODE_GLOBAL`](crate::ffi::CU_STREAM_CAPTURE_MODE_GLOBAL)).
    ///
    /// # Errors
    ///
    /// * [`CudaError::NotInitialized`] when no driver is loaded.
    /// * [`CudaError::NotSupported`] when the driver predates stream capture.
    /// * Any [`CudaError`] mapped from `cuStreamBeginCapture_v2`.
    pub fn begin(stream: &'s Stream, mode: crate::ffi::CUstreamCaptureMode) -> CudaResult<Self> {
        let api = crate::loader::try_driver()?;
        let begin = api.cu_stream_begin_capture.ok_or(CudaError::NotSupported)?;
        // SAFETY: `begin` was resolved from the driver; `stream.raw()` is a
        // live `CUstream` and `mode` is a documented capture-mode value.
        crate::error::check(unsafe { begin(stream.raw(), mode) })?;
        Ok(Self {
            stream,
            active: true,
        })
    }

    /// Returns whether the capture is still active (not yet ended).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Reports the driver's capture status for the stream
    /// (`cuStreamIsCapturing`).
    ///
    /// # Errors
    ///
    /// Propagates driver-load failures and any error from
    /// `cuStreamIsCapturing`.
    pub fn capture_status(&self) -> CudaResult<crate::ffi::CUstreamCaptureStatus> {
        let api = crate::loader::try_driver()?;
        let is_capturing = api.cu_stream_is_capturing.ok_or(CudaError::NotSupported)?;
        let mut status = crate::ffi::CU_STREAM_CAPTURE_STATUS_NONE;
        // SAFETY: resolved from the driver; live stream, valid out-pointer.
        crate::error::check(unsafe { is_capturing(self.stream.raw(), &mut status) })?;
        Ok(status)
    }

    /// Ends capture and instantiates the captured `CUgraph` into a launchable
    /// [`GraphExec`].
    ///
    /// The returned exec's [`node_count`](GraphExec::node_count) reflects the
    /// number of nodes the driver actually captured (queried via
    /// `cuGraphGetNodes`), and [`launch`](GraphExec::launch) replays the
    /// captured work via `cuGraphLaunch`.
    ///
    /// # Errors
    ///
    /// Propagates driver-load failures and any error from `cuStreamEndCapture`
    /// or graph instantiation. On instantiation failure the captured graph is
    /// destroyed before returning so the driver object does not leak.
    pub fn end(mut self) -> CudaResult<GraphExec> {
        let api = crate::loader::try_driver()?;
        let end = api.cu_stream_end_capture.ok_or(CudaError::NotSupported)?;
        let mut raw_graph = crate::ffi::CUgraph::default();
        // SAFETY: resolved from the driver; live stream, valid out-pointer.
        crate::error::check(unsafe { end(self.stream.raw(), &mut raw_graph) })?;
        self.active = false;

        let build = || -> CudaResult<GraphExec> {
            // Build a CPU-side model with the captured node count so
            // `GraphExec::node_count()` is truthful. The driver-side topology
            // is the source of truth for launch; the model is a linear chain
            // of that many empty nodes (the captured DAG's exact edges are not
            // re-queried — only its size).
            let node_count = Self::query_node_count(api, raw_graph).unwrap_or(0);
            let mut model = Graph::new();
            let mut prev: Option<usize> = None;
            for _ in 0..node_count {
                let idx = model.add_empty_node();
                if let Some(p) = prev {
                    model.add_dependency(p, idx)?;
                }
                prev = Some(idx);
            }
            let execution_order = model.topological_sort().unwrap_or_default();
            // `instantiate_driver_graph` ignores `&self`; a throwaway graph
            // gives access to the shared instantiation path.
            let raw_exec = Graph::new().instantiate_driver_graph(api, raw_graph)?;
            Ok(GraphExec {
                graph: model,
                execution_order,
                raw_graph: Some(raw_graph),
                raw_exec: Some(raw_exec),
                owner: crate::context::current_ctx_owner(),
            })
        };

        match build() {
            Ok(exec) => Ok(exec),
            Err(e) => {
                if let Some(destroy) = api.cu_graph_destroy {
                    // SAFETY: `destroy` resolved from the driver; `raw_graph`
                    // is the live handle just produced by end-capture.
                    let rc = unsafe { destroy(raw_graph) };
                    if rc != 0 {
                        tracing::warn!(
                            cuda_error = rc,
                            "cuGraphDestroy failed while unwinding end_capture"
                        );
                    }
                }
                Err(e)
            }
        }
    }

    /// Queries the number of nodes in a captured graph via `cuGraphGetNodes`
    /// (null node pointer returns just the count).
    fn query_node_count(
        api: &crate::loader::DriverApi,
        graph: crate::ffi::CUgraph,
    ) -> CudaResult<usize> {
        let get_nodes = api.cu_graph_get_nodes.ok_or(CudaError::NotSupported)?;
        let mut count: usize = 0;
        // SAFETY: resolved from the driver; `graph` is live, a null nodes
        // pointer with a valid count out-pointer is the documented
        // count-query form.
        crate::error::check(unsafe { get_nodes(graph, std::ptr::null_mut(), &mut count) })?;
        Ok(count)
    }
}

impl Drop for StreamGraphCapture<'_> {
    fn drop(&mut self) {
        // If the caller dropped the capture without ending it, terminate the
        // capture so the stream is not left in a capturing state, and destroy
        // any graph the driver hands back.
        if !self.active {
            return;
        }
        let Ok(api) = crate::loader::try_driver() else {
            return;
        };
        if let Some(end) = api.cu_stream_end_capture {
            let mut g = crate::ffi::CUgraph::default();
            // SAFETY: resolved from the driver; live stream, valid out-pointer.
            let _ = unsafe { end(self.stream.raw(), &mut g) };
            if !g.0.is_null() {
                if let Some(destroy) = api.cu_graph_destroy {
                    // SAFETY: `g` is the live graph returned by end-capture.
                    let _ = unsafe { destroy(g) };
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_new_is_empty() {
        let g = Graph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.dependency_count(), 0);
        assert!(g.nodes().is_empty());
        assert!(g.dependencies().is_empty());
    }

    #[test]
    fn graph_default_is_empty() {
        let g = Graph::default();
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn add_kernel_node_returns_sequential_indices() {
        let mut g = Graph::new();
        let n0 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let n1 = g.add_kernel_node("k1", (2, 1, 1), (64, 1, 1), 128);
        assert_eq!(n0, 0);
        assert_eq!(n1, 1);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn add_memcpy_node_records_direction_and_size() {
        let mut g = Graph::new();
        let idx = g.add_memcpy_node(MemcpyDirection::HostToDevice, 4096);
        assert_eq!(idx, 0);
        let node = g.get_node(0);
        assert!(node.is_some());
        if let Some(GraphNode::Memcpy { direction, size }) = node {
            assert_eq!(*direction, MemcpyDirection::HostToDevice);
            assert_eq!(*size, 4096);
        } else {
            panic!("expected Memcpy node");
        }
    }

    #[test]
    fn add_memset_node_records_size_and_value() {
        let mut g = Graph::new();
        let idx = g.add_memset_node(8192, 0xAB);
        assert_eq!(idx, 0);
        if let Some(GraphNode::Memset { size, value }) = g.get_node(idx) {
            assert_eq!(*size, 8192);
            assert_eq!(*value, 0xAB);
        } else {
            panic!("expected Memset node");
        }
    }

    #[test]
    fn add_empty_node_works() {
        let mut g = Graph::new();
        let idx = g.add_empty_node();
        assert_eq!(idx, 0);
        assert_eq!(g.get_node(idx), Some(&GraphNode::Empty));
    }

    #[test]
    fn add_dependency_valid() {
        let mut g = Graph::new();
        let n0 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let n1 = g.add_kernel_node("k1", (1, 1, 1), (32, 1, 1), 0);
        assert!(g.add_dependency(n0, n1).is_ok());
        assert_eq!(g.dependency_count(), 1);
        assert_eq!(g.dependencies()[0], (0, 1));
    }

    #[test]
    fn add_dependency_out_of_bounds() {
        let mut g = Graph::new();
        let _n0 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let result = g.add_dependency(0, 5);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn add_dependency_self_loop() {
        let mut g = Graph::new();
        let n0 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let result = g.add_dependency(n0, n0);
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn topological_sort_linear_chain() {
        let mut g = Graph::new();
        let n0 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let n1 = g.add_kernel_node("k1", (1, 1, 1), (32, 1, 1), 0);
        let n2 = g.add_kernel_node("k2", (1, 1, 1), (32, 1, 1), 0);
        g.add_dependency(n0, n1).ok();
        g.add_dependency(n1, n2).ok();

        let order = g.topological_sort();
        assert!(order.is_ok());
        let order = order.ok();
        assert!(order.is_some());
        let order = order.unwrap_or_default();
        // n0 must come before n1, n1 before n2
        let pos = |n: usize| -> usize { order.iter().position(|&x| x == n).unwrap_or(usize::MAX) };
        assert!(pos(n0) < pos(n1));
        assert!(pos(n1) < pos(n2));
    }

    #[test]
    fn topological_sort_detects_cycle() {
        let mut g = Graph::new();
        let n0 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let n1 = g.add_kernel_node("k1", (1, 1, 1), (32, 1, 1), 0);
        g.add_dependency(n0, n1).ok();
        g.add_dependency(n1, n0).ok();

        let result = g.topological_sort();
        assert_eq!(result, Err(CudaError::InvalidValue));
    }

    #[test]
    fn topological_sort_no_deps() {
        let mut g = Graph::new();
        g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        g.add_kernel_node("k1", (1, 1, 1), (32, 1, 1), 0);
        g.add_kernel_node("k2", (1, 1, 1), (32, 1, 1), 0);

        let order = g.topological_sort();
        assert!(order.is_ok());
        let order = order.unwrap_or_default();
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn instantiate_valid_graph() {
        let mut g = Graph::new();
        let n0 = g.add_memcpy_node(MemcpyDirection::HostToDevice, 1024);
        let n1 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let n2 = g.add_memcpy_node(MemcpyDirection::DeviceToHost, 1024);
        g.add_dependency(n0, n1).ok();
        g.add_dependency(n1, n2).ok();

        let exec = g.instantiate();
        assert!(exec.is_ok());
        let exec = exec.ok();
        assert!(exec.is_some());
        if let Some(exec) = exec {
            assert_eq!(exec.node_count(), 3);
            assert_eq!(exec.execution_order().len(), 3);
        }
    }

    #[test]
    fn instantiate_cyclic_graph_fails() {
        let mut g = Graph::new();
        let n0 = g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        let n1 = g.add_kernel_node("k1", (1, 1, 1), (32, 1, 1), 0);
        g.add_dependency(n0, n1).ok();
        g.add_dependency(n1, n0).ok();

        let result = g.instantiate();
        assert!(result.is_err());
    }

    #[test]
    fn graph_display() {
        let mut g = Graph::new();
        g.add_kernel_node("k0", (1, 1, 1), (32, 1, 1), 0);
        g.add_memcpy_node(MemcpyDirection::HostToDevice, 512);
        let disp = format!("{g}");
        assert!(disp.contains("2 nodes"));
        assert!(disp.contains("0 deps"));
    }

    #[test]
    fn node_display() {
        let node = GraphNode::KernelLaunch {
            function_name: "foo".to_owned(),
            grid: (4, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
        };
        let disp = format!("{node}");
        assert!(disp.contains("foo"));

        let node = GraphNode::Memcpy {
            direction: MemcpyDirection::DeviceToHost,
            size: 1024,
        };
        let disp = format!("{node}");
        assert!(disp.contains("DtoH"));

        let node = GraphNode::Memset {
            size: 256,
            value: 0xFF,
        };
        let disp = format!("{node}");
        assert!(disp.contains("0xff"));

        let node = GraphNode::Empty;
        let disp = format!("{node}");
        assert!(disp.contains("Empty"));
    }

    #[test]
    fn memcpy_direction_display() {
        assert_eq!(format!("{}", MemcpyDirection::HostToDevice), "HtoD");
        assert_eq!(format!("{}", MemcpyDirection::DeviceToHost), "DtoH");
        assert_eq!(format!("{}", MemcpyDirection::DeviceToDevice), "DtoD");
    }

    #[test]
    fn graph_get_node_out_of_bounds() {
        let g = Graph::new();
        assert!(g.get_node(0).is_none());
        assert!(g.get_node(100).is_none());
    }

    #[test]
    fn graph_diamond_dag() {
        // Diamond: n0 -> n1, n0 -> n2, n1 -> n3, n2 -> n3
        let mut g = Graph::new();
        let n0 = g.add_empty_node();
        let n1 = g.add_kernel_node("k1", (1, 1, 1), (32, 1, 1), 0);
        let n2 = g.add_kernel_node("k2", (1, 1, 1), (32, 1, 1), 0);
        let n3 = g.add_empty_node();
        g.add_dependency(n0, n1).ok();
        g.add_dependency(n0, n2).ok();
        g.add_dependency(n1, n3).ok();
        g.add_dependency(n2, n3).ok();

        let order = g.topological_sort().unwrap_or_default();
        assert_eq!(order.len(), 4);
        let pos = |n: usize| -> usize { order.iter().position(|&x| x == n).unwrap_or(usize::MAX) };
        assert!(pos(n0) < pos(n1));
        assert!(pos(n0) < pos(n2));
        assert!(pos(n1) < pos(n3));
        assert!(pos(n2) < pos(n3));

        let exec = g.instantiate();
        assert!(exec.is_ok());
    }

    #[test]
    fn graph_exec_debug() {
        let mut g = Graph::new();
        g.add_empty_node();
        let exec = g.instantiate().ok();
        assert!(exec.is_some());
        if let Some(exec) = exec {
            let dbg = format!("{exec:?}");
            assert!(dbg.contains("GraphExec"));
            // The debug output advertises the driver-backed status.
            assert!(dbg.contains("driver_backed"));
        }
    }

    // -- Driver-backed instantiation ---------------------------------------
    //
    // `instantiate` builds a real `CUgraph`/`CUgraphExec` when a driver is
    // present, and a CPU-side-only `GraphExec` otherwise.  On a host with no
    // CUDA driver every path below must still produce a valid `GraphExec`
    // (clean fallback) — never a panic, never an error from the missing
    // driver alone.

    /// Returns `true` when a real CUDA driver is loadable on this host.
    fn driver_present() -> bool {
        crate::loader::try_driver().is_ok()
    }

    /// Instantiating an empty graph succeeds; without a driver the result
    /// is a CPU-side-only `GraphExec`.
    #[test]
    fn instantiate_empty_graph_driver_state() {
        let g = Graph::new();
        let exec = g.instantiate().expect("empty graph instantiates");
        assert_eq!(exec.node_count(), 0);
        if driver_present() {
            // A live driver either backs the graph or, on a graphless
            // driver, leaves it CPU-side — both are valid, typed outcomes.
            let _ = exec.is_driver_backed();
        } else {
            assert!(!exec.is_driver_backed());
        }
    }

    /// A linear-chain graph instantiates and preserves topology; the
    /// `GraphExec` reports a consistent driver-backed flag.
    #[test]
    fn instantiate_chain_preserves_topology() {
        let mut g = Graph::new();
        let n0 = g.add_memset_node(256, 0);
        let n1 = g.add_kernel_node("k", (1, 1, 1), (32, 1, 1), 0);
        let n2 = g.add_memcpy_node(MemcpyDirection::DeviceToHost, 256);
        g.add_dependency(n0, n1).ok();
        g.add_dependency(n1, n2).ok();

        let exec = g.instantiate().expect("chain instantiates");
        assert_eq!(exec.node_count(), 3);
        assert_eq!(exec.execution_order().len(), 3);
        if !driver_present() {
            assert!(!exec.is_driver_backed());
        }
    }

    /// A diamond DAG instantiates without a driver to a CPU-side `GraphExec`.
    #[test]
    fn instantiate_diamond_without_driver_is_clean() {
        let mut g = Graph::new();
        let n0 = g.add_empty_node();
        let n1 = g.add_kernel_node("k1", (1, 1, 1), (32, 1, 1), 0);
        let n2 = g.add_kernel_node("k2", (1, 1, 1), (32, 1, 1), 0);
        let n3 = g.add_empty_node();
        g.add_dependency(n0, n1).ok();
        g.add_dependency(n0, n2).ok();
        g.add_dependency(n1, n3).ok();
        g.add_dependency(n2, n3).ok();

        let exec = g.instantiate();
        assert!(exec.is_ok(), "diamond DAG must instantiate cleanly");
        if !driver_present() {
            if let Ok(exec) = exec {
                assert!(!exec.is_driver_backed());
            }
        }
    }

    /// `build_driver_graph` surfaces a clean typed error on a host with no
    /// driver — `NotInitialized`, never a panic.
    #[test]
    fn build_driver_graph_absent_driver_is_clean() {
        let mut g = Graph::new();
        g.add_empty_node();
        let result = g.build_driver_graph();
        if driver_present() {
            // Live driver: either real handles, or a typed driver error.
            match result {
                Ok((raw_graph, raw_exec)) => {
                    assert_eq!(raw_graph.is_some(), raw_exec.is_some());
                }
                Err(_) => { /* typed driver error is acceptable */ }
            }
        } else {
            assert_eq!(result.err(), Some(CudaError::NotInitialized));
        }
    }

    /// Dropping a CPU-side-only `GraphExec` must not panic (the `Drop` impl
    /// only touches driver handles when both they and the driver exist).
    #[test]
    fn graph_exec_drop_without_driver_is_safe() {
        let mut g = Graph::new();
        g.add_empty_node();
        g.add_empty_node();
        let exec = g.instantiate().expect("instantiates");
        // Explicit drop — must complete without panicking.
        drop(exec);
    }

    /// A cyclic graph fails instantiation at the topological-sort stage,
    /// before any driver call is attempted.
    #[test]
    fn instantiate_cycle_fails_before_driver() {
        let mut g = Graph::new();
        let n0 = g.add_empty_node();
        let n1 = g.add_empty_node();
        g.add_dependency(n0, n1).ok();
        g.add_dependency(n1, n0).ok();
        assert_eq!(g.instantiate().err(), Some(CudaError::InvalidValue));
    }

    // -- End-to-end real-GPU graph execution -------------------------------
    //
    // When this host has a usable GPU, build a CUDA context (which makes it
    // current), instantiate a real driver-backed graph, and launch it via
    // `cuGraphLaunch`.  On a host without a GPU the test is a clean no-op.

    /// Instantiate and launch a real diamond-DAG graph on the GPU.
    #[test]
    fn real_graph_instantiate_and_launch() {
        use crate::context::Context;
        use crate::device::Device;

        // No GPU on this host — nothing to exercise.
        let device = match Device::get(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // Creating the context makes it current on this thread, which the
        // CUDA Graph API requires.
        let ctx = match Context::new(&device) {
            Ok(c) => std::sync::Arc::new(c),
            Err(_) => return,
        };
        let stream = match Stream::new(&ctx) {
            Ok(s) => s,
            Err(_) => return,
        };

        // Diamond DAG: n0 -> {n1, n2} -> n3.
        let mut g = Graph::new();
        let n0 = g.add_empty_node();
        let n1 = g.add_kernel_node("k1", (1, 1, 1), (32, 1, 1), 0);
        let n2 = g.add_kernel_node("k2", (1, 1, 1), (32, 1, 1), 0);
        let n3 = g.add_empty_node();
        g.add_dependency(n0, n1).ok();
        g.add_dependency(n0, n2).ok();
        g.add_dependency(n1, n3).ok();
        g.add_dependency(n2, n3).ok();

        let exec = g.instantiate().expect("diamond DAG instantiates");
        assert_eq!(exec.node_count(), 4);

        // A context is current and any driver from the CUDA 10.0+ era exposes
        // the Graph API (`cuGraphAddEmptyNode` / `cuGraphInstantiate` /
        // `cuGraphLaunch`). With a real device present the graph MUST therefore
        // be driver-backed — otherwise the launch below would be silently
        // skipped and the test would pass vacuously without ever exercising the
        // driver path.
        assert!(
            exec.is_driver_backed(),
            "a real CUDA device is present but the graph is not driver-backed; \
             the cuGraph* FFI entry points failed to load"
        );
        exec.launch(&stream)
            .expect("cuGraphLaunch on a real graph succeeds");
        stream
            .synchronize()
            .expect("stream synchronises after graph launch");
    }

    /// A driver-backed graph can be relaunched repeatedly on the same stream.
    #[test]
    fn real_graph_repeated_launch() {
        use crate::context::Context;
        use crate::device::Device;

        let device = match Device::get(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let ctx = match Context::new(&device) {
            Ok(c) => std::sync::Arc::new(c),
            Err(_) => return,
        };
        let stream = match Stream::new(&ctx) {
            Ok(s) => s,
            Err(_) => return,
        };

        let mut g = Graph::new();
        let a = g.add_empty_node();
        let b = g.add_empty_node();
        g.add_dependency(a, b).ok();

        let exec = g.instantiate().expect("chain instantiates");
        // A real device is present, so the Graph API must be driver-backed
        // (see `real_graph_instantiate_and_launch`); a CPU-only fallback here
        // would make the repeated-launch assertion vacuous.
        assert!(
            exec.is_driver_backed(),
            "a real CUDA device is present but the graph is not driver-backed"
        );
        // The whole point of a graph: cheap repeated submission.
        for _ in 0..8 {
            exec.launch(&stream)
                .expect("repeated cuGraphLaunch succeeds");
        }
        stream.synchronize().expect("stream synchronises");
    }

    /// End-to-end **real stream capture** round-trip on the GPU: capture an
    /// async memset of a device buffer, instantiate the captured graph, launch
    /// it, and verify the replayed memset wrote the expected pattern to device
    /// memory — i.e. `cuGraphLaunch` output matches the CPU simulation of the
    /// captured op (the `oxicuda-graph` TODO:164 hardware check).
    #[test]
    fn real_stream_capture_memset_round_trip() {
        use crate::context::Context;
        use crate::device::Device;
        use crate::ffi::{
            CU_STREAM_CAPTURE_MODE_GLOBAL, CU_STREAM_CAPTURE_STATUS_ACTIVE, CUdeviceptr,
        };

        let device = match Device::get(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let ctx = match Context::new(&device) {
            Ok(c) => std::sync::Arc::new(c),
            Err(_) => return,
        };
        let stream = match Stream::new(&ctx) {
            Ok(s) => s,
            Err(_) => return,
        };

        let api = crate::loader::try_driver().expect("driver present");
        // Skip cleanly if this driver lacks stream capture or async memset.
        if api.cu_stream_begin_capture.is_none() || api.cu_memset_d32_async.is_none() {
            return;
        }

        const N: usize = 256;
        const MAGIC: u32 = 0xABCD_1234;
        let bytes = N * std::mem::size_of::<u32>();

        // Allocate and zero-initialise the device buffer.
        let mut dptr: CUdeviceptr = 0;
        crate::error::check(unsafe { (api.cu_mem_alloc_v2)(&mut dptr, bytes) }).expect("alloc");
        crate::error::check(unsafe { (api.cu_memset_d32_v2)(dptr, 0, N) }).expect("zero-init");

        let run = || -> CudaResult<Vec<u32>> {
            // Begin capture and record an async memset to MAGIC on the stream.
            let cap = StreamGraphCapture::begin(&stream, CU_STREAM_CAPTURE_MODE_GLOBAL)?;
            if let Ok(status) = cap.capture_status() {
                assert_eq!(
                    status, CU_STREAM_CAPTURE_STATUS_ACTIVE,
                    "stream must report active capture after begin"
                );
            }
            let memset_async = api.cu_memset_d32_async.expect("async memset");
            // SAFETY: `dptr` is a live N-element u32 allocation; while capture
            // is active this call is recorded into the graph, not executed.
            crate::error::check(unsafe { memset_async(dptr, MAGIC, N, stream.raw()) })?;
            let exec = cap.end()?;

            assert!(
                exec.is_driver_backed(),
                "captured graph must be driver-backed"
            );
            assert!(exec.node_count() >= 1, "capture recorded no nodes");

            // Capture records but does not execute: the buffer is still zero.
            // (This synchronous copy runs after capture has ended, so it does
            // not invalidate the capture.)
            let mut pre = vec![0u32; N];
            crate::error::check(unsafe {
                (api.cu_memcpy_dtoh_v2)(pre.as_mut_ptr().cast(), dptr, bytes)
            })?;
            assert!(
                pre.iter().all(|&v| v == 0),
                "capture must not execute the recorded memset"
            );

            // Replaying the graph performs the memset on the device.
            exec.launch(&stream)?;
            crate::error::check(unsafe { (api.cu_stream_synchronize)(stream.raw()) })?;
            let mut out = vec![0u32; N];
            crate::error::check(unsafe {
                (api.cu_memcpy_dtoh_v2)(out.as_mut_ptr().cast(), dptr, bytes)
            })?;
            Ok(out)
        };

        let result = run();
        // Always release the device allocation, even on failure.
        let _ = unsafe { (api.cu_mem_free_v2)(dptr) };

        let out = result.expect("stream-capture round-trip");
        // CPU simulation of the captured op: every element becomes MAGIC.
        assert!(
            out.iter().all(|&v| v == MAGIC),
            "replayed captured graph did not memset device memory to MAGIC"
        );
    }
}
