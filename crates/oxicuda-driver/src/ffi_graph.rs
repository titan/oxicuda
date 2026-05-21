//! CUDA Graph and stream-ordered memory-pool FFI types.
//!
//! Opaque graph handles (`CUgraph`, `CUgraphExec`, `CUgraphNode`), the node
//! parameter descriptors consumed by `cuGraphAdd*Node` (`CUDA_KERNEL_NODE_PARAMS`,
//! `CUDA_MEMCPY3D`, `CUDA_MEMSET_NODE_PARAMS`), and the `CUmemPoolAttribute`
//! discriminant used by `cuMemPoolSetAttribute` / `cuMemPoolGetAttribute`.
//!
//! All structs are `#[repr(C)]` and mirror the layout of the corresponding
//! types in `cuda.h`; trailing reserved fields are part of the published ABI.

use std::ffi::{c_char, c_void};

use super::{CUarray, CUdeviceptr, CUfunction};

// =========================================================================
// CUgraph / CUgraphExec / CUgraphNode — opaque graph handles
// =========================================================================

/// Opaque handle to a CUDA graph (`CUgraph`).
///
/// A graph is a mutable DAG of operations created by `cuGraphCreate`,
/// populated by `cuGraphAdd*Node`, and finalised into an executable form
/// via `cuGraphInstantiate`.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CUgraph(pub *mut c_void);

// SAFETY: CUDA graph handles are opaque driver-side identifiers; treating
// the handle as Send+Sync mirrors the C-side pointer, which the driver may
// inspect from any thread when properly synchronised.
unsafe impl Send for CUgraph {}
unsafe impl Sync for CUgraph {}

impl CUgraph {
    /// Returns `true` if the handle is null (uninitialised).
    #[inline]
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
}

impl Default for CUgraph {
    #[inline]
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl std::fmt::Debug for CUgraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUgraph({:p})", self.0)
    }
}

/// Opaque handle to an instantiated, executable CUDA graph (`CUgraphExec`).
///
/// Produced by `cuGraphInstantiate`; submitted to a stream by `cuGraphLaunch`
/// and destroyed by `cuGraphExecDestroy`.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CUgraphExec(pub *mut c_void);

// SAFETY: see [`CUgraph`].
unsafe impl Send for CUgraphExec {}
unsafe impl Sync for CUgraphExec {}

impl CUgraphExec {
    /// Returns `true` if the handle is null (uninitialised).
    #[inline]
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
}

impl Default for CUgraphExec {
    #[inline]
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl std::fmt::Debug for CUgraphExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUgraphExec({:p})", self.0)
    }
}

/// Opaque handle to a single node within a [`CUgraph`] (`CUgraphNode`).
///
/// Returned by every `cuGraphAdd*Node` call and used as a dependency
/// endpoint when wiring graph edges.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CUgraphNode(pub *mut c_void);

// SAFETY: see [`CUgraph`].
unsafe impl Send for CUgraphNode {}
unsafe impl Sync for CUgraphNode {}

impl CUgraphNode {
    /// Returns `true` if the handle is null (uninitialised).
    #[inline]
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
}

impl Default for CUgraphNode {
    #[inline]
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl std::fmt::Debug for CUgraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUgraphNode({:p})", self.0)
    }
}

// =========================================================================
// CUDA_KERNEL_NODE_PARAMS — kernel-launch node descriptor
// =========================================================================

/// Parameters for a kernel-launch graph node, consumed by
/// `cuGraphAddKernelNode`.
///
/// Mirrors `CUDA_KERNEL_NODE_PARAMS` (the pre-12.0 layout, which remains
/// ABI-stable and accepted by the driver). `kernel_params` points to an
/// array of pointers to the individual kernel arguments; `extra` is the
/// alternative `CU_LAUNCH_PARAM_*` packing mechanism and is normally null.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUDA_KERNEL_NODE_PARAMS {
    /// Kernel function to launch.
    pub func: CUfunction,
    /// Grid dimension X (number of blocks).
    pub grid_dim_x: u32,
    /// Grid dimension Y (number of blocks).
    pub grid_dim_y: u32,
    /// Grid dimension Z (number of blocks).
    pub grid_dim_z: u32,
    /// Block dimension X (threads per block).
    pub block_dim_x: u32,
    /// Block dimension Y (threads per block).
    pub block_dim_y: u32,
    /// Block dimension Z (threads per block).
    pub block_dim_z: u32,
    /// Dynamic shared-memory size in bytes.
    pub shared_mem_bytes: u32,
    /// Array of pointers to kernel arguments; null when `extra` is used.
    pub kernel_params: *mut *mut c_void,
    /// Alternative argument-packing buffer (`CU_LAUNCH_PARAM_*`); usually null.
    pub extra: *mut *mut c_void,
}

// SAFETY: the struct carries raw pointers to caller-owned argument buffers;
// the driver treats them as opaque. Mirroring the C struct, it is logically
// Send+Sync.
unsafe impl Send for CUDA_KERNEL_NODE_PARAMS {}
unsafe impl Sync for CUDA_KERNEL_NODE_PARAMS {}

impl Default for CUDA_KERNEL_NODE_PARAMS {
    fn default() -> Self {
        Self {
            func: CUfunction::default(),
            grid_dim_x: 0,
            grid_dim_y: 0,
            grid_dim_z: 0,
            block_dim_x: 0,
            block_dim_y: 0,
            block_dim_z: 0,
            shared_mem_bytes: 0,
            kernel_params: std::ptr::null_mut(),
            extra: std::ptr::null_mut(),
        }
    }
}

// =========================================================================
// CUDA_MEMCPY3D — descriptor for `cuGraphAddMemcpyNode` / `cuMemcpy3D`
// =========================================================================

/// Descriptor for a 3-D memory copy, consumed by `cuGraphAddMemcpyNode`
/// (and `cuMemcpy3D`).
///
/// Mirrors `CUDA_MEMCPY3D` in `cuda.h`. The driver inspects only the fields
/// appropriate for the chosen source / destination memory types; the rest
/// **must** be zeroed. Use [`CUDA_MEMCPY3D::default`] to obtain a
/// zero-initialised descriptor and set only the fields you need.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUDA_MEMCPY3D {
    /// Source X offset in bytes.
    pub src_x_in_bytes: usize,
    /// Source Y offset in rows.
    pub src_y: usize,
    /// Source Z offset in slices.
    pub src_z: usize,
    /// Source LOD (level of detail).
    pub src_lod: usize,
    /// Source memory type; see [`super::CUmemorytype`].
    pub src_memory_type: u32,
    /// Source host pointer (valid when `src_memory_type == Host`).
    pub src_host: *const c_void,
    /// Source device pointer (valid when `src_memory_type == Device`).
    pub src_device: CUdeviceptr,
    /// Source CUDA array (valid when `src_memory_type == Array`).
    pub src_array: CUarray,
    /// Reserved; must be null.
    pub reserved0: *mut c_void,
    /// Source pitch in bytes (`0` selects a tightly-packed layout).
    pub src_pitch: usize,
    /// Source height in rows (`0` selects a tightly-packed layout).
    pub src_height: usize,
    /// Destination X offset in bytes.
    pub dst_x_in_bytes: usize,
    /// Destination Y offset in rows.
    pub dst_y: usize,
    /// Destination Z offset in slices.
    pub dst_z: usize,
    /// Destination LOD (level of detail).
    pub dst_lod: usize,
    /// Destination memory type; see [`super::CUmemorytype`].
    pub dst_memory_type: u32,
    /// Destination host pointer (valid when `dst_memory_type == Host`).
    pub dst_host: *mut c_void,
    /// Destination device pointer (valid when `dst_memory_type == Device`).
    pub dst_device: CUdeviceptr,
    /// Destination CUDA array (valid when `dst_memory_type == Array`).
    pub dst_array: CUarray,
    /// Reserved; must be null.
    pub reserved1: *mut c_void,
    /// Destination pitch in bytes (`0` selects a tightly-packed layout).
    pub dst_pitch: usize,
    /// Destination height in rows (`0` selects a tightly-packed layout).
    pub dst_height: usize,
    /// Width of the copied region in bytes.
    pub width_in_bytes: usize,
    /// Height of the copied region in rows.
    pub height: usize,
    /// Depth of the copied region in slices.
    pub depth: usize,
}

// SAFETY: carries raw pointers / array handles to caller-owned memory; the
// driver treats them as opaque. Mirroring the C struct, it is Send+Sync.
unsafe impl Send for CUDA_MEMCPY3D {}
unsafe impl Sync for CUDA_MEMCPY3D {}

impl Default for CUDA_MEMCPY3D {
    fn default() -> Self {
        Self {
            src_x_in_bytes: 0,
            src_y: 0,
            src_z: 0,
            src_lod: 0,
            src_memory_type: 0,
            src_host: std::ptr::null(),
            src_device: 0,
            src_array: CUarray::default(),
            reserved0: std::ptr::null_mut(),
            src_pitch: 0,
            src_height: 0,
            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_z: 0,
            dst_lod: 0,
            dst_memory_type: 0,
            dst_host: std::ptr::null_mut(),
            dst_device: 0,
            dst_array: CUarray::default(),
            reserved1: std::ptr::null_mut(),
            dst_pitch: 0,
            dst_height: 0,
            width_in_bytes: 0,
            height: 0,
            depth: 0,
        }
    }
}

// =========================================================================
// CUDA_MEMSET_NODE_PARAMS — descriptor for `cuGraphAddMemsetNode`
// =========================================================================

/// Parameters for a memset graph node, consumed by `cuGraphAddMemsetNode`.
///
/// Mirrors `CUDA_MEMSET_NODE_PARAMS` in `cuda.h`. For a 1-D (linear) memset
/// set `height = 1` and `pitch = 0`; `element_size` is `1`, `2`, or `4`
/// bytes and `width` is the number of elements per row.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CUDA_MEMSET_NODE_PARAMS {
    /// Destination device pointer.
    pub dst: CUdeviceptr,
    /// Destination pitch in bytes (`0` for a tightly-packed 1-D memset).
    pub pitch: usize,
    /// Value to write, interpreted according to `element_size`.
    pub value: u32,
    /// Size of each element in bytes (`1`, `2`, or `4`).
    pub element_size: u32,
    /// Width of the region in elements.
    pub width: usize,
    /// Height of the region in rows (`1` for a 1-D memset).
    pub height: usize,
}

impl Default for CUDA_MEMSET_NODE_PARAMS {
    fn default() -> Self {
        Self {
            dst: 0,
            pitch: 0,
            value: 0,
            element_size: 1,
            width: 0,
            height: 1,
        }
    }
}

// =========================================================================
// CUDA_HOST_NODE_PARAMS — descriptor for `cuGraphAddHostNode`
// =========================================================================

/// Parameters for a host-callback graph node, consumed by
/// `cuGraphAddHostNode`.
///
/// Mirrors `CUDA_HOST_NODE_PARAMS` in `cuda.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUDA_HOST_NODE_PARAMS {
    /// Host function to execute, of type `void (*)(void *userData)`.
    pub fn_ptr: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
    /// Argument forwarded to `fn_ptr`.
    pub user_data: *mut c_void,
}

// SAFETY: the struct holds a function pointer and an opaque user-data
// pointer; both are caller-managed and the driver treats them as opaque.
unsafe impl Send for CUDA_HOST_NODE_PARAMS {}
unsafe impl Sync for CUDA_HOST_NODE_PARAMS {}

impl Default for CUDA_HOST_NODE_PARAMS {
    fn default() -> Self {
        Self {
            fn_ptr: None,
            user_data: std::ptr::null_mut(),
        }
    }
}

// =========================================================================
// CUmemPoolAttribute — `cuMemPoolSetAttribute` / `cuMemPoolGetAttribute`
// =========================================================================

/// Attribute discriminant for `cuMemPoolSetAttribute` /
/// `cuMemPoolGetAttribute`.
///
/// Mirrors `CUmemPoolAttribute` in `cuda.h`. The numeric values match the
/// CUDA header exactly so the enum can be passed straight to the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[non_exhaustive]
pub enum CUmemPoolAttribute {
    /// `(value type = int)` Allow reuse of memory still in use by an
    /// operation scheduled with an event dependency.
    ReuseFollowEventDependencies = 1,
    /// `(value type = int)` Allow reuse of completed frees with no explicit
    /// event dependency (opportunistic reuse).
    ReuseAllowOpportunistic = 2,
    /// `(value type = int)` Allow the driver to insert internal stream
    /// dependencies to enable reuse.
    ReuseAllowInternalDependencies = 3,
    /// `(value type = cuuint64_t)` Amount of reserved memory (bytes) to hold
    /// onto before trying to release memory back to the OS.
    ReleaseThreshold = 4,
    /// `(value type = cuuint64_t, read-only)` Amount of backing memory
    /// currently allocated for the pool.
    ReservedMemCurrent = 5,
    /// `(value type = cuuint64_t, read/write)` High-water mark of backing
    /// memory allocated for the pool since the last reset.
    ReservedMemHigh = 6,
    /// `(value type = cuuint64_t, read-only)` Amount of memory from the pool
    /// currently in use by the application.
    UsedMemCurrent = 7,
    /// `(value type = cuuint64_t, read/write)` High-water mark of memory in
    /// use from the pool since the last reset.
    UsedMemHigh = 8,
}

// =========================================================================
// CUgraphInstantiate_flags — flags for `cuGraphInstantiateWithFlags`
// =========================================================================

/// Instantiate a graph in auto-free-on-launch mode (a finished graph frees
/// its memory-allocation nodes before the next launch).
pub const CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH: u64 = 1;
/// Upload the graph during instantiation.
pub const CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD: u64 = 2;
/// Instantiate the graph for launch from the device.
pub const CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH: u64 = 4;
/// Run the graph using per-node priorities from the stream it is captured on.
pub const CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY: u64 = 8;

// =========================================================================
// CUgraphNodeType — node-type discriminant (informational)
// =========================================================================

/// Type of a graph node, as reported by `cuGraphNodeGetType`.
///
/// Mirrors `CUgraphNodeType` in `cuda.h`. Provided for completeness and
/// node-type queries; `cuGraphAdd*Node` calls do not require it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[non_exhaustive]
pub enum CUgraphNodeType {
    /// GPU kernel-launch node.
    Kernel = 0,
    /// Memory-copy node.
    Memcpy = 1,
    /// Memory-set node.
    Memset = 2,
    /// Host (CPU) callback node.
    Host = 3,
    /// Node that executes an embedded child graph.
    Graph = 4,
    /// Empty (no-op) node used as a synchronisation barrier.
    Empty = 5,
}

// =========================================================================
// CU_LAUNCH_PARAM_* sentinels (for the `extra` kernel-arg packing buffer)
// =========================================================================

/// Terminator for the `extra` kernel-argument buffer.
pub const CU_LAUNCH_PARAM_END: *mut c_void = std::ptr::null_mut();

// `c_char` is referenced by FFI signatures that consume node names; keep the
// import meaningful so the module stays warning-free.
#[allow(dead_code)]
type GraphNodeName = *const c_char;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_handles_default_to_null() {
        assert!(CUgraph::default().is_null());
        assert!(CUgraphExec::default().is_null());
        assert!(CUgraphNode::default().is_null());
    }

    #[test]
    fn kernel_node_params_default_is_zeroed() {
        let p = CUDA_KERNEL_NODE_PARAMS::default();
        assert!(p.func.is_null());
        assert_eq!(p.grid_dim_x, 0);
        assert_eq!(p.block_dim_x, 0);
        assert_eq!(p.shared_mem_bytes, 0);
        assert!(p.kernel_params.is_null());
        assert!(p.extra.is_null());
    }

    #[test]
    fn memset_node_params_default_is_linear() {
        let p = CUDA_MEMSET_NODE_PARAMS::default();
        assert_eq!(p.dst, 0);
        assert_eq!(p.pitch, 0);
        assert_eq!(p.element_size, 1);
        assert_eq!(p.height, 1);
    }

    #[test]
    fn memcpy3d_default_is_zeroed() {
        let m = CUDA_MEMCPY3D::default();
        assert_eq!(m.src_memory_type, 0);
        assert_eq!(m.dst_memory_type, 0);
        assert_eq!(m.width_in_bytes, 0);
        assert_eq!(m.depth, 0);
        assert!(m.src_host.is_null());
        assert!(m.reserved0.is_null());
        assert!(m.reserved1.is_null());
    }

    #[test]
    fn host_node_params_default_is_empty() {
        let p = CUDA_HOST_NODE_PARAMS::default();
        assert!(p.fn_ptr.is_none());
        assert!(p.user_data.is_null());
    }

    #[test]
    fn mem_pool_attribute_discriminants_match_cuda() {
        assert_eq!(CUmemPoolAttribute::ReuseFollowEventDependencies as u32, 1);
        assert_eq!(CUmemPoolAttribute::ReleaseThreshold as u32, 4);
        assert_eq!(CUmemPoolAttribute::UsedMemHigh as u32, 8);
    }

    #[test]
    fn graph_node_type_discriminants_match_cuda() {
        assert_eq!(CUgraphNodeType::Kernel as u32, 0);
        assert_eq!(CUgraphNodeType::Empty as u32, 5);
    }
}
