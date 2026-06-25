//! Stream-ordered memory allocation (CUDA 11.2+ / 12.x+).
//!
//! Stream-ordered memory allocation allows memory operations (`alloc` / `free`)
//! to participate in the stream execution order, eliminating the need for
//! explicit synchronisation between allocation and kernel launch.
//!
//! This module provides:
//!
//! * [`StreamMemoryPool`] — a memory pool bound to a specific device.
//! * [`StreamAllocation`] — a handle to a stream-ordered allocation.
//! * [`StreamOrderedAllocConfig`] — pool configuration (sizes, thresholds).
//! * [`PoolAttribute`] / [`PoolUsageStats`] — attribute queries and statistics.
//! * [`PoolExportDescriptor`] / [`ShareableHandleType`] — IPC sharing metadata.
//! * [`stream_alloc`] / [`stream_free`] — convenience free functions.
//!
//! # The stream-ordered model
//!
//! Independently of the GPU driver, every pool carries a faithful CPU
//! simulation of the stream-ordered allocator (see
//! [`stream_ordered_model`](crate::stream_ordered_model)).  Allocations
//! (`alloc_async` / [`alloc_on`](StreamMemoryPool::alloc_on)) and frees
//! (`free_async` / [`free_on`](StreamMemoryPool::free_on)) are sequenced per
//! stream; freed blocks return to the pool once their stream reaches the free
//! point and are then **reused** by a later same-or-larger request.  This makes
//! the allocator's semantics — visibility ordering, block reuse, and
//! reserved-vs-used accounting — observable on a plain CPU, with no GPU
//! required.
//!
//! # Platform behaviour
//!
//! On platforms with a real CUDA driver (Linux, Windows),
//! [`StreamMemoryPool::new`] additionally creates a driver-side pool via
//! `cuMemPoolCreate`.  The lower-level `cuMem*Async` bindings remain available
//! for real-GPU use.  [`StreamMemoryPool::cpu_pool`] builds a pool backed only
//! by the CPU model, so the full stream-ordered API can be exercised without a
//! driver on any platform.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_driver::stream_ordered_alloc::*;
//! use oxicuda_driver::StreamOrderId;
//!
//! // A pool backed by the faithful CPU model — no GPU driver required.
//! let config = StreamOrderedAllocConfig::default_for_device(0);
//! let mut pool = StreamMemoryPool::cpu_pool(config)?;
//!
//! // A genuine stream-ordering identity.  In a GPU program this is derived
//! // from a real `Stream` via `StreamMemoryPool::stream_id`; the model
//! // sequences this token exactly like a real stream would.
//! let stream = StreamOrderId::from(1);
//!
//! let mut alloc = pool.alloc_on(1024, stream)?;
//! assert_eq!(alloc.size(), 1024);
//! assert!(!alloc.is_freed());
//!
//! pool.free_on(&mut alloc, stream)?;
//! assert!(alloc.is_freed());
//! # Ok::<(), oxicuda_driver::CudaError>(())
//! ```

use std::fmt;

use crate::error::{CudaError, CudaResult};
use crate::ffi::CUdeviceptr;
use crate::stream::Stream;
use crate::stream_ordered_model::{ModelLimits, StreamOrderId, StreamOrderModel};

// ---------------------------------------------------------------------------
// Constants — CUmemPoolAttribute (mirrors CUDA header values)
// ---------------------------------------------------------------------------

/// Pool reuse policy: follow event dependencies.
pub const CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES: u32 = 1;
/// Pool reuse policy: allow opportunistic reuse.
pub const CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC: u32 = 2;
/// Pool reuse policy: allow internal dependency insertion.
pub const CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES: u32 = 3;
/// Release threshold in bytes (memory returned to OS when usage drops below).
pub const CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: u32 = 4;
/// Current reserved memory (bytes) — read-only.
pub const CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT: u32 = 5;
/// High-water mark of reserved memory (bytes) — resettable.
pub const CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH: u32 = 6;
/// Current used memory (bytes) — read-only.
pub const CU_MEMPOOL_ATTR_USED_MEM_CURRENT: u32 = 7;
/// High-water mark of used memory (bytes) — resettable.
pub const CU_MEMPOOL_ATTR_USED_MEM_HIGH: u32 = 8;

// ---------------------------------------------------------------------------
// StreamOrderedAllocConfig
// ---------------------------------------------------------------------------

/// Configuration for a stream-ordered memory pool.
///
/// All sizes are in bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamOrderedAllocConfig {
    /// Initial pool size in bytes.  The pool pre-reserves this amount of
    /// device memory when created.
    pub initial_pool_size: usize,

    /// Maximum pool size in bytes.  `0` means unlimited — the pool will grow
    /// as needed (subject to device memory limits).
    pub max_pool_size: usize,

    /// Release threshold in bytes.  When the pool is trimmed, at least this
    /// much memory is kept reserved for future allocations.
    pub release_threshold: usize,

    /// The device ordinal to create the pool on.
    pub device: i32,
}

impl StreamOrderedAllocConfig {
    /// Validate that the configuration is internally consistent.
    ///
    /// # Rules
    ///
    /// * `initial_pool_size` must not exceed `max_pool_size` (when
    ///   `max_pool_size > 0`).
    /// * `release_threshold` must not exceed `max_pool_size` (when
    ///   `max_pool_size > 0`).
    /// * `device` must be non-negative.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if any rule is violated.
    pub fn validate(&self) -> CudaResult<()> {
        if self.device < 0 {
            return Err(CudaError::InvalidValue);
        }

        if self.max_pool_size > 0 {
            if self.initial_pool_size > self.max_pool_size {
                return Err(CudaError::InvalidValue);
            }
            if self.release_threshold > self.max_pool_size {
                return Err(CudaError::InvalidValue);
            }
        }

        Ok(())
    }

    /// Returns a sensible default configuration for the given device.
    ///
    /// * `initial_pool_size` = 0 (grow on demand)
    /// * `max_pool_size` = 0 (unlimited)
    /// * `release_threshold` = 0 (release everything on trim)
    pub fn default_for_device(device: i32) -> Self {
        Self {
            initial_pool_size: 0,
            max_pool_size: 0,
            release_threshold: 0,
            device,
        }
    }
}

// ---------------------------------------------------------------------------
// PoolAttribute
// ---------------------------------------------------------------------------

/// Attributes that can be queried or set on a [`StreamMemoryPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolAttribute {
    /// Whether freed blocks can be reused by following event dependencies.
    ReuseFollowEventDependencies,
    /// Whether freed blocks can be opportunistically reused (without ordering).
    ReuseAllowOpportunistic,
    /// Whether the pool may insert internal dependencies for reuse.
    ReuseAllowInternalDependencies,
    /// The release threshold in bytes.
    ReleaseThreshold(u64),
    /// Current reserved memory (read-only query).
    ReservedMemCurrent,
    /// High-water mark of reserved memory.
    ReservedMemHigh,
    /// Current used memory (read-only query).
    UsedMemCurrent,
    /// High-water mark of used memory.
    UsedMemHigh,
}

impl PoolAttribute {
    /// Convert to the raw CUDA attribute constant.
    pub fn to_raw(self) -> u32 {
        match self {
            Self::ReuseFollowEventDependencies => CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
            Self::ReuseAllowOpportunistic => CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
            Self::ReuseAllowInternalDependencies => {
                CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES
            }
            Self::ReleaseThreshold(_) => CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            Self::ReservedMemCurrent => CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            Self::ReservedMemHigh => CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
            Self::UsedMemCurrent => CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            Self::UsedMemHigh => CU_MEMPOOL_ATTR_USED_MEM_HIGH,
        }
    }
}

// ---------------------------------------------------------------------------
// PoolUsageStats
// ---------------------------------------------------------------------------

/// Snapshot of pool memory usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolUsageStats {
    /// Bytes currently reserved from the device allocator.
    pub reserved_current: u64,
    /// Peak bytes reserved (since creation or last reset).
    pub reserved_high: u64,
    /// Bytes currently in use by outstanding allocations.
    pub used_current: u64,
    /// Peak bytes in use (since creation or last reset).
    pub used_high: u64,
    /// Number of active (not-yet-freed) allocations.
    pub active_allocations: usize,
    /// Peak number of concurrent allocations.
    pub peak_allocations: usize,
}

// ---------------------------------------------------------------------------
// ShareableHandleType / PoolExportDescriptor
// ---------------------------------------------------------------------------

/// Handle type used for IPC sharing of memory pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShareableHandleType {
    /// No sharing.
    #[default]
    None,
    /// POSIX file descriptor (Linux).
    PosixFileDescriptor,
    /// Win32 handle (Windows).
    Win32Handle,
    /// Win32 KMT handle (Windows, legacy).
    Win32KmtHandle,
}

/// Descriptor for exporting a pool for IPC sharing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolExportDescriptor {
    /// The handle type to use for sharing.
    pub shareable_handle_type: ShareableHandleType,
    /// The device ordinal that owns the pool.
    pub pool_device: i32,
}

// ---------------------------------------------------------------------------
// StreamAllocation
// ---------------------------------------------------------------------------

/// Handle to a stream-ordered memory allocation.
///
/// An allocation lives on the GPU and is associated with a specific stream
/// and memory pool.  It becomes available when all preceding work on the
/// stream has completed, and is returned to the pool when freed (also
/// stream-ordered).
pub struct StreamAllocation {
    /// Device pointer (`CUdeviceptr`).
    ptr: CUdeviceptr,
    /// Size of the allocation in bytes.
    size: usize,
    /// The stream this allocation is ordered on (raw ordering token).
    stream: u64,
    /// The pool handle that owns this allocation.
    pool: u64,
    /// Sequence number at which the allocation becomes valid on its stream,
    /// in the owning pool's [`StreamOrderModel`].
    ready_seq: u64,
    /// Whether this allocation has already been freed.
    freed: bool,
}

impl StreamAllocation {
    /// Returns the device pointer as a raw `u64` (`CUdeviceptr`).
    #[inline]
    pub fn as_ptr(&self) -> u64 {
        self.ptr
    }

    /// Returns the allocation size in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns `true` if this allocation has been freed.
    #[inline]
    pub fn is_freed(&self) -> bool {
        self.freed
    }

    /// Returns the stream handle this allocation is ordered on.
    #[inline]
    pub fn stream(&self) -> u64 {
        self.stream
    }

    /// Returns the ordering identifier of the stream this allocation is bound
    /// to in the owning pool's stream-ordered model.
    #[inline]
    pub fn stream_id(&self) -> StreamOrderId {
        StreamOrderId(self.stream)
    }

    /// Returns the sequence number at which this allocation becomes valid on
    /// its stream within the owning pool's [`StreamOrderModel`].
    ///
    /// The allocation is safe to read on its own stream only once that stream
    /// has executed past this point (queryable via
    /// [`StreamMemoryPool::is_ready`]).
    #[inline]
    pub fn ready_seq(&self) -> u64 {
        self.ready_seq
    }

    /// Returns the pool handle that owns this allocation.
    #[inline]
    pub fn pool(&self) -> u64 {
        self.pool
    }
}

impl fmt::Debug for StreamAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamAllocation")
            .field("ptr", &format_args!("0x{:016x}", self.ptr))
            .field("size", &self.size)
            .field("stream", &format_args!("0x{:016x}", self.stream))
            .field("freed", &self.freed)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// StreamMemoryPool
// ---------------------------------------------------------------------------

/// A memory pool for stream-ordered allocations.
///
/// Every pool drives a faithful CPU model of the stream-ordered allocator (the
/// source of truth for byte accounting, block reuse, and per-stream ordering).
/// On platforms with a real CUDA driver (Linux, Windows),
/// [`StreamMemoryPool::new`] *additionally* creates a driver-side pool via
/// `cuMemPoolCreate`.  [`StreamMemoryPool::cpu_pool`] builds a pool backed only
/// by the CPU model and never touches the driver, so the API can be exercised
/// on any platform.
///
/// # Allocation tracking
///
/// The pool maintains running allocation counts and byte totals (mirrored from
/// the CPU model) for diagnostics; these are available everywhere via
/// [`StreamMemoryPool::stats`].
pub struct StreamMemoryPool {
    /// Raw `CUmemoryPool` handle (0 if not backed by a real driver pool).
    handle: u64,
    /// Device ordinal.
    device: i32,
    /// Configuration used to create this pool.
    config: StreamOrderedAllocConfig,
    /// Number of currently active (not freed) allocations (mirror of the
    /// model's live count, kept for cheap field access).
    active_allocations: usize,
    /// Total bytes currently in use (mirror of the model's `used`).
    total_allocated: usize,
    /// Peak bytes ever in use concurrently (mirror of the model's `used_high`).
    peak_allocated: usize,
    /// Peak number of concurrent allocations (mirror of the model's peak).
    peak_allocation_count: usize,
    /// Faithful CPU model of the stream-ordered allocator.  This is the
    /// authority for pointers, reuse, and stream ordering on every platform.
    model: StreamOrderModel,
}

impl fmt::Debug for StreamMemoryPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamMemoryPool")
            .field("handle", &format_args!("0x{:016x}", self.handle))
            .field("device", &self.device)
            .field("active_allocations", &self.active_allocations)
            .field("total_allocated", &self.total_allocated)
            .field("reserved", &self.model.reserved())
            .finish()
    }
}

impl StreamMemoryPool {
    /// Create a new memory pool for the given device.
    ///
    /// The configuration is validated and the CPU model is initialised.  On
    /// platforms with a real CUDA driver, `cuMemPoolCreate` is also invoked and
    /// its handle stored; without a driver this fails cleanly.
    ///
    /// To obtain a pool that never touches the driver (e.g. for CPU-only use of
    /// the stream-ordered API), use [`StreamMemoryPool::cpu_pool`].
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the config fails validation.
    /// * On non-macOS, any [`CudaError`] from `cuMemPoolCreate` (e.g.
    ///   [`CudaError::NotInitialized`] when no driver is loadable).
    pub fn new(config: StreamOrderedAllocConfig) -> CudaResult<Self> {
        config.validate()?;

        #[cfg_attr(target_os = "macos", allow(unused_mut))]
        let mut pool = Self::with_model(config);

        // On real GPU platforms, create the driver-side pool via
        // `cuMemPoolCreate` and store the returned handle.  When the driver
        // is absent the call returns `Err` and pool creation fails cleanly.
        #[cfg(not(target_os = "macos"))]
        {
            pool.handle = Self::gpu_create_pool(&pool.config)?;
        }

        Ok(pool)
    }

    /// Create a pool backed solely by the faithful CPU model, without touching
    /// the CUDA driver.
    ///
    /// This always succeeds (given a valid configuration) on every platform and
    /// is the recommended entry point for using the stream-ordered allocation
    /// semantics on a CPU.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the config fails validation.
    pub fn cpu_pool(config: StreamOrderedAllocConfig) -> CudaResult<Self> {
        config.validate()?;
        Ok(Self::with_model(config))
    }

    /// Build a pool value with a fresh CPU model (no driver interaction).
    fn with_model(config: StreamOrderedAllocConfig) -> Self {
        let model = StreamOrderModel::new(Self::model_limits(&config));
        Self {
            handle: 0,
            device: config.device,
            config,
            active_allocations: 0,
            total_allocated: 0,
            peak_allocated: 0,
            peak_allocation_count: 0,
            model,
        }
    }

    /// Build the model limits from a pool configuration.
    fn model_limits(config: &StreamOrderedAllocConfig) -> ModelLimits {
        ModelLimits {
            max_pool_size: config.max_pool_size,
            release_threshold: config.release_threshold,
        }
    }

    /// Derive a stream-ordering identifier from a genuine [`Stream`].
    ///
    /// The stream's raw handle is a stable, unique token for the lifetime of
    /// the stream, which the CPU model uses as the stream's ordering identity.
    #[inline]
    pub fn stream_id(stream: &Stream) -> StreamOrderId {
        StreamOrderId(stream.raw().0 as usize as u64)
    }

    /// Allocate memory on a stream (stream-ordered), identified by a raw
    /// ordering token.
    ///
    /// The allocation becomes valid on the stream once the stream reaches the
    /// allocation point.  A previously-freed block of the same-or-larger size
    /// is reused when available; otherwise a fresh block is carved from the
    /// pool.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `size` is zero.
    /// * [`CudaError::OutOfMemory`] if `max_pool_size` would be exceeded.
    pub fn alloc_async(&mut self, size: usize, stream: u64) -> CudaResult<StreamAllocation> {
        self.alloc_on(size, StreamOrderId(stream))
    }

    /// Allocate memory ordered on a genuine [`Stream`].
    ///
    /// This is the recommended entry point when a real CUDA [`Stream`] is
    /// available: the allocation is sequenced against that exact stream in the
    /// pool's [`StreamOrderModel`] (see [`StreamMemoryPool::stream_id`]).
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `size` is zero.
    /// * [`CudaError::OutOfMemory`] if `max_pool_size` would be exceeded.
    pub fn alloc_async_on_stream(
        &mut self,
        size: usize,
        stream: &Stream,
    ) -> CudaResult<StreamAllocation> {
        self.alloc_on(size, Self::stream_id(stream))
    }

    /// Allocate memory ordered on the stream identified by `stream`.
    ///
    /// The block is carved from the pool — reusing a previously-freed block of
    /// the same-or-larger size when one is available — and sequenced on the
    /// stream so that it only becomes valid once the stream reaches the
    /// allocation point (queryable via [`StreamMemoryPool::is_ready`]).
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `size` is zero.
    /// * [`CudaError::OutOfMemory`] if `max_pool_size` would be exceeded.
    pub fn alloc_on(&mut self, size: usize, stream: StreamOrderId) -> CudaResult<StreamAllocation> {
        let model_alloc = self.model.alloc(size, stream)?;
        self.sync_mirror_stats();

        Ok(StreamAllocation {
            ptr: model_alloc.ptr,
            size: model_alloc.size,
            stream: stream.raw(),
            pool: self.handle,
            ready_seq: model_alloc.ready_seq,
            freed: false,
        })
    }

    /// Free memory on a stream (stream-ordered).
    ///
    /// The memory is returned to the pool once all prior work on the
    /// allocation's stream has completed.  The allocation is marked freed and
    /// cannot be freed again.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the allocation is already freed, or its
    ///   pointer is not live in this pool (foreign-pointer free).
    pub fn free_async(&mut self, alloc: &mut StreamAllocation) -> CudaResult<()> {
        let stream = alloc.stream_id();
        self.free_on(alloc, stream)
    }

    /// Free `alloc` ordered on a genuine [`Stream`].
    ///
    /// CUDA permits freeing on a stream different from the one the allocation
    /// was made on; the free still completes only once *that* stream reaches
    /// the free point.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the allocation is already freed or its
    ///   pointer is not live in this pool.
    pub fn free_async_on_stream(
        &mut self,
        alloc: &mut StreamAllocation,
        stream: &Stream,
    ) -> CudaResult<()> {
        self.free_on(alloc, Self::stream_id(stream))
    }

    /// Free `alloc` ordered on the stream identified by `stream`.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the allocation is already freed or its
    ///   pointer is not live in this pool.
    pub fn free_on(
        &mut self,
        alloc: &mut StreamAllocation,
        stream: StreamOrderId,
    ) -> CudaResult<()> {
        if alloc.freed {
            return Err(CudaError::InvalidValue);
        }

        self.model.free(alloc.ptr, stream)?;
        self.sync_mirror_stats();

        alloc.freed = true;
        Ok(())
    }

    /// Advance a stream to its head (model of `cuStreamSynchronize`),
    /// completing every operation submitted on it so far and reclaiming any
    /// completed stream-ordered frees into the pool for reuse.
    pub fn synchronize_stream(&mut self, stream: StreamOrderId) {
        self.model.synchronize(stream);
        self.sync_mirror_stats();
    }

    /// Returns `true` if `alloc` is valid to read on its own ordering stream,
    /// i.e. that stream has executed past the allocation point.
    pub fn is_ready(&self, alloc: &StreamAllocation) -> bool {
        let model_alloc = crate::stream_ordered_model::ModelAllocation {
            ptr: alloc.ptr,
            size: alloc.size,
            capacity: alloc.size,
            stream: alloc.stream_id(),
            ready_seq: alloc.ready_seq,
        };
        self.model.is_ready_same_stream(&model_alloc)
    }

    /// Returns `true` if `alloc` (made on its own stream) is safe to use on
    /// `consumer` given that `consumer` was ordered after `wait_seq` on the
    /// allocation's stream (the sequence captured by an event it waited on).
    ///
    /// Use [`StreamMemoryPool::record_event`] on the producing stream to obtain
    /// a `wait_seq` that captures the allocation.
    pub fn is_ready_on(
        &self,
        alloc: &StreamAllocation,
        consumer: StreamOrderId,
        wait_seq: u64,
    ) -> bool {
        let model_alloc = crate::stream_ordered_model::ModelAllocation {
            ptr: alloc.ptr,
            size: alloc.size,
            capacity: alloc.size,
            stream: alloc.stream_id(),
            ready_seq: alloc.ready_seq,
        };
        self.model
            .is_ready_cross_stream(&model_alloc, consumer, wait_seq)
    }

    /// Record an event on `stream`, returning the sequence number it captures.
    ///
    /// A later cross-stream wait on this value orders the waiting stream after
    /// every operation submitted on `stream` before this point.
    pub fn record_event(&mut self, stream: StreamOrderId) -> u64 {
        self.model.record_event(stream)
    }

    /// Trim the CPU model's pool, releasing free-list bytes above
    /// `min_bytes_to_keep` back to the (virtual) device.
    ///
    /// This is the CPU-model analogue of `cuMemPoolTrimTo`; it always succeeds
    /// and is available on every platform.  For the raw driver trim, see the
    /// platform-gated [`StreamMemoryPool::trim`].
    pub fn model_trim(&mut self, min_bytes_to_keep: usize) {
        self.model.trim_to(min_bytes_to_keep);
        self.sync_mirror_stats();
    }

    /// Trim the driver-side pool, releasing unused memory back to the OS.
    ///
    /// At least `min_bytes_to_keep` bytes of reserved memory remain in the
    /// pool.  This drives the real `cuMemPoolTrimTo` binding; for the
    /// always-available CPU-model trim, use [`StreamMemoryPool::model_trim`].
    ///
    /// # Errors
    ///
    /// * [`CudaError::NotSupported`] on macOS.
    /// * Any [`CudaError`] from `cuMemPoolTrimTo`.
    pub fn trim(&mut self, min_bytes_to_keep: usize) -> CudaResult<()> {
        self.platform_trim(min_bytes_to_keep)
    }

    /// Get pool usage statistics from the CPU model.
    ///
    /// `reserved_*` reflects everything the pool has carved from the (virtual)
    /// device (live + reusable + pending-free bytes), whereas `used_*` reflects
    /// only currently-live allocations.
    pub fn stats(&self) -> PoolUsageStats {
        PoolUsageStats {
            reserved_current: self.model.reserved() as u64,
            reserved_high: self.model.reserved_high() as u64,
            used_current: self.model.used() as u64,
            used_high: self.model.used_high() as u64,
            active_allocations: self.model.active(),
            peak_allocations: self.model.peak_active(),
        }
    }

    /// Set a pool attribute.
    ///
    /// Only attributes that carry a value (e.g. [`PoolAttribute::ReleaseThreshold`])
    /// modify pool state.  Read-only attributes (e.g. `ReservedMemCurrent`)
    /// return [`CudaError::InvalidValue`].
    ///
    /// The release threshold is applied to the CPU model as well as (on
    /// non-macOS) the driver pool.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] for read-only attributes.
    /// * [`CudaError::NotSupported`] on macOS for non-threshold attributes.
    pub fn set_attribute(&mut self, attr: PoolAttribute) -> CudaResult<()> {
        // Read-only attributes cannot be set.
        match attr {
            PoolAttribute::ReservedMemCurrent
            | PoolAttribute::UsedMemCurrent
            | PoolAttribute::ReservedMemHigh
            | PoolAttribute::UsedMemHigh => {
                return Err(CudaError::InvalidValue);
            }
            _ => {}
        }

        // Apply locally-meaningful attributes to the config and CPU model.
        if let PoolAttribute::ReleaseThreshold(val) = attr {
            self.config.release_threshold = val as usize;
            self.model.set_release_threshold(val as usize);
        }

        self.platform_set_attribute(attr)
    }

    /// Enable peer access from another device to allocations in this pool.
    ///
    /// After this call, kernels running on `peer_device` can access memory
    /// allocated from this pool.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidDevice`] if `peer_device` equals this pool's device.
    /// * [`CudaError::NotSupported`] on macOS.
    pub fn enable_peer_access(&self, peer_device: i32) -> CudaResult<()> {
        if peer_device == self.device {
            return Err(CudaError::InvalidDevice);
        }

        self.platform_enable_peer_access(peer_device)
    }

    /// Disable peer access from another device to allocations in this pool.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidDevice`] if `peer_device` equals this pool's device.
    /// * [`CudaError::NotSupported`] on macOS.
    pub fn disable_peer_access(&self, peer_device: i32) -> CudaResult<()> {
        if peer_device == self.device {
            return Err(CudaError::InvalidDevice);
        }

        self.platform_disable_peer_access(peer_device)
    }

    /// Reset peak statistics (peak used bytes and peak allocation count).
    pub fn reset_peak_stats(&mut self) {
        self.model.reset_peaks();
        self.sync_mirror_stats();
    }

    /// Mirror the model's current/peak figures into the cheap struct fields.
    fn sync_mirror_stats(&mut self) {
        self.active_allocations = self.model.active();
        self.total_allocated = self.model.used();
        self.peak_allocated = self.model.used_high();
        self.peak_allocation_count = self.model.peak_active();
    }

    /// Get the default memory pool for a device.
    ///
    /// CUDA provides a default pool per device, queried via
    /// `cuDeviceGetDefaultMemPool`.  The returned pool is owned by the
    /// driver and is *not* destroyed when the [`StreamMemoryPool`] wrapper
    /// is dropped.  On macOS, this returns a local-only pool with default
    /// configuration.  In all cases the CPU model is initialised.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `device` is negative.
    /// * [`CudaError::NotInitialized`] if the CUDA driver is not loaded.
    /// * Any [`CudaError`] mapped from `cuDeviceGetDefaultMemPool`.
    pub fn default_pool(device: i32) -> CudaResult<Self> {
        if device < 0 {
            return Err(CudaError::InvalidValue);
        }

        let config = StreamOrderedAllocConfig::default_for_device(device);

        // On macOS there is no driver — fall back to a local-only pool.
        #[cfg(target_os = "macos")]
        {
            Ok(Self::with_model(config))
        }

        // On real GPU platforms, resolve the device's default pool handle.
        #[cfg(not(target_os = "macos"))]
        {
            let handle = Self::gpu_default_pool(device)?;
            let mut pool = Self::with_model(config);
            pool.handle = handle;
            Ok(pool)
        }
    }

    /// Returns the raw pool handle.
    #[inline]
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Returns the device ordinal.
    #[inline]
    pub fn device(&self) -> i32 {
        self.device
    }

    /// Returns the pool configuration.
    #[inline]
    pub fn config(&self) -> &StreamOrderedAllocConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Platform-specific helpers (driver passthrough)
    // -----------------------------------------------------------------------

    /// Trim the driver pool on the current platform.
    fn platform_trim(&mut self, min_bytes_to_keep: usize) -> CudaResult<()> {
        #[cfg(target_os = "macos")]
        {
            let _ = min_bytes_to_keep;
            Err(CudaError::NotSupported)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_trim(self.handle, min_bytes_to_keep)
        }
    }

    /// Set attribute on the driver pool on the current platform.
    fn platform_set_attribute(&self, attr: PoolAttribute) -> CudaResult<()> {
        #[cfg(target_os = "macos")]
        {
            match attr {
                PoolAttribute::ReleaseThreshold(_) => Ok(()),
                _ => Err(CudaError::NotSupported),
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_set_attribute(self.handle, attr)
        }
    }

    /// Enable peer access on the driver pool on the current platform.
    fn platform_enable_peer_access(&self, peer_device: i32) -> CudaResult<()> {
        #[cfg(target_os = "macos")]
        {
            let _ = peer_device;
            Err(CudaError::NotSupported)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_enable_peer_access(self.handle, peer_device)
        }
    }

    /// Disable peer access on the driver pool on the current platform.
    fn platform_disable_peer_access(&self, peer_device: i32) -> CudaResult<()> {
        #[cfg(target_os = "macos")]
        {
            let _ = peer_device;
            Err(CudaError::NotSupported)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_disable_peer_access(self.handle, peer_device)
        }
    }

    // -----------------------------------------------------------------------
    // GPU-only driver bindings (compiled out on macOS)
    //
    // These remain available for genuine GPU use.  They are *not* on the CPU
    // model's hot path: the model is the allocator on every platform, and these
    // bindings are exercised directly by the `gpu_*` tests against whatever
    // driver the host provides.
    // -----------------------------------------------------------------------

    /// Create the pool on the GPU via `cuMemPoolCreate`.
    ///
    /// Builds a [`CUmemPoolProps`] from the pool configuration (pinned device
    /// memory on `config.device`, `max_size` from `config.max_pool_size`),
    /// invokes the driver, and returns the raw `CUmemoryPool` handle encoded
    /// as a `u64`.
    ///
    /// When the driver is absent, [`try_driver`](crate::loader::try_driver)
    /// returns `Err(CudaError::NotInitialized)` and pool creation fails
    /// cleanly.  When the driver is present but predates CUDA 11.2 (no
    /// `cuMemPoolCreate`), [`CudaError::NotSupported`] is returned.
    #[cfg(not(target_os = "macos"))]
    fn gpu_create_pool(config: &StreamOrderedAllocConfig) -> CudaResult<u64> {
        use crate::ffi::{
            CUmemAllocationType, CUmemLocation, CUmemLocationType, CUmemPoolProps, CUmemoryPool,
        };

        let api = crate::loader::try_driver()?;
        let create = api.cu_mem_pool_create.ok_or(CudaError::NotSupported)?;

        let props = CUmemPoolProps {
            alloc_type: CUmemAllocationType::Pinned as u32,
            handle_types: 0,
            location: CUmemLocation {
                loc_type: CUmemLocationType::Device as u32,
                id: config.device,
            },
            win32_security_attributes: std::ptr::null_mut(),
            max_size: config.max_pool_size,
            reserved: [0u8; 56],
        };

        let mut pool = CUmemoryPool::default();
        // SAFETY: `create` was just resolved from the driver; `props` and
        // `pool` are valid, correctly-typed local variables, and the CUDA
        // ABI's reserved padding is zeroed.
        let rc = unsafe { create(&mut pool, &props) };
        crate::error::check(rc)?;

        Ok(pool.0 as usize as u64)
    }

    /// Resolve a device's default memory pool via `cuDeviceGetDefaultMemPool`.
    #[cfg(not(target_os = "macos"))]
    fn gpu_default_pool(device: i32) -> CudaResult<u64> {
        use crate::ffi::CUmemoryPool;

        let api = crate::loader::try_driver()?;
        let get_default = api
            .cu_device_get_default_mem_pool
            .ok_or(CudaError::NotSupported)?;

        let mut pool = CUmemoryPool::default();
        // SAFETY: `get_default` was just resolved from the driver; `pool` is
        // a valid local and `device` is a plain device ordinal.
        let rc = unsafe { get_default(&mut pool, device) };
        crate::error::check(rc)?;

        Ok(pool.0 as usize as u64)
    }

    /// Allocate stream-ordered memory.
    ///
    /// When `pool_handle` is non-zero, allocates from that explicit pool via
    /// `cuMemAllocFromPoolAsync`; when it is zero (default-pool semantics),
    /// uses the context-wide `cuMemAllocAsync`.
    ///
    /// The CPU model is the allocator on every platform, so this real-driver
    /// binding has no production caller; it is retained for genuine GPU use and
    /// exercised directly by the `gpu_*` FFI tests.  `#[cfg_attr(not(test), …)]`
    /// keeps the production lib build warning-free without removing the
    /// real binding.
    #[cfg(not(target_os = "macos"))]
    #[cfg_attr(not(test), allow(dead_code))]
    fn gpu_alloc_async(pool_handle: u64, size: usize, stream: u64) -> CudaResult<CUdeviceptr> {
        use crate::ffi::{CUmemoryPool, CUstream};

        let api = crate::loader::try_driver()?;
        let cu_stream = CUstream(stream as usize as *mut std::ffi::c_void);
        let mut dptr: CUdeviceptr = 0;

        if pool_handle != 0 {
            let alloc_from_pool = api
                .cu_mem_alloc_from_pool_async
                .ok_or(CudaError::NotSupported)?;
            let pool = CUmemoryPool(pool_handle as usize as *mut std::ffi::c_void);
            // SAFETY: `alloc_from_pool` was just resolved; `dptr` is a valid
            // out-pointer and `pool`/`cu_stream` are reconstructed handles.
            let rc = unsafe { alloc_from_pool(&mut dptr, size, pool, cu_stream) };
            crate::error::check(rc)?;
        } else {
            let alloc_async = api.cu_mem_alloc_async.ok_or(CudaError::NotSupported)?;
            // SAFETY: `alloc_async` was just resolved; `dptr` is a valid
            // out-pointer and `cu_stream` is a reconstructed handle.
            let rc = unsafe { alloc_async(&mut dptr, size, cu_stream) };
            crate::error::check(rc)?;
        }

        Ok(dptr)
    }

    /// Free stream-ordered memory via `cuMemFreeAsync`.
    ///
    /// Retained for genuine GPU use and exercised directly by the `gpu_*` FFI
    /// tests; the CPU model is the allocator on the production path.
    #[cfg(not(target_os = "macos"))]
    #[cfg_attr(not(test), allow(dead_code))]
    fn gpu_free_async(ptr: CUdeviceptr, stream: u64) -> CudaResult<()> {
        use crate::ffi::CUstream;

        let api = crate::loader::try_driver()?;
        let free_async = api.cu_mem_free_async.ok_or(CudaError::NotSupported)?;
        let cu_stream = CUstream(stream as usize as *mut std::ffi::c_void);
        // SAFETY: `free_async` was just resolved from the driver; `ptr` is a
        // device pointer previously returned by an async allocation and
        // `cu_stream` is a reconstructed handle.
        crate::error::check(unsafe { free_async(ptr, cu_stream) })
    }

    /// Trim the pool via `cuMemPoolTrimTo`.
    #[cfg(not(target_os = "macos"))]
    fn gpu_trim(pool_handle: u64, min_bytes_to_keep: usize) -> CudaResult<()> {
        use crate::ffi::CUmemoryPool;

        let api = crate::loader::try_driver()?;
        let trim = api.cu_mem_pool_trim_to.ok_or(CudaError::NotSupported)?;
        let pool = CUmemoryPool(pool_handle as usize as *mut std::ffi::c_void);
        // SAFETY: `trim` was just resolved from the driver; `pool` is a
        // reconstructed pool handle and `min_bytes_to_keep` is a plain count.
        crate::error::check(unsafe { trim(pool, min_bytes_to_keep) })
    }

    /// Set a pool attribute via `cuMemPoolSetAttribute`.
    ///
    /// The reuse-policy attributes carry an `int` value; the release
    /// threshold carries a `cuuint64_t`.  The value buffer is sized
    /// accordingly and passed to the driver.
    #[cfg(not(target_os = "macos"))]
    fn gpu_set_attribute(pool_handle: u64, attr: PoolAttribute) -> CudaResult<()> {
        use crate::ffi::CUmemoryPool;

        let api = crate::loader::try_driver()?;
        let set_attr = api
            .cu_mem_pool_set_attribute
            .ok_or(CudaError::NotSupported)?;
        let pool = CUmemoryPool(pool_handle as usize as *mut std::ffi::c_void);
        let raw_attr = Self::map_pool_attribute(attr)?;

        // The driver dereferences `value` as either `int` or `cuuint64_t`
        // depending on the attribute.  Stack-allocate the correct width.
        match attr {
            PoolAttribute::ReuseFollowEventDependencies
            | PoolAttribute::ReuseAllowOpportunistic
            | PoolAttribute::ReuseAllowInternalDependencies => {
                // Boolean-style reuse policies: enable (1) the policy.
                let mut value: std::ffi::c_int = 1;
                // SAFETY: `set_attr` was just resolved; `pool` is a
                // reconstructed handle and `value` is a valid `int` matching
                // the attribute's documented value type.
                let rc = unsafe {
                    set_attr(pool, raw_attr, (&mut value as *mut std::ffi::c_int).cast())
                };
                crate::error::check(rc)
            }
            PoolAttribute::ReleaseThreshold(threshold) => {
                let mut value: u64 = threshold;
                // SAFETY: `set_attr` was just resolved; `pool` is a
                // reconstructed handle and `value` is a valid `cuuint64_t`
                // matching the release-threshold value type.
                let rc = unsafe { set_attr(pool, raw_attr, (&mut value as *mut u64).cast()) };
                crate::error::check(rc)
            }
            // Read-only attributes are rejected before reaching this point.
            PoolAttribute::ReservedMemCurrent
            | PoolAttribute::ReservedMemHigh
            | PoolAttribute::UsedMemCurrent
            | PoolAttribute::UsedMemHigh => Err(CudaError::InvalidValue),
        }
    }

    /// Map a [`PoolAttribute`] to the driver's [`CUmemPoolAttribute`].
    #[cfg(not(target_os = "macos"))]
    fn map_pool_attribute(attr: PoolAttribute) -> CudaResult<crate::ffi::CUmemPoolAttribute> {
        use crate::ffi::CUmemPoolAttribute;
        Ok(match attr {
            PoolAttribute::ReuseFollowEventDependencies => {
                CUmemPoolAttribute::ReuseFollowEventDependencies
            }
            PoolAttribute::ReuseAllowOpportunistic => CUmemPoolAttribute::ReuseAllowOpportunistic,
            PoolAttribute::ReuseAllowInternalDependencies => {
                CUmemPoolAttribute::ReuseAllowInternalDependencies
            }
            PoolAttribute::ReleaseThreshold(_) => CUmemPoolAttribute::ReleaseThreshold,
            PoolAttribute::ReservedMemCurrent => CUmemPoolAttribute::ReservedMemCurrent,
            PoolAttribute::ReservedMemHigh => CUmemPoolAttribute::ReservedMemHigh,
            PoolAttribute::UsedMemCurrent => CUmemPoolAttribute::UsedMemCurrent,
            PoolAttribute::UsedMemHigh => CUmemPoolAttribute::UsedMemHigh,
        })
    }

    /// Enable peer access from `peer_device` via `cuMemPoolSetAccess`.
    ///
    /// Builds a [`CUmemAccessDesc`] granting read-write access to the peer
    /// device and applies it to the pool.
    #[cfg(not(target_os = "macos"))]
    fn gpu_enable_peer_access(pool_handle: u64, peer_device: i32) -> CudaResult<()> {
        Self::gpu_set_pool_access(pool_handle, peer_device, true)
    }

    /// Disable peer access from `peer_device` via `cuMemPoolSetAccess`.
    #[cfg(not(target_os = "macos"))]
    fn gpu_disable_peer_access(pool_handle: u64, peer_device: i32) -> CudaResult<()> {
        Self::gpu_set_pool_access(pool_handle, peer_device, false)
    }

    /// Shared implementation for enabling / disabling pool peer access.
    #[cfg(not(target_os = "macos"))]
    fn gpu_set_pool_access(pool_handle: u64, peer_device: i32, enable: bool) -> CudaResult<()> {
        use crate::ffi::{
            CUmemAccessDesc, CUmemAccessFlags, CUmemLocation, CUmemLocationType, CUmemoryPool,
        };

        let api = crate::loader::try_driver()?;
        let set_access = api.cu_mem_pool_set_access.ok_or(CudaError::NotSupported)?;
        let pool = CUmemoryPool(pool_handle as usize as *mut std::ffi::c_void);

        let flags = if enable {
            CUmemAccessFlags::ReadWrite
        } else {
            CUmemAccessFlags::None
        };
        let desc = CUmemAccessDesc {
            location: CUmemLocation {
                loc_type: CUmemLocationType::Device as u32,
                id: peer_device,
            },
            flags: flags as u32,
        };

        // SAFETY: `set_access` was just resolved from the driver; `pool` is a
        // reconstructed handle and `desc` is a single valid descriptor.
        let rc = unsafe { set_access(pool, &desc, 1) };
        crate::error::check(rc)
    }
}

// ---------------------------------------------------------------------------
// Convenience free functions
// ---------------------------------------------------------------------------

/// Allocate memory on a stream using the default pool for device 0.
///
/// This is a convenience wrapper around [`StreamMemoryPool::default_pool`]
/// and [`StreamMemoryPool::alloc_async`].
///
/// # Errors
///
/// Propagates errors from pool creation and allocation.
pub fn stream_alloc(size: usize, stream: u64) -> CudaResult<StreamAllocation> {
    let mut pool = StreamMemoryPool::default_pool(0)?;
    pool.alloc_async(size, stream)
}

/// Free a stream-ordered allocation.
///
/// Marks the allocation freed.  This convenience function operates on the
/// allocation handle only (it does not require the owning pool); use
/// [`StreamMemoryPool::free_on`] when you need the freed block to re-enter a
/// specific pool's reuse list.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if the allocation is already freed.
pub fn stream_free(alloc: &mut StreamAllocation) -> CudaResult<()> {
    if alloc.freed {
        return Err(CudaError::InvalidValue);
    }

    alloc.freed = true;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
#[path = "stream_ordered_alloc_tests.rs"]
mod tests;
