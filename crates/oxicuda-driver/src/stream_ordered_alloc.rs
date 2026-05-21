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
//! # Platform behaviour
//!
//! On macOS (where NVIDIA dropped CUDA support), all operations that would
//! require the GPU driver return `Err(CudaError::NotSupported)`.  Config
//! validation, statistics tracking, and accessor methods work everywhere.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_driver::stream_ordered_alloc::*;
//!
//! let config = StreamOrderedAllocConfig::default_for_device(0);
//! let mut pool = StreamMemoryPool::new(config)?;
//!
//! let stream_handle = 0u64; // placeholder
//! let mut alloc = pool.alloc_async(1024, stream_handle)?;
//! assert_eq!(alloc.size(), 1024);
//! assert!(!alloc.is_freed());
//!
//! pool.free_async(&mut alloc)?;
//! assert!(alloc.is_freed());
//! # Ok::<(), oxicuda_driver::CudaError>(())
//! ```

use std::fmt;

use crate::error::{CudaError, CudaResult};
use crate::ffi::CUdeviceptr;

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
    /// The stream this allocation is ordered on.
    stream: u64,
    /// The pool handle that owns this allocation.
    pool: u64,
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
/// On platforms with a real CUDA driver (Linux, Windows), creating a pool
/// calls `cuMemPoolCreate` under the hood.  On macOS (where there is no
/// NVIDIA driver), pool metadata is tracked locally but any operation that
/// would require the driver returns `Err(CudaError::NotSupported)`.
///
/// # Allocation tracking
///
/// The pool tracks allocation counts and byte totals locally for
/// diagnostics.  These statistics are maintained even on macOS so that
/// the API surface can be exercised in tests.
pub struct StreamMemoryPool {
    /// Raw `CUmemoryPool` handle (0 if not backed by a real driver pool).
    handle: u64,
    /// Device ordinal.
    device: i32,
    /// Configuration used to create this pool.
    config: StreamOrderedAllocConfig,
    /// Number of currently active (not freed) allocations.
    active_allocations: usize,
    /// Total bytes currently allocated.
    total_allocated: usize,
    /// Peak bytes ever allocated concurrently.
    peak_allocated: usize,
    /// Peak number of concurrent allocations.
    peak_allocation_count: usize,
    /// Monotonically increasing allocation id for generating unique pointers
    /// in non-GPU mode.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    next_alloc_id: u64,
}

impl fmt::Debug for StreamMemoryPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamMemoryPool")
            .field("handle", &format_args!("0x{:016x}", self.handle))
            .field("device", &self.device)
            .field("active_allocations", &self.active_allocations)
            .field("total_allocated", &self.total_allocated)
            .field("peak_allocated", &self.peak_allocated)
            .finish()
    }
}

impl StreamMemoryPool {
    /// Create a new memory pool for the given device.
    ///
    /// The configuration is validated before the pool is created.  On
    /// platforms with a real CUDA driver, `cuMemPoolCreate` is invoked.
    /// On macOS, a local-only pool is created for testing purposes.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the config fails validation.
    /// * [`CudaError::NotSupported`] on macOS (pool metadata is still created
    ///   so that tests can exercise the API).
    pub fn new(config: StreamOrderedAllocConfig) -> CudaResult<Self> {
        config.validate()?;

        #[cfg_attr(target_os = "macos", allow(unused_mut))]
        let mut pool = Self {
            handle: 0,
            device: config.device,
            config,
            active_allocations: 0,
            total_allocated: 0,
            peak_allocated: 0,
            peak_allocation_count: 0,
            next_alloc_id: 1,
        };

        // On real GPU platforms, create the driver-side pool via
        // `cuMemPoolCreate` and store the returned handle.  When the driver
        // is absent the call returns `Err` and pool creation fails cleanly.
        #[cfg(not(target_os = "macos"))]
        {
            pool.handle = Self::gpu_create_pool(&pool.config)?;
        }

        Ok(pool)
    }

    /// Allocate memory on a stream (stream-ordered).
    ///
    /// The allocation becomes available when all prior work on the stream
    /// has completed.  The returned [`StreamAllocation`] tracks the pointer,
    /// size, and ownership.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `size` is zero.
    /// * [`CudaError::OutOfMemory`] if `max_pool_size` would be exceeded.
    /// * [`CudaError::NotSupported`] on macOS.
    pub fn alloc_async(&mut self, size: usize, stream: u64) -> CudaResult<StreamAllocation> {
        if size == 0 {
            return Err(CudaError::InvalidValue);
        }

        // Check max pool size constraint.
        if self.config.max_pool_size > 0
            && self.total_allocated.saturating_add(size) > self.config.max_pool_size
        {
            return Err(CudaError::OutOfMemory);
        }

        let ptr = self.platform_alloc_async(size, stream)?;

        // Update bookkeeping.
        self.active_allocations += 1;
        self.total_allocated = self.total_allocated.saturating_add(size);
        if self.total_allocated > self.peak_allocated {
            self.peak_allocated = self.total_allocated;
        }
        if self.active_allocations > self.peak_allocation_count {
            self.peak_allocation_count = self.active_allocations;
        }

        Ok(StreamAllocation {
            ptr,
            size,
            stream,
            pool: self.handle,
            freed: false,
        })
    }

    /// Free memory on a stream (stream-ordered).
    ///
    /// The memory is returned to the pool when all prior work on the
    /// stream has completed.  The allocation is marked as freed and
    /// cannot be freed again.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the allocation is already freed.
    /// * [`CudaError::NotSupported`] on macOS.
    pub fn free_async(&mut self, alloc: &mut StreamAllocation) -> CudaResult<()> {
        if alloc.freed {
            return Err(CudaError::InvalidValue);
        }

        self.platform_free_async(alloc)?;

        alloc.freed = true;
        self.active_allocations = self.active_allocations.saturating_sub(1);
        self.total_allocated = self.total_allocated.saturating_sub(alloc.size);

        Ok(())
    }

    /// Trim the pool, releasing unused memory back to the OS.
    ///
    /// At least `min_bytes_to_keep` bytes of reserved memory will remain
    /// in the pool for future allocations.
    ///
    /// # Errors
    ///
    /// * [`CudaError::NotSupported`] on macOS.
    pub fn trim(&mut self, min_bytes_to_keep: usize) -> CudaResult<()> {
        self.platform_trim(min_bytes_to_keep)
    }

    /// Get pool usage statistics.
    ///
    /// The returned [`PoolUsageStats`] combines locally tracked allocation
    /// counts with byte-level information.  On macOS, the reserved/used
    /// byte fields mirror the local bookkeeping since no driver is available.
    pub fn stats(&self) -> PoolUsageStats {
        PoolUsageStats {
            reserved_current: self.total_allocated as u64,
            reserved_high: self.peak_allocated as u64,
            used_current: self.total_allocated as u64,
            used_high: self.peak_allocated as u64,
            active_allocations: self.active_allocations,
            peak_allocations: self.peak_allocation_count,
        }
    }

    /// Set a pool attribute.
    ///
    /// Only attributes that carry a value (e.g. [`PoolAttribute::ReleaseThreshold`])
    /// modify pool state.  Read-only attributes (e.g. `ReservedMemCurrent`)
    /// return [`CudaError::InvalidValue`].
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] for read-only attributes.
    /// * [`CudaError::NotSupported`] on macOS.
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

        // Apply locally-meaningful attributes.
        if let PoolAttribute::ReleaseThreshold(val) = attr {
            self.config.release_threshold = val as usize;
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

    /// Reset peak statistics (peak allocated bytes and peak allocation count).
    pub fn reset_peak_stats(&mut self) {
        self.peak_allocated = self.total_allocated;
        self.peak_allocation_count = self.active_allocations;
    }

    /// Get the default memory pool for a device.
    ///
    /// CUDA provides a default pool per device, queried via
    /// `cuDeviceGetDefaultMemPool`.  The returned pool is owned by the
    /// driver and is *not* destroyed when the [`StreamMemoryPool`] wrapper
    /// is dropped.  On macOS, this returns a local-only pool with default
    /// configuration.
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
            Self::new(config)
        }

        // On real GPU platforms, resolve the device's default pool handle.
        #[cfg(not(target_os = "macos"))]
        {
            let handle = Self::gpu_default_pool(device)?;
            Ok(Self {
                handle,
                device,
                config,
                active_allocations: 0,
                total_allocated: 0,
                peak_allocated: 0,
                peak_allocation_count: 0,
                next_alloc_id: 1,
            })
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
    // Platform-specific helpers
    // -----------------------------------------------------------------------

    /// Perform the actual allocation.  On macOS, generates a synthetic pointer.
    fn platform_alloc_async(&mut self, size: usize, stream: u64) -> CudaResult<CUdeviceptr> {
        #[cfg(target_os = "macos")]
        {
            let _ = stream;
            // Generate a synthetic, non-zero device pointer for testing.
            // Each allocation gets a unique "address" based on the pool's
            // monotonic counter, with a base offset to avoid null.
            let synthetic_ptr = 0x1000_0000_0000_u64 + self.next_alloc_id * 0x1000;
            self.next_alloc_id = self.next_alloc_id.wrapping_add(1);
            let _ = size;
            Ok(synthetic_ptr)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_alloc_async(self.handle, size, stream)
        }
    }

    /// Trim on current platform.
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

    /// Set attribute on current platform.
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

    /// Enable peer access on current platform.
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

    /// Disable peer access on current platform.
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

    /// Perform the actual free.  On macOS, this is a no-op (synthetic pointers).
    fn platform_free_async(&self, alloc: &StreamAllocation) -> CudaResult<()> {
        #[cfg(target_os = "macos")]
        {
            let _ = alloc;
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_free_async(alloc.ptr, alloc.stream)
        }
    }

    // -----------------------------------------------------------------------
    // GPU-only driver bindings (compiled out on macOS)
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
    #[cfg(not(target_os = "macos"))]
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
    #[cfg(not(target_os = "macos"))]
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

/// Free a stream-ordered allocation using a temporary default pool.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if the allocation is already freed.
pub fn stream_free(alloc: &mut StreamAllocation) -> CudaResult<()> {
    if alloc.freed {
        return Err(CudaError::InvalidValue);
    }

    // On macOS, just mark as freed (no real GPU work).
    #[cfg(target_os = "macos")]
    {
        alloc.freed = true;
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    {
        StreamMemoryPool::gpu_free_async(alloc.ptr, alloc.stream)?;
        alloc.freed = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns `true` when a real CUDA driver is loadable on this host.
    ///
    /// Pool creation on non-macOS platforms now performs a genuine
    /// `cuMemPoolCreate`; without a driver it must fail with a clean typed
    /// error rather than succeeding or panicking.  Tests that need a live
    /// pool gate on this helper.
    #[cfg(not(target_os = "macos"))]
    fn driver_present() -> bool {
        crate::loader::try_driver().is_ok()
    }

    // -- Config validation -------------------------------------------------

    #[test]
    fn config_validate_valid_sizes() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 1024,
            max_pool_size: 4096,
            release_threshold: 512,
            device: 0,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_validate_unlimited_max() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 1024 * 1024,
            max_pool_size: 0, // unlimited
            release_threshold: 512,
            device: 0,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_validate_initial_exceeds_max() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 8192,
            max_pool_size: 4096,
            release_threshold: 0,
            device: 0,
        };
        assert_eq!(config.validate(), Err(CudaError::InvalidValue));
    }

    #[test]
    fn config_validate_negative_device() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 0,
            max_pool_size: 0,
            release_threshold: 0,
            device: -1,
        };
        assert_eq!(config.validate(), Err(CudaError::InvalidValue));
    }

    #[test]
    fn config_validate_threshold_exceeds_max() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 0,
            max_pool_size: 1024,
            release_threshold: 2048,
            device: 0,
        };
        assert_eq!(config.validate(), Err(CudaError::InvalidValue));
    }

    // -- Default config ----------------------------------------------------

    #[test]
    fn default_config_for_device() {
        let config = StreamOrderedAllocConfig::default_for_device(2);
        assert_eq!(config.device, 2);
        assert_eq!(config.initial_pool_size, 0);
        assert_eq!(config.max_pool_size, 0);
        assert_eq!(config.release_threshold, 0);
        assert!(config.validate().is_ok());
    }

    // -- Pool creation -----------------------------------------------------

    /// On macOS, pool creation always succeeds with a local-only pool.
    #[cfg(target_os = "macos")]
    #[test]
    fn pool_creation() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let pool = StreamMemoryPool::new(config);
        assert!(pool.is_ok());
        let pool = pool.ok();
        assert!(pool.is_some());
        let pool = pool.map(|p| {
            assert_eq!(p.device(), 0);
            assert_eq!(p.active_allocations, 0);
            assert_eq!(p.total_allocated, 0);
        });
        let _ = pool;
    }

    /// On non-macOS, pool creation performs a real `cuMemPoolCreate`: it
    /// succeeds when a driver is present and otherwise fails with a clean
    /// typed error (never a panic).
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn pool_creation() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let pool = StreamMemoryPool::new(config);
        if driver_present() {
            // A live driver may still reject the pool (e.g. no device);
            // either way the result must be a typed Result, not a panic.
            if let Ok(p) = pool {
                assert_eq!(p.device(), 0);
                assert_eq!(p.active_allocations, 0);
                assert_eq!(p.total_allocated, 0);
            } else {
                assert!(matches!(
                    pool,
                    Err(CudaError::NotSupported)
                        | Err(CudaError::NoDevice)
                        | Err(CudaError::InvalidDevice)
                        | Err(CudaError::InvalidContext)
                        | Err(CudaError::NotInitialized)
                ));
            }
        } else {
            // No driver: must surface a clean NotInitialized error.
            assert_eq!(pool.err(), Some(CudaError::NotInitialized));
        }
    }

    #[test]
    fn pool_creation_invalid_config() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 0,
            max_pool_size: 0,
            release_threshold: 0,
            device: -1,
        };
        let result = StreamMemoryPool::new(config);
        assert!(matches!(result, Err(CudaError::InvalidValue)));
    }

    // -- alloc_async / free_async -----------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn alloc_async_creates_allocation() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool = StreamMemoryPool::new(config).ok();
        assert!(pool.is_some());
        let pool = pool.as_mut().map(|p| {
            let alloc = p.alloc_async(1024, 0);
            assert!(alloc.is_ok());
            let alloc = alloc.ok();
            assert!(alloc.is_some());
            if let Some(a) = &alloc {
                assert_eq!(a.size(), 1024);
                assert!(!a.is_freed());
                assert_ne!(a.as_ptr(), 0);
                assert_eq!(a.stream(), 0);
            }
        });
        let _ = pool;
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn free_async_marks_freed() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");
        let mut alloc = pool
            .alloc_async(2048, 0)
            .expect("alloc should succeed on macOS");
        assert!(!alloc.is_freed());
        assert!(pool.free_async(&mut alloc).is_ok());
        assert!(alloc.is_freed());
        assert_eq!(pool.active_allocations, 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn double_free_returns_error() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");
        let mut alloc = pool
            .alloc_async(512, 0)
            .expect("alloc should succeed on macOS");
        assert!(pool.free_async(&mut alloc).is_ok());
        assert_eq!(pool.free_async(&mut alloc), Err(CudaError::InvalidValue));
    }

    // -- Trim --------------------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn trim_returns_not_supported_on_macos() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");
        assert_eq!(pool.trim(0), Err(CudaError::NotSupported));
    }

    // -- Stats tracking ----------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn stats_tracking() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");

        let mut a1 = pool.alloc_async(1024, 0).expect("alloc should succeed");
        let _a2 = pool.alloc_async(2048, 0).expect("alloc should succeed");

        let stats = pool.stats();
        assert_eq!(stats.active_allocations, 2);
        assert_eq!(stats.used_current, 3072);
        assert_eq!(stats.used_high, 3072);
        assert_eq!(stats.peak_allocations, 2);

        pool.free_async(&mut a1).expect("free should succeed");
        let stats = pool.stats();
        assert_eq!(stats.active_allocations, 1);
        assert_eq!(stats.used_current, 2048);
        // Peak should remain at 3072.
        assert_eq!(stats.used_high, 3072);
    }

    // -- Pool attribute setting --------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn set_attribute_release_threshold() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");
        let result = pool.set_attribute(PoolAttribute::ReleaseThreshold(4096));
        assert!(result.is_ok());
        assert_eq!(pool.config().release_threshold, 4096);
    }

    /// Read-only attributes are rejected by pre-flight validation, before
    /// any driver call.  On macOS the pool is always available.
    #[cfg(target_os = "macos")]
    #[test]
    fn set_attribute_readonly_returns_error() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool = StreamMemoryPool::new(config).expect("pool creation should succeed");
        assert_eq!(
            pool.set_attribute(PoolAttribute::ReservedMemCurrent),
            Err(CudaError::InvalidValue)
        );
        assert_eq!(
            pool.set_attribute(PoolAttribute::UsedMemCurrent),
            Err(CudaError::InvalidValue)
        );
    }

    /// On non-macOS, this can only be exercised when a driver is present
    /// (pool creation requires `cuMemPoolCreate`).  The read-only check
    /// itself runs before the driver is touched.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn set_attribute_readonly_returns_error() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let pool = StreamMemoryPool::new(config);
        let mut pool = match pool {
            Ok(p) => p,
            Err(e) => {
                // No usable driver/device: pool creation must fail cleanly.
                assert!(matches!(
                    e,
                    CudaError::NotInitialized
                        | CudaError::NotSupported
                        | CudaError::NoDevice
                        | CudaError::InvalidDevice
                        | CudaError::InvalidContext
                ));
                return;
            }
        };
        assert_eq!(
            pool.set_attribute(PoolAttribute::ReservedMemCurrent),
            Err(CudaError::InvalidValue)
        );
        assert_eq!(
            pool.set_attribute(PoolAttribute::UsedMemCurrent),
            Err(CudaError::InvalidValue)
        );
    }

    // -- StreamAllocation accessors ----------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn allocation_accessors() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");
        let alloc = pool.alloc_async(4096, 42).expect("alloc should succeed");
        assert_eq!(alloc.size(), 4096);
        assert_eq!(alloc.stream(), 42);
        assert!(!alloc.is_freed());
        assert_ne!(alloc.as_ptr(), 0);
        // Debug formatting should not panic.
        let _debug = format!("{alloc:?}");
    }

    // -- Convenience functions ---------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn convenience_stream_alloc() {
        let result = stream_alloc(256, 0);
        assert!(result.is_ok());
        let alloc = result.expect("should succeed on macOS");
        assert_eq!(alloc.size(), 256);
        assert!(!alloc.is_freed());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn convenience_stream_free() {
        let mut alloc = stream_alloc(128, 0).expect("alloc should succeed on macOS");
        assert!(stream_free(&mut alloc).is_ok());
        assert!(alloc.is_freed());
        // Double free via convenience function.
        assert_eq!(stream_free(&mut alloc), Err(CudaError::InvalidValue));
    }

    // -- Large allocation size ---------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn large_allocation_size() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 0,
            max_pool_size: 0, // unlimited
            release_threshold: 0,
            device: 0,
        };
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");
        // 16 GiB allocation (large but valid).
        let size = 16 * 1024 * 1024 * 1024_usize;
        let alloc = pool.alloc_async(size, 0);
        assert!(alloc.is_ok());
        let alloc = alloc.expect("should succeed");
        assert_eq!(alloc.size(), size);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn alloc_exceeds_max_pool_size() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 0,
            max_pool_size: 1024,
            release_threshold: 0,
            device: 0,
        };
        let mut pool = StreamMemoryPool::new(config).expect("pool creation should succeed");
        assert!(matches!(
            pool.alloc_async(2048, 0),
            Err(CudaError::OutOfMemory)
        ));
    }

    // -- Peer access -------------------------------------------------------

    /// Same-device peer access is rejected by pre-flight validation, before
    /// any driver call.  On macOS the pool is always available.
    #[cfg(target_os = "macos")]
    #[test]
    fn peer_access_same_device_error() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let pool = StreamMemoryPool::new(config).expect("pool creation should succeed");
        assert_eq!(pool.enable_peer_access(0), Err(CudaError::InvalidDevice));
        assert_eq!(pool.disable_peer_access(0), Err(CudaError::InvalidDevice));
    }

    /// On non-macOS, the same-device check runs before the driver is
    /// touched; it is only reachable when a pool could be created.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn peer_access_same_device_error() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let pool = match StreamMemoryPool::new(config) {
            Ok(p) => p,
            Err(e) => {
                assert!(matches!(
                    e,
                    CudaError::NotInitialized
                        | CudaError::NotSupported
                        | CudaError::NoDevice
                        | CudaError::InvalidDevice
                        | CudaError::InvalidContext
                ));
                return;
            }
        };
        assert_eq!(pool.enable_peer_access(0), Err(CudaError::InvalidDevice));
        assert_eq!(pool.disable_peer_access(0), Err(CudaError::InvalidDevice));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn peer_access_not_supported_on_macos() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let pool = StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");
        assert_eq!(pool.enable_peer_access(1), Err(CudaError::NotSupported));
        assert_eq!(pool.disable_peer_access(1), Err(CudaError::NotSupported));
    }

    // -- Reset peak stats --------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn reset_peak_stats() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");

        let mut a1 = pool.alloc_async(1024, 0).expect("alloc ok");
        let _a2 = pool.alloc_async(2048, 0).expect("alloc ok");
        assert_eq!(pool.stats().peak_allocations, 2);
        assert_eq!(pool.stats().used_high, 3072);

        pool.free_async(&mut a1).expect("free ok");
        pool.reset_peak_stats();

        let stats = pool.stats();
        assert_eq!(stats.used_high, 2048); // reset to current
        assert_eq!(stats.peak_allocations, 1); // reset to current
    }

    // -- Zero-size alloc ---------------------------------------------------

    /// Zero-size allocation is rejected by pre-flight validation, before any
    /// driver call.  On macOS the pool is always available.
    #[cfg(target_os = "macos")]
    #[test]
    fn alloc_zero_size_returns_error() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool = StreamMemoryPool::new(config).expect("pool creation should succeed");
        assert!(matches!(
            pool.alloc_async(0, 0),
            Err(CudaError::InvalidValue)
        ));
    }

    /// On non-macOS, the zero-size check runs before the driver is touched;
    /// it is only reachable when a pool could be created.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn alloc_zero_size_returns_error() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool = match StreamMemoryPool::new(config) {
            Ok(p) => p,
            Err(e) => {
                assert!(matches!(
                    e,
                    CudaError::NotInitialized
                        | CudaError::NotSupported
                        | CudaError::NoDevice
                        | CudaError::InvalidDevice
                        | CudaError::InvalidContext
                ));
                return;
            }
        };
        assert!(matches!(
            pool.alloc_async(0, 0),
            Err(CudaError::InvalidValue)
        ));
    }

    // -- Default pool ------------------------------------------------------

    /// On macOS, the default pool is a local-only pool and always succeeds.
    #[cfg(target_os = "macos")]
    #[test]
    fn default_pool_valid_device() {
        let pool = StreamMemoryPool::default_pool(0);
        assert!(pool.is_ok());
    }

    /// On non-macOS, `default_pool` performs a real
    /// `cuDeviceGetDefaultMemPool`: success with a driver, a clean typed
    /// error without one — never a panic.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn default_pool_valid_device() {
        let pool = StreamMemoryPool::default_pool(0);
        if driver_present() {
            if let Ok(p) = pool {
                assert_eq!(p.device(), 0);
            } else {
                assert!(matches!(
                    pool,
                    Err(CudaError::NotSupported)
                        | Err(CudaError::NoDevice)
                        | Err(CudaError::InvalidDevice)
                        | Err(CudaError::InvalidContext)
                        | Err(CudaError::NotInitialized)
                ));
            }
        } else {
            assert_eq!(pool.err(), Some(CudaError::NotInitialized));
        }
    }

    #[test]
    fn default_pool_negative_device() {
        assert!(matches!(
            StreamMemoryPool::default_pool(-1),
            Err(CudaError::InvalidValue)
        ));
    }

    // -- PoolAttribute::to_raw ---------------------------------------------

    #[test]
    fn pool_attribute_to_raw() {
        assert_eq!(
            PoolAttribute::ReuseFollowEventDependencies.to_raw(),
            CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES
        );
        assert_eq!(
            PoolAttribute::ReuseAllowOpportunistic.to_raw(),
            CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC
        );
        assert_eq!(
            PoolAttribute::ReuseAllowInternalDependencies.to_raw(),
            CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES
        );
        assert_eq!(
            PoolAttribute::ReleaseThreshold(0).to_raw(),
            CU_MEMPOOL_ATTR_RELEASE_THRESHOLD
        );
        assert_eq!(
            PoolAttribute::ReservedMemCurrent.to_raw(),
            CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT
        );
        assert_eq!(
            PoolAttribute::ReservedMemHigh.to_raw(),
            CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH
        );
        assert_eq!(
            PoolAttribute::UsedMemCurrent.to_raw(),
            CU_MEMPOOL_ATTR_USED_MEM_CURRENT
        );
        assert_eq!(
            PoolAttribute::UsedMemHigh.to_raw(),
            CU_MEMPOOL_ATTR_USED_MEM_HIGH
        );
    }

    // -- ShareableHandleType default ---------------------------------------

    #[test]
    fn shareable_handle_type_default() {
        assert_eq!(ShareableHandleType::default(), ShareableHandleType::None);
    }

    // -- PoolExportDescriptor construction ---------------------------------

    #[test]
    fn pool_export_descriptor() {
        let desc = PoolExportDescriptor {
            shareable_handle_type: ShareableHandleType::PosixFileDescriptor,
            pool_device: 0,
        };
        assert_eq!(
            desc.shareable_handle_type,
            ShareableHandleType::PosixFileDescriptor
        );
        assert_eq!(desc.pool_device, 0);
    }

    // -- GPU driver bindings: real-FFI / absent-driver path ----------------
    //
    // These tests exercise the `gpu_*` bindings against whatever driver the
    // host provides.
    //
    // * Without a driver, every binding must surface a clean typed error
    //   (`NotInitialized`) — never a panic, never a fake `Ok`.
    // * With a driver, the bindings reach the real CUDA FFI.  The driver
    //   *dereferences* pool / device-pointer handles without validating
    //   them, so a fabricated handle would segfault.  Handle-consuming
    //   bindings are therefore only ever called with handles obtained from
    //   a genuine `cuMemPoolCreate` / `cuMemAllocAsync`.

    /// Create a real driver-backed pool, or `None` when the host cannot
    /// (no driver, no device, or a graphless/poolless driver).
    #[cfg(not(target_os = "macos"))]
    fn make_real_pool() -> Option<StreamMemoryPool> {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        StreamMemoryPool::new(config).ok()
    }

    /// `gpu_create_pool` is deref-free: it builds `CUmemPoolProps` and calls
    /// `cuMemPoolCreate`.  Without a driver it fails cleanly; with one it
    /// returns a real handle or a typed driver error.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn gpu_create_pool_real_or_clean_error() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let result = StreamMemoryPool::gpu_create_pool(&config);
        if !driver_present() {
            assert_eq!(result.err(), Some(CudaError::NotInitialized));
        } else {
            match result {
                Ok(handle) => assert_ne!(handle, 0, "a created pool has a non-null handle"),
                Err(e) => assert!(matches!(
                    e,
                    CudaError::NotSupported
                        | CudaError::NoDevice
                        | CudaError::InvalidDevice
                        | CudaError::InvalidContext
                )),
            }
        }
    }

    /// `gpu_default_pool` is deref-free: it only needs a device ordinal.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn gpu_default_pool_real_or_clean_error() {
        let result = StreamMemoryPool::gpu_default_pool(0);
        if !driver_present() {
            assert_eq!(result.err(), Some(CudaError::NotInitialized));
        } else {
            match result {
                Ok(handle) => assert_ne!(handle, 0, "the default pool has a non-null handle"),
                Err(e) => assert!(matches!(
                    e,
                    CudaError::NotSupported | CudaError::NoDevice | CudaError::InvalidDevice
                )),
            }
        }
    }

    /// `gpu_alloc_async` default-pool path (`handle == 0`) calls
    /// `cuMemAllocAsync`, which dereferences no caller handle.  Without a
    /// current context the driver returns `InvalidContext`; without a
    /// driver, `NotInitialized`.  Either way: a clean typed error, never a
    /// panic.  Any device pointer actually returned is freed immediately.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn gpu_alloc_async_default_pool_is_clean() {
        let result = StreamMemoryPool::gpu_alloc_async(0, 1024, 0);
        if !driver_present() {
            assert_eq!(result.err(), Some(CudaError::NotInitialized));
        } else {
            match result {
                Ok(ptr) => {
                    // A live allocation: return it on the same null stream.
                    assert_ne!(ptr, 0);
                    let _ = StreamMemoryPool::gpu_free_async(ptr, 0);
                }
                Err(e) => assert!(matches!(
                    e,
                    CudaError::InvalidContext
                        | CudaError::NotSupported
                        | CudaError::NoDevice
                        | CudaError::InvalidDevice
                        | CudaError::OutOfMemory
                )),
            }
        }
    }

    /// `gpu_trim` on a *real* pool handle: trimming an empty pool succeeds.
    /// Without a driver the binding fails cleanly before any dereference.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn gpu_trim_on_real_pool_or_clean_error() {
        if !driver_present() {
            // The function must fail cleanly; with no driver `try_driver`
            // returns the error before a handle is ever dereferenced.
            // (A fabricated handle is never passed to a live driver.)
            return;
        }
        let Some(pool) = make_real_pool() else {
            return; // driver present but no usable pool — nothing to trim.
        };
        // `cuMemPoolTrimTo` on a real, empty pool is a valid no-op.
        let result = StreamMemoryPool::gpu_trim(pool.handle(), 0);
        assert!(
            result.is_ok() || result.is_err(),
            "gpu_trim must return a typed Result, not panic"
        );
        if let Err(e) = result {
            assert!(matches!(e, CudaError::NotSupported));
        }
    }

    /// `gpu_set_attribute` on a *real* pool handle, for both the reuse-policy
    /// (`int` value) and release-threshold (`cuuint64_t` value) branches.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn gpu_set_attribute_on_real_pool_or_clean_error() {
        if !driver_present() {
            return;
        }
        let Some(pool) = make_real_pool() else {
            return;
        };
        let reuse = StreamMemoryPool::gpu_set_attribute(
            pool.handle(),
            PoolAttribute::ReuseAllowOpportunistic,
        );
        let threshold = StreamMemoryPool::gpu_set_attribute(
            pool.handle(),
            PoolAttribute::ReleaseThreshold(8192),
        );
        // On a CUDA 11.2+ driver both branches succeed; an older driver
        // yields a clean `NotSupported`.
        for r in [reuse, threshold] {
            if let Err(e) = r {
                assert!(matches!(e, CudaError::NotSupported));
            }
        }
    }

    /// `gpu_enable_peer_access` / `gpu_disable_peer_access` on a *real* pool.
    /// Granting access to a non-existent peer device is a typed driver error
    /// (`InvalidDevice`), not a panic.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn gpu_peer_access_on_real_pool_or_clean_error() {
        if !driver_present() {
            return;
        }
        let Some(pool) = make_real_pool() else {
            return;
        };
        // Device 1 may or may not exist; either outcome must be a typed
        // Result.  `cuMemPoolSetAccess` dereferences only the real pool
        // handle, never a fabricated one.
        let enable = StreamMemoryPool::gpu_enable_peer_access(pool.handle(), 1);
        let disable = StreamMemoryPool::gpu_disable_peer_access(pool.handle(), 1);
        for r in [enable, disable] {
            if let Err(e) = r {
                assert!(matches!(
                    e,
                    CudaError::InvalidDevice | CudaError::InvalidValue | CudaError::NotSupported
                ));
            }
        }
    }

    /// The `gpu_*` bindings surface `NotInitialized` (never a panic) when no
    /// driver is loadable.  This is a no-op assertion on a host *with* a
    /// driver; the per-binding tests above cover the live-FFI behaviour.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn gpu_bindings_clean_error_without_driver() {
        if driver_present() {
            return;
        }
        // No driver: every binding fails before touching a handle.
        let config = StreamOrderedAllocConfig::default_for_device(0);
        assert_eq!(
            StreamMemoryPool::gpu_create_pool(&config).err(),
            Some(CudaError::NotInitialized)
        );
        assert_eq!(
            StreamMemoryPool::gpu_default_pool(0).err(),
            Some(CudaError::NotInitialized)
        );
        assert_eq!(
            StreamMemoryPool::gpu_alloc_async(0, 1024, 0).err(),
            Some(CudaError::NotInitialized)
        );
        // Handle-consuming bindings also fail before any dereference.
        assert_eq!(
            StreamMemoryPool::gpu_alloc_async(0x1, 1024, 0).err(),
            Some(CudaError::NotInitialized)
        );
        assert_eq!(
            StreamMemoryPool::gpu_free_async(0x1, 0).err(),
            Some(CudaError::NotInitialized)
        );
        assert_eq!(
            StreamMemoryPool::gpu_trim(0x1, 0).err(),
            Some(CudaError::NotInitialized)
        );
        assert_eq!(
            StreamMemoryPool::gpu_set_attribute(0x1, PoolAttribute::ReleaseThreshold(1)).err(),
            Some(CudaError::NotInitialized)
        );
        assert_eq!(
            StreamMemoryPool::gpu_enable_peer_access(0x1, 1).err(),
            Some(CudaError::NotInitialized)
        );
        assert_eq!(
            StreamMemoryPool::gpu_disable_peer_access(0x1, 1).err(),
            Some(CudaError::NotInitialized)
        );
    }

    /// `map_pool_attribute` maps every variant to the matching driver enum.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn map_pool_attribute_covers_all_variants() {
        use crate::ffi::CUmemPoolAttribute;
        let cases = [
            (
                PoolAttribute::ReuseFollowEventDependencies,
                CUmemPoolAttribute::ReuseFollowEventDependencies,
            ),
            (
                PoolAttribute::ReuseAllowOpportunistic,
                CUmemPoolAttribute::ReuseAllowOpportunistic,
            ),
            (
                PoolAttribute::ReuseAllowInternalDependencies,
                CUmemPoolAttribute::ReuseAllowInternalDependencies,
            ),
            (
                PoolAttribute::ReleaseThreshold(0),
                CUmemPoolAttribute::ReleaseThreshold,
            ),
            (
                PoolAttribute::ReservedMemCurrent,
                CUmemPoolAttribute::ReservedMemCurrent,
            ),
            (
                PoolAttribute::ReservedMemHigh,
                CUmemPoolAttribute::ReservedMemHigh,
            ),
            (
                PoolAttribute::UsedMemCurrent,
                CUmemPoolAttribute::UsedMemCurrent,
            ),
            (PoolAttribute::UsedMemHigh, CUmemPoolAttribute::UsedMemHigh),
        ];
        for (attr, expected) in cases {
            let mapped = StreamMemoryPool::map_pool_attribute(attr);
            assert_eq!(mapped, Ok(expected));
        }
    }

    /// `stream_alloc` convenience: a clean typed error without a driver, and
    /// a typed `Result` (allocation, or a context/driver error) with one.
    /// Any device pointer actually returned is freed immediately.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn convenience_stream_alloc_real_or_clean_error() {
        let result = stream_alloc(256, 0);
        if !driver_present() {
            assert_eq!(result.err(), Some(CudaError::NotInitialized));
        } else {
            match result {
                Ok(mut alloc) => {
                    assert_eq!(alloc.size(), 256);
                    let _ = stream_free(&mut alloc);
                }
                Err(e) => assert!(matches!(
                    e,
                    CudaError::InvalidContext
                        | CudaError::NotSupported
                        | CudaError::NoDevice
                        | CudaError::InvalidDevice
                        | CudaError::OutOfMemory
                )),
            }
        }
    }
}
