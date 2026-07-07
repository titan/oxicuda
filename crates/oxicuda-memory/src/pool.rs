//! Stream-ordered memory pool for efficient async allocation.
//!
//! Requires CUDA 11.2+ driver.  Gated behind the `pool` feature.
//!
//! Stream-ordered memory pools allow allocation and deallocation to be
//! ordered relative to other operations on a CUDA stream, enabling the
//! driver to reuse memory more aggressively and avoid synchronisation
//! barriers that would otherwise be needed for conventional
//! `cuMemAlloc` / `cuMemFree` calls.
//!
//! # Implementation note
//!
//! This implementation provides a practical fallback pool that reuses freed
//! allocations by size and uses `cuMemAlloc_v2` / `cuMemFree_v2` under the
//! hood.  It keeps the same API surface as a stream-ordered pool, but does
//! not yet expose native CUDA mempool handles.
//!
//! Reuse is stream-ordered via a recycle event: dropping a [`PooledBuffer`]
//! records a `CU_EVENT_DISABLE_TIMING` event on the stream it was allocated
//! from rather than immediately handing the pointer back out, so a second
//! concurrent allocation can only reuse the pointer once all GPU work
//! enqueued before the drop has actually completed. Each [`MemoryPool`] also
//! retains its device's primary context for its lifetime and binds every
//! allocation/free/event operation to that context, so the pool is safe to
//! share across threads without mixing pointers from different devices.
//!
//! # API
//!
//! ```rust,ignore
//! let pool = MemoryPool::new(device)?;
//! let buf = PooledBuffer::<f32>::alloc_async(&pool, 1024, &stream)?;
//! // … use buf in kernels on `stream` …
//! // `buf` is dropped here: a recycle event is enqueued on `stream`, and the
//! // pointer becomes reusable by a later `alloc_async` only once that event
//! // (i.e. all prior work on `stream`) has completed.
//! ```

#![cfg(feature = "pool")]

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use oxicuda_driver::error::{CudaError, CudaResult, check};
use oxicuda_driver::ffi::{
    CU_EVENT_DISABLE_TIMING, CUDA_ERROR_NOT_READY, CUcontext, CUdeviceptr, CUevent,
    CUmemAllocationHandleType, CUmemAllocationType, CUmemLocation, CUmemLocationType,
    CUmemPoolProps, CUmemoryPool, CUstream,
};
use oxicuda_driver::loader::{DriverApi, try_driver};
use oxicuda_driver::stream::Stream;
use tracing::warn;

// ---------------------------------------------------------------------------
// MemoryPool
// ---------------------------------------------------------------------------

/// A stream-ordered memory pool (CUDA 11.2+).
///
/// Memory pools allow the driver to reuse freed allocations without
/// returning them to the OS, reducing allocation latency and avoiding
/// the implicit synchronisation of `cuMemFree`.
///
/// # Status
///
/// `MemoryPool` is a software pool layered on top of `cuMemAlloc_v2`.
/// For a thin wrapper over the *native* CUDA stream-ordered memory pool
/// API (`cuMemPoolCreate`, `cuMemPoolDestroy`, `cuMemAllocFromPoolAsync`,
/// `cuMemFreeAsync`), use [`NativeMemoryPool`].
///
/// Statistics for a memory pool's allocation behaviour.
///
/// These statistics track the total bytes allocated, peak usage,
/// allocation count, and free count for a given pool.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Total bytes currently allocated from the pool.
    pub allocated_bytes: usize,
    /// Peak bytes allocated at any point during the pool's lifetime.
    pub peak_bytes: usize,
    /// Total number of allocations performed.
    pub allocation_count: u64,
    /// Total number of frees performed.
    pub free_count: u64,
}

#[derive(Debug)]
struct MemoryPoolInner {
    handle: u64,
    device_ordinal: i32,
    /// The device's primary context, retained for the lifetime of the pool.
    ///
    /// All allocations, frees, and event operations issued by this pool are
    /// performed with this context made current on the calling thread (see
    /// [`with_device_context`](Self::with_device_context)), so that a pool
    /// shared across threads always targets `device_ordinal` regardless of
    /// whatever context happens to already be current on a given thread.
    ctx: CUcontext,
    threshold_bytes: AtomicUsize,
    cached_bytes: AtomicUsize,
    stats: Mutex<PoolStats>,
    /// Free-list bins keyed by allocation size.  Each entry pairs a device
    /// pointer with the (possibly null) recycle-safety event that was
    /// recorded on the freeing `PooledBuffer`'s stream; the pointer must not
    /// be handed out again until that event has completed (see
    /// [`try_pop_reuse`](Self::try_pop_reuse)).
    free_bins: Mutex<HashMap<usize, Vec<(CUdeviceptr, CUevent)>>>,
}

impl MemoryPoolInner {
    /// Runs `f` with this pool's device context current on the calling
    /// thread, restoring the caller's previous context before returning.
    ///
    /// This is the mechanism that binds every driver call issued by the pool
    /// to `device_ordinal`'s context, so a `MemoryPool` shared across threads
    /// (or used from a thread whose current context targets a different
    /// device) never mixes pointers from different devices in one free-bin
    /// map.
    fn with_device_context<R>(
        &self,
        f: impl FnOnce(&'static DriverApi) -> CudaResult<R>,
    ) -> CudaResult<R> {
        let api = try_driver()?;
        let mut prev = CUcontext::default();
        check(unsafe { (api.cu_ctx_get_current)(&mut prev) })?;
        check(unsafe { (api.cu_ctx_set_current)(self.ctx) })?;

        let result = f(api);

        let restore_rc = check(unsafe { (api.cu_ctx_set_current)(prev) });
        match result {
            Ok(value) => restore_rc.map(|()| value),
            Err(e) => {
                if let Err(restore_err) = restore_rc {
                    warn!(
                        "failed to restore previous CUDA context after pool operation \
                         (original error: {e}): {restore_err}"
                    );
                }
                Err(e)
            }
        }
    }

    fn allocate_fresh(&self, bytes: usize) -> CudaResult<CUdeviceptr> {
        self.with_device_context(|api| {
            let mut ptr: CUdeviceptr = 0;
            let rc = unsafe { (api.cu_mem_alloc_v2)(&mut ptr, bytes) };
            oxicuda_driver::check(rc)?;
            Ok(ptr)
        })
    }

    fn free_ptr(&self, ptr: CUdeviceptr) -> CudaResult<()> {
        self.with_device_context(|api| {
            let rc = unsafe { (api.cu_mem_free_v2)(ptr) };
            oxicuda_driver::check(rc)
        })
    }

    /// Creates a `CU_EVENT_DISABLE_TIMING` event and records it on `stream`.
    ///
    /// Returns the raw event handle; the caller owns it and must eventually
    /// pass it to [`destroy_event`](Self::destroy_event).
    fn record_recycle_event(&self, stream: CUstream) -> CudaResult<CUevent> {
        self.with_device_context(|api| {
            let mut event = CUevent::default();
            check(unsafe { (api.cu_event_create)(&mut event, CU_EVENT_DISABLE_TIMING) })?;
            if let Err(e) = check(unsafe { (api.cu_event_record)(event, stream) }) {
                let _ = unsafe { (api.cu_event_destroy_v2)(event) };
                return Err(e);
            }
            Ok(event)
        })
    }

    /// Returns `true` if `event` is null (nothing to wait for) or has
    /// completed; `false` if it is still pending on its stream.
    fn event_ready(&self, event: CUevent) -> bool {
        if event.is_null() {
            return true;
        }
        self.with_device_context(|api| {
            let rc = unsafe { (api.cu_event_query)(event) };
            if rc == 0 {
                Ok(true)
            } else if rc == CUDA_ERROR_NOT_READY {
                Ok(false)
            } else {
                Err(CudaError::from_raw(rc))
            }
        })
        // A query error (rather than "not ready") is treated conservatively
        // as "not yet safe to reuse" so the pointer is never handed out
        // early; the entry simply remains in the free bin.
        .unwrap_or(false)
    }

    /// Blocks until `event` has completed. A null event is a no-op.
    fn synchronize_event(&self, event: CUevent) -> CudaResult<()> {
        if event.is_null() {
            return Ok(());
        }
        self.with_device_context(|api| check(unsafe { (api.cu_event_synchronize)(event) }))
    }

    /// Destroys `event`, logging (but not propagating) any driver error.
    /// A null event is a no-op.
    fn destroy_event(&self, event: CUevent) {
        if event.is_null() {
            return;
        }
        let result =
            self.with_device_context(|api| check(unsafe { (api.cu_event_destroy_v2)(event) }));
        if let Err(e) = result {
            warn!("cuEventDestroy_v2 failed for pooled-buffer recycle event: {e}");
        }
    }

    fn try_pop_reuse(&self, bytes: usize) -> CudaResult<Option<CUdeviceptr>> {
        let popped = {
            let mut bins = self.free_bins.lock().map_err(|_| CudaError::Unknown(0))?;
            let Some(vec) = bins.get_mut(&bytes) else {
                return Ok(None);
            };
            // Only hand out an entry whose recycle event has completed —
            // i.e. all GPU work enqueued before the corresponding `Drop`
            // has actually finished — so a second concurrent user can never
            // receive a pointer that is still in flight. Entries that are
            // not yet ready are left in the bin for a later call.
            let ready_idx = vec.iter().position(|(_, event)| self.event_ready(*event));
            ready_idx.map(|idx| vec.swap_remove(idx))
        };

        let Some((ptr, event)) = popped else {
            return Ok(None);
        };
        self.destroy_event(event);
        self.cached_bytes.fetch_sub(bytes, Ordering::Relaxed);
        Ok(Some(ptr))
    }

    fn stash_freed(&self, ptr: CUdeviceptr, bytes: usize, event: CUevent) -> CudaResult<()> {
        let mut bins = self.free_bins.lock().map_err(|_| CudaError::Unknown(0))?;
        bins.entry(bytes).or_default().push((ptr, event));
        self.cached_bytes.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }

    fn release_cached_until(&self, keep_bytes: usize) -> CudaResult<()> {
        loop {
            let cached = self.cached_bytes.load(Ordering::Relaxed);
            if cached <= keep_bytes {
                return Ok(());
            }

            let popped = {
                let mut bins = self.free_bins.lock().map_err(|_| CudaError::Unknown(0))?;
                let mut candidate: Option<(usize, CUdeviceptr, CUevent)> = None;
                for (size, vec) in bins.iter_mut() {
                    if let Some((ptr, event)) = vec.pop() {
                        candidate = Some((*size, ptr, event));
                        break;
                    }
                }
                candidate
            };

            let Some((size, ptr, event)) = popped else {
                return Ok(());
            };
            // A cached pointer may still have GPU work in flight against it;
            // block until its recorded recycle event completes before
            // handing the memory back to `cuMemFree_v2`.
            self.synchronize_event(event)?;
            self.destroy_event(event);
            self.free_ptr(ptr)?;
            self.cached_bytes.fetch_sub(size, Ordering::Relaxed);
        }
    }

    fn update_alloc_stats(&self, bytes: usize) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.allocated_bytes = stats.allocated_bytes.saturating_add(bytes);
            stats.allocation_count = stats.allocation_count.saturating_add(1);
            if stats.allocated_bytes > stats.peak_bytes {
                stats.peak_bytes = stats.allocated_bytes;
            }
        }
    }

    fn update_free_stats(&self, bytes: usize) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.allocated_bytes = stats.allocated_bytes.saturating_sub(bytes);
            stats.free_count = stats.free_count.saturating_add(1);
        }
    }
}

impl Drop for MemoryPoolInner {
    fn drop(&mut self) {
        let Ok(mut bins) = self.free_bins.lock() else {
            return;
        };
        let mut to_free: Vec<(CUdeviceptr, CUevent)> = Vec::new();
        for vec in bins.values_mut() {
            to_free.append(vec);
        }
        drop(bins);

        for (ptr, event) in to_free {
            // Cached pointers may still have GPU work in flight; wait for
            // their recycle event before freeing the underlying memory.
            if let Err(e) = self.synchronize_event(event) {
                warn!("cuEventSynchronize failed while draining pool on drop: {e}");
            }
            self.destroy_event(event);
            if let Err(e) = self.free_ptr(ptr) {
                warn!("failed to free pooled pointer {ptr:#x} during drop: {e}");
            }
        }

        if !self.ctx.is_null() {
            if let Ok(api) = try_driver() {
                let rc = unsafe { (api.cu_device_primary_ctx_release_v2)(self.device_ordinal) };
                if rc != 0 {
                    warn!(
                        cuda_error = rc,
                        device_ordinal = self.device_ordinal,
                        "cuDevicePrimaryCtxRelease_v2 failed while dropping MemoryPool"
                    );
                }
            }
        }
    }
}

/// A stream-ordered memory pool (CUDA 11.2+).
pub struct MemoryPool {
    inner: Arc<MemoryPoolInner>,
}

impl MemoryPool {
    /// Creates a new memory pool on the given device.
    ///
    /// This retains the device's primary context for the lifetime of the
    /// pool (released when the pool is dropped), and binds every subsequent
    /// allocation and free issued through this pool to that context, so the
    /// pool is safe to share across threads without mixing pointers from
    /// different devices.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidDevice`] if `device_ordinal` is negative.
    /// * [`CudaError::NotInitialized`] if no CUDA driver is available.
    /// * Other [`CudaError`] variants if `cuDevicePrimaryCtxRetain` fails
    ///   (e.g. an out-of-range device ordinal).
    pub fn new(device_ordinal: i32) -> CudaResult<Self> {
        if device_ordinal < 0 {
            return Err(CudaError::InvalidDevice);
        }
        let api = try_driver()?;
        let mut ctx = CUcontext::default();
        check(unsafe { (api.cu_device_primary_ctx_retain)(&mut ctx, device_ordinal) })?;
        Ok(Self {
            inner: Arc::new(MemoryPoolInner {
                handle: 0,
                device_ordinal,
                ctx,
                threshold_bytes: AtomicUsize::new(0),
                cached_bytes: AtomicUsize::new(0),
                stats: Mutex::new(PoolStats::default()),
                free_bins: Mutex::new(HashMap::new()),
            }),
        })
    }

    /// Returns the raw pool handle.
    ///
    /// # Status
    ///
    /// Returns `0` until the pool is properly initialised.
    #[inline]
    pub fn raw_handle(&self) -> u64 {
        self.inner.handle
    }

    /// Returns the device ordinal this pool targets.
    #[inline]
    pub fn device_ordinal(&self) -> i32 {
        self.inner.device_ordinal
    }

    /// Returns current pool statistics.
    ///
    /// The statistics track allocation behaviour over the pool's lifetime.
    #[inline]
    pub fn stats(&self) -> PoolStats {
        self.inner.stats.lock().map(|s| *s).unwrap_or_default()
    }

    /// Trims the pool, releasing unused memory back to the OS.
    ///
    /// Attempts to release memory such that the pool retains at most
    /// `min_bytes` of unused memory.
    ///
    /// # Errors
    ///
    pub fn trim(&mut self, min_bytes: usize) -> CudaResult<()> {
        self.inner.release_cached_until(min_bytes)
    }

    /// Sets the threshold at which the pool will automatically release
    /// memory back to the OS.
    ///
    /// When the pool's unused memory exceeds `bytes`, subsequent frees
    /// will trigger automatic trimming.
    ///
    /// # Errors
    ///
    pub fn set_threshold(&mut self, bytes: usize) -> CudaResult<()> {
        self.inner.threshold_bytes.store(bytes, Ordering::Relaxed);
        self.inner.release_cached_until(bytes)
    }
}

// ---------------------------------------------------------------------------
// PooledBuffer<T>
// ---------------------------------------------------------------------------

/// A device buffer allocated from a [`MemoryPool`].
///
/// Unlike [`DeviceBuffer`](crate::DeviceBuffer), a `PooledBuffer` is freed
/// asynchronously — the free operation is enqueued on the stream rather
/// than blocking the CPU.  This enables overlap of allocation, computation,
/// and deallocation across multiple stream operations.
///
/// # Stream-ordering
///
/// Dropping a `PooledBuffer` does not immediately reuse its device pointer.
/// Instead, a `CU_EVENT_DISABLE_TIMING` event is recorded on the stream the
/// buffer was allocated on (the `stream` argument to
/// [`alloc_async`](Self::alloc_async)); the pointer is only handed back out
/// to a later [`alloc_async`](Self::alloc_async) call once that event has completed, i.e. once
/// all work enqueued on the stream before the drop has actually finished
/// executing on the device. A `PooledBuffer` must therefore not outlive its
/// stream. If the recycle event cannot be created or recorded, `Drop` falls
/// back to a blocking `cuStreamSynchronize` on the owning stream before
/// freeing the pointer directly, so correctness is preserved at the cost of
/// losing the pool's overlap benefits for that allocation.
///
/// # Status
///
/// This type allocates from an in-process memory pool and returns buffers to
/// that pool on drop.
pub struct PooledBuffer<T: Copy> {
    /// Raw device pointer to the pooled allocation.
    ptr: CUdeviceptr,
    /// Number of `T` elements.
    len: usize,
    /// Number of bytes in this allocation.
    bytes: usize,
    /// Owning pool.
    pool: Arc<MemoryPoolInner>,
    /// The stream this allocation was requested on. The recycle-safety
    /// event enqueued in `Drop` is recorded against this stream.
    stream: CUstream,
    /// Marker for the element type.
    _phantom: PhantomData<T>,
}

impl<T: Copy> PooledBuffer<T> {
    /// Asynchronously allocates a buffer of `n` elements from the given pool.
    ///
    /// The allocation is ordered relative to other operations on `stream`;
    /// `stream` is also recorded against so that [`Drop`] can establish that
    /// all work using this buffer has completed before its pointer is
    /// recycled to a later caller (see the type-level docs).
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `n` is zero or `n * size_of::<T>()`
    ///   overflows.
    /// * Other [`CudaError`] variants if the underlying allocation fails.
    pub fn alloc_async(pool: &MemoryPool, n: usize, stream: &Stream) -> CudaResult<Self> {
        if n == 0 {
            return Err(CudaError::InvalidValue);
        }
        let bytes = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(CudaError::InvalidValue)?;
        let ptr = if let Some(reused) = pool.inner.try_pop_reuse(bytes)? {
            reused
        } else {
            pool.inner.allocate_fresh(bytes)?
        };
        pool.inner.update_alloc_stats(bytes);

        Ok(Self {
            ptr,
            len: n,
            bytes,
            pool: Arc::clone(&pool.inner),
            stream: stream.raw(),
            _phantom: PhantomData,
        })
    }

    /// Returns the number of `T` elements in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the total size of the allocation in bytes.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.bytes
    }

    /// Returns the raw [`CUdeviceptr`] handle.
    #[inline]
    pub fn as_device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}

impl<T: Copy> Drop for PooledBuffer<T> {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }

        let event = match self.pool.record_recycle_event(self.stream) {
            Ok(event) => event,
            Err(e) => {
                // Without an event we cannot prove the pointer is safe to
                // reuse without blocking, so synchronise the owning stream
                // directly and free the pointer rather than risk handing it
                // to a second concurrent user while GPU work may still be
                // in flight.
                warn!(
                    "failed to record pooled-buffer recycle event ({e}); falling back to a \
                     blocking stream synchronize before freeing directly"
                );
                if let Ok(api) = try_driver() {
                    let rc = unsafe { (api.cu_stream_synchronize)(self.stream) };
                    if rc != 0 {
                        warn!(cuda_error = rc, "fallback cuStreamSynchronize failed");
                    }
                }
                if let Err(free_err) = self.pool.free_ptr(self.ptr) {
                    warn!("direct free of pooled pointer failed: {free_err}");
                }
                self.pool.update_free_stats(self.bytes);
                self.ptr = 0;
                return;
            }
        };

        if let Err(e) = self.pool.stash_freed(self.ptr, self.bytes, event) {
            warn!("failed to return pooled pointer to free list: {e}; freeing directly");
            self.pool.destroy_event(event);
            if let Err(free_err) = self.pool.free_ptr(self.ptr) {
                warn!("direct free of pooled pointer failed: {free_err}");
            }
            self.pool.update_free_stats(self.bytes);
            self.ptr = 0;
            return;
        }

        self.pool.update_free_stats(self.bytes);
        let threshold = self.pool.threshold_bytes.load(Ordering::Relaxed);
        if let Err(e) = self.pool.release_cached_until(threshold) {
            warn!("pool threshold trim failed: {e}");
        }
        self.ptr = 0;
    }
}

// ---------------------------------------------------------------------------
// NativeMemoryPool — thin wrapper over the CUDA stream-ordered pool API
// ---------------------------------------------------------------------------

/// Configuration for a [`NativeMemoryPool`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeMemoryPoolProps {
    /// Device ordinal that physically backs the pool.
    pub device_ordinal: i32,
    /// Maximum aggregate size (bytes) the pool may hold.  `0` = unlimited.
    pub max_size_bytes: usize,
}

/// Thin wrapper around the CUDA driver's stream-ordered memory pool
/// (`cuMemPoolCreate` / `cuMemPoolDestroy`).
///
/// Allocations are issued via [`NativeMemoryPool::alloc_async`] which
/// invokes `cuMemAllocFromPoolAsync`; frees are issued via
/// [`NativeMemoryPool::free_async`] which invokes `cuMemFreeAsync`.
///
/// # Stream-ordering
///
/// The CUDA stream-ordered pool API requires the caller to ensure all
/// outstanding work on the stream has completed before destroying the
/// pool.  The [`Drop`] implementation calls `cuMemPoolDestroy` and
/// silently swallows any error to honour the standard Drop convention.
/// Call [`NativeMemoryPool::destroy`] explicitly to surface destruction
/// errors.
///
/// # Status
///
/// On systems without a CUDA driver (e.g. macOS), [`NativeMemoryPool::new`]
/// fails with [`CudaError::NotInitialized`].  On older drivers that lack
/// the pool entry points it fails with [`CudaError::NotSupported`].
pub struct NativeMemoryPool {
    raw: CUmemoryPool,
    device_ordinal: i32,
}

// SAFETY: `CUmemoryPool` is an opaque driver handle.  The CUDA driver is
// thread-safe; multiple threads may issue stream-ordered allocations from
// the same pool concurrently.
unsafe impl Send for NativeMemoryPool {}
unsafe impl Sync for NativeMemoryPool {}

impl NativeMemoryPool {
    /// Creates a new native memory pool on the device described by `props`.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `device_ordinal` is negative.
    /// * [`CudaError::NotInitialized`] if no CUDA driver is available.
    /// * [`CudaError::NotSupported`] if the driver does not export
    ///   `cuMemPoolCreate`.
    /// * Other [`CudaError`] variants on driver failure.
    pub fn new(props: NativeMemoryPoolProps) -> CudaResult<Self> {
        if props.device_ordinal < 0 {
            return Err(CudaError::InvalidDevice);
        }

        let api = try_driver()?;
        let f = api.cu_mem_pool_create.ok_or(CudaError::NotSupported)?;

        let pool_props = CUmemPoolProps {
            alloc_type: CUmemAllocationType::Pinned as u32,
            handle_types: CUmemAllocationHandleType::None as u32,
            location: CUmemLocation {
                loc_type: CUmemLocationType::Device as u32,
                id: props.device_ordinal,
            },
            max_size: props.max_size_bytes,
            ..CUmemPoolProps::default()
        };

        let mut raw = CUmemoryPool::default();
        check(unsafe { f(&mut raw, &pool_props) })?;

        Ok(Self {
            raw,
            device_ordinal: props.device_ordinal,
        })
    }

    /// Returns the raw [`CUmemoryPool`] handle.
    #[inline]
    pub fn raw(&self) -> CUmemoryPool {
        self.raw
    }

    /// Returns the device ordinal that backs this pool.
    #[inline]
    pub fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    /// Asynchronously allocates `bytes` of memory from the pool, ordered
    /// against `stream`.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `bytes` is zero.
    /// * [`CudaError::NotInitialized`] if no CUDA driver is available.
    /// * [`CudaError::NotSupported`] if the driver does not export
    ///   `cuMemAllocFromPoolAsync`.
    /// * Other [`CudaError`] variants on driver failure.
    pub fn alloc_async(&self, bytes: usize, stream: &Stream) -> CudaResult<CUdeviceptr> {
        if bytes == 0 {
            return Err(CudaError::InvalidValue);
        }
        let api = try_driver()?;
        let f = api
            .cu_mem_alloc_from_pool_async
            .ok_or(CudaError::NotSupported)?;
        let mut ptr: CUdeviceptr = 0;
        check(unsafe { f(&mut ptr, bytes, self.raw, stream.raw()) })?;
        Ok(ptr)
    }

    /// Asynchronously frees a pointer previously returned by
    /// [`alloc_async`](Self::alloc_async), ordered against `stream`.
    ///
    /// # Errors
    ///
    /// * [`CudaError::NotInitialized`] if no CUDA driver is available.
    /// * [`CudaError::NotSupported`] if the driver does not export
    ///   `cuMemFreeAsync`.
    /// * Other [`CudaError`] variants on driver failure.
    pub fn free_async(&self, ptr: CUdeviceptr, stream: &Stream) -> CudaResult<()> {
        let api = try_driver()?;
        let f = api.cu_mem_free_async.ok_or(CudaError::NotSupported)?;
        check(unsafe { f(ptr, stream.raw()) })
    }

    /// Destroys the pool, returning any driver error to the caller.
    ///
    /// The caller is responsible for ensuring all outstanding work on
    /// streams that allocated from this pool has completed before calling
    /// `destroy`.
    ///
    /// After this call returns, the [`Drop`] implementation will be a
    /// no-op.
    ///
    /// # Errors
    ///
    /// * [`CudaError::NotInitialized`] if no CUDA driver is available.
    /// * [`CudaError::NotSupported`] if the driver does not export
    ///   `cuMemPoolDestroy`.
    /// * Other [`CudaError`] variants on driver failure.
    pub fn destroy(mut self) -> CudaResult<()> {
        self.destroy_inner()
    }

    fn destroy_inner(&mut self) -> CudaResult<()> {
        if self.raw.is_null() {
            return Ok(());
        }
        let api = try_driver()?;
        let f = api.cu_mem_pool_destroy.ok_or(CudaError::NotSupported)?;
        let result = check(unsafe { f(self.raw) });
        // Always clear the handle so Drop is a no-op even if destroy fails.
        self.raw = CUmemoryPool::default();
        result
    }
}

impl Drop for NativeMemoryPool {
    fn drop(&mut self) {
        if let Err(e) = self.destroy_inner() {
            warn!("failed to destroy native memory pool during drop: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn is_driver_unavailable(err: &CudaError) -> bool {
        matches!(err, CudaError::NotInitialized | CudaError::NotSupported)
    }

    #[test]
    fn native_memory_pool_props_default() {
        let props = NativeMemoryPoolProps::default();
        assert_eq!(props.device_ordinal, 0);
        assert_eq!(props.max_size_bytes, 0);
    }

    #[test]
    fn native_memory_pool_new_negative_device_fails() {
        let props = NativeMemoryPoolProps {
            device_ordinal: -1,
            max_size_bytes: 0,
        };
        let result = NativeMemoryPool::new(props);
        assert_eq!(result.err(), Some(CudaError::InvalidDevice));
    }

    /// Without a CUDA driver, `NativeMemoryPool::new` must fail with one of
    /// the driver-unavailability error kinds rather than panicking.
    #[test]
    fn native_memory_pool_new_no_driver_returns_driver_unavailable() {
        let result = NativeMemoryPool::new(NativeMemoryPoolProps::default());
        match result {
            Ok(pool) => {
                // CUDA available: explicit destroy must succeed too.
                let destroy = pool.destroy();
                assert!(destroy.is_ok(), "destroy failed: {destroy:?}");
            }
            Err(e) => assert!(
                is_driver_unavailable(&e),
                "expected driver-unavailable error, got {e:?}"
            ),
        }
    }

    /// On macOS specifically, every driver-calling method must return
    /// [`CudaError::NotInitialized`] (no library to load).
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_native_pool_returns_not_initialized() {
        let result = NativeMemoryPool::new(NativeMemoryPoolProps::default());
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected NotInitialized on macOS, got Ok"),
        };
        assert!(
            matches!(err, CudaError::NotInitialized),
            "expected NotInitialized, got {err:?}"
        );
    }

    // -- MemoryPool / PooledBuffer tests -------------------------------------

    /// The negative-ordinal check must reject before any driver call, so
    /// this must hold regardless of whether a CUDA driver is present.
    #[test]
    fn memory_pool_new_negative_device_fails() {
        let result = MemoryPool::new(-1);
        assert_eq!(result.err(), Some(CudaError::InvalidDevice));
    }

    #[cfg(feature = "gpu-tests")]
    mod gpu_tests {
        use super::*;
        use std::ffi::c_void;
        use std::sync::Arc;

        /// Bootstraps a real CUDA context on device 0. Returns `None` if no
        /// driver/GPU is available so callers can skip gracefully.
        fn real_context() -> Option<Arc<oxicuda_driver::context::Context>> {
            if oxicuda_driver::init().is_err()
                || oxicuda_driver::device::Device::count().unwrap_or(0) == 0
            {
                return None;
            }
            let dev = oxicuda_driver::device::Device::get(0).ok()?;
            oxicuda_driver::context::Context::new(&dev)
                .ok()
                .map(Arc::new)
        }

        /// `MemoryPool::new` must bind to the requested device (retaining
        /// its primary context — the F075 fix) and allocations issued
        /// through the pool must round-trip real data correctly (a
        /// regression guard that the save/restore-context wrapping in
        /// `with_device_context` does not corrupt normal HtoD/DtoH traffic).
        #[test]
        fn memory_pool_binds_device_and_round_trips_data() {
            let Some(ctx) = real_context() else {
                return;
            };
            let Ok(pool) = MemoryPool::new(0) else {
                return;
            };
            assert_eq!(pool.device_ordinal(), 0);
            let Ok(stream) = Stream::new(&ctx) else {
                return;
            };

            let host_in: Vec<f32> = (0..256).map(|i| i as f32).collect();
            let api = try_driver().expect("driver must be present under gpu-tests");

            let buf =
                PooledBuffer::<f32>::alloc_async(&pool, 256, &stream).expect("alloc_async failed");
            assert_eq!(buf.len(), 256);
            assert_eq!(buf.byte_size(), 256 * std::mem::size_of::<f32>());

            let rc = unsafe {
                (api.cu_memcpy_htod_v2)(
                    buf.as_device_ptr(),
                    host_in.as_ptr().cast::<c_void>(),
                    buf.byte_size(),
                )
            };
            check(rc).expect("HtoD copy failed");

            let mut host_out = vec![0.0f32; 256];
            let rc = unsafe {
                (api.cu_memcpy_dtoh_v2)(
                    host_out.as_mut_ptr().cast::<c_void>(),
                    buf.as_device_ptr(),
                    buf.byte_size(),
                )
            };
            check(rc).expect("DtoH copy failed");
            assert_eq!(host_out, host_in);

            drop(buf);
            let stats = pool.stats();
            assert_eq!(stats.allocation_count, 1);
            assert_eq!(stats.free_count, 1);
        }

        /// Drops a `PooledBuffer` and immediately requests a new allocation
        /// of the same size, then writes and reads back fresh data. This
        /// exercises the event-gated recycle path (F010): whether or not
        /// the pointer is physically reused, the second buffer's contents
        /// must reflect exactly what was written to it, never stale data
        /// from a still-in-flight free.
        #[test]
        fn memory_pool_reuse_after_drop_preserves_data_integrity() {
            let Some(ctx) = real_context() else {
                return;
            };
            let Ok(pool) = MemoryPool::new(0) else {
                return;
            };
            let Ok(stream) = Stream::new(&ctx) else {
                return;
            };
            let api = try_driver().expect("driver must be present under gpu-tests");

            let first_pattern: Vec<u32> = vec![0xAAAA_AAAA; 128];
            {
                let buf = PooledBuffer::<u32>::alloc_async(&pool, 128, &stream)
                    .expect("first alloc_async failed");
                let rc = unsafe {
                    (api.cu_memcpy_htod_v2)(
                        buf.as_device_ptr(),
                        first_pattern.as_ptr().cast::<c_void>(),
                        buf.byte_size(),
                    )
                };
                check(rc).expect("first HtoD copy failed");
                // `buf` drops here, enqueuing a recycle event on `stream`.
            }

            let second_pattern: Vec<u32> = (0..128u32).collect();
            let buf2 = PooledBuffer::<u32>::alloc_async(&pool, 128, &stream)
                .expect("second alloc_async failed");
            let rc = unsafe {
                (api.cu_memcpy_htod_v2)(
                    buf2.as_device_ptr(),
                    second_pattern.as_ptr().cast::<c_void>(),
                    buf2.byte_size(),
                )
            };
            check(rc).expect("second HtoD copy failed");

            let mut readback = vec![0u32; 128];
            let rc = unsafe {
                (api.cu_memcpy_dtoh_v2)(
                    readback.as_mut_ptr().cast::<c_void>(),
                    buf2.as_device_ptr(),
                    buf2.byte_size(),
                )
            };
            check(rc).expect("DtoH copy failed");
            assert_eq!(readback, second_pattern);

            let stats = pool.stats();
            assert_eq!(stats.allocation_count, 2);
            assert_eq!(stats.free_count, 1);
        }
    }
}
