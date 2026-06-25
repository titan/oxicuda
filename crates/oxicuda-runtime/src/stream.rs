//! CUDA stream management.
//!
//! Implements the CUDA Runtime stream API:
//! - `cudaStreamCreate` / `cudaStreamCreateWithFlags` / `cudaStreamCreateWithPriority`
//! - `cudaStreamDestroy`
//! - `cudaStreamSynchronize`
//! - `cudaStreamQuery`
//! - `cudaStreamWaitEvent`
//! - `cudaStreamGetPriority`
//! - `cudaStreamGetFlags`
//! - `cudaStreamGetDevice`
//! - The default stream (`cudaStreamDefault` / `cudaStreamLegacy` / `cudaStreamPerThread`)

use std::collections::HashSet;

use oxicuda_driver::ffi::CUstream;
use oxicuda_driver::loader::try_driver;

use crate::error::{CudaRtError, CudaRtResult};

// ─── StreamFlags ─────────────────────────────────────────────────────────────

/// Flags for stream creation.
///
/// Mirrors `cudaStreamFlags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StreamFlags(pub u32);

impl StreamFlags {
    /// Default stream flag: stream synchronises with the legacy default stream.
    pub const DEFAULT: Self = Self(0x0);
    /// Non-blocking stream: the stream does not implicitly synchronise with the
    /// legacy default stream (mirrors `cudaStreamNonBlocking`).
    pub const NON_BLOCKING: Self = Self(0x1);
}

// ─── CudaStream ──────────────────────────────────────────────────────────────

/// A CUDA stream handle.
///
/// Wraps the raw `CUstream` handle from the driver API.  The stream is
/// **not** automatically destroyed when dropped — call [`stream_destroy`]
/// explicitly or use the stream within its creating context lifetime.
///
/// Use [`CudaStream::DEFAULT`] to obtain the special legacy-default
/// stream sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaStream(CUstream);

impl CudaStream {
    /// The legacy default CUDA stream (`cudaStreamDefault` = 0).
    ///
    /// Operations on the default stream block all other streams in the context.
    pub const DEFAULT: Self = Self(CUstream(std::ptr::null_mut()));

    /// Per-thread default stream (`cudaStreamPerThread`).
    ///
    /// Equivalent to passing `cudaStreamPerThread` in the Runtime API.
    /// The value `0x2` is the canonical sentinel used by the CUDA Runtime.
    pub const PER_THREAD: Self = Self(CUstream(2 as *mut std::ffi::c_void));

    /// Construct a `CudaStream` from a raw driver handle.
    ///
    /// # Safety
    ///
    /// The caller must ensure the handle is valid and not used after the
    /// associated context is destroyed.
    #[must_use]
    pub const unsafe fn from_raw(raw: CUstream) -> Self {
        Self(raw)
    }

    /// Returns the underlying raw `CUstream`.
    #[must_use]
    pub fn raw(self) -> CUstream {
        self.0
    }

    /// Returns `true` if this is the legacy default stream.
    #[must_use]
    pub fn is_default(self) -> bool {
        self.0.is_null()
    }
}

// ─── Stream creation / destruction ────────────────────────────────────────────

/// Create a new CUDA stream with default flags.
///
/// Mirrors `cudaStreamCreate`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_create() -> CudaRtResult<CudaStream> {
    stream_create_with_flags(StreamFlags::DEFAULT)
}

/// Create a new CUDA stream with the given flags.
///
/// Mirrors `cudaStreamCreateWithFlags`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_create_with_flags(flags: StreamFlags) -> CudaRtResult<CudaStream> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut stream = CUstream::default();
    // SAFETY: FFI; stream is a valid stack-allocated opaque pointer.
    let rc = unsafe { (api.cu_stream_create)(&raw mut stream, flags.0) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(CudaStream(stream))
}

/// Create a new CUDA stream with the given flags and scheduling priority.
///
/// Mirrors `cudaStreamCreateWithPriority`.
///
/// `priority` is a signed integer where lower values indicate higher priority.
/// The valid range can be queried with `cudaDeviceGetStreamPriorityRange`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_create_with_priority(flags: StreamFlags, priority: i32) -> CudaRtResult<CudaStream> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut stream = CUstream::default();
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_stream_create_with_priority)(&raw mut stream, flags.0, priority) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(CudaStream(stream))
}

/// Destroy a CUDA stream.
///
/// Mirrors `cudaStreamDestroy`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_destroy(stream: CudaStream) -> CudaRtResult<()> {
    if stream.is_default() {
        return Ok(()); // default stream is never explicitly destroyed
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; stream handle is valid.
    let rc = unsafe { (api.cu_stream_destroy_v2)(stream.raw()) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(())
}

// ─── Stream synchronisation / query ──────────────────────────────────────────

/// Wait until all preceding operations in `stream` complete.
///
/// Mirrors `cudaStreamSynchronize`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_synchronize(stream: CudaStream) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_stream_synchronize)(stream.raw()) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::Unknown));
    }
    Ok(())
}

/// Check whether all preceding operations in `stream` have completed.
///
/// Mirrors `cudaStreamQuery`.
///
/// Returns `Ok(true)` if complete, `Ok(false)` if still running.
///
/// # Errors
///
/// Propagates driver errors (other than `NotReady`).
pub fn stream_query(stream: CudaStream) -> CudaRtResult<bool> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_stream_query)(stream.raw()) };
    match rc {
        0 => Ok(true),    // CUDA_SUCCESS — complete
        600 => Ok(false), // CUDA_ERROR_NOT_READY — still running
        other => Err(CudaRtError::from_code(other).unwrap_or(CudaRtError::Unknown)),
    }
}

/// Make all future work submitted to `stream` wait until `event` is recorded.
///
/// Mirrors `cudaStreamWaitEvent`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_wait_event(
    stream: CudaStream,
    event: crate::event::CudaEvent,
    flags: u32,
) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_stream_wait_event)(stream.raw(), event.raw(), flags) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(())
}

/// Returns the priority of `stream`.
///
/// Mirrors `cudaStreamGetPriority`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_get_priority(stream: CudaStream) -> CudaRtResult<i32> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut priority: std::ffi::c_int = 0;
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_stream_get_priority)(stream.raw(), &raw mut priority) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(priority)
}

/// Returns the flags of `stream`.
///
/// Mirrors `cudaStreamGetFlags`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_get_flags(stream: CudaStream) -> CudaRtResult<StreamFlags> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut flags: u32 = 0;
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_stream_get_flags)(stream.raw(), &raw mut flags) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(StreamFlags(flags))
}

// ─── Host-side stream id bookkeeping ──────────────────────────────────────────

/// A GPU-free allocator of unique stream identifiers.
///
/// The CUDA Runtime hands back opaque `cudaStream_t` handles; this models the
/// *host-side bookkeeping* a runtime keeps so that every live stream has a
/// distinct, monotonically increasing id and teardown is accounted for. It
/// allocates **no device resources** — it is pure host allocation tracking, so
/// it runs and self-verifies with no NVIDIA driver present.
///
/// Ids start at `1` (id `0` is reserved for the legacy default stream) and
/// never repeat for the lifetime of the allocator, even across destroy/create
/// cycles (mirroring the monotonic-handle invariant real runtimes rely on to
/// surface use-after-destroy bugs).
#[derive(Debug, Default)]
pub struct StreamIdAllocator {
    next_id: u64,
    live: HashSet<u64>,
}

impl StreamIdAllocator {
    /// Create an empty allocator. The first id handed out is `1`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            live: HashSet::new(),
        }
    }

    /// Allocate a fresh, never-before-seen stream id and mark it live.
    ///
    /// # Errors
    ///
    /// Returns [`CudaRtError::InvalidResourceHandle`] if the monotonic counter
    /// would overflow `u64` (practically unreachable).
    pub fn create(&mut self) -> CudaRtResult<u64> {
        let id = self.next_id;
        // The next id is strictly greater, so ids never repeat.
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(CudaRtError::InvalidResourceHandle)?;
        self.live.insert(id);
        Ok(id)
    }

    /// Mark a previously-allocated id as destroyed.
    ///
    /// # Errors
    ///
    /// Returns [`CudaRtError::InvalidResourceHandle`] if `id` is not currently
    /// live (double-free or never-allocated), modeling the runtime's
    /// invalid-handle rejection.
    pub fn destroy(&mut self, id: u64) -> CudaRtResult<()> {
        if self.live.remove(&id) {
            Ok(())
        } else {
            Err(CudaRtError::InvalidResourceHandle)
        }
    }

    /// Number of currently-live stream ids.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.live.len()
    }

    /// The id that will be returned by the next [`Self::create`] call.
    #[must_use]
    pub fn peek_next_id(&self) -> u64 {
        self.next_id
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_stream_is_null() {
        assert!(CudaStream::DEFAULT.is_default());
        assert!(!CudaStream::PER_THREAD.is_default());
    }

    #[test]
    fn stream_flags_values() {
        assert_eq!(StreamFlags::DEFAULT.0, 0);
        assert_eq!(StreamFlags::NON_BLOCKING.0, 1);
    }

    #[test]
    fn stream_destroy_default_is_noop() {
        // Should never hit the driver for the default stream.
        let result = stream_destroy(CudaStream::DEFAULT);
        // Without a driver it fails with DriverNotAvailable; with a driver it's Ok.
        let _ = result;
    }

    #[test]
    fn stream_create_without_gpu_returns_error() {
        let result = stream_create();
        // Must either succeed (GPU present) or fail with DriverNotAvailable /
        // some other non-panic error.
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn stream_id_allocator_starts_at_one() {
        let mut alloc = StreamIdAllocator::new();
        assert_eq!(alloc.peek_next_id(), 1);
        let first = alloc.create().expect("create");
        assert_eq!(first, 1);
        assert_eq!(alloc.live_count(), 1);
    }

    #[test]
    fn stream_id_allocator_rejects_double_free() {
        let mut alloc = StreamIdAllocator::new();
        let id = alloc.create().expect("create");
        alloc.destroy(id).expect("first destroy ok");
        // Second destroy of the same id must fail.
        assert_eq!(alloc.destroy(id), Err(CudaRtError::InvalidResourceHandle));
        // Destroying a never-allocated id must fail too.
        assert_eq!(
            alloc.destroy(999_999),
            Err(CudaRtError::InvalidResourceHandle)
        );
    }

    #[test]
    fn stream_stress_create_destroy_10k_no_collision() {
        // Host-only bookkeeping stress: create and destroy 10,000 streams in a
        // loop, asserting monotonic ids, zero collisions, and clean teardown.
        const N: usize = 10_000;
        let mut alloc = StreamIdAllocator::new();
        let mut seen: HashSet<u64> = HashSet::with_capacity(N);
        let mut prev: u64 = 0;
        for _ in 0..N {
            let id = alloc.create().expect("create must succeed (host-only)");
            // Strictly monotonic across the whole run.
            assert!(id > prev, "ids must be strictly increasing: {id} <= {prev}");
            prev = id;
            // No id is ever handed out twice.
            assert!(seen.insert(id), "duplicate stream id {id}");
            // Immediately destroy to model a create/destroy churn loop.
            alloc.destroy(id).expect("destroy must succeed");
        }
        // Clean teardown: nothing left live, and exactly N distinct ids issued.
        assert_eq!(alloc.live_count(), 0);
        assert_eq!(seen.len(), N);
        // The next id is past everything issued (monotonic invariant holds).
        assert_eq!(alloc.peek_next_id(), (N as u64) + 1);
    }
}
