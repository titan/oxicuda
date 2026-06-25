//! CUDA event management.
//!
//! Implements the CUDA Runtime event API:
//! - `cudaEventCreate` / `cudaEventCreateWithFlags`
//! - `cudaEventDestroy`
//! - `cudaEventRecord`
//! - `cudaEventSynchronize`
//! - `cudaEventQuery`
//! - `cudaEventElapsedTime`

use std::collections::HashSet;

use oxicuda_driver::ffi::CUevent;
use oxicuda_driver::loader::try_driver;

use crate::error::{CudaRtError, CudaRtResult};
use crate::stream::CudaStream;

// ─── EventFlags ──────────────────────────────────────────────────────────────

/// Flags for event creation.
///
/// Mirrors `cudaEventFlags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EventFlags(pub u32);

impl EventFlags {
    /// Default flags.
    pub const DEFAULT: Self = Self(0x0);
    /// Use blocking synchronisation: `cudaEventSynchronize` yields the CPU
    /// instead of spinning (mirrors `cudaEventBlockingSync`).
    pub const BLOCKING_SYNC: Self = Self(0x1);
    /// Event will not record timing data (lower overhead).
    pub const DISABLE_TIMING: Self = Self(0x2);
    /// Event can be used for interprocess synchronisation.
    pub const INTERPROCESS: Self = Self(0x4);
}

// ─── CudaEvent ───────────────────────────────────────────────────────────────

/// A CUDA event handle.
///
/// Wraps the raw `CUevent` from the driver API.  The event is **not**
/// automatically destroyed on drop — call [`event_destroy`] explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaEvent(CUevent);

impl CudaEvent {
    /// Construct from a raw driver event handle.
    ///
    /// # Safety
    ///
    /// The handle must be valid and not be used after the owning context is
    /// destroyed.
    #[must_use]
    pub const unsafe fn from_raw(raw: CUevent) -> Self {
        Self(raw)
    }

    /// Returns the underlying raw `CUevent`.
    #[must_use]
    pub fn raw(self) -> CUevent {
        self.0
    }

    /// Returns `true` if the event handle is null (invalid).
    #[must_use]
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
}

// ─── Event creation / destruction ────────────────────────────────────────────

/// Create a CUDA event with default flags.
///
/// Mirrors `cudaEventCreate`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn event_create() -> CudaRtResult<CudaEvent> {
    event_create_with_flags(EventFlags::DEFAULT)
}

/// Create a CUDA event with the given flags.
///
/// Mirrors `cudaEventCreateWithFlags`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn event_create_with_flags(flags: EventFlags) -> CudaRtResult<CudaEvent> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut event = CUevent::default();
    // SAFETY: FFI; event pointer is valid.
    let rc = unsafe { (api.cu_event_create)(&raw mut event, flags.0) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(CudaEvent(event))
}

/// Destroy a CUDA event.
///
/// Mirrors `cudaEventDestroy`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn event_destroy(event: CudaEvent) -> CudaRtResult<()> {
    if event.is_null() {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; event is valid.
    let rc = unsafe { (api.cu_event_destroy_v2)(event.raw()) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(())
}

// ─── Event recording and synchronisation ─────────────────────────────────────

/// Record `event` at the current position in `stream`.
///
/// Mirrors `cudaEventRecord`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn event_record(event: CudaEvent, stream: CudaStream) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; event and stream are valid.
    let rc = unsafe { (api.cu_event_record)(event.raw(), stream.raw()) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(())
}

/// Record `event` at the current position in `stream` with flags.
///
/// Mirrors `cudaEventRecordWithFlags`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn event_record_with_flags(
    event: CudaEvent,
    stream: CudaStream,
    flags: u32,
) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI. cu_event_record_with_flags is optional (CUDA 11.1+).
    let f = api
        .cu_event_record_with_flags
        .ok_or(CudaRtError::NotSupported)?;
    let rc = unsafe { f(event.raw(), stream.raw(), flags) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(())
}

/// Block the calling thread until `event` is recorded.
///
/// Mirrors `cudaEventSynchronize`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn event_synchronize(event: CudaEvent) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_event_synchronize)(event.raw()) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::NotReady));
    }
    Ok(())
}

/// Query whether `event` has been recorded.
///
/// Mirrors `cudaEventQuery`.
///
/// Returns `Ok(true)` if complete, `Ok(false)` if not yet reached.
///
/// # Errors
///
/// Propagates driver errors other than `NotReady`.
pub fn event_query(event: CudaEvent) -> CudaRtResult<bool> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_event_query)(event.raw()) };
    match rc {
        0 => Ok(true),
        600 => Ok(false), // CUDA_ERROR_NOT_READY
        other => Err(CudaRtError::from_code(other).unwrap_or(CudaRtError::Unknown)),
    }
}

/// Compute the elapsed time between two events in milliseconds.
///
/// Mirrors `cudaEventElapsedTime`.
///
/// Both events must have been recorded.  If either was created with
/// `EventFlags::DISABLE_TIMING`, this returns [`CudaRtError::InvalidResourceHandle`].
///
/// # Errors
///
/// Propagates driver errors.
pub fn event_elapsed_time(start: CudaEvent, end: CudaEvent) -> CudaRtResult<f32> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut ms: f32 = 0.0;
    // SAFETY: FFI; ms is a valid stack-allocated f32.
    let rc = unsafe { (api.cu_event_elapsed_time)(&raw mut ms, start.raw(), end.raw()) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(ms)
}

// ─── Host-side event id bookkeeping ───────────────────────────────────────────

/// A GPU-free allocator of unique event identifiers.
///
/// Mirrors [`crate::stream::StreamIdAllocator`] for `cudaEvent_t` handles:
/// pure host-side allocation tracking with strictly-monotonic, never-repeating
/// ids and accounted teardown, allocating **no device resources**. Ids start at
/// `1`.
#[derive(Debug, Default)]
pub struct EventIdAllocator {
    next_id: u64,
    live: HashSet<u64>,
}

impl EventIdAllocator {
    /// Create an empty allocator. The first id handed out is `1`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            live: HashSet::new(),
        }
    }

    /// Allocate a fresh, never-before-seen event id and mark it live.
    ///
    /// # Errors
    ///
    /// Returns [`CudaRtError::InvalidResourceHandle`] if the counter would
    /// overflow `u64` (practically unreachable).
    pub fn create(&mut self) -> CudaRtResult<u64> {
        let id = self.next_id;
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
    /// Returns [`CudaRtError::InvalidResourceHandle`] if `id` is not live
    /// (double-free or never-allocated).
    pub fn destroy(&mut self, id: u64) -> CudaRtResult<()> {
        if self.live.remove(&id) {
            Ok(())
        } else {
            Err(CudaRtError::InvalidResourceHandle)
        }
    }

    /// Number of currently-live event ids.
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
    fn event_flags_values() {
        assert_eq!(EventFlags::DEFAULT.0, 0x0);
        assert_eq!(EventFlags::BLOCKING_SYNC.0, 0x1);
        assert_eq!(EventFlags::DISABLE_TIMING.0, 0x2);
        assert_eq!(EventFlags::INTERPROCESS.0, 0x4);
    }

    #[test]
    fn event_create_without_gpu_returns_error() {
        // Without a GPU this will fail, but must not panic.
        let _ = event_create();
    }

    #[test]
    fn event_destroy_null_is_noop() {
        // SAFETY: null is a well-defined state; no FFI call made.
        let ev = unsafe { CudaEvent::from_raw(CUevent::default()) };
        let _ = event_destroy(ev); // must not panic
    }

    #[test]
    fn event_id_allocator_starts_at_one() {
        let mut alloc = EventIdAllocator::new();
        assert_eq!(alloc.peek_next_id(), 1);
        assert_eq!(alloc.create().expect("create"), 1);
        assert_eq!(alloc.live_count(), 1);
    }

    #[test]
    fn event_id_allocator_rejects_double_free() {
        let mut alloc = EventIdAllocator::new();
        let id = alloc.create().expect("create");
        alloc.destroy(id).expect("first destroy ok");
        assert_eq!(alloc.destroy(id), Err(CudaRtError::InvalidResourceHandle));
        assert_eq!(
            alloc.destroy(123_456),
            Err(CudaRtError::InvalidResourceHandle)
        );
    }

    #[test]
    fn event_stress_create_destroy_10k_no_collision() {
        // Host-only bookkeeping stress: create and destroy 10,000 events,
        // asserting monotonic ids, zero collisions, and clean teardown.
        const N: usize = 10_000;
        let mut alloc = EventIdAllocator::new();
        let mut seen: HashSet<u64> = HashSet::with_capacity(N);
        let mut prev: u64 = 0;
        for _ in 0..N {
            let id = alloc.create().expect("create must succeed (host-only)");
            assert!(id > prev, "ids must be strictly increasing: {id} <= {prev}");
            prev = id;
            assert!(seen.insert(id), "duplicate event id {id}");
            alloc.destroy(id).expect("destroy must succeed");
        }
        assert_eq!(alloc.live_count(), 0);
        assert_eq!(seen.len(), N);
        assert_eq!(alloc.peek_next_id(), (N as u64) + 1);
    }

    #[test]
    fn event_stress_retain_then_teardown_10k() {
        // Variant: keep all 10,000 live, then tear down — asserts no id is lost
        // and teardown fully drains the live set.
        const N: usize = 10_000;
        let mut alloc = EventIdAllocator::new();
        let mut ids: Vec<u64> = Vec::with_capacity(N);
        for _ in 0..N {
            ids.push(alloc.create().expect("create"));
        }
        assert_eq!(alloc.live_count(), N);
        // All ids distinct.
        let distinct: HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(distinct.len(), N);
        // Tear down every id; each destroy must succeed exactly once.
        for id in ids {
            alloc.destroy(id).expect("destroy live id");
        }
        assert_eq!(alloc.live_count(), 0);
    }
}
