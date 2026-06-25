//! CUPTI-lite — runtime-loaded CUDA Profiling Tools Interface stubs.
//!
//! The CUDA Profiling Tools Interface (`libcupti.so`) ships alongside the
//! driver and exposes the *Activity API* — an asynchronous, buffer-based
//! firehose of timestamped records (kernel launches, memcpys, API calls) that
//! tools like Nsight Systems consume.  Just like [`crate::loader`] loads the
//! driver itself, this module loads CUPTI **at runtime** via
//! [`libloading`](https://crates.io/crates/libloading) — no `cupti.h`, no
//! link-time dependency — and resolves the three pillars of the Activity API:
//!
//! | Symbol                             | Role                                    |
//! |------------------------------------|-----------------------------------------|
//! | `cuptiActivityEnable`              | turn a record kind on/off               |
//! | `cuptiActivityRegisterCallbacks`   | hand CUPTI buffer-alloc / drain hooks   |
//! | `cuptiActivityFlushAll`            | force completion of pending buffers     |
//!
//! # Two layers
//!
//! 1. **`CuptiLibrary`** — the libloading layer.  Searches the platform's
//!    CUPTI library names and resolves the activity entry points.  On a host
//!    without CUPTI installed (every CI box here) `CuptiLibrary::load`
//!    returns a [`DriverLoadError`](crate::error::DriverLoadError), which is the honest, testable behaviour —
//!    we never fabricate a "profiler attached" success.
//!
//! 2. **[`ActivitySession`]** — a host-side behavioural model of the Activity
//!    API's buffer protocol.  CUPTI's double-buffer ownership transfer (it asks
//!    the tool for a buffer, fills it, hands it back on flush) is a precise
//!    state machine that is fully deterministic and therefore CPU-testable
//!    without any GPU or CUPTI present.  This is what lets the rest of OxiCUDA
//!    unit-test profiling logic offline.

#[cfg(not(target_os = "macos"))]
use crate::error::DriverLoadError;
#[cfg(not(target_os = "macos"))]
use libloading::Library;

// ---------------------------------------------------------------------------
// CuptiActivityKind
// ---------------------------------------------------------------------------

/// The category of activity record requested via `cuptiActivityEnable`.
///
/// Mirrors the most commonly used members of the C enum `CUpti_ActivityKind`.
/// The discriminants match the CUPTI header so a future hardware-backed path
/// can pass them straight through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CuptiActivityKind {
    /// Memory copies between host and device (`CUPTI_ACTIVITY_KIND_MEMCPY`).
    Memcpy = 1,
    /// Device-side memory sets (`CUPTI_ACTIVITY_KIND_MEMSET`).
    Memset = 2,
    /// Kernel executions (`CUPTI_ACTIVITY_KIND_KERNEL`).
    Kernel = 3,
    /// Driver API call durations (`CUPTI_ACTIVITY_KIND_DRIVER`).
    Driver = 4,
    /// Runtime API call durations (`CUPTI_ACTIVITY_KIND_RUNTIME`).
    Runtime = 5,
    /// Concurrent kernel executions (`CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL`).
    ConcurrentKernel = 6,
    /// CUDA stream / device / context names (`CUPTI_ACTIVITY_KIND_NAME`).
    Name = 7,
    /// User / driver markers (`CUPTI_ACTIVITY_KIND_MARKER`).
    Marker = 8,
    /// Stream-ordered memory pool operations
    /// (`CUPTI_ACTIVITY_KIND_MEMORY_POOL`).
    MemoryPool = 9,
}

impl CuptiActivityKind {
    /// The raw `CUpti_ActivityKind` discriminant.
    #[must_use]
    pub fn raw(self) -> u32 {
        self as u32
    }
}

// ---------------------------------------------------------------------------
// CuptiLibrary — the libloading layer
// ---------------------------------------------------------------------------

/// Candidate file names for the CUPTI shared library, in search order.
///
/// CUPTI is versioned with the CUDA major release; we try the unversioned
/// `SONAME` first, then a spread of recent CUDA majors.
#[cfg(not(target_os = "macos"))]
const CUPTI_LIBRARY_NAMES: &[&str] = &[
    "libcupti.so",
    "libcupti.so.12",
    "libcupti.so.11",
    "cupti64_2025.1.0.dll",
    "cupti64_2024.1.0.dll",
];

/// A runtime-loaded handle to `libcupti.so` exposing the Activity API.
///
/// Loading is entirely lazy and fallible: there is no CUPTI on a machine
/// without the CUDA toolkit profiling components, in which case
/// [`CuptiLibrary::load`] returns an `Err`.  The handle keeps the [`Library`]
/// alive for the lifetime of the resolved function pointers.
#[cfg(not(target_os = "macos"))]
pub struct CuptiLibrary {
    /// Keeps the dynamic library mapped; the resolved fn pointers borrow it.
    _lib: Library,
    has_activity_enable: bool,
    has_register_callbacks: bool,
    has_flush_all: bool,
}

#[cfg(not(target_os = "macos"))]
impl CuptiLibrary {
    /// Attempt to load CUPTI and confirm the Activity API symbols are present.
    ///
    /// # Errors
    ///
    /// * [`DriverLoadError::LibraryNotFound`] if no CUPTI library can be
    ///   opened (the common case on a host without the profiling components).
    /// * [`DriverLoadError::SymbolNotFound`] if the library opens but is
    ///   missing a required Activity API entry point.
    pub fn load() -> Result<Self, DriverLoadError> {
        let mut last_error = String::new();
        let lib = 'open: {
            for name in CUPTI_LIBRARY_NAMES {
                // SAFETY: opening a shared library runs its initialisers; CUPTI
                // is designed to be dlopen'd by profiling tools.
                match unsafe { Library::new(*name) } {
                    Ok(lib) => break 'open lib,
                    Err(e) => last_error = e.to_string(),
                }
            }
            return Err(DriverLoadError::LibraryNotFound {
                candidates: CUPTI_LIBRARY_NAMES
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
                last_error,
            });
        };

        // Probe the three Activity API symbols.  We only record presence here;
        // calling them requires the matching C ABI, which is out of scope for
        // the host model and gated behind real hardware.
        let has_activity_enable = Self::has_symbol(&lib, b"cuptiActivityEnable");
        let has_register_callbacks = Self::has_symbol(&lib, b"cuptiActivityRegisterCallbacks");
        let has_flush_all = Self::has_symbol(&lib, b"cuptiActivityFlushAll");

        if !has_activity_enable {
            return Err(DriverLoadError::SymbolNotFound {
                symbol: "cuptiActivityEnable",
                reason: "symbol absent from loaded CUPTI library".to_string(),
            });
        }

        Ok(Self {
            _lib: lib,
            has_activity_enable,
            has_register_callbacks,
            has_flush_all,
        })
    }

    /// Whether a named symbol resolves in the given library.
    fn has_symbol(lib: &Library, name: &[u8]) -> bool {
        // SAFETY: we never call the resolved pointer here; we only test for
        // its existence.  The fn signature is a placeholder of the right shape.
        unsafe { lib.get::<unsafe extern "C" fn()>(name) }.is_ok()
    }

    /// Whether `cuptiActivityEnable` resolved.
    #[must_use]
    pub fn supports_activity_enable(&self) -> bool {
        self.has_activity_enable
    }

    /// Whether `cuptiActivityRegisterCallbacks` resolved.
    #[must_use]
    pub fn supports_register_callbacks(&self) -> bool {
        self.has_register_callbacks
    }

    /// Whether `cuptiActivityFlushAll` resolved.
    #[must_use]
    pub fn supports_flush_all(&self) -> bool {
        self.has_flush_all
    }
}

/// Convenience: attempt to load CUPTI, returning whether it is available.
///
/// Never panics; on any platform without CUPTI it simply returns `false`.
#[must_use]
pub fn cupti_available() -> bool {
    #[cfg(not(target_os = "macos"))]
    {
        CuptiLibrary::load().is_ok()
    }
    #[cfg(target_os = "macos")]
    {
        false
    }
}

// ---------------------------------------------------------------------------
// ActivityRecord — the modelled record payload
// ---------------------------------------------------------------------------

/// A single timestamped activity record, the host-model analogue of one
/// `CUpti_Activity*` struct drained from a completed buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityRecord {
    /// The record category.
    pub kind: CuptiActivityKind,
    /// Start timestamp in nanoseconds (device clock domain).
    pub start_ns: u64,
    /// End timestamp in nanoseconds.
    pub end_ns: u64,
    /// Correlation id linking this record to its launching API call.
    pub correlation_id: u32,
}

impl ActivityRecord {
    /// Duration of the activity in nanoseconds (`end - start`, saturating).
    #[must_use]
    pub fn duration_ns(&self) -> u64 {
        self.end_ns.saturating_sub(self.start_ns)
    }
}

// ---------------------------------------------------------------------------
// ActivitySession — host-side Activity-buffer state machine
// ---------------------------------------------------------------------------

/// Errors specific to the modelled Activity buffer protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CuptiActivityError {
    /// An operation requires callbacks that were never registered.
    #[error("activity buffer callbacks not registered")]
    CallbacksNotRegistered,
    /// A record kind was emitted while disabled (CUPTI would silently drop it).
    #[error("activity kind {kind:?} is not enabled")]
    KindNotEnabled {
        /// The kind that was rejected.
        kind: CuptiActivityKind,
    },
    /// The pending buffer is full and must be flushed before more records fit.
    #[error("activity buffer full ({capacity} records); flush required")]
    BufferFull {
        /// The fixed record capacity of one buffer.
        capacity: usize,
    },
}

/// A host-side model of CUPTI's asynchronous Activity buffer protocol.
///
/// The real flow is: the tool registers a *requested* callback (CUPTI calls it
/// to obtain an empty buffer) and a *completed* callback (CUPTI calls it to
/// hand back a full buffer for draining), enables one or more
/// [`CuptiActivityKind`]s, runs GPU work (which fills buffers), then calls
/// `cuptiActivityFlushAll` to force any partial buffer to be completed.
///
/// This struct reproduces that ownership dance deterministically: `record`
/// fills the pending buffer, `flush` (and a full buffer) moves records into the
/// *completed* queue which [`ActivitySession::drain`] empties — modelling the
/// completed-buffer callback delivering records to the tool.
#[derive(Debug)]
pub struct ActivitySession {
    enabled: Vec<CuptiActivityKind>,
    callbacks_registered: bool,
    buffer_capacity: usize,
    pending: Vec<ActivityRecord>,
    completed: Vec<ActivityRecord>,
    dropped: u64,
}

impl ActivitySession {
    /// Default per-buffer record capacity, echoing CUPTI's habit of using a
    /// few-MB buffer that holds many fixed-size records.
    pub const DEFAULT_BUFFER_CAPACITY: usize = 256;

    /// Create an idle session with no kinds enabled and no callbacks set.
    #[must_use]
    pub fn new() -> Self {
        Self::with_buffer_capacity(Self::DEFAULT_BUFFER_CAPACITY)
    }

    /// Create a session with an explicit per-buffer record capacity.
    #[must_use]
    pub fn with_buffer_capacity(buffer_capacity: usize) -> Self {
        Self {
            enabled: Vec::new(),
            callbacks_registered: false,
            buffer_capacity: buffer_capacity.max(1),
            pending: Vec::new(),
            completed: Vec::new(),
            dropped: 0,
        }
    }

    /// Model `cuptiActivityRegisterCallbacks`: install the buffer request /
    /// complete hooks.  Idempotent.
    pub fn register_callbacks(&mut self) {
        self.callbacks_registered = true;
    }

    /// Whether buffer callbacks have been registered.
    #[must_use]
    pub fn callbacks_registered(&self) -> bool {
        self.callbacks_registered
    }

    /// Model `cuptiActivityEnable(kind)`: turn a record kind on.
    ///
    /// # Errors
    ///
    /// [`CuptiActivityError::CallbacksNotRegistered`] — CUPTI rejects enabling
    /// activity before buffer callbacks exist (there would be nowhere to put
    /// the records).
    pub fn enable(&mut self, kind: CuptiActivityKind) -> Result<(), CuptiActivityError> {
        if !self.callbacks_registered {
            return Err(CuptiActivityError::CallbacksNotRegistered);
        }
        if !self.enabled.contains(&kind) {
            self.enabled.push(kind);
        }
        Ok(())
    }

    /// Model `cuptiActivityDisable(kind)`: turn a record kind off.
    pub fn disable(&mut self, kind: CuptiActivityKind) {
        self.enabled.retain(|k| *k != kind);
    }

    /// Whether the given kind is currently enabled.
    #[must_use]
    pub fn is_enabled(&self, kind: CuptiActivityKind) -> bool {
        self.enabled.contains(&kind)
    }

    /// The set of currently enabled kinds.
    #[must_use]
    pub fn enabled_kinds(&self) -> &[CuptiActivityKind] {
        &self.enabled
    }

    /// Emit one activity record into the pending buffer.
    ///
    /// Models the GPU filling a buffer obtained from the requested callback.
    /// If the record's kind is not enabled it is silently dropped (CUPTI does
    /// the same) and counted in [`ActivitySession::dropped_records`].
    ///
    /// # Errors
    ///
    /// * [`CuptiActivityError::CallbacksNotRegistered`] if no callbacks exist.
    /// * [`CuptiActivityError::BufferFull`] if the pending buffer is already at
    ///   capacity — the caller must [`ActivitySession::flush`] first.
    pub fn record(&mut self, record: ActivityRecord) -> Result<(), CuptiActivityError> {
        if !self.callbacks_registered {
            return Err(CuptiActivityError::CallbacksNotRegistered);
        }
        if !self.enabled.contains(&record.kind) {
            self.dropped = self.dropped.saturating_add(1);
            return Ok(());
        }
        if self.pending.len() >= self.buffer_capacity {
            return Err(CuptiActivityError::BufferFull {
                capacity: self.buffer_capacity,
            });
        }
        self.pending.push(record);
        Ok(())
    }

    /// Model `cuptiActivityFlushAll`: move every pending record into the
    /// completed queue, as CUPTI does when it hands a full / forced buffer to
    /// the completed callback.  Returns the number of records flushed.
    ///
    /// # Errors
    ///
    /// [`CuptiActivityError::CallbacksNotRegistered`] if no callbacks exist —
    /// `cuptiActivityFlushAll` is meaningless without a completed callback to
    /// deliver buffers to.
    pub fn flush(&mut self) -> Result<usize, CuptiActivityError> {
        if !self.callbacks_registered {
            return Err(CuptiActivityError::CallbacksNotRegistered);
        }
        let n = self.pending.len();
        self.completed.append(&mut self.pending);
        Ok(n)
    }

    /// Drain the completed queue, returning all records delivered to the tool
    /// since the last drain.  Models the tool consuming the completed buffer.
    pub fn drain(&mut self) -> Vec<ActivityRecord> {
        std::mem::take(&mut self.completed)
    }

    /// Number of records currently waiting in the (un-flushed) pending buffer.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Number of records flushed and waiting to be drained.
    #[must_use]
    pub fn completed_len(&self) -> usize {
        self.completed.len()
    }

    /// Count of records dropped because their kind was disabled.
    #[must_use]
    pub fn dropped_records(&self) -> u64 {
        self.dropped
    }
}

impl Default for ActivitySession {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(kind: CuptiActivityKind, start: u64, end: u64, cid: u32) -> ActivityRecord {
        ActivityRecord {
            kind,
            start_ns: start,
            end_ns: end,
            correlation_id: cid,
        }
    }

    #[test]
    fn activity_kind_raw_discriminants() {
        assert_eq!(CuptiActivityKind::Memcpy.raw(), 1);
        assert_eq!(CuptiActivityKind::Kernel.raw(), 3);
        assert_eq!(CuptiActivityKind::ConcurrentKernel.raw(), 6);
        assert_eq!(CuptiActivityKind::MemoryPool.raw(), 9);
    }

    #[test]
    fn cupti_available_does_not_panic() {
        // On CI there is no libcupti; this must return false (or true on a box
        // that happens to have it) without panicking.
        let _ = cupti_available();
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn load_without_cupti_yields_library_not_found() {
        // The CI host has no CUPTI; the load path must fail honestly with
        // LibraryNotFound rather than fabricating a session.
        match CuptiLibrary::load() {
            Err(DriverLoadError::LibraryNotFound { candidates, .. }) => {
                assert!(candidates.iter().any(|c| c.contains("cupti")));
            }
            Err(DriverLoadError::SymbolNotFound { .. }) => {
                // Acceptable if a stub libcupti exists but lacks symbols.
            }
            Ok(lib) => {
                // If CUPTI genuinely exists, enable must have resolved.
                assert!(lib.supports_activity_enable());
            }
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn record_requires_registered_callbacks() {
        let mut s = ActivitySession::new();
        let r = rec(CuptiActivityKind::Kernel, 0, 100, 1);
        assert_eq!(s.record(r), Err(CuptiActivityError::CallbacksNotRegistered));
    }

    #[test]
    fn enable_requires_registered_callbacks() {
        let mut s = ActivitySession::new();
        assert_eq!(
            s.enable(CuptiActivityKind::Kernel),
            Err(CuptiActivityError::CallbacksNotRegistered)
        );
    }

    #[test]
    fn flush_requires_registered_callbacks() {
        let mut s = ActivitySession::new();
        assert_eq!(s.flush(), Err(CuptiActivityError::CallbacksNotRegistered));
    }

    #[test]
    fn full_enable_record_flush_drain_cycle() {
        let mut s = ActivitySession::new();
        s.register_callbacks();
        s.enable(CuptiActivityKind::Kernel).unwrap();
        s.enable(CuptiActivityKind::Memcpy).unwrap();

        s.record(rec(CuptiActivityKind::Kernel, 0, 500, 1)).unwrap();
        s.record(rec(CuptiActivityKind::Memcpy, 500, 700, 2))
            .unwrap();
        assert_eq!(s.pending_len(), 2);
        assert_eq!(s.completed_len(), 0);

        let flushed = s.flush().unwrap();
        assert_eq!(flushed, 2);
        assert_eq!(s.pending_len(), 0);
        assert_eq!(s.completed_len(), 2);

        let records = s.drain();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].duration_ns(), 500);
        assert_eq!(records[1].duration_ns(), 200);
        assert_eq!(s.completed_len(), 0);
    }

    #[test]
    fn disabled_kind_is_dropped_and_counted() {
        let mut s = ActivitySession::new();
        s.register_callbacks();
        s.enable(CuptiActivityKind::Kernel).unwrap();
        // Memcpy is NOT enabled → dropped.
        s.record(rec(CuptiActivityKind::Memcpy, 0, 10, 1)).unwrap();
        s.record(rec(CuptiActivityKind::Kernel, 0, 10, 2)).unwrap();
        assert_eq!(s.pending_len(), 1);
        assert_eq!(s.dropped_records(), 1);
    }

    #[test]
    fn disable_stops_recording() {
        let mut s = ActivitySession::new();
        s.register_callbacks();
        s.enable(CuptiActivityKind::Kernel).unwrap();
        assert!(s.is_enabled(CuptiActivityKind::Kernel));
        s.disable(CuptiActivityKind::Kernel);
        assert!(!s.is_enabled(CuptiActivityKind::Kernel));
        s.record(rec(CuptiActivityKind::Kernel, 0, 10, 1)).unwrap();
        assert_eq!(s.pending_len(), 0);
        assert_eq!(s.dropped_records(), 1);
    }

    #[test]
    fn buffer_full_requires_flush() {
        let mut s = ActivitySession::with_buffer_capacity(2);
        s.register_callbacks();
        s.enable(CuptiActivityKind::Kernel).unwrap();
        s.record(rec(CuptiActivityKind::Kernel, 0, 1, 1)).unwrap();
        s.record(rec(CuptiActivityKind::Kernel, 1, 2, 2)).unwrap();
        // Third record overflows the 2-slot buffer.
        assert_eq!(
            s.record(rec(CuptiActivityKind::Kernel, 2, 3, 3)),
            Err(CuptiActivityError::BufferFull { capacity: 2 })
        );
        // After flushing, there is room again.
        s.flush().unwrap();
        s.record(rec(CuptiActivityKind::Kernel, 2, 3, 3)).unwrap();
        assert_eq!(s.pending_len(), 1);
    }

    #[test]
    fn enable_is_idempotent() {
        let mut s = ActivitySession::new();
        s.register_callbacks();
        s.enable(CuptiActivityKind::Kernel).unwrap();
        s.enable(CuptiActivityKind::Kernel).unwrap();
        assert_eq!(s.enabled_kinds().len(), 1);
    }

    #[test]
    fn register_callbacks_is_idempotent() {
        let mut s = ActivitySession::new();
        s.register_callbacks();
        s.register_callbacks();
        assert!(s.callbacks_registered());
    }

    #[test]
    fn buffer_capacity_floors_at_one() {
        let s = ActivitySession::with_buffer_capacity(0);
        // capacity 0 would be unusable; it must floor to 1.
        assert_eq!(s.pending_len(), 0);
        let mut s = s;
        s.register_callbacks();
        s.enable(CuptiActivityKind::Kernel).unwrap();
        s.record(rec(CuptiActivityKind::Kernel, 0, 1, 1)).unwrap();
        assert_eq!(
            s.record(rec(CuptiActivityKind::Kernel, 1, 2, 2)),
            Err(CuptiActivityError::BufferFull { capacity: 1 })
        );
    }

    #[test]
    fn duration_saturates_on_inverted_timestamps() {
        let r = rec(CuptiActivityKind::Kernel, 100, 50, 1);
        assert_eq!(r.duration_ns(), 0);
    }
}
