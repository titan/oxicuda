//! CUDA context management with RAII semantics.
//!
//! A CUDA **context** is the primary interface through which a CPU thread
//! interacts with a GPU. It owns driver state such as loaded modules, allocated
//! memory, and streams. This module provides the [`Context`] type, an RAII
//! wrapper around `CUcontext` that automatically calls `cuCtxDestroy` on drop.
//!
//! # Thread safety
//!
//! The CUDA Driver API is thread-safe, and (since CUDA 4.0) a context may be
//! current on multiple threads simultaneously; "current-ness" is a per-thread
//! property set with `cuCtxSetCurrent`. Accordingly [`Context`] is both
//! [`Send`] and [`Sync`] (auto-derived from its fields), so it can be wrapped
//! in an [`Arc<Context>`](std::sync::Arc) and shared across threads. Each
//! thread that issues driver calls must first make the context current on
//! itself via [`set_current`](Context::set_current).
//!
//! # Examples
//!
//! ```no_run
//! use oxicuda_driver::context::Context;
//! use oxicuda_driver::device::Device;
//!
//! oxicuda_driver::init()?;
//! let device = Device::get(0)?;
//! let ctx = Context::new(&device)?;
//! ctx.set_current()?;
//! // ... launch kernels, allocate memory ...
//! ctx.synchronize()?;
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::device::Device;
use crate::error::CudaResult;
use crate::ffi::CUcontext;
use crate::loader::try_driver;

// ---------------------------------------------------------------------------
// Live-context registry
// ---------------------------------------------------------------------------
//
// A `Context` owns a `CUcontext` created with `cuCtxCreate` and destroys it via
// `cuCtxDestroy` on drop. Child resources (`Event`, `Module`, `GraphExec`)
// carry raw driver handles whose validity is tied to the context that was
// current when they were created. Safe code can drop the `Context` before the
// child, after which the child's own `Drop` would call the driver destroy on a
// now-stale handle — a use-after-free.
//
// To make this sound without changing any public signature, every `Context`
// registers its raw address together with a unique generation here. A child
// captures the owning `(addr, generation)` at construction and, in its `Drop`,
// only calls the driver destroy while that owner is still registered with the
// same generation. The shared `Mutex` serialises a child `Drop` against
// `Context::drop`, and the generation defeats `CUcontext` address recycling.

/// Monotonic source of per-context generations (never zero).
static CTX_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Process-wide map of live `CUcontext` address to its generation.
fn live_ctxs() -> &'static Mutex<HashMap<usize, u64>> {
    static LIVE: OnceLock<Mutex<HashMap<usize, u64>>> = OnceLock::new();
    LIVE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Lock the registry, recovering the guard if a previous holder panicked (the
/// map is plain data, so a poisoned lock is safe to keep using).
pub(crate) fn lock_live_ctxs() -> MutexGuard<'static, HashMap<usize, u64>> {
    live_ctxs().lock().unwrap_or_else(|e| e.into_inner())
}

/// Identity of the CUDA context owning a driver resource, captured at the
/// resource's construction from the thread's current context.
pub(crate) type CtxOwner = Option<(usize, u64)>;

/// Capture the current thread's context as a resource owner, if that context
/// was created through this crate (and is therefore tracked). Returns `None`
/// when no context is current or the current context is untracked — in which
/// case the resource keeps the pre-existing "always destroy on drop" behaviour.
pub(crate) fn current_ctx_owner() -> CtxOwner {
    let driver = try_driver().ok()?;
    let mut ctx = CUcontext::default();
    // SAFETY: `cu_ctx_get_current` was resolved from the driver; `ctx` is a
    // valid out-pointer.
    let rc = unsafe { (driver.cu_ctx_get_current)(&mut ctx) };
    if rc != 0 || ctx.is_null() {
        return None;
    }
    let addr = ctx.0 as usize;
    let map = lock_live_ctxs();
    map.get(&addr).map(|&generation| (addr, generation))
}

/// Whether a resource owned by `owner` may still safely call its driver
/// destroy: the owner is either untracked (`None`) or its context is still live
/// with the same generation. Callers must hold the registry lock across the
/// subsequent destroy so the check stays race-free.
pub(crate) fn owner_is_live(map: &HashMap<usize, u64>, owner: CtxOwner) -> bool {
    match owner {
        None => true,
        Some((addr, generation)) => map.get(&addr) == Some(&generation),
    }
}

// ---------------------------------------------------------------------------
// Scheduling flags
// ---------------------------------------------------------------------------

/// Context scheduling flags passed to [`Context::with_flags`].
///
/// These control how the CPU thread behaves while waiting for GPU operations.
pub mod flags {
    /// Let the driver choose the optimal scheduling policy.
    pub const SCHED_AUTO: u32 = 0x00;

    /// Actively spin (busy-wait) while waiting for GPU results. Lowest latency
    /// but consumes a full CPU core.
    pub const SCHED_SPIN: u32 = 0x01;

    /// Yield the CPU time-slice to other threads while waiting. Good for
    /// multi-threaded applications.
    pub const SCHED_YIELD: u32 = 0x02;

    /// Block the calling thread on a synchronisation primitive. Lowest CPU
    /// usage but slightly higher latency.
    pub const SCHED_BLOCKING_SYNC: u32 = 0x04;

    /// Enable mapped pinned allocations in this context.
    pub const MAP_HOST: u32 = 0x08;

    /// Keep local memory allocation after launch (deprecated flag kept for
    /// completeness).
    pub const LMEM_RESIZE_TO_MAX: u32 = 0x10;
}

// ---------------------------------------------------------------------------
// Scoped-context restore guard
// ---------------------------------------------------------------------------

/// RAII guard that restores a previously-current context on drop.
///
/// Used by [`Context::scoped`] so that the previous context is re-installed on
/// the calling thread at scope exit regardless of how the scope is left —
/// normal return, early error, or a panic unwinding through the closure.
struct CurrentCtxRestore {
    /// The context to restore (may be a null handle to detach all contexts).
    prev: CUcontext,
}

impl Drop for CurrentCtxRestore {
    fn drop(&mut self) {
        if let Ok(driver) = try_driver() {
            if let Err(e) = crate::error::check(unsafe { (driver.cu_ctx_set_current)(self.prev) }) {
                tracing::warn!("failed to restore previous context: {e}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// RAII wrapper for a CUDA context.
///
/// A context is created on a specific [`Device`] and becomes the active
/// context for the calling thread. When the `Context` is dropped,
/// `cuCtxDestroy_v2` is called automatically.
///
/// # Examples
///
/// ```no_run
/// use oxicuda_driver::context::Context;
/// use oxicuda_driver::device::Device;
///
/// oxicuda_driver::init()?;
/// let dev = Device::get(0)?;
/// let ctx = Context::new(&dev)?;
/// println!("Context on device {}", ctx.device().ordinal());
/// ctx.synchronize()?;
/// // ctx is destroyed when it goes out of scope
/// # Ok::<(), oxicuda_driver::error::CudaError>(())
/// ```
pub struct Context {
    /// The raw CUDA context handle.
    raw: CUcontext,
    /// The device this context was created on.
    device: Device,
    /// Generation under which this context is registered in the live-context
    /// registry; used to tether child resources (see the module-level
    /// registry documentation). Unused (`0`) for borrowed, non-owning wrappers.
    generation: u64,
    /// Whether this wrapper owns the underlying `CUcontext`. Owning contexts
    /// (created via [`Context::new`] / [`Context::with_flags`]) call
    /// `cuCtxDestroy` on drop and are tracked in the live-context registry;
    /// borrowed wrappers created via [`Context::from_raw_borrowed`] do neither.
    owned: bool,
}

impl Context {
    // -- Construction --------------------------------------------------------

    /// Create a new context on the given device with default flags
    /// ([`flags::SCHED_AUTO`]).
    ///
    /// The new context is automatically pushed onto the calling thread's
    /// context stack and becomes the current context.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver cannot create the context (e.g., device
    /// is invalid, out of resources).
    pub fn new(device: &Device) -> CudaResult<Self> {
        Self::with_flags(device, flags::SCHED_AUTO)
    }

    /// Create a new context on the given device with specific scheduling flags.
    ///
    /// See the [`flags`] module for available values. Multiple flags can be
    /// combined with bitwise OR.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver cannot create the context.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oxicuda_driver::context::{Context, flags};
    /// use oxicuda_driver::device::Device;
    ///
    /// oxicuda_driver::init()?;
    /// let dev = Device::get(0)?;
    /// let ctx = Context::with_flags(&dev, flags::SCHED_BLOCKING_SYNC)?;
    /// # Ok::<(), oxicuda_driver::error::CudaError>(())
    /// ```
    pub fn with_flags(device: &Device, flags: u32) -> CudaResult<Self> {
        let driver = try_driver()?;
        let mut raw = CUcontext::default();
        crate::error::check(unsafe { (driver.cu_ctx_create_v2)(&mut raw, flags, device.raw()) })?;
        // Register this context so child resources can tether their drops to it.
        let generation = CTX_GENERATION.fetch_add(1, Ordering::Relaxed);
        lock_live_ctxs().insert(raw.0 as usize, generation);
        Ok(Self {
            raw,
            device: *device,
            generation,
            owned: true,
        })
    }

    /// Wraps an already-live `CUcontext` in a **non-owning** [`Context`].
    ///
    /// Unlike [`Context::new`] / [`Context::with_flags`], the returned wrapper
    /// does *not* own the underlying context: on drop it never calls
    /// `cuCtxDestroy`, and it is not entered into the live-context generation
    /// registry. Its sole purpose is to hand an externally-owned context (for
    /// example a device **primary context** retained via
    /// [`PrimaryContext`](crate::primary_context::PrimaryContext)) to APIs that
    /// require an `Arc<Context>` lifetime token, without transferring ownership
    /// of the context's lifetime.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `raw` refers to a valid, live CUDA
    /// context and that it remains live for at least as long as this `Context`
    /// wrapper — and any resource (stream, module, …) created from it — is in
    /// use. Because the wrapper is untracked, resources created while it is
    /// current fall back to the driver's default "destroy on drop" behaviour,
    /// so the borrowed context must outlive them.
    #[must_use]
    pub unsafe fn from_raw_borrowed(raw: CUcontext, device: Device) -> Self {
        Self {
            raw,
            device,
            generation: 0,
            owned: false,
        }
    }

    // -- Current context management -----------------------------------------

    /// Set this context as the current context for the calling thread.
    ///
    /// Any previous context on this thread is detached (but not destroyed).
    ///
    /// # Errors
    ///
    /// Returns an error if the driver call fails.
    pub fn set_current(&self) -> CudaResult<()> {
        let driver = try_driver()?;
        crate::error::check(unsafe { (driver.cu_ctx_set_current)(self.raw) })
    }

    /// Get the raw handle of the current context for the calling thread.
    ///
    /// Returns `None` if no context is bound to the current thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver call fails.
    pub fn current_raw() -> CudaResult<Option<CUcontext>> {
        let driver = try_driver()?;
        let mut ctx = CUcontext::default();
        crate::error::check(unsafe { (driver.cu_ctx_get_current)(&mut ctx) })?;
        if ctx.is_null() {
            Ok(None)
        } else {
            Ok(Some(ctx))
        }
    }

    /// Pop the context currently on top of the calling thread's context stack,
    /// leaving it "floating" (owned but not bound to any thread).
    ///
    /// This is the standard driver-API idiom for undoing the implicit
    /// `cuCtxCreate` push: a freshly created context is both pushed and made
    /// current, so callers that want to manage current-ness explicitly pop it
    /// straight away. The popped handle is discarded because the caller already
    /// owns it via the [`Context`] wrapper.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver call fails (e.g. the stack is empty).
    pub(crate) fn pop_current() -> CudaResult<()> {
        let driver = try_driver()?;
        let mut popped = CUcontext::default();
        crate::error::check(unsafe { (driver.cu_ctx_pop_current_v2)(&mut popped) })
    }

    // -- Synchronisation ----------------------------------------------------

    /// Block until all pending GPU operations in this context have completed.
    ///
    /// This sets the context as current before synchronising to ensure the
    /// correct context is targeted.
    ///
    /// # Errors
    ///
    /// Returns an error if any GPU operation failed or the driver call fails.
    pub fn synchronize(&self) -> CudaResult<()> {
        self.set_current()?;
        let driver = try_driver()?;
        crate::error::check(unsafe { (driver.cu_ctx_synchronize)() })
    }

    // -- Scoped execution ---------------------------------------------------

    /// Execute a closure with this context set as current, then restore the
    /// previous context.
    ///
    /// This is useful when temporarily switching contexts. The previous
    /// context (if any) is restored even if the closure returns an error or
    /// panics — restoration is performed by an RAII guard at scope exit.
    ///
    /// # Errors
    ///
    /// Propagates any error from the closure. Context-restoration errors are
    /// logged but do not override the closure result.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oxicuda_driver::context::Context;
    /// use oxicuda_driver::device::Device;
    ///
    /// oxicuda_driver::init()?;
    /// let dev = Device::get(0)?;
    /// let ctx = Context::new(&dev)?;
    /// let result = ctx.scoped(|| {
    ///     // ctx is current here
    ///     Ok(42)
    /// })?;
    /// assert_eq!(result, 42);
    /// # Ok::<(), oxicuda_driver::error::CudaError>(())
    /// ```
    pub fn scoped<F, R>(&self, f: F) -> CudaResult<R>
    where
        F: FnOnce() -> CudaResult<R>,
    {
        // Save the currently active context (may be None).
        let prev = Self::current_raw()?;

        // Activate this context.
        self.set_current()?;

        // Install an RAII guard that restores the previous context at scope
        // exit — on the normal path, on an error return from the closure, and
        // even if the closure panics (which would otherwise leave *this*
        // context current on the thread). A null CUcontext detaches any
        // context from the current thread, which is the correct behaviour when
        // there was no previous context.
        let _restore = CurrentCtxRestore {
            prev: prev.unwrap_or_default(),
        };

        // Run the user closure. `_restore` fires when it goes out of scope.
        f()
    }

    // -- Accessors ----------------------------------------------------------

    /// Get a reference to the [`Device`] this context was created on.
    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the raw `CUcontext` handle for use with FFI calls.
    #[inline]
    pub fn raw(&self) -> CUcontext {
        self.raw
    }

    /// Returns `true` if this context is the current context on the calling
    /// thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver call fails.
    pub fn is_current(&self) -> CudaResult<bool> {
        match Self::current_raw()? {
            Some(ctx) => Ok(ctx == self.raw),
            None => Ok(false),
        }
    }
}

// ---------------------------------------------------------------------------
// Drop
// ---------------------------------------------------------------------------

impl Drop for Context {
    /// Destroy the CUDA context.
    ///
    /// Errors during destruction are logged via `tracing::warn` but never
    /// propagated (destructors must not panic).
    fn drop(&mut self) {
        // Borrowed, non-owning wrappers (see `from_raw_borrowed`) neither own
        // the context nor registered a generation, so there is nothing to
        // destroy or unregister — the real owner handles teardown.
        if !self.owned {
            return;
        }
        // Hold the registry lock across the destroy so any concurrent
        // child-resource `Drop` is serialised with it (closing the
        // use-after-free race). Removing our entry first means children created
        // under this context will observe it as gone and skip their own
        // destroy (which `cuCtxDestroy` has already performed for them).
        let mut map = lock_live_ctxs();
        let addr = self.raw.0 as usize;
        if map.get(&addr) == Some(&self.generation) {
            map.remove(&addr);
        }
        if let Ok(driver) = try_driver() {
            let result = unsafe { (driver.cu_ctx_destroy_v2)(self.raw) };
            if result != 0 {
                tracing::warn!(
                    "cuCtxDestroy_v2 failed with error code {result} during Context drop \
                     (device ordinal {})",
                    self.device.ordinal()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

// `Context` is `Send + Sync` by auto-derivation: its fields are a `CUcontext`
// handle (a plain driver-side identifier) and a `Device` (two integers). The
// CUDA Driver API is thread-safe, so no manual `unsafe impl` is needed — and
// relying on the auto-trait keeps the compiler's guard if a non-thread-safe
// field is ever added.

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("raw", &self.raw)
            .field("device", &self.device)
            .finish()
    }
}
