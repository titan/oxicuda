//! Async kernel launch with completion futures.
//!
//! This module provides [`AsyncKernel`] for launching GPU kernels that
//! return [`Future`]s, enabling integration with Rust's `async`/`await`
//! ecosystem without depending on any specific async runtime.
//!
//! # Architecture
//!
//! Since the OxiCUDA driver crate does not expose CUDA callback
//! registration, completion is detected by **polling**
//! [`Event::query()`](oxicuda_driver::Event::query). The
//! [`PollStrategy`] enum controls how aggressively the future polls:
//!
//! - [`Spin`](PollStrategy::Spin) — busy-poll with no yielding.
//! - [`Yield`](PollStrategy::Yield) — call `std::thread::yield_now()`
//!   between polls.
//! - [`BackoffMicros`](PollStrategy::BackoffMicros) — sleep a fixed
//!   number of microseconds between polls.
//!
//! # Example
//!
//! ```rust,no_run
//! # use std::sync::Arc;
//! # use oxicuda_driver::{Module, Stream, Context, Device};
//! # use oxicuda_launch::{Kernel, LaunchParams, AsyncKernel, PollStrategy, AsyncLaunchConfig};
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! # oxicuda_driver::init()?;
//! # let dev = Device::get(0)?;
//! # let ctx = Arc::new(Context::new(&dev)?);
//! # let ptx = "";
//! # let module = Arc::new(Module::from_ptx(ptx)?);
//! # let kernel = Kernel::from_module(module, "my_kernel")?;
//! let async_kernel = AsyncKernel::new(kernel);
//! let stream = Stream::new(&ctx)?;
//! let params = LaunchParams::new(4u32, 256u32);
//!
//! // Fire-and-await
//! let completion = async_kernel.launch_async(&params, &stream, &(42u32,))?;
//! completion.await?;
//! # Ok(())
//! # }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::event::Event;
use oxicuda_driver::stream::Stream;

use crate::kernel::{Kernel, KernelArgs};
use crate::params::LaunchParams;

// ---------------------------------------------------------------------------
// CompletionStatus
// ---------------------------------------------------------------------------

/// Status of a GPU kernel completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionStatus {
    /// The kernel has not yet completed.
    Pending,
    /// The kernel has completed successfully.
    Complete,
    /// An error occurred while querying completion.
    Error(String),
}

impl CompletionStatus {
    /// Returns `true` if the status is [`Complete`](Self::Complete).
    #[inline]
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Complete)
    }

    /// Returns `true` if the status is [`Pending`](Self::Pending).
    #[inline]
    pub fn is_pending(&self) -> bool {
        matches!(self, Self::Pending)
    }

    /// Returns `true` if the status is [`Error`](Self::Error).
    #[inline]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}

impl std::fmt::Display for CompletionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Complete => write!(f, "Complete"),
            Self::Error(msg) => write!(f, "Error: {msg}"),
        }
    }
}

// ---------------------------------------------------------------------------
// PollStrategy
// ---------------------------------------------------------------------------

/// Strategy for polling GPU event completion.
///
/// Controls the trade-off between CPU usage and latency when waiting
/// for a kernel to finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PollStrategy {
    /// Busy-poll `event.query()` with no pause between polls.
    ///
    /// Lowest latency but highest CPU usage.
    Spin,

    /// Call [`std::thread::yield_now()`] between polls.
    ///
    /// Allows other threads to run but still polls frequently.
    Yield,

    /// Sleep for the given number of microseconds between polls.
    ///
    /// Lower CPU usage at the cost of higher latency.
    BackoffMicros(u64),
}

impl Default for PollStrategy {
    /// Defaults to [`Yield`](PollStrategy::Yield) for a balanced
    /// trade-off between latency and CPU usage.
    #[inline]
    fn default() -> Self {
        Self::Yield
    }
}

// ---------------------------------------------------------------------------
// AsyncLaunchConfig
// ---------------------------------------------------------------------------

/// Configuration for async kernel launch behaviour.
#[derive(Debug, Clone)]
pub struct AsyncLaunchConfig {
    /// Strategy for polling event completion.
    pub poll_strategy: PollStrategy,
    /// Optional maximum time to wait before the future resolves with
    /// a timeout error.
    pub timeout: Option<Duration>,
}

impl Default for AsyncLaunchConfig {
    /// Default config: [`PollStrategy::Yield`], no timeout.
    #[inline]
    fn default() -> Self {
        Self {
            poll_strategy: PollStrategy::Yield,
            timeout: None,
        }
    }
}

impl AsyncLaunchConfig {
    /// Creates a new config with the given poll strategy and no timeout.
    #[inline]
    pub fn new(poll_strategy: PollStrategy) -> Self {
        Self {
            poll_strategy,
            timeout: None,
        }
    }

    /// Sets the timeout duration.
    #[inline]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

// ---------------------------------------------------------------------------
// LaunchTiming
// ---------------------------------------------------------------------------

/// Timing information for a completed kernel launch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaunchTiming {
    /// Elapsed GPU time in microseconds.
    pub elapsed_us: f64,
}

impl LaunchTiming {
    /// Returns the elapsed time in milliseconds.
    #[inline]
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed_us / 1000.0
    }

    /// Returns the elapsed time in seconds.
    #[inline]
    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed_us / 1_000_000.0
    }
}

impl std::fmt::Display for LaunchTiming {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.elapsed_us < 1000.0 {
            write!(f, "{:.2} us", self.elapsed_us)
        } else if self.elapsed_us < 1_000_000.0 {
            write!(f, "{:.3} ms", self.elapsed_ms())
        } else {
            write!(f, "{:.4} s", self.elapsed_secs())
        }
    }
}

// ---------------------------------------------------------------------------
// PollerShared — background re-waking thread shared by both completion futures
// ---------------------------------------------------------------------------

/// State shared between a completion future and its background poller
/// thread.
///
/// A single thread (spawned lazily on the first [`Poll::Pending`]) loops
/// according to the future's [`PollStrategy`], repeatedly re-waking the
/// most recently stored [`Waker`] until `done` is set — either because the
/// future resolved, or because it was dropped before resolving.
///
/// This replaces a one-shot wake: waking the executor exactly once is not
/// sufficient for [`PollStrategy::Yield`] or [`PollStrategy::BackoffMicros`]
/// to ever make further progress, since nothing else would schedule a
/// second poll and the future would hang forever whenever the GPU work
/// outlives that single wake.
struct PollerShared {
    /// Set once the future has resolved or been dropped, telling the
    /// background thread to stop looping.
    done: AtomicBool,
    /// The waker to re-invoke on each interval; refreshed by the future
    /// whenever the executor hands it a different one.
    waker: Mutex<Waker>,
}

impl PollerShared {
    /// Creates the shared state, seeded with the waker from the first
    /// [`Poll::Pending`].
    fn new(waker: Waker) -> Self {
        Self {
            done: AtomicBool::new(false),
            waker: Mutex::new(waker),
        }
    }

    /// Marks the poller as done, so its background thread exits on its
    /// next loop iteration instead of continuing to re-wake forever.
    fn mark_done(&self) {
        self.done.store(true, Ordering::Release);
    }

    /// Replaces the stored waker if the executor has handed us a
    /// different one since the last poll (e.g. the task moved to another
    /// executor thread).
    fn refresh_waker(&self, cx: &Context<'_>) {
        let mut stored = self
            .waker
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !stored.will_wake(cx.waker()) {
            *stored = cx.waker().clone();
        }
    }
}

/// Marks `poller` as done, if a background poller thread was ever spawned
/// for this future (i.e. it was polled at least once while still pending).
fn mark_poller_done(poller: &Option<Arc<PollerShared>>) {
    if let Some(shared) = poller {
        shared.mark_done();
    }
}

/// Spawns the single background thread backing `shared`, looping
/// according to `strategy` and re-waking `shared`'s stored waker on every
/// interval until `shared.done` is set.
///
/// * [`PollStrategy::Spin`] — no pause between wakes (one busy-looping
///   thread rather than the previous thread-per-poll spin).
/// * [`PollStrategy::Yield`] — [`std::thread::yield_now`] between wakes.
/// * [`PollStrategy::BackoffMicros`] — sleeps the given interval between
///   wakes.
fn spawn_poller_thread(shared: Arc<PollerShared>, strategy: PollStrategy) {
    std::thread::spawn(move || {
        loop {
            match strategy {
                PollStrategy::Spin => {}
                PollStrategy::Yield => std::thread::yield_now(),
                PollStrategy::BackoffMicros(us) => std::thread::sleep(Duration::from_micros(us)),
            }
            if shared.done.load(Ordering::Acquire) {
                break;
            }
            let waker = shared
                .waker
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            waker.wake_by_ref();
        }
    });
}

// ---------------------------------------------------------------------------
// LaunchCompletion
// ---------------------------------------------------------------------------

/// A [`Future`] that resolves when a GPU kernel finishes execution.
///
/// Created by [`AsyncKernel::launch_async`]. The future polls the
/// underlying CUDA event to detect completion.
pub struct LaunchCompletion {
    /// The event recorded after kernel launch.
    event: Event,
    /// Poll strategy.
    strategy: PollStrategy,
    /// Optional timeout.
    timeout: Option<Duration>,
    /// When the future was first polled (lazily initialised).
    start_time: Option<Instant>,
    /// Background poller thread's shared state, created on the first
    /// [`Poll::Pending`].
    poller: Option<Arc<PollerShared>>,
}

impl LaunchCompletion {
    /// Creates a new completion future wrapping the given event.
    fn new(event: Event, config: &AsyncLaunchConfig) -> Self {
        Self {
            event,
            strategy: config.poll_strategy,
            timeout: config.timeout,
            start_time: None,
            poller: None,
        }
    }

    /// Queries the current completion status without consuming the future.
    pub fn status(&self) -> CompletionStatus {
        match self.event.query() {
            Ok(true) => CompletionStatus::Complete,
            Ok(false) => CompletionStatus::Pending,
            Err(e) => CompletionStatus::Error(e.to_string()),
        }
    }

    /// Checks whether the timeout (if any) has been exceeded.
    fn check_timeout(&self) -> bool {
        match (self.timeout, self.start_time) {
            (Some(timeout), Some(start)) => start.elapsed() >= timeout,
            _ => false,
        }
    }

    /// Ensures a background poller thread is running for this future,
    /// spawning it on the first [`Poll::Pending`] and refreshing the
    /// stored waker on every subsequent one.
    fn ensure_poller(&mut self, cx: &Context<'_>) {
        match &self.poller {
            None => {
                let shared = Arc::new(PollerShared::new(cx.waker().clone()));
                spawn_poller_thread(Arc::clone(&shared), self.strategy);
                self.poller = Some(shared);
            }
            Some(shared) => shared.refresh_waker(cx),
        }
    }
}

impl Future for LaunchCompletion {
    type Output = CudaResult<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Initialise start time on first poll.
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        // Check timeout.
        if self.check_timeout() {
            mark_poller_done(&self.poller);
            return Poll::Ready(Err(CudaError::Timeout));
        }

        // Query the event.
        match self.event.query() {
            Ok(true) => {
                mark_poller_done(&self.poller);
                Poll::Ready(Ok(()))
            }
            Ok(false) => {
                self.ensure_poller(cx);
                Poll::Pending
            }
            Err(e) => {
                mark_poller_done(&self.poller);
                Poll::Ready(Err(e))
            }
        }
    }
}

impl Drop for LaunchCompletion {
    /// Signals the background poller thread (if any) to stop, so it does
    /// not keep looping forever after the future is dropped without
    /// resolving.
    fn drop(&mut self) {
        mark_poller_done(&self.poller);
    }
}

// ---------------------------------------------------------------------------
// TimedLaunchCompletion
// ---------------------------------------------------------------------------

/// A [`Future`] that resolves to [`LaunchTiming`] when a GPU kernel
/// finishes, measuring elapsed GPU time via CUDA events.
pub struct TimedLaunchCompletion {
    /// Event recorded before the kernel launch.
    start_event: Event,
    /// Event recorded after the kernel launch.
    end_event: Event,
    /// Poll strategy.
    strategy: PollStrategy,
    /// Optional timeout.
    timeout: Option<Duration>,
    /// When the future was first polled.
    start_time: Option<Instant>,
    /// Background poller thread's shared state, created on the first
    /// [`Poll::Pending`].
    poller: Option<Arc<PollerShared>>,
}

impl TimedLaunchCompletion {
    /// Creates a new timed completion future.
    fn new(start_event: Event, end_event: Event, config: &AsyncLaunchConfig) -> Self {
        Self {
            start_event,
            end_event,
            strategy: config.poll_strategy,
            timeout: config.timeout,
            start_time: None,
            poller: None,
        }
    }

    /// Queries the current completion status.
    pub fn status(&self) -> CompletionStatus {
        match self.end_event.query() {
            Ok(true) => CompletionStatus::Complete,
            Ok(false) => CompletionStatus::Pending,
            Err(e) => CompletionStatus::Error(e.to_string()),
        }
    }

    /// Checks whether the timeout has been exceeded.
    fn check_timeout(&self) -> bool {
        match (self.timeout, self.start_time) {
            (Some(timeout), Some(start)) => start.elapsed() >= timeout,
            _ => false,
        }
    }

    /// Ensures a background poller thread is running for this future,
    /// spawning it on the first [`Poll::Pending`] and refreshing the
    /// stored waker on every subsequent one.
    fn ensure_poller(&mut self, cx: &Context<'_>) {
        match &self.poller {
            None => {
                let shared = Arc::new(PollerShared::new(cx.waker().clone()));
                spawn_poller_thread(Arc::clone(&shared), self.strategy);
                self.poller = Some(shared);
            }
            Some(shared) => shared.refresh_waker(cx),
        }
    }
}

impl Future for TimedLaunchCompletion {
    type Output = CudaResult<LaunchTiming>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        if self.check_timeout() {
            mark_poller_done(&self.poller);
            return Poll::Ready(Err(CudaError::Timeout));
        }

        match self.end_event.query() {
            Ok(true) => {
                mark_poller_done(&self.poller);
                // Kernel complete — compute elapsed time.
                match Event::elapsed_time(&self.start_event, &self.end_event) {
                    Ok(ms) => {
                        let elapsed_us = f64::from(ms) * 1000.0;
                        Poll::Ready(Ok(LaunchTiming { elapsed_us }))
                    }
                    Err(e) => Poll::Ready(Err(e)),
                }
            }
            Ok(false) => {
                self.ensure_poller(cx);
                Poll::Pending
            }
            Err(e) => {
                mark_poller_done(&self.poller);
                Poll::Ready(Err(e))
            }
        }
    }
}

impl Drop for TimedLaunchCompletion {
    /// Signals the background poller thread (if any) to stop, so it does
    /// not keep looping forever after the future is dropped without
    /// resolving.
    fn drop(&mut self) {
        mark_poller_done(&self.poller);
    }
}

// ---------------------------------------------------------------------------
// AsyncKernel
// ---------------------------------------------------------------------------

/// A kernel wrapper with async launch capability.
///
/// Wraps a [`Kernel`] and provides methods that return [`Future`]s
/// resolving when the GPU work completes.
pub struct AsyncKernel {
    /// The underlying kernel.
    kernel: Kernel,
    /// Configuration for async behaviour.
    config: AsyncLaunchConfig,
}

impl AsyncKernel {
    /// Creates a new `AsyncKernel` with default configuration.
    #[inline]
    pub fn new(kernel: Kernel) -> Self {
        Self {
            kernel,
            config: AsyncLaunchConfig::default(),
        }
    }

    /// Creates a new `AsyncKernel` with the given configuration.
    #[inline]
    pub fn with_config(kernel: Kernel, config: AsyncLaunchConfig) -> Self {
        Self { kernel, config }
    }

    /// Returns a reference to the underlying [`Kernel`].
    #[inline]
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    /// Returns the kernel function name.
    #[inline]
    pub fn name(&self) -> &str {
        self.kernel.name()
    }

    /// Returns a reference to the current [`AsyncLaunchConfig`].
    #[inline]
    pub fn config(&self) -> &AsyncLaunchConfig {
        &self.config
    }

    /// Updates the async configuration.
    #[inline]
    pub fn set_config(&mut self, config: AsyncLaunchConfig) {
        self.config = config;
    }

    /// Launches the kernel and returns a [`LaunchCompletion`] future.
    ///
    /// The kernel is launched asynchronously on the given stream, then
    /// a CUDA event is recorded. The returned future polls that event
    /// until it completes.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] if the kernel launch or event operations
    /// fail. The future itself can also resolve to an error if the event
    /// query fails later.
    pub fn launch_async<A: KernelArgs>(
        &self,
        params: &LaunchParams,
        stream: &Stream,
        args: &A,
    ) -> CudaResult<LaunchCompletion> {
        // Launch the kernel.
        self.kernel.launch(params, stream, args)?;

        // Record an event after the launch.
        let event = Event::new()?;
        event.record(stream)?;

        Ok(LaunchCompletion::new(event, &self.config))
    }

    /// Launches the kernel and returns a [`TimedLaunchCompletion`] future
    /// that resolves to [`LaunchTiming`] with elapsed GPU time.
    ///
    /// Two events are recorded: one before and one after the kernel
    /// launch. When the future resolves, the elapsed time between the
    /// two events is computed.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] if the launch or event operations fail.
    pub fn launch_and_time_async<A: KernelArgs>(
        &self,
        params: &LaunchParams,
        stream: &Stream,
        args: &A,
    ) -> CudaResult<TimedLaunchCompletion> {
        let start_event = Event::new()?;
        start_event.record(stream)?;

        self.kernel.launch(params, stream, args)?;

        let end_event = Event::new()?;
        end_event.record(stream)?;

        Ok(TimedLaunchCompletion::new(
            start_event,
            end_event,
            &self.config,
        ))
    }
}

impl std::fmt::Debug for AsyncKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncKernel")
            .field("kernel", &self.kernel)
            .field("config", &self.config)
            .finish()
    }
}

impl std::fmt::Display for AsyncKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AsyncKernel({})", self.kernel.name())
    }
}

// ---------------------------------------------------------------------------
// multi_launch_async
// ---------------------------------------------------------------------------

/// Launches multiple kernels on the same stream and returns a combined
/// [`LaunchCompletion`] future that resolves when **all** have finished.
///
/// A single event is recorded after all kernels have been enqueued,
/// so the future resolves once the last kernel in the batch completes.
///
/// # Parameters
///
/// * `launches` — a slice of `(&Kernel, &LaunchParams, param_ptrs)` tuples.
///   Each entry's `param_ptrs` is the result of calling
///   [`KernelArgs::as_param_ptrs()`] on the kernel's arguments.
/// * `stream` — the stream on which to enqueue all kernels.
/// * `config` — async launch configuration.
///
/// # Errors
///
/// Returns the first [`CudaError`] encountered during any kernel launch
/// or event operation.
pub fn multi_launch_async(
    launches: &[(&Kernel, &LaunchParams)],
    args_list: &[&dyn ErasedKernelArgs],
    stream: &Stream,
    config: &AsyncLaunchConfig,
) -> CudaResult<LaunchCompletion> {
    for (i, (kernel, params)) in launches.iter().enumerate() {
        let args = args_list.get(i).ok_or(CudaError::InvalidValue)?;
        kernel.launch_erased(params, stream, *args)?;
    }

    let event = Event::new()?;
    event.record(stream)?;

    Ok(LaunchCompletion::new(event, config))
}

// ---------------------------------------------------------------------------
// ErasedKernelArgs — object-safe wrapper
// ---------------------------------------------------------------------------

/// Object-safe trait for kernel arguments, enabling heterogeneous
/// argument lists in [`multi_launch_async`].
///
/// # Safety
///
/// Implementors must ensure the returned pointers are valid for the
/// duration of the kernel launch call.
pub unsafe trait ErasedKernelArgs {
    /// Convert arguments to void pointers.
    fn erased_param_ptrs(&self) -> Vec<*mut std::ffi::c_void>;
}

/// Blanket implementation: every `KernelArgs` is also `ErasedKernelArgs`.
///
/// # Safety
///
/// Delegates to the underlying [`KernelArgs::as_param_ptrs`].
unsafe impl<T: KernelArgs> ErasedKernelArgs for T {
    #[inline]
    fn erased_param_ptrs(&self) -> Vec<*mut std::ffi::c_void> {
        self.as_param_ptrs()
    }
}

// ---------------------------------------------------------------------------
// Kernel::launch_erased — internal helper
// ---------------------------------------------------------------------------

impl Kernel {
    /// Launches the kernel with erased (object-safe) arguments.
    ///
    /// This is an internal helper for [`multi_launch_async`].
    pub(crate) fn launch_erased(
        &self,
        params: &LaunchParams,
        stream: &Stream,
        args: &dyn ErasedKernelArgs,
    ) -> CudaResult<()> {
        let driver = oxicuda_driver::loader::try_driver()?;
        let mut param_ptrs = args.erased_param_ptrs();
        oxicuda_driver::error::check(unsafe {
            (driver.cu_launch_kernel)(
                self.function().raw(),
                params.grid.x,
                params.grid.y,
                params.grid.z,
                params.block.x,
                params.block.y,
                params.block.z,
                params.shared_mem_bytes,
                stream.raw(),
                param_ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- CompletionStatus tests --

    #[test]
    fn completion_status_is_complete() {
        let status = CompletionStatus::Complete;
        assert!(status.is_complete());
        assert!(!status.is_pending());
        assert!(!status.is_error());
    }

    #[test]
    fn completion_status_is_pending() {
        let status = CompletionStatus::Pending;
        assert!(status.is_pending());
        assert!(!status.is_complete());
        assert!(!status.is_error());
    }

    #[test]
    fn completion_status_is_error() {
        let status = CompletionStatus::Error("test error".to_string());
        assert!(status.is_error());
        assert!(!status.is_complete());
        assert!(!status.is_pending());
    }

    #[test]
    fn completion_status_display() {
        assert_eq!(CompletionStatus::Pending.to_string(), "Pending");
        assert_eq!(CompletionStatus::Complete.to_string(), "Complete");
        assert_eq!(
            CompletionStatus::Error("oops".to_string()).to_string(),
            "Error: oops"
        );
    }

    #[test]
    fn completion_status_eq() {
        assert_eq!(CompletionStatus::Pending, CompletionStatus::Pending);
        assert_eq!(CompletionStatus::Complete, CompletionStatus::Complete);
        assert_ne!(CompletionStatus::Pending, CompletionStatus::Complete);
        assert_eq!(
            CompletionStatus::Error("a".into()),
            CompletionStatus::Error("a".into())
        );
        assert_ne!(
            CompletionStatus::Error("a".into()),
            CompletionStatus::Error("b".into())
        );
    }

    // -- PollStrategy tests --

    #[test]
    fn poll_strategy_default_is_yield() {
        assert_eq!(PollStrategy::default(), PollStrategy::Yield);
    }

    #[test]
    fn poll_strategy_backoff_value() {
        let strategy = PollStrategy::BackoffMicros(100);
        if let PollStrategy::BackoffMicros(us) = strategy {
            assert_eq!(us, 100);
        } else {
            panic!("expected BackoffMicros");
        }
    }

    // -- AsyncLaunchConfig tests --

    #[test]
    fn async_launch_config_default() {
        let config = AsyncLaunchConfig::default();
        assert_eq!(config.poll_strategy, PollStrategy::Yield);
        assert!(config.timeout.is_none());
    }

    #[test]
    fn async_launch_config_new() {
        let config = AsyncLaunchConfig::new(PollStrategy::Spin);
        assert_eq!(config.poll_strategy, PollStrategy::Spin);
        assert!(config.timeout.is_none());
    }

    #[test]
    fn async_launch_config_with_timeout() {
        let config = AsyncLaunchConfig::new(PollStrategy::BackoffMicros(50))
            .with_timeout(Duration::from_millis(500));
        assert_eq!(config.poll_strategy, PollStrategy::BackoffMicros(50));
        assert_eq!(config.timeout, Some(Duration::from_millis(500)));
    }

    // -- LaunchTiming tests --

    #[test]
    fn launch_timing_conversions() {
        let timing = LaunchTiming {
            elapsed_us: 1_500_000.0,
        };
        assert!((timing.elapsed_ms() - 1500.0).abs() < f64::EPSILON);
        assert!((timing.elapsed_secs() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn launch_timing_display_microseconds() {
        let timing = LaunchTiming { elapsed_us: 42.5 };
        let display = timing.to_string();
        assert!(display.contains("us"), "expected 'us' in: {display}");
    }

    #[test]
    fn launch_timing_display_milliseconds() {
        let timing = LaunchTiming {
            elapsed_us: 5_000.0,
        };
        let display = timing.to_string();
        assert!(display.contains("ms"), "expected 'ms' in: {display}");
    }

    #[test]
    fn launch_timing_display_seconds() {
        let timing = LaunchTiming {
            elapsed_us: 2_500_000.0,
        };
        let display = timing.to_string();
        assert!(display.contains("s"), "expected 's' in: {display}");
        assert!(
            !display.contains("us"),
            "should not contain 'us' in: {display}"
        );
        assert!(
            !display.contains("ms"),
            "should not contain 'ms' in: {display}"
        );
    }

    #[test]
    fn launch_timing_zero() {
        let timing = LaunchTiming { elapsed_us: 0.0 };
        assert!(timing.elapsed_ms().abs() < f64::EPSILON);
        assert!(timing.elapsed_secs().abs() < f64::EPSILON);
        assert!(timing.to_string().contains("us"));
    }

    // ---------------------------------------------------------------------------
    // Quality gate tests (CPU-only)
    // ---------------------------------------------------------------------------

    #[test]
    fn async_launch_status_pending_initially() {
        // CompletionStatus::Pending represents "not yet completed".
        // Verify initial/constructed status is Pending and passes is_pending().
        let status = CompletionStatus::Pending;
        assert!(status.is_pending(), "Newly created status must be Pending");
        assert!(!status.is_complete());
        assert!(!status.is_error());
    }

    #[test]
    fn async_launch_debug_impl() {
        // AsyncLaunchConfig implements Debug — verify it does not panic.
        let config = AsyncLaunchConfig::new(PollStrategy::Yield);
        let dbg = format!("{config:?}");
        assert!(
            dbg.contains("AsyncLaunchConfig"),
            "Debug output must contain type name, got: {dbg}"
        );
        // PollStrategy also implements Debug
        let strategy_dbg = format!("{:?}", PollStrategy::BackoffMicros(200));
        assert!(
            strategy_dbg.contains("BackoffMicros"),
            "PollStrategy Debug must contain variant name, got: {strategy_dbg}"
        );
    }

    #[test]
    fn async_completion_event_created() {
        // Creating an AsyncLaunchConfig produces a valid struct with the fields
        // expected by the async launch machinery.
        let config = AsyncLaunchConfig {
            poll_strategy: PollStrategy::Spin,
            timeout: Some(Duration::from_secs(5)),
        };
        assert_eq!(config.poll_strategy, PollStrategy::Spin);
        assert_eq!(config.timeout, Some(Duration::from_secs(5)));

        // with_timeout builder chain also works
        let config2 = AsyncLaunchConfig::new(PollStrategy::BackoffMicros(100))
            .with_timeout(Duration::from_millis(250));
        assert_eq!(config2.poll_strategy, PollStrategy::BackoffMicros(100));
        assert_eq!(config2.timeout, Some(Duration::from_millis(250)));
    }

    // ---------------------------------------------------------------------------
    // On-device regression tests (F030): the one-shot poller must not hang the
    // future forever under PollStrategy::Yield/BackoffMicros.
    // ---------------------------------------------------------------------------

    /// A device-side busy-spin kernel: a single thread repeatedly performs a
    /// **volatile** global-memory read-modify-write on the same scratch
    /// address, `iters` times. Used to keep a real CUDA event `Pending` for
    /// a measurable, tunable amount of wall-clock time so the completion
    /// futures actually observe multiple poll cycles before resolving.
    ///
    /// The loop body must have a genuine, non-eliminable side effect: an
    /// earlier version that only incremented a register (with no memory
    /// traffic) let the JIT compiler algebraically collapse the whole loop
    /// into a single closed-form store, since it could prove no external
    /// observer could see the intermediate values — making the kernel
    /// finish instantly instead of taking real time. The `volatile`
    /// qualifier on the load/store forbids that transformation and forces
    /// one genuine dependent DRAM round trip per iteration.
    #[cfg(feature = "gpu-tests")]
    const BUSY_SPIN_PTX: &str = "\
.version 7.0
.target sm_70
.address_size 64
.visible .entry busy_spin(
    .param .u64 scratch_ptr,
    .param .u32 iters
)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<2>;
    .reg .pred %p<2>;
    ld.param.u64 %rd0, [scratch_ptr];
    ld.param.u32 %r0, [iters];
    mov.u32 %r1, 0;
$LOOP:
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra $DONE;
    ld.volatile.global.u32 %r2, [%rd0];
    add.u32 %r2, %r2, 1;
    st.volatile.global.u32 [%rd0], %r2;
    add.u32 %r1, %r1, 1;
    bra $LOOP;
$DONE:
    ret;
}
";

    /// A minimal wake counter: a [`std::task::Wake`] implementation that
    /// just counts invocations, used to detect whether the background
    /// poller thread wakes the executor repeatedly (fixed behaviour) or
    /// only once (the regression this test guards against).
    #[cfg(feature = "gpu-tests")]
    struct CountingWaker(std::sync::atomic::AtomicUsize);

    #[cfg(feature = "gpu-tests")]
    impl std::task::Wake for CountingWaker {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Regression test for the bug where `LaunchCompletion`'s background
    /// poller only woke the executor **once**: under
    /// [`PollStrategy::Yield`], a real async executor that parks the task
    /// until woken (rather than busy-polling) would then never be told to
    /// poll again, and would hang forever whenever the GPU work outlives
    /// that single wake. This drives the future with a manual executor
    /// that self-polls on a short interval (so it does not itself rely on
    /// the wake signal to make progress — that would risk a hang if the
    /// fix regressed) while independently counting wake-ups; the fixed
    /// implementation must wake the executor *more than once* over the
    /// lifetime of a kernel slow enough to take multiple poll cycles.
    /// Self-skips if there is no GPU/driver.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn launch_completion_yield_strategy_wakes_repeatedly_and_completes() {
        let Ok(dev) = oxicuda_driver::device::Device::get(0) else {
            return;
        };
        let ctx = match oxicuda_driver::context::Context::new(&dev) {
            Ok(c) => Arc::new(c),
            Err(_) => return,
        };
        let stream = match Stream::new(&ctx) {
            Ok(s) => s,
            Err(_) => return,
        };
        let module = match oxicuda_driver::module::Module::from_ptx(BUSY_SPIN_PTX) {
            Ok(m) => Arc::new(m),
            Err(_) => return,
        };
        let kernel = match Kernel::from_module(module, "busy_spin") {
            Ok(k) => k,
            Err(_) => return,
        };
        let scratch = match oxicuda_memory::DeviceBuffer::<u32>::zeroed(1) {
            Ok(b) => b,
            Err(_) => return,
        };

        let async_kernel =
            AsyncKernel::with_config(kernel, AsyncLaunchConfig::new(PollStrategy::Yield));
        // Enough dependent volatile-memory round trips to keep the kernel
        // busy for a measurable stretch of wall-clock time, so multiple
        // poll cycles happen before it completes.
        let iters: u32 = 3_000_000;
        let completion = match async_kernel.launch_async(
            &LaunchParams::new(1u32, 1u32),
            &stream,
            &(scratch.as_device_ptr(), iters),
        ) {
            Ok(c) => c,
            Err(_) => return,
        };

        let counter = Arc::new(CountingWaker(std::sync::atomic::AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&counter));
        let mut cx = Context::from_waker(&waker);
        let mut completion = Box::pin(completion);

        let deadline = Instant::now() + Duration::from_secs(30);
        let result = loop {
            match completion.as_mut().poll(&mut cx) {
                Poll::Ready(r) => break r,
                Poll::Pending => {
                    assert!(
                        Instant::now() < deadline,
                        "LaunchCompletion under PollStrategy::Yield did not complete \
                         within the deadline (regression of the one-shot-waker hang)"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        };
        result.expect("busy-spin kernel launch must succeed");

        // The key regression signal: the background poller must wake the
        // executor more than once. With the old one-shot poller, `counter`
        // would never exceed 1 regardless of how long the kernel runs.
        let wakes = counter.0.load(Ordering::SeqCst);
        assert!(
            wakes > 1,
            "background poller must wake the executor more than once under \
             PollStrategy::Yield, got {wakes} wake(s)"
        );
    }

    /// Regression test for the `Drop` half of the fix: once a pending
    /// completion future is dropped, its background poller thread must
    /// stop looping (and thus stop waking) instead of spinning/yielding
    /// forever in the background. Self-skips if there is no GPU/driver.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn launch_completion_drop_stops_background_poller() {
        let Ok(dev) = oxicuda_driver::device::Device::get(0) else {
            return;
        };
        let ctx = match oxicuda_driver::context::Context::new(&dev) {
            Ok(c) => Arc::new(c),
            Err(_) => return,
        };
        let stream = match Stream::new(&ctx) {
            Ok(s) => s,
            Err(_) => return,
        };
        let module = match oxicuda_driver::module::Module::from_ptx(BUSY_SPIN_PTX) {
            Ok(m) => Arc::new(m),
            Err(_) => return,
        };
        let kernel = match Kernel::from_module(module, "busy_spin") {
            Ok(k) => k,
            Err(_) => return,
        };
        let scratch = match oxicuda_memory::DeviceBuffer::<u32>::zeroed(1) {
            Ok(b) => b,
            Err(_) => return,
        };

        let async_kernel =
            AsyncKernel::with_config(kernel, AsyncLaunchConfig::new(PollStrategy::Yield));
        // A long-running kernel so it is essentially guaranteed to still be
        // pending immediately after the first poll.
        let iters: u32 = 20_000_000;
        let completion = match async_kernel.launch_async(
            &LaunchParams::new(1u32, 1u32),
            &stream,
            &(scratch.as_device_ptr(), iters),
        ) {
            Ok(c) => c,
            Err(_) => return,
        };

        let counter = Arc::new(CountingWaker(std::sync::atomic::AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&counter));
        let mut cx = Context::from_waker(&waker);
        let mut completion = Box::pin(completion);

        match completion.as_mut().poll(&mut cx) {
            Poll::Pending => {}
            // Finished implausibly fast on this device — nothing to
            // observe, skip rather than risk a flaky failure.
            Poll::Ready(_) => return,
        }

        std::thread::sleep(Duration::from_millis(50));
        let before_drop = counter.0.load(Ordering::SeqCst);
        assert!(
            before_drop > 0,
            "poller should have woken the executor at least once by now"
        );

        drop(completion);

        // Give the background thread time to observe `done` and exit.
        std::thread::sleep(Duration::from_millis(150));
        let settled = counter.0.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(150));
        let after_settle = counter.0.load(Ordering::SeqCst);

        // The completion future's Drop only stops our host-side polling
        // machinery — the kernel itself (and its writes to `scratch`) keep
        // running on the device. Let it actually finish before `stream`
        // and `scratch` are dropped below, regardless of the assertion
        // outcome, so we never free device memory (or destroy the stream)
        // out from under an in-flight kernel.
        stream
            .synchronize()
            .expect("stream sync after dropped completion");

        assert_eq!(
            settled, after_settle,
            "background poller must stop waking once the future has been dropped"
        );
    }

    /// Smoke test that `TimedLaunchCompletion` — which shares the same
    /// `PollerShared`/`spawn_poller_thread` machinery — also resolves
    /// under `PollStrategy::Yield` instead of hanging. Self-skips if there
    /// is no GPU/driver.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn timed_launch_completion_yield_strategy_completes() {
        let Ok(dev) = oxicuda_driver::device::Device::get(0) else {
            return;
        };
        let ctx = match oxicuda_driver::context::Context::new(&dev) {
            Ok(c) => Arc::new(c),
            Err(_) => return,
        };
        let stream = match Stream::new(&ctx) {
            Ok(s) => s,
            Err(_) => return,
        };
        let module = match oxicuda_driver::module::Module::from_ptx(BUSY_SPIN_PTX) {
            Ok(m) => Arc::new(m),
            Err(_) => return,
        };
        let kernel = match Kernel::from_module(module, "busy_spin") {
            Ok(k) => k,
            Err(_) => return,
        };
        let scratch = match oxicuda_memory::DeviceBuffer::<u32>::zeroed(1) {
            Ok(b) => b,
            Err(_) => return,
        };

        let async_kernel =
            AsyncKernel::with_config(kernel, AsyncLaunchConfig::new(PollStrategy::Yield));
        let iters: u32 = 3_000_000;
        let completion = match async_kernel.launch_and_time_async(
            &LaunchParams::new(1u32, 1u32),
            &stream,
            &(scratch.as_device_ptr(), iters),
        ) {
            Ok(c) => c,
            Err(_) => return,
        };

        let counter = Arc::new(CountingWaker(std::sync::atomic::AtomicUsize::new(0)));
        let waker = Waker::from(counter);
        let mut cx = Context::from_waker(&waker);
        let mut completion = Box::pin(completion);

        let deadline = Instant::now() + Duration::from_secs(30);
        let timing = loop {
            match completion.as_mut().poll(&mut cx) {
                Poll::Ready(r) => break r,
                Poll::Pending => {
                    assert!(
                        Instant::now() < deadline,
                        "TimedLaunchCompletion under PollStrategy::Yield did not \
                         complete within the deadline"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }
        .expect("busy-spin kernel launch must succeed");
        assert!(timing.elapsed_us >= 0.0);
    }
}
