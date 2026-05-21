//! High-level FFT executor.
//!
//! [`FftHandle`] is the primary entry point for executing FFT transforms.
//! It owns the CUDA context, stream, PTX cache, and architecture info
//! needed to compile and launch FFT kernels.
//!
//! # Usage
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use oxicuda_driver::{Context, Device, Stream, init};
//! use oxicuda_fft::prelude::*;
//!
//! init().expect("CUDA init");
//! let dev = Device::get(0).expect("device");
//! let ctx = Arc::new(Context::new(&dev).expect("context"));
//! let handle = FftHandle::new(&ctx).expect("handle");
//!
//! let plan = FftPlan::new_1d(1024, FftType::C2C, 1).expect("plan");
//! // handle.execute(&plan, input_ptr, output_ptr, FftDirection::Forward)
//! ```
#![allow(dead_code)]

use std::sync::Arc;

use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_driver::{Context, Stream};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::cache::PtxCache;

use crate::error::{FftError, FftResult};
use crate::plan::FftPlan;
use crate::transforms::{c2c, c2r, fft2d, fft3d, r2c};
use crate::types::{FftDirection, FftType};

// ---------------------------------------------------------------------------
// FftHandle
// ---------------------------------------------------------------------------

/// A handle for executing FFT transforms on the GPU.
///
/// The handle holds all the state necessary to compile and launch FFT
/// kernels: a reference to the CUDA context, a dedicated stream, a PTX
/// cache for avoiding redundant code generation, and the detected GPU
/// architecture version.
///
/// # Thread Safety
///
/// `FftHandle` is `Send` but not `Sync`.  Each thread should create its
/// own handle (or synchronise externally).
pub struct FftHandle {
    /// Reference-counted CUDA context.
    context: Arc<Context>,
    /// CUDA stream for FFT kernel launches.
    stream: Stream,
    /// Disk-backed PTX cache for compiled kernels.
    ptx_cache: PtxCache,
    /// Target GPU architecture (e.g. `Sm80` for Ampere).
    sm_version: SmVersion,
}

impl FftHandle {
    /// Creates a new FFT handle bound to the given CUDA context.
    ///
    /// This creates a new CUDA stream and initialises the PTX cache.
    /// The SM version is detected from the context's device.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::Cuda`] if stream creation fails.
    /// Returns [`FftError::InternalError`] if the PTX cache cannot be
    /// initialised or the device architecture cannot be queried.
    pub fn new(ctx: &Arc<Context>) -> FftResult<Self> {
        let stream = Stream::new(ctx).map_err(FftError::Cuda)?;

        let ptx_cache = PtxCache::new()
            .map_err(|e| FftError::InternalError(format!("PTX cache init failed: {e}")))?;

        // Detect the GPU architecture from the context's device by querying
        // its CUDA compute capability (`COMPUTE_CAPABILITY_MAJOR`/`MINOR`).
        // Falls back to Ampere (sm_80) only if the query genuinely fails or
        // reports a capability this build does not recognise.
        let sm_version = detect_sm_version(ctx);

        Ok(Self {
            context: Arc::clone(ctx),
            stream,
            ptx_cache,
            sm_version,
        })
    }

    /// Creates a handle with an explicit SM version (useful for testing).
    ///
    /// # Errors
    ///
    /// Returns [`FftError::Cuda`] if stream creation fails.
    /// Returns [`FftError::InternalError`] if the PTX cache cannot be
    /// initialised.
    pub fn with_sm_version(ctx: &Arc<Context>, sm: SmVersion) -> FftResult<Self> {
        let stream = Stream::new(ctx).map_err(FftError::Cuda)?;

        let ptx_cache = PtxCache::new()
            .map_err(|e| FftError::InternalError(format!("PTX cache init failed: {e}")))?;

        Ok(Self {
            context: Arc::clone(ctx),
            stream,
            ptx_cache,
            sm_version: sm,
        })
    }

    /// Returns the CUDA context associated with this handle.
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Returns the CUDA stream used by this handle.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Returns the PTX cache used by this handle.
    pub fn ptx_cache(&self) -> &PtxCache {
        &self.ptx_cache
    }

    /// Returns the target SM version.
    pub fn sm_version(&self) -> SmVersion {
        self.sm_version
    }

    // -----------------------------------------------------------------------
    // Execution
    // -----------------------------------------------------------------------

    /// Executes an FFT transform according to the given plan.
    ///
    /// This dispatches to the appropriate transform implementation based
    /// on the plan's dimensionality and transform type.
    ///
    /// # Arguments
    ///
    /// * `plan`      - A compiled [`FftPlan`].
    /// * `input`     - Device pointer to the input data.
    /// * `output`    - Device pointer to the output data.
    /// * `direction` - [`FftDirection::Forward`] or [`FftDirection::Inverse`].
    ///
    /// # Errors
    ///
    /// Returns an [`FftError`] if the kernel launch or synchronisation fails.
    pub fn execute(
        &self,
        plan: &FftPlan,
        input: CUdeviceptr,
        output: CUdeviceptr,
        direction: FftDirection,
    ) -> FftResult<()> {
        match plan.ndim() {
            1 => self.execute_1d(plan, input, output, direction),
            2 => fft2d::fft_2d(plan, input, output, direction, &self.stream),
            3 => fft3d::fft_3d(plan, input, output, direction, &self.stream),
            n => Err(FftError::UnsupportedTransform(format!(
                "{n}-D FFT is not supported (max 3)"
            ))),
        }
    }

    /// Executes a batched FFT transform.
    ///
    /// This is equivalent to [`execute`](Self::execute) but processes
    /// `batch_count` independent transforms. Each batch element is
    /// assumed to be contiguous in memory.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::InvalidBatch`] if `batch_count` is zero.
    pub fn execute_batch(
        &self,
        plan: &FftPlan,
        input: CUdeviceptr,
        output: CUdeviceptr,
        direction: FftDirection,
        batch_count: usize,
    ) -> FftResult<()> {
        if batch_count == 0 {
            return Err(FftError::InvalidBatch(
                "batch_count must be >= 1".to_string(),
            ));
        }

        match plan.ndim() {
            1 => self.execute_1d_batched(plan, input, output, direction, batch_count),
            2 => fft2d::fft_2d_batched(plan, input, output, direction, batch_count, &self.stream),
            3 => fft3d::fft_3d_batched(plan, input, output, direction, batch_count, &self.stream),
            n => Err(FftError::UnsupportedTransform(format!(
                "{n}-D batched FFT is not supported (max 3)"
            ))),
        }
    }

    // -----------------------------------------------------------------------
    // 1-D dispatch
    // -----------------------------------------------------------------------

    /// Dispatches a 1-D FFT based on the transform type.
    fn execute_1d(
        &self,
        plan: &FftPlan,
        input: CUdeviceptr,
        output: CUdeviceptr,
        direction: FftDirection,
    ) -> FftResult<()> {
        match plan.transform_type {
            FftType::C2C => c2c::fft_c2c(plan, input, output, direction, &self.stream),
            FftType::R2C => r2c::fft_r2c(plan, input, output, &self.stream),
            FftType::C2R => c2r::fft_c2r(plan, input, output, &self.stream),
        }
    }

    /// Dispatches a batched 1-D FFT based on the transform type.
    fn execute_1d_batched(
        &self,
        plan: &FftPlan,
        input: CUdeviceptr,
        output: CUdeviceptr,
        direction: FftDirection,
        batch_count: usize,
    ) -> FftResult<()> {
        match plan.transform_type {
            FftType::C2C => {
                c2c::fft_c2c_batched(plan, input, output, direction, batch_count, &self.stream)
            }
            FftType::R2C => r2c::fft_r2c_batched(plan, input, output, batch_count, &self.stream),
            FftType::C2R => c2r::fft_c2r_batched(plan, input, output, batch_count, &self.stream),
        }
    }
}

impl std::fmt::Debug for FftHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftHandle")
            .field("sm_version", &self.sm_version)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// SM version detection
// ---------------------------------------------------------------------------

/// Detects the target [`SmVersion`] for the device backing `ctx`.
///
/// This queries the CUDA compute capability of the context's device via
/// `cuDeviceGetAttribute` (the `COMPUTE_CAPABILITY_MAJOR` / `MINOR`
/// attributes, wrapped by [`oxicuda_driver::device::Device::compute_capability`])
/// and maps the resulting `(major, minor)` pair to an `SmVersion`:
///
/// | Compute capability | `SmVersion`        |
/// |--------------------|--------------------|
/// | 7.5                | [`SmVersion::Sm75`] |
/// | 8.0                | [`SmVersion::Sm80`] |
/// | 8.6                | [`SmVersion::Sm86`] |
/// | 8.9                | [`SmVersion::Sm89`] |
/// | 9.0                | [`SmVersion::Sm90`] |
/// | 10.0               | [`SmVersion::Sm100`] |
/// | 12.0               | [`SmVersion::Sm120`] |
///
/// The `9.0a` accelerated profile ([`SmVersion::Sm90a`]) is not a distinct
/// hardware capability — the driver reports plain `9.0` for Hopper parts — so
/// it is never produced by automatic detection; callers that need the `a`
/// profile select it explicitly via [`FftHandle::with_sm_version`].
///
/// Falls back to [`SmVersion::Sm80`] (Ampere) only when the attribute query
/// genuinely fails or reports a capability this build does not recognise.
fn detect_sm_version(ctx: &Arc<Context>) -> SmVersion {
    match ctx.device().compute_capability() {
        Ok((major, minor)) => {
            SmVersion::from_compute_capability(major, minor).unwrap_or(SmVersion::Sm80)
        }
        Err(_) => SmVersion::Sm80,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Attempts to build a CUDA context on device 0.
    ///
    /// Returns `None` so the caller skips the test when no CUDA device is
    /// available, following the crate's device-test skip convention.
    fn gpu_context() -> Option<Arc<Context>> {
        if oxicuda_driver::init().is_err() {
            return None;
        }
        let device = oxicuda_driver::device::Device::get(0).ok()?;
        let context = Context::new(&device).ok()?;
        Some(Arc::new(context))
    }

    /// The compute-capability mapping must cover every recognised pair.
    #[test]
    fn from_compute_capability_covers_known_arches() {
        assert_eq!(
            SmVersion::from_compute_capability(7, 5),
            Some(SmVersion::Sm75)
        );
        assert_eq!(
            SmVersion::from_compute_capability(8, 0),
            Some(SmVersion::Sm80)
        );
        assert_eq!(
            SmVersion::from_compute_capability(8, 6),
            Some(SmVersion::Sm86)
        );
        assert_eq!(
            SmVersion::from_compute_capability(8, 9),
            Some(SmVersion::Sm89)
        );
        assert_eq!(
            SmVersion::from_compute_capability(9, 0),
            Some(SmVersion::Sm90)
        );
        assert_eq!(
            SmVersion::from_compute_capability(10, 0),
            Some(SmVersion::Sm100)
        );
        assert_eq!(
            SmVersion::from_compute_capability(12, 0),
            Some(SmVersion::Sm120)
        );
        // Unknown capabilities have no mapping (detection falls back).
        assert_eq!(SmVersion::from_compute_capability(6, 1), None);
    }

    /// `with_sm_version` must honour the explicitly-requested architecture.
    #[test]
    fn with_sm_version_round_trips() {
        let Some(ctx) = gpu_context() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let handle = FftHandle::with_sm_version(&ctx, SmVersion::Sm75)
            .expect("handle creation must succeed");
        assert_eq!(handle.sm_version(), SmVersion::Sm75);
    }

    /// `FftHandle::new` must detect the *real* compute capability of the
    /// installed GPU instead of hardcoding `Sm80`.
    ///
    /// On this box (an NVIDIA RTX A4000) the device reports compute
    /// capability 8.6, so detection must yield [`SmVersion::Sm86`].
    #[test]
    fn new_detects_real_sm_version() {
        let Some(ctx) = gpu_context() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // Cross-check against a direct compute-capability query.
        let expected = match ctx.device().compute_capability() {
            Ok((major, minor)) => {
                SmVersion::from_compute_capability(major, minor).unwrap_or(SmVersion::Sm80)
            }
            Err(_) => {
                eprintln!("skipping: compute capability query failed");
                return;
            }
        };

        let handle = FftHandle::new(&ctx).expect("handle creation must succeed");
        assert_eq!(
            handle.sm_version(),
            expected,
            "FftHandle::new must detect the device's real SM version"
        );

        // The RTX A4000 is an Ampere GA10x part (sm_86).  When this test runs
        // on that hardware, confirm detection did not silently fall back to
        // the hardcoded Sm80 default.
        if let Ok(name) = ctx.device().name() {
            if name.contains("A4000") {
                assert_eq!(
                    handle.sm_version(),
                    SmVersion::Sm86,
                    "RTX A4000 must be detected as sm_86, not the Sm80 fallback"
                );
            }
        }
    }
}
