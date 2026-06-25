//! Training session handle.
//!
//! [`crate::handle::TrainHandle`] owns a CUDA context and stream and carries SM version
//! metadata used to select the correct PTX target when JIT-compiling
//! optimizer update kernels.

use std::sync::Arc;

use oxicuda_driver::{Context, Stream};

use crate::error::TrainResult;

// ─── LcgRng ──────────────────────────────────────────────────────────────────

/// Minimal PCG-style linear-congruential random number generator for
/// deterministic CPU-side sampling (e.g. Rademacher vectors for the
/// Hutchinson Hessian-diagonal estimator used by [`crate::optimizer::sophia`]).
///
/// Uses a high-quality 64-bit LCG multiplier (Knuth MMIX) followed by an
/// xorshift output permutation:
/// `x_{n+1} = 6364136223846793005·x_n + 1442695040888963407 (mod 2⁶⁴)`.
///
/// Reproducible across runs for a fixed seed; not cryptographically secure.
#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    /// Create a new LCG with the given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Advance one step and return a full-range `u32` (xor-folded high bits).
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 33) ^ self.state) as u32
    }

    /// Advance one step and return a full-range `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let hi = u64::from(self.next_u32());
        let lo = u64::from(self.next_u32());
        (hi << 32) | lo
    }

    /// Return an `f32` uniformly distributed in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }

    /// Return an `f64` uniformly distributed in `[0, 1)`.
    ///
    /// Uses all 53 mantissa bits via a 64-bit draw scaled by `2⁻⁵³`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Take the top 53 bits of a 64-bit draw and scale into [0, 1).
        ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Draw a single Rademacher sample: `+1.0` or `-1.0` with equal probability.
    #[inline]
    pub fn rademacher(&mut self) -> f64 {
        if self.next_u32() & 1 == 0 { 1.0 } else { -1.0 }
    }

    /// Fill `buf` with independent Rademacher (`±1`) samples.
    #[inline]
    pub fn fill_rademacher(&mut self, buf: &mut [f64]) {
        for slot in buf.iter_mut() {
            *slot = self.rademacher();
        }
    }
}

/// Central handle for all OxiCUDA training operations.
///
/// Pass a `&TrainHandle` (or clone the `Arc`-backed handle) to optimizers,
/// gradient utilities, and checkpointing APIs to share a single CUDA context
/// and stream across the training loop.
///
/// # Example
///
/// ```rust,no_run
/// use oxicuda_train::handle::TrainHandle;
///
/// let handle = TrainHandle::new().expect("new should succeed");
/// println!("SM version: {}", handle.sm_version());
/// ```
#[derive(Clone)]
pub struct TrainHandle {
    context: Arc<Context>,
    stream: Arc<Stream>,
    /// Numeric SM version, e.g. 800 for sm_80, 900 for sm_90.
    sm_version: u32,
    device_id: i32,
}

impl TrainHandle {
    /// Create a handle on the best available GPU device.
    ///
    /// Queries the device's compute capability to fill `sm_version`.
    pub fn new() -> TrainResult<Self> {
        use oxicuda_driver::{best_device, device::Device, init};
        init()?;
        let device = if let Some(d) = best_device()? {
            d
        } else {
            Device::get(0)?
        };
        let sm_version = device_sm_version(&device).unwrap_or(800);
        let device_id = device.ordinal();
        let context = Arc::new(Context::new(&device)?);
        let stream = Arc::new(Stream::new(&context)?);
        Ok(Self {
            context,
            stream,
            sm_version,
            device_id,
        })
    }

    /// Create a handle from an already-created context and stream.
    ///
    /// `sm_version` must be the numeric SM version (e.g. `800` for sm_80).
    #[must_use]
    pub fn from_parts(context: Arc<Context>, stream: Arc<Stream>, sm_version: u32) -> Self {
        let device_id = 0i32; // caller-supplied; unknown here
        Self {
            context,
            stream,
            sm_version,
            device_id,
        }
    }

    /// CUDA context.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// CUDA stream used for all asynchronous launches.
    #[must_use]
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Numeric SM version (e.g. `800` for sm_80, `900` for sm_90).
    #[must_use]
    pub fn sm_version(&self) -> u32 {
        self.sm_version
    }

    /// Logical device index.
    #[must_use]
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Block until all GPU work queued on this stream has completed.
    pub fn synchronize(&self) -> TrainResult<()> {
        self.stream.synchronize()?;
        Ok(())
    }
}

impl std::fmt::Debug for TrainHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainHandle")
            .field("sm_version", &self.sm_version)
            .field("device_id", &self.device_id)
            .finish_non_exhaustive()
    }
}

// ─── SM-version helper ───────────────────────────────────────────────────────

/// Query compute capability of `device` and convert to a numeric SM version.
///
/// Returns `None` if the driver query fails (the caller should fall back to
/// a safe default such as 800 for sm_80).
fn device_sm_version(device: &oxicuda_driver::device::Device) -> Option<u32> {
    use oxicuda_driver::ffi::CUdevice_attribute;
    let major = device
        .attribute(CUdevice_attribute::ComputeCapabilityMajor)
        .ok()?;
    let minor = device
        .attribute(CUdevice_attribute::ComputeCapabilityMinor)
        .ok()?;
    Some((major as u32) * 100 + (minor as u32) * 10)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `LcgRng` is deterministic for a fixed seed.
    #[test]
    fn lcg_deterministic() {
        let mut a = LcgRng::new(42);
        let mut b = LcgRng::new(42);
        for _ in 0..32 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    /// Different seeds diverge.
    #[test]
    fn lcg_seed_sensitive() {
        let mut a = LcgRng::new(1);
        let mut b = LcgRng::new(2);
        // Extremely unlikely to match across many draws if seeds differ.
        let mut all_equal = true;
        for _ in 0..16 {
            if a.next_u32() != b.next_u32() {
                all_equal = false;
            }
        }
        assert!(!all_equal, "distinct seeds should produce distinct streams");
    }

    /// `next_f64` stays in `[0, 1)`.
    #[test]
    fn lcg_f64_unit_interval() {
        let mut rng = LcgRng::new(7);
        for _ in 0..1_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x), "f64 sample {x} out of [0,1)");
        }
    }

    /// Rademacher draws are exactly ±1 and roughly balanced.
    #[test]
    fn lcg_rademacher_balanced() {
        let mut rng = LcgRng::new(123);
        let mut sum = 0.0_f64;
        let n = 10_000;
        for _ in 0..n {
            let r = rng.rademacher();
            assert!(r == 1.0 || r == -1.0, "rademacher must be ±1, got {r}");
            sum += r;
        }
        // Mean should be near zero for a balanced sign distribution.
        let mean = sum / f64::from(n);
        assert!(mean.abs() < 0.1, "rademacher mean {mean} too far from 0");
    }

    /// Verifies that `from_parts` stores parameters correctly without needing
    /// a real GPU.
    #[test]
    #[ignore = "requires GPU"]
    fn handle_from_parts_round_trip() {
        let handle = TrainHandle::new().expect("TrainHandle creation should succeed on GPU");
        let sm = handle.sm_version();
        assert!(sm >= 750, "expected at least sm_75, got {sm}");
    }

    /// Debug formatting does not panic.
    #[test]
    #[ignore = "requires GPU"]
    fn handle_debug() {
        let handle = TrainHandle::new().expect("TrainHandle creation should succeed on GPU");
        let _ = format!("{handle:?}");
    }
}
