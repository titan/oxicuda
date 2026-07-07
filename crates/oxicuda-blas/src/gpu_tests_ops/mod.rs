//! On-device GPU validation for the non-GEMM BLAS kernels (Level-1, Level-2,
//! reductions, elementwise, complex) emitted by `oxicuda-blas`.
//!
//! The sibling `gpu_tests` module covers GEMM; this module covers everything
//! else the public ops launch in production. Each test drives the production op
//! (which JITs + launches the kernel on the device), copies results back, and
//! asserts equivalence to an independent CPU oracle. Every test skips when no
//! CUDA device is present.

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Stream};
use oxicuda_ptx::arch::SmVersion;

mod batched;
mod elementwise;
mod level1;
mod level2;
mod level3;
mod reduction;

/// A live CUDA context + stream + the device SM version.
pub(crate) struct GpuFixture {
    pub ctx: Arc<Context>,
    pub stream: Stream,
    pub sm: SmVersion,
}

/// Acquire a GPU fixture, or `None` when no driver / device is present.
pub(crate) fn gpu_fixture() -> Option<GpuFixture> {
    oxicuda_driver::init().ok()?;
    if Device::count().ok()? == 0 {
        return None;
    }
    let dev = Device::get(0).ok()?;
    let (major, minor) = dev.compute_capability().ok()?;
    let sm = SmVersion::from_compute_capability(major, minor).unwrap_or(SmVersion::Sm80);
    let ctx = Arc::new(Context::new(&dev).ok()?);
    let stream = Stream::new(&ctx).ok()?;
    Some(GpuFixture { ctx, stream, sm })
}

/// Relative-with-absolute-floor closeness test for FP32.
pub(crate) fn close_f32(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

/// Asserts two FP32 slices agree within tolerance, reporting the first mismatch.
pub(crate) fn assert_close_f32(gpu: &[f32], cpu: &[f32], rel: f32, abs: f32, tag: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{tag}: length mismatch");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!(
            close_f32(g, c, rel, abs),
            "{tag}: element {i} mismatch gpu={g} cpu={c} (rel={rel:e} abs={abs:e})"
        );
    }
}

/// Asserts two FP64 slices agree within tolerance, reporting the first mismatch.
pub(crate) fn assert_close_f64(gpu: &[f64], cpu: &[f64], rel: f64, abs: f64, tag: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{tag}: length mismatch");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!(
            (g - c).abs() <= rel * g.abs().max(c.abs()) + abs,
            "{tag}: element {i} mismatch gpu={g} cpu={c} (rel={rel:e} abs={abs:e})"
        );
    }
}

/// A small deterministic LCG. Normalisation divides by `2^32` (never `2^31`).
pub(crate) struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    /// Uniform in `[0, 1)` via division by `2^32`.
    pub fn unit(&mut self) -> f64 {
        f64::from(self.next_u32()) / 4_294_967_296.0
    }
    /// Uniform `f32` in `[lo, hi)`.
    pub fn range_f32(&mut self, lo: f64, hi: f64) -> f32 {
        (lo + (hi - lo) * self.unit()) as f32
    }
    /// Uniform `f64` in `[lo, hi)`.
    pub fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}
