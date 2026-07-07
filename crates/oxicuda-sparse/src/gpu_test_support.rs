//! Shared on-device test fixtures for the `gpu-tests` feature.
//!
//! These helpers acquire a live CUDA context, build a [`SparseHandle`], and
//! compare device results against CPU oracles. The module is compiled only
//! under `#[cfg(all(test, feature = "gpu-tests"))]`, so it never affects a
//! normal (non-test) build and carries no runtime cost.

use std::sync::Arc;

use oxicuda_driver::Context;

use crate::handle::SparseHandle;

/// Acquires a live CUDA context bound to the calling thread and a ready
/// [`SparseHandle`], or returns `None` when no GPU is present so the test can
/// skip cleanly.
///
/// `oxicuda_driver::Context::new` calls `cuCtxCreate`, which both creates the
/// context and makes it current for the calling thread; the returned handle
/// keeps an `Arc<Context>` alive internally, so the context survives for the
/// lifetime of the handle.
pub(crate) fn gpu_handle() -> Option<SparseHandle> {
    oxicuda_driver::init().ok()?;
    if oxicuda_driver::Device::count().ok()? == 0 {
        return None;
    }
    let dev = oxicuda_driver::Device::get(0).ok()?;
    let ctx = Arc::new(Context::new(&dev).ok()?);
    Some(SparseHandle::new(&ctx).expect("test: build SparseHandle"))
}

/// Asserts each element of `got` matches `want` within `tol`, scaled by the
/// larger of `|want|` and 1 so the bound is meaningful for both small and
/// large magnitudes.
pub(crate) fn assert_close(got: &[f64], want: &[f64], tol: f64, ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        let scale = w.abs().max(1.0);
        assert!(
            diff <= tol * scale,
            "{ctx}: element {i}: got {g}, want {w} (|diff| {diff} > {bound})",
            bound = tol * scale
        );
    }
}
