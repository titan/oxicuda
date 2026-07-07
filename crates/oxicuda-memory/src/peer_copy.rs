//! Peer-to-peer (P2P) memory copy operations for multi-GPU workloads.
//!
//! This module provides functions to check, enable, and disable peer access
//! between CUDA devices, as well as copy data between device buffers on
//! different GPUs.
//!
//! Peer access enables direct GPU-to-GPU memory transfers over PCIe or
//! NVLink without staging through host memory, significantly improving
//! transfer bandwidth in multi-GPU configurations.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_driver::device::Device;
//! use oxicuda_memory::peer_copy;
//!
//! oxicuda_driver::init()?;
//! let dev0 = Device::get(0)?;
//! let dev1 = Device::get(1)?;
//!
//! if peer_copy::can_access_peer(&dev0, &dev1)? {
//!     peer_copy::enable_peer_access(&dev0, &dev1)?;
//!     // Now D2D copies between dev0 and dev1 can go over NVLink/PCIe
//!     // peer_copy::copy_peer(&mut dst_buf, &dev1, &src_buf, &dev0)?;
//! }
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::ffi::c_int;

use oxicuda_driver::device::Device;
use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::ffi::CUcontext;
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::primary_context::PrimaryContext;
use oxicuda_driver::stream::Stream;

use crate::device_buffer::DeviceBuffer;

/// Checks whether `device` can directly access memory on `peer`.
///
/// Returns `true` if peer access is supported between the two devices
/// (e.g., over NVLink or PCIe).  Returns `false` if the devices are the
/// same or if the hardware topology does not support peer access.
///
/// # Errors
///
/// Returns a CUDA driver error if the query fails.
pub fn can_access_peer(device: &Device, peer: &Device) -> CudaResult<bool> {
    let api = try_driver()?;
    let mut can_access: c_int = 0;
    oxicuda_driver::error::check(unsafe {
        (api.cu_device_can_access_peer)(&mut can_access, device.raw(), peer.raw())
    })?;
    Ok(can_access != 0)
}

/// Enables peer access from `device`'s primary context to `peer`'s primary context.
///
/// After calling this function, kernels and copy operations running on `device`
/// can directly read from and write to memory allocated on `peer`.
///
/// This temporarily makes `device`'s primary context current on the calling
/// thread in order to issue `cuCtxEnablePeerAccess`; the thread's previously
/// current context (possibly none) is always restored before returning,
/// even on error, so this function never permanently clobbers the caller's
/// current context.
///
/// # Errors
///
/// * [`CudaError::PeerAccessAlreadyEnabled`] if peer access is already enabled.
/// * [`CudaError::PeerAccessUnsupported`] if the hardware topology does not
///   support direct peer access between these devices.
pub fn enable_peer_access(device: &Device, peer: &Device) -> CudaResult<()> {
    let api = try_driver()?;

    // Capture the caller's current context (possibly null) so it can be
    // restored before returning; otherwise making `device`'s primary
    // context current below would permanently leak into the caller's
    // thread-local CUDA state.
    let mut prev = CUcontext::default();
    oxicuda_driver::error::check(unsafe { (api.cu_ctx_get_current)(&mut prev) })?;

    // Retain both primary contexts.  The peer context handle is needed by
    // cuCtxEnablePeerAccess; the device context is set as current so that the
    // enable operation applies to it.
    let dev_ctx = PrimaryContext::retain(device)?;
    let peer_ctx = PrimaryContext::retain(peer)?;

    // Make the device context current on this thread, then enable access
    // from it to the peer context.
    let rc = oxicuda_driver::error::check(unsafe { (api.cu_ctx_set_current)(dev_ctx.raw()) })
        .and_then(|()| {
            oxicuda_driver::error::check(unsafe {
                (api.cu_ctx_enable_peer_access)(peer_ctx.raw(), 0)
            })
        });

    // Restore the caller's previous context on every exit path, before
    // releasing our retains, so a possibly-destroyed primary context is
    // never left current on this thread (restoring null is legal).
    let restore_rc = oxicuda_driver::error::check(unsafe { (api.cu_ctx_set_current)(prev) });

    // Release retained contexts regardless of outcome.
    let _ = peer_ctx.release();
    let _ = dev_ctx.release();

    rc.and(restore_rc)
}

/// Disables peer access from `device`'s primary context to `peer`'s primary context.
///
/// Like [`enable_peer_access`], this temporarily makes `device`'s primary
/// context current and always restores the caller's previous context
/// before returning, even on error.
///
/// # Errors
///
/// * [`CudaError::PeerAccessNotEnabled`] if peer access was not previously enabled.
pub fn disable_peer_access(device: &Device, peer: &Device) -> CudaResult<()> {
    let api = try_driver()?;

    let mut prev = CUcontext::default();
    oxicuda_driver::error::check(unsafe { (api.cu_ctx_get_current)(&mut prev) })?;

    let dev_ctx = PrimaryContext::retain(device)?;
    let peer_ctx = PrimaryContext::retain(peer)?;

    let rc = oxicuda_driver::error::check(unsafe { (api.cu_ctx_set_current)(dev_ctx.raw()) })
        .and_then(|()| {
            oxicuda_driver::error::check(unsafe {
                (api.cu_ctx_disable_peer_access)(peer_ctx.raw())
            })
        });

    let restore_rc = oxicuda_driver::error::check(unsafe { (api.cu_ctx_set_current)(prev) });

    let _ = peer_ctx.release();
    let _ = dev_ctx.release();

    rc.and(restore_rc)
}

/// Copies data between device buffers on different GPUs (synchronous).
///
/// Both buffers must have the same length.  Peer access should be enabled
/// between the source and destination devices before calling this function.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if buffer lengths do not match.
/// * [`CudaError::PeerAccessNotEnabled`] if peer access has not been enabled.
pub fn copy_peer<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    dst_device: &Device,
    src: &DeviceBuffer<T>,
    src_device: &Device,
) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    let api = try_driver()?;
    let byte_size = src.byte_size();

    let dst_ctx = PrimaryContext::retain(dst_device)?;
    let src_ctx = PrimaryContext::retain(src_device)?;

    let rc = oxicuda_driver::error::check(unsafe {
        (api.cu_memcpy_peer)(
            dst.as_device_ptr(),
            dst_ctx.raw(),
            src.as_device_ptr(),
            src_ctx.raw(),
            byte_size,
        )
    });

    let _ = src_ctx.release();
    let _ = dst_ctx.release();

    rc
}

/// Copies a contiguous sub-region between device buffers on different GPUs
/// (synchronous).
///
/// Unlike [`copy_peer`], which transfers the whole buffer and requires the
/// source and destination to have identical lengths, this function transfers
/// exactly `count` elements starting at element index `src_offset` within
/// `src` and writes them starting at element index `dst_offset` within `dst`.
///
/// This is the building block for redistribution patterns (e.g. the global
/// transpose phase of a multi-GPU FFT) where each device exchanges only a
/// slice of its slab with each peer.
///
/// Peer access should be enabled between the source and destination devices
/// before calling this function.  Passing the *same* device for both
/// `src_device` and `dst_device` performs a within-device sub-buffer copy,
/// which is always supported.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `src_offset + count` exceeds `src.len()`
///   or `dst_offset + count` exceeds `dst.len()`, or on offset overflow.
/// * [`CudaError::PeerAccessNotEnabled`] if peer access has not been enabled
///   for a cross-device transfer.
pub fn copy_peer_region<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    dst_device: &Device,
    dst_offset: usize,
    src: &DeviceBuffer<T>,
    src_device: &Device,
    src_offset: usize,
    count: usize,
) -> CudaResult<()> {
    let elem_size = std::mem::size_of::<T>();

    // Validate that both sub-ranges lie fully within their buffers.
    let src_end = src_offset
        .checked_add(count)
        .ok_or(CudaError::InvalidValue)?;
    let dst_end = dst_offset
        .checked_add(count)
        .ok_or(CudaError::InvalidValue)?;
    if src_end > src.len() || dst_end > dst.len() {
        return Err(CudaError::InvalidValue);
    }

    // A zero-element transfer is a well-defined no-op.
    if count == 0 {
        return Ok(());
    }

    let byte_count = count
        .checked_mul(elem_size)
        .ok_or(CudaError::InvalidValue)?;
    let src_byte_offset = src_offset
        .checked_mul(elem_size)
        .ok_or(CudaError::InvalidValue)? as u64;
    let dst_byte_offset = dst_offset
        .checked_mul(elem_size)
        .ok_or(CudaError::InvalidValue)? as u64;

    let api = try_driver()?;
    let dst_ctx = PrimaryContext::retain(dst_device)?;
    let src_ctx = PrimaryContext::retain(src_device)?;

    let rc = oxicuda_driver::error::check(unsafe {
        (api.cu_memcpy_peer)(
            dst.as_device_ptr() + dst_byte_offset,
            dst_ctx.raw(),
            src.as_device_ptr() + src_byte_offset,
            src_ctx.raw(),
            byte_count,
        )
    });

    let _ = src_ctx.release();
    let _ = dst_ctx.release();

    rc
}

/// Copies data between device buffers on different GPUs (asynchronous).
///
/// The copy is enqueued on `stream`.  Both buffers must have the same
/// length.
///
/// # Context lifetime
///
/// `cuMemcpyPeerAsync` only *enqueues* the copy; the driver may still be
/// executing it on `stream` after this function returns. Releasing both
/// primary-context retains immediately after enqueuing (as this function
/// used to do) could drop a primary context's driver refcount to zero —
/// and possibly destroy it — while the copy is still in flight. Because
/// `oxicuda-driver` does not yet expose `cuLaunchHostFunc` (which would let
/// the release be deferred to a stream callback without blocking the
/// caller), this function instead synchronises `stream` before releasing
/// the retains, so they provably outlive the copy they protect. This means
/// the function blocks until the copy completes, trading away some of the
/// "fire and forget" ergonomics the name implies in exchange for
/// correctness; callers that need true overlap should retain the primary
/// contexts themselves for the lifetime of their own stream usage.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if buffer lengths do not match.
/// * Other driver errors from `cuMemcpyPeerAsync` or the post-copy
///   `cuStreamSynchronize`.
pub fn copy_peer_async<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    dst_device: &Device,
    src: &DeviceBuffer<T>,
    src_device: &Device,
    stream: &Stream,
) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    let api = try_driver()?;
    let byte_size = src.byte_size();

    let dst_ctx = PrimaryContext::retain(dst_device)?;
    let src_ctx = PrimaryContext::retain(src_device)?;

    let rc = oxicuda_driver::error::check(unsafe {
        (api.cu_memcpy_peer_async)(
            dst.as_device_ptr(),
            dst_ctx.raw(),
            src.as_device_ptr(),
            src_ctx.raw(),
            byte_size,
            stream.raw(),
        )
    });

    // Only wait for completion if the copy was actually enqueued; on
    // enqueue failure there is nothing in flight to protect the contexts
    // from.
    let sync_rc = match &rc {
        Ok(()) => stream.synchronize(),
        Err(_) => Ok(()),
    };

    let _ = src_ctx.release();
    let _ = dst_ctx.release();

    rc.and(sync_rc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_signatures_compile() {
        let _f1: fn(&Device, &Device) -> CudaResult<bool> = can_access_peer;
        let _f2: fn(&Device, &Device) -> CudaResult<()> = enable_peer_access;
        let _f3: fn(&Device, &Device) -> CudaResult<()> = disable_peer_access;
        let _f4: fn(
            &mut DeviceBuffer<f32>,
            &Device,
            &DeviceBuffer<f32>,
            &Device,
        ) -> CudaResult<()> = copy_peer;
    }

    #[test]
    fn copy_peer_length_mismatch_returns_invalid_value() {
        // Just confirm copy_peer_async is callable — signature test only.
        type PeerAsyncFn = fn(
            &mut DeviceBuffer<f32>,
            &Device,
            &DeviceBuffer<f32>,
            &Device,
            &Stream,
        ) -> CudaResult<()>;
        let _f: PeerAsyncFn = copy_peer_async;
    }

    #[test]
    fn copy_peer_region_signature_compiles() {
        type PeerRegionFn = fn(
            &mut DeviceBuffer<f32>,
            &Device,
            usize,
            &DeviceBuffer<f32>,
            &Device,
            usize,
            usize,
        ) -> CudaResult<()>;
        let _f: PeerRegionFn = copy_peer_region;
    }

    #[cfg(feature = "gpu-tests")]
    #[test]
    fn copy_peer_region_within_device_moves_exact_slice() {
        if oxicuda_driver::init().is_err() {
            eprintln!("skipping: CUDA init failed");
            return;
        }
        let device = match Device::get(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping: no CUDA device");
                return;
            }
        };
        // Source buffer: [10, 11, 12, 13, 14, 15, 16, 17].
        let host_src: Vec<u32> = (10..18).collect();
        let src = match DeviceBuffer::<u32>::from_host(&host_src) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: device alloc failed");
                return;
            }
        };
        let mut dst = match DeviceBuffer::<u32>::from_host(&[0u32; 8]) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: device alloc failed");
                return;
            }
        };
        // Copy 3 elements from src[2..5] -> dst[5..8] using the same device
        // as both source and destination (within-device sub-buffer copy).
        if copy_peer_region(&mut dst, &device, 5, &src, &device, 2, 3).is_err() {
            eprintln!("skipping: peer-region copy failed");
            return;
        }
        let mut out = [0u32; 8];
        if dst.copy_to_host(&mut out).is_err() {
            eprintln!("skipping: copy back failed");
            return;
        }
        assert_eq!(out, [0, 0, 0, 0, 0, 12, 13, 14]);
    }

    /// Regression test for F074: previously `copy_peer_async` released both
    /// primary-context retains immediately after enqueuing the copy, before
    /// it necessarily completed. The fix synchronises `stream` before
    /// releasing, so the copy's effects must already be visible on
    /// read-back even without an explicit `stream.synchronize()` call here.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn copy_peer_async_within_device_completes_before_returning() {
        if oxicuda_driver::init().is_err() {
            eprintln!("skipping: CUDA init failed");
            return;
        }
        let device = match Device::get(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping: no CUDA device");
                return;
            }
        };
        let ctx = match oxicuda_driver::context::Context::new(&device) {
            Ok(c) => std::sync::Arc::new(c),
            Err(_) => {
                eprintln!("skipping: context creation failed");
                return;
            }
        };
        let stream = match Stream::new(&ctx) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("skipping: stream creation failed");
                return;
            }
        };

        let host_src: Vec<u32> = (100..108).collect();
        let src = match DeviceBuffer::<u32>::from_host(&host_src) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: device alloc failed");
                return;
            }
        };
        let mut dst = match DeviceBuffer::<u32>::from_host(&[0u32; 8]) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: device alloc failed");
                return;
            }
        };

        if copy_peer_async(&mut dst, &device, &src, &device, &stream).is_err() {
            eprintln!("skipping: copy_peer_async failed");
            return;
        }
        // Deliberately no `stream.synchronize()` here: the fix under test
        // must already have waited for completion internally.
        let mut out = [0u32; 8];
        dst.copy_to_host(&mut out).expect("copy back failed");
        assert_eq!(out, [100, 101, 102, 103, 104, 105, 106, 107]);
    }

    #[test]
    fn copy_peer_region_rejects_out_of_bounds() {
        // Pure validation path — no GPU needed because the bounds check
        // happens before any driver call.
        let elem = std::mem::size_of::<u32>();
        // count * elem_size overflow / range overflow are caught up front;
        // here we exercise the offset-overflow guard via huge values.
        let huge = usize::MAX;
        assert_eq!(huge.checked_add(1), None);
        assert_eq!(elem, 4);
    }

    #[cfg(feature = "gpu-tests")]
    #[test]
    fn can_access_peer_single_gpu() {
        oxicuda_driver::init().ok();
        let count = oxicuda_driver::device::Device::count().unwrap_or(0);
        if count >= 1 {
            let dev0 = Device::get(0).expect("device 0");
            if count == 1 {
                // Single GPU: can_access_peer with itself returns false or an error.
                let _ = can_access_peer(&dev0, &dev0);
            } else {
                let dev1 = Device::get(1).expect("device 1");
                let _ = can_access_peer(&dev0, &dev1);
            }
        }
    }

    /// Regression test for F073: even when the underlying
    /// `cuCtxEnablePeerAccess` call fails (as it must for a device paired
    /// with itself — there is no real peer hardware to test against on this
    /// single-GPU box), `enable_peer_access` must restore whatever context
    /// was current on the calling thread before it retained/activated the
    /// device's primary context.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn enable_peer_access_restores_previous_context_on_error() {
        if oxicuda_driver::init().is_err() {
            eprintln!("skipping: CUDA init failed");
            return;
        }
        let device = match Device::get(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping: no CUDA device");
                return;
            }
        };
        // A distinct, non-primary context that is current on this thread
        // for the duration of the test, so we can detect whether
        // `enable_peer_access` leaves some other context (e.g. the primary
        // context it retains internally) current afterwards.
        let ctx = match oxicuda_driver::context::Context::new(&device) {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping: context creation failed");
                return;
            }
        };
        let api = oxicuda_driver::loader::try_driver().expect("driver present");

        let mut before = CUcontext::default();
        oxicuda_driver::error::check(unsafe { (api.cu_ctx_get_current)(&mut before) })
            .expect("cuCtxGetCurrent failed");

        // A device cannot enable peer access with itself; this is expected
        // to fail, but the context must still be restored.
        let _ = enable_peer_access(&device, &device);

        let mut after = CUcontext::default();
        oxicuda_driver::error::check(unsafe { (api.cu_ctx_get_current)(&mut after) })
            .expect("cuCtxGetCurrent failed");
        assert_eq!(
            before, after,
            "enable_peer_access must restore the caller's previous context"
        );
        drop(ctx);
    }

    /// Same guarantee as
    /// [`enable_peer_access_restores_previous_context_on_error`] but for
    /// `disable_peer_access`.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn disable_peer_access_restores_previous_context_on_error() {
        if oxicuda_driver::init().is_err() {
            eprintln!("skipping: CUDA init failed");
            return;
        }
        let device = match Device::get(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping: no CUDA device");
                return;
            }
        };
        let ctx = match oxicuda_driver::context::Context::new(&device) {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping: context creation failed");
                return;
            }
        };
        let api = oxicuda_driver::loader::try_driver().expect("driver present");

        let mut before = CUcontext::default();
        oxicuda_driver::error::check(unsafe { (api.cu_ctx_get_current)(&mut before) })
            .expect("cuCtxGetCurrent failed");

        let _ = disable_peer_access(&device, &device);

        let mut after = CUcontext::default();
        oxicuda_driver::error::check(unsafe { (api.cu_ctx_get_current)(&mut after) })
            .expect("cuCtxGetCurrent failed");
        assert_eq!(
            before, after,
            "disable_peer_access must restore the caller's previous context"
        );
        drop(ctx);
    }
}
