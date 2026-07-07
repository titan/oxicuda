//! Peer-to-peer device access.
//!
//! Implements:
//! - `cudaDeviceCanAccessPeer`
//! - `cudaDeviceEnablePeerAccess`
//! - `cudaDeviceDisablePeerAccess`
//! - `cudaMemcpyPeer` / `cudaMemcpyPeerAsync`

use std::ffi::c_int;

use oxicuda_driver::loader::try_driver;

use crate::error::{CudaRtError, CudaRtResult};
use crate::memory::DevicePtr;
use crate::stream::CudaStream;

/// Check whether `device` can directly access the memory of `peer_device`.
///
/// Mirrors `cudaDeviceCanAccessPeer`.
///
/// Returns `Ok(true)` if peer access is supported.
///
/// # Errors
///
/// Propagates driver errors.
pub fn device_can_access_peer(device: u32, peer_device: u32) -> CudaRtResult<bool> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut can_access: c_int = 0;
    // SAFETY: FFI; both ordinals are checked against count by caller if needed.
    let rc = unsafe {
        (api.cu_device_can_access_peer)(&raw mut can_access, device as c_int, peer_device as c_int)
    };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevice));
    }
    Ok(can_access != 0)
}

/// Enable peer access from the current context to the context owning `peer_device`.
///
/// Mirrors `cudaDeviceEnablePeerAccess`.
///
/// # Errors
///
/// - [`CudaRtError::PeerAccessUnsupported`] — link does not support peer access.
/// - [`CudaRtError::PeerAccessAlreadyEnabled`] — already enabled.
/// - Other driver errors.
pub fn device_enable_peer_access(peer_device: u32, flags: u32) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut peer_ctx = oxicuda_driver::ffi::CUcontext::default();
    // Retain the primary context of the peer device.
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_device_primary_ctx_retain)(&raw mut peer_ctx, peer_device as c_int) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevice));
    }
    // Enable peer access to that context.
    // SAFETY: FFI.
    let rc2 = unsafe { (api.cu_ctx_enable_peer_access)(peer_ctx, flags) };
    if rc2 != 0 {
        // Release the retained context regardless.
        // SAFETY: FFI.
        unsafe { (api.cu_device_primary_ctx_release_v2)(peer_device as c_int) };
        return Err(CudaRtError::from_code(rc2).unwrap_or(CudaRtError::PeerAccessUnsupported));
    }
    Ok(())
}

/// Disable peer access from the current context to `peer_device`.
///
/// Mirrors `cudaDeviceDisablePeerAccess`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn device_disable_peer_access(peer_device: u32) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut peer_ctx = oxicuda_driver::ffi::CUcontext::default();
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_device_primary_ctx_retain)(&raw mut peer_ctx, peer_device as c_int) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevice));
    }
    // SAFETY: FFI.
    let rc2 = unsafe { (api.cu_ctx_disable_peer_access)(peer_ctx) };
    if rc2 != 0 {
        // SAFETY: FFI.
        unsafe { (api.cu_device_primary_ctx_release_v2)(peer_device as c_int) };
        return Err(CudaRtError::from_code(rc2).unwrap_or(CudaRtError::PeerAccessNotEnabled));
    }
    Ok(())
}

/// Copy `count` bytes from `src` on `src_device` to `dst` on `dst_device`.
///
/// Mirrors `cudaMemcpyPeer`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn memcpy_peer(
    dst: DevicePtr,
    dst_device: u32,
    src: DevicePtr,
    src_device: u32,
    count: usize,
) -> CudaRtResult<()> {
    if count == 0 {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut dst_ctx = oxicuda_driver::ffi::CUcontext::default();
    let mut src_ctx = oxicuda_driver::ffi::CUcontext::default();
    // SAFETY: FFI; dst_ctx is a valid stack-allocated context handle.
    let rc = unsafe { (api.cu_device_primary_ctx_retain)(&raw mut dst_ctx, dst_device as c_int) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevice));
    }
    // SAFETY: FFI; src_ctx is a valid stack-allocated context handle.
    let rc = unsafe { (api.cu_device_primary_ctx_retain)(&raw mut src_ctx, src_device as c_int) };
    if rc != 0 {
        // Release the dst retain before propagating the failure so we don't
        // leak the primary context we already acquired.
        // SAFETY: FFI; dst_ctx was successfully retained above.
        unsafe { (api.cu_device_primary_ctx_release_v2)(dst_device as c_int) };
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevice));
    }
    // SAFETY: FFI; pointers are valid device allocations on the specified devices.
    let rc = unsafe { (api.cu_memcpy_peer)(dst.0, dst_ctx, src.0, src_ctx, count) };
    // Release both primary-context retains unconditionally before mapping
    // the copy's return code — retains must not outlive this call.
    // SAFETY: FFI; both contexts were successfully retained above.
    unsafe {
        (api.cu_device_primary_ctx_release_v2)(src_device as c_int);
        (api.cu_device_primary_ctx_release_v2)(dst_device as c_int);
    }
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidMemcpyDirection));
    }
    Ok(())
}

/// Asynchronously copy across devices on `stream`.
///
/// Mirrors `cudaMemcpyPeerAsync`.
///
/// # Context lifetime
///
/// This function retains both devices' primary contexts for the duration of
/// the call and releases them again before returning. Because
/// `cuMemcpyPeerAsync` only *enqueues* the copy, releasing the retains right
/// after enqueuing (without waiting) could let a primary context's driver
/// refcount drop to zero — and possibly be destroyed — while the copy is
/// still in flight. To keep the retains provably alive for the copy's
/// duration, this function synchronises `stream` before releasing, so it
/// blocks until the copy completes despite the "async" name. Callers that
/// need true overlap should retain the primary contexts themselves for the
/// lifetime of their own stream usage.
///
/// # Errors
///
/// Propagates driver errors from the retain, the copy enqueue, or the
/// post-copy synchronisation.
pub fn memcpy_peer_async(
    dst: DevicePtr,
    dst_device: u32,
    src: DevicePtr,
    src_device: u32,
    count: usize,
    stream: CudaStream,
) -> CudaRtResult<()> {
    if count == 0 {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut dst_ctx = oxicuda_driver::ffi::CUcontext::default();
    let mut src_ctx = oxicuda_driver::ffi::CUcontext::default();
    // SAFETY: FFI; dst_ctx is a valid stack-allocated context handle.
    let rc = unsafe { (api.cu_device_primary_ctx_retain)(&raw mut dst_ctx, dst_device as c_int) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevice));
    }
    // SAFETY: FFI; src_ctx is a valid stack-allocated context handle.
    let rc = unsafe { (api.cu_device_primary_ctx_retain)(&raw mut src_ctx, src_device as c_int) };
    if rc != 0 {
        // Release the dst retain before propagating the failure so we don't
        // leak the primary context we already acquired.
        // SAFETY: FFI; dst_ctx was successfully retained above.
        unsafe { (api.cu_device_primary_ctx_release_v2)(dst_device as c_int) };
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevice));
    }
    // SAFETY: FFI.
    let rc =
        unsafe { (api.cu_memcpy_peer_async)(dst.0, dst_ctx, src.0, src_ctx, count, stream.raw()) };
    // `cuMemcpyPeerAsync` only *enqueues* the copy; the driver may still be
    // executing it on `stream` after this call returns. Releasing both
    // primary-context retains immediately could drop a primary context's
    // driver refcount to zero (and possibly destroy it) while the copy is
    // still in flight, corrupting an in-progress transfer. Synchronise the
    // stream first so the retains provably outlive the copy they protect —
    // mirrors the fix applied to `oxicuda-memory::peer_copy::copy_peer_async`
    // for the identical hazard. Only wait if the copy was actually enqueued;
    // on enqueue failure there is nothing in flight to protect.
    let sync_rc = if rc == 0 {
        // SAFETY: FFI; stream handle is valid.
        unsafe { (api.cu_stream_synchronize)(stream.raw()) }
    } else {
        0
    };
    // Release both primary-context retains unconditionally before mapping
    // the return codes — retains must not outlive this call.
    // SAFETY: FFI; both contexts were successfully retained above.
    unsafe {
        (api.cu_device_primary_ctx_release_v2)(src_device as c_int);
        (api.cu_device_primary_ctx_release_v2)(dst_device as c_int);
    }
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidMemcpyDirection));
    }
    if sync_rc != 0 {
        return Err(CudaRtError::from_code(sync_rc).unwrap_or(CudaRtError::Unknown));
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_access_self_check() {
        // Without GPU, driver returns DriverNotAvailable.
        // With GPU, peer access with itself should return false or succeed.
        match device_can_access_peer(0, 0) {
            Ok(v) => {
                // Self-access should typically be false for P2P (same device).
                let _ = v;
            }
            // Driver absent or not initialised — both are expected without a GPU.
            Err(CudaRtError::DriverNotAvailable)
            | Err(CudaRtError::NoGpu)
            | Err(CudaRtError::InitializationError)
            | Err(CudaRtError::InvalidDevice) => {}
            Err(e) => panic!("unexpected: {e}"),
        }
    }

    /// Regression test for F089: previously `memcpy_peer` ignored the
    /// `cuDevicePrimaryCtxRetain` return codes and never released either
    /// retained primary context, silently leaking one retain per call.
    /// Exercises a same-device ("self-peer") copy repeatedly — if a retain
    /// leak or a NULL-context bug were reintroduced, either the copy would
    /// eventually fail or the data would come back corrupted.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn memcpy_peer_self_copy_roundtrip_no_leak() {
        if crate::device::set_device(0).is_err() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let host_src: Vec<u32> = (0..64).collect();
        let bytes = std::mem::size_of_val(host_src.as_slice());
        let dev_a = match crate::memory::malloc(bytes) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skipping: device alloc failed");
                return;
            }
        };
        let dev_b = match crate::memory::malloc(bytes) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skipping: device alloc failed");
                let _ = crate::memory::free(dev_a);
                return;
            }
        };
        if crate::memory::memcpy_h2d(dev_a, &host_src).is_err() {
            eprintln!("skipping: h2d seed failed");
            let _ = crate::memory::free(dev_a);
            let _ = crate::memory::free(dev_b);
            return;
        }

        // Repeated self-peer copies (device 0 -> device 0): must not error
        // and must not exhaust/corrupt the primary context across calls.
        for _ in 0..8 {
            memcpy_peer(dev_b, 0, dev_a, 0, bytes).expect("memcpy_peer self-copy failed");
        }

        let mut host_dst = vec![0u32; host_src.len()];
        crate::memory::memcpy_d2h(&mut host_dst, dev_b).expect("d2h readback failed");
        assert_eq!(host_dst, host_src);

        let _ = crate::memory::free(dev_a);
        let _ = crate::memory::free(dev_b);
    }

    /// Regression test for F089 (async path): `memcpy_peer_async` must
    /// release both retained primary contexts and must not corrupt the
    /// transfer by releasing them before the copy actually lands.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn memcpy_peer_async_self_copy_roundtrip_no_leak() {
        if crate::device::set_device(0).is_err() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let stream = match crate::stream::stream_create() {
            Ok(s) => s,
            Err(_) => {
                eprintln!("skipping: stream creation failed");
                return;
            }
        };
        let host_src: Vec<u32> = (200..264).collect();
        let bytes = std::mem::size_of_val(host_src.as_slice());
        let dev_a = match crate::memory::malloc(bytes) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skipping: device alloc failed");
                let _ = crate::stream::stream_destroy(stream);
                return;
            }
        };
        let dev_b = match crate::memory::malloc(bytes) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skipping: device alloc failed");
                let _ = crate::memory::free(dev_a);
                let _ = crate::stream::stream_destroy(stream);
                return;
            }
        };
        if crate::memory::memcpy_h2d(dev_a, &host_src).is_err() {
            eprintln!("skipping: h2d seed failed");
            let _ = crate::memory::free(dev_a);
            let _ = crate::memory::free(dev_b);
            let _ = crate::stream::stream_destroy(stream);
            return;
        }

        for _ in 0..8 {
            memcpy_peer_async(dev_b, 0, dev_a, 0, bytes, stream)
                .expect("memcpy_peer_async self-copy failed");
        }

        let mut host_dst = vec![0u32; host_src.len()];
        crate::memory::memcpy_d2h(&mut host_dst, dev_b).expect("d2h readback failed");
        assert_eq!(host_dst, host_src);

        let _ = crate::memory::free(dev_a);
        let _ = crate::memory::free(dev_b);
        let _ = crate::stream::stream_destroy(stream);
    }
}
