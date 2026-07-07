//! Explicit memory copy operations between host and device.
//!
//! This module provides freestanding functions for copying data between
//! host memory, device memory, and pinned host memory.  Each function
//! validates that the source and destination have matching lengths before
//! issuing the underlying CUDA driver call.
//!
//! For simple cases, the methods on [`DeviceBuffer`]
//! (e.g. [`DeviceBuffer::copy_from_host`]) are more
//! ergonomic.  These freestanding functions are useful when you want to be
//! explicit about the direction of the transfer or when working with
//! [`PinnedBuffer`] for async operations.
//!
//! # Length validation
//!
//! All functions return [`CudaError::InvalidValue`] if the element counts
//! of source and destination do not match.

use std::ffi::c_void;

use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::stream::Stream;

use crate::device_buffer::DeviceBuffer;
use crate::host_buffer::PinnedBuffer;

// ---------------------------------------------------------------------------
// Synchronous copies
// ---------------------------------------------------------------------------

/// Copies data from a host slice into a device buffer (host-to-device).
///
/// This is a synchronous operation: it blocks the calling thread until the
/// transfer completes.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `src.len() != dst.len()`.
/// * Other driver errors from `cuMemcpyHtoD_v2`.
pub fn copy_htod<T: Copy>(dst: &mut DeviceBuffer<T>, src: &[T]) -> CudaResult<()> {
    if src.len() != dst.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = dst.byte_size();
    let api = try_driver()?;
    // SAFETY: `src` is a valid host slice, `dst` owns a valid device allocation,
    // and the byte counts match.
    let rc = unsafe {
        (api.cu_memcpy_htod_v2)(
            dst.as_device_ptr(),
            src.as_ptr().cast::<c_void>(),
            byte_size,
        )
    };
    oxicuda_driver::check(rc)
}

/// Copies data from a device buffer into a host slice (device-to-host).
///
/// This is a synchronous operation: it blocks the calling thread until the
/// transfer completes.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `dst.len() != src.len()`.
/// * Other driver errors from `cuMemcpyDtoH_v2`.
pub fn copy_dtoh<T: Copy>(dst: &mut [T], src: &DeviceBuffer<T>) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = src.byte_size();
    let api = try_driver()?;
    // SAFETY: `dst` is a valid host slice, `src` owns a valid device allocation,
    // and the byte counts match.
    let rc = unsafe {
        (api.cu_memcpy_dtoh_v2)(
            dst.as_mut_ptr().cast::<c_void>(),
            src.as_device_ptr(),
            byte_size,
        )
    };
    oxicuda_driver::check(rc)
}

/// Copies data from one device buffer to another (device-to-device).
///
/// This is a synchronous operation that blocks until the copy completes.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `dst.len() != src.len()`.
/// * Other driver errors from `cuMemcpyDtoD_v2`.
pub fn copy_dtod<T: Copy>(dst: &mut DeviceBuffer<T>, src: &DeviceBuffer<T>) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = src.byte_size();
    let api = try_driver()?;
    // SAFETY: both buffers own valid device allocations of the same size.
    let rc =
        unsafe { (api.cu_memcpy_dtod_v2)(dst.as_device_ptr(), src.as_device_ptr(), byte_size) };
    oxicuda_driver::check(rc)
}

// ---------------------------------------------------------------------------
// Asynchronous copies
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Asynchronous copies (raw slice variants)
// ---------------------------------------------------------------------------

/// Asynchronously copies data from a host slice into a device buffer.
///
/// The copy is enqueued on `stream` and may not be complete when this
/// function returns.  The caller must ensure that `src` remains valid
/// (i.e., is not moved or dropped) until the stream has been synchronised.
/// For guaranteed correctness with DMA, prefer using a [`PinnedBuffer`]
/// as the source.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `src.len() != dst.len()`.
/// * Other driver errors from `cuMemcpyHtoDAsync_v2`.
pub fn copy_htod_async_raw<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    src: &[T],
    stream: &Stream,
) -> CudaResult<()> {
    if src.len() != dst.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = dst.byte_size();
    let api = try_driver()?;
    let rc = unsafe {
        (api.cu_memcpy_htod_async_v2)(
            dst.as_device_ptr(),
            src.as_ptr().cast::<c_void>(),
            byte_size,
            stream.raw(),
        )
    };
    oxicuda_driver::check(rc)
}

/// Asynchronously copies data from a device buffer into a host slice.
///
/// The copy is enqueued on `stream` and may not be complete when this
/// function returns.  The caller must ensure that `dst` remains valid
/// and is not read until the stream has been synchronised.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `dst.len() != src.len()`.
/// * Other driver errors from `cuMemcpyDtoHAsync_v2`.
pub fn copy_dtoh_async_raw<T: Copy>(
    dst: &mut [T],
    src: &DeviceBuffer<T>,
    stream: &Stream,
) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = src.byte_size();
    let api = try_driver()?;
    let rc = unsafe {
        (api.cu_memcpy_dtoh_async_v2)(
            dst.as_mut_ptr().cast::<c_void>(),
            src.as_device_ptr(),
            byte_size,
            stream.raw(),
        )
    };
    oxicuda_driver::check(rc)
}

/// Asynchronously copies data from one device buffer to another.
///
/// Both buffers must have the same length.  The copy is enqueued on
/// `stream` via `cuMemcpyDtoDAsync_v2` and may not be complete when this
/// function returns; the caller must synchronise `stream` (or otherwise
/// order subsequent accesses) before reading `dst` or reusing `src`.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `dst.len() != src.len()`.
/// * [`CudaError::NotSupported`] if the loaded driver predates CUDA 4.0 and
///   does not export `cuMemcpyDtoDAsync_v2`.
/// * Other driver errors from `cuMemcpyDtoDAsync_v2`.
pub fn copy_dtod_async<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    src: &DeviceBuffer<T>,
    stream: &Stream,
) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = src.byte_size();
    oxicuda_driver::memory_info::memcpy_device_to_device_async(
        dst.as_device_ptr(),
        src.as_device_ptr(),
        byte_size,
        stream,
    )
}

// ---------------------------------------------------------------------------
// Asynchronous copies (pinned buffer variants)
// ---------------------------------------------------------------------------

/// Asynchronously copies data from a pinned host buffer into a device buffer.
///
/// The copy is enqueued on `stream` and may not be complete when this
/// function returns.  The caller must not modify `src` or read `dst` until
/// the stream has been synchronised.
///
/// Using a [`PinnedBuffer`] as the source guarantees that the host memory
/// is page-locked, which is required for correct async DMA transfers.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `src.len() != dst.len()`.
/// * Other driver errors from `cuMemcpyHtoDAsync_v2`.
pub fn copy_htod_async<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    src: &PinnedBuffer<T>,
    stream: &Stream,
) -> CudaResult<()> {
    if src.len() != dst.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = dst.byte_size();
    let api = try_driver()?;
    // SAFETY: `src` is pinned host memory, `dst` is a valid device allocation,
    // byte counts match, and the stream will order the transfer.
    let rc = unsafe {
        (api.cu_memcpy_htod_async_v2)(
            dst.as_device_ptr(),
            src.as_ptr().cast::<c_void>(),
            byte_size,
            stream.raw(),
        )
    };
    oxicuda_driver::check(rc)
}

/// Asynchronously copies data from a device buffer into a pinned host buffer.
///
/// The copy is enqueued on `stream` and may not be complete when this
/// function returns.  The caller must not read `dst` until the stream
/// has been synchronised.
///
/// Using a [`PinnedBuffer`] as the destination guarantees that the host
/// memory is page-locked, which is required for correct async DMA transfers.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `dst.len() != src.len()`.
/// * Other driver errors from `cuMemcpyDtoHAsync_v2`.
pub fn copy_dtoh_async<T: Copy>(
    dst: &mut PinnedBuffer<T>,
    src: &DeviceBuffer<T>,
    stream: &Stream,
) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = src.byte_size();
    let api = try_driver()?;
    // SAFETY: `dst` is pinned host memory, `src` is a valid device allocation,
    // byte counts match, and the stream will order the transfer.
    let rc = unsafe {
        (api.cu_memcpy_dtoh_async_v2)(
            dst.as_mut_ptr().cast::<c_void>(),
            src.as_device_ptr(),
            byte_size,
            stream.raw(),
        )
    };
    oxicuda_driver::check(rc)
}

// ---------------------------------------------------------------------------
// Asynchronous sub-region copies (pinned buffer staging)
// ---------------------------------------------------------------------------

/// Asynchronously copies a contiguous sub-region of a device buffer into a
/// pinned host buffer.
///
/// Exactly `count` elements starting at element index `src_offset` within
/// `src` are copied into `dst[0..count]`.  The pinned buffer must be large
/// enough to receive `count` elements.
///
/// This is the device→host leg of a host-staged inter-device transfer: the
/// caller stages a slab slice into pinned memory here, then pushes it onto a
/// different device with [`copy_htod_region_async`].
///
/// The copy is enqueued on `stream`; the caller must synchronise the stream
/// before reading `dst`.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `src_offset + count` exceeds `src.len()`,
///   if `count` exceeds `dst.len()`, or on offset overflow.
/// * Other driver errors from `cuMemcpyDtoHAsync_v2`.
pub fn copy_dtoh_region_async<T: Copy>(
    dst: &mut PinnedBuffer<T>,
    src: &DeviceBuffer<T>,
    src_offset: usize,
    count: usize,
    stream: &Stream,
) -> CudaResult<()> {
    let elem_size = std::mem::size_of::<T>();
    let src_end = src_offset
        .checked_add(count)
        .ok_or(CudaError::InvalidValue)?;
    if src_end > src.len() || count > dst.len() {
        return Err(CudaError::InvalidValue);
    }
    if count == 0 {
        return Ok(());
    }
    let byte_count = count
        .checked_mul(elem_size)
        .ok_or(CudaError::InvalidValue)?;
    let src_byte_offset = src_offset
        .checked_mul(elem_size)
        .ok_or(CudaError::InvalidValue)? as u64;
    let api = try_driver()?;
    // SAFETY: `dst` is pinned host memory with room for `count` elements,
    // the source sub-range lies within `src`, and byte counts match.
    let rc = unsafe {
        (api.cu_memcpy_dtoh_async_v2)(
            dst.as_mut_ptr().cast::<c_void>(),
            src.as_device_ptr() + src_byte_offset,
            byte_count,
            stream.raw(),
        )
    };
    oxicuda_driver::check(rc)
}

/// Asynchronously copies from a pinned host buffer into a contiguous
/// sub-region of a device buffer.
///
/// The first `count` elements of `src` are written into `dst` starting at
/// element index `dst_offset`.
///
/// This is the host→device leg of a host-staged inter-device transfer; see
/// [`copy_dtoh_region_async`] for the device→host leg.
///
/// The copy is enqueued on `stream`; the caller must synchronise the stream
/// before reusing `src`.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `dst_offset + count` exceeds `dst.len()`,
///   if `count` exceeds `src.len()`, or on offset overflow.
/// * Other driver errors from `cuMemcpyHtoDAsync_v2`.
pub fn copy_htod_region_async<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    dst_offset: usize,
    src: &PinnedBuffer<T>,
    count: usize,
    stream: &Stream,
) -> CudaResult<()> {
    let elem_size = std::mem::size_of::<T>();
    let dst_end = dst_offset
        .checked_add(count)
        .ok_or(CudaError::InvalidValue)?;
    if dst_end > dst.len() || count > src.len() {
        return Err(CudaError::InvalidValue);
    }
    if count == 0 {
        return Ok(());
    }
    let byte_count = count
        .checked_mul(elem_size)
        .ok_or(CudaError::InvalidValue)?;
    let dst_byte_offset = dst_offset
        .checked_mul(elem_size)
        .ok_or(CudaError::InvalidValue)? as u64;
    let api = try_driver()?;
    // SAFETY: `src` is pinned host memory holding at least `count` elements,
    // the destination sub-range lies within `dst`, and byte counts match.
    let rc = unsafe {
        (api.cu_memcpy_htod_async_v2)(
            dst.as_device_ptr() + dst_byte_offset,
            src.as_ptr().cast::<c_void>(),
            byte_count,
            stream.raw(),
        )
    };
    oxicuda_driver::check(rc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn copy_htod_signature_compiles() {
        let _f: fn(&mut super::DeviceBuffer<f32>, &[f32]) -> super::CudaResult<()> =
            super::copy_htod;
        let _f2: fn(&mut [f32], &super::DeviceBuffer<f32>) -> super::CudaResult<()> =
            super::copy_dtoh;
    }

    #[test]
    fn copy_dtod_signature_compiles() {
        let _f: fn(
            &mut super::DeviceBuffer<f32>,
            &super::DeviceBuffer<f32>,
        ) -> super::CudaResult<()> = super::copy_dtod;
    }

    #[test]
    fn async_raw_htod_signature_compiles() {
        let _f: fn(
            &mut super::DeviceBuffer<f32>,
            &[f32],
            &oxicuda_driver::stream::Stream,
        ) -> super::CudaResult<()> = super::copy_htod_async_raw;
    }

    #[test]
    fn async_raw_dtoh_signature_compiles() {
        let _f: fn(
            &mut [f32],
            &super::DeviceBuffer<f32>,
            &oxicuda_driver::stream::Stream,
        ) -> super::CudaResult<()> = super::copy_dtoh_async_raw;
    }

    #[test]
    fn async_dtod_signature_compiles() {
        let _f: fn(
            &mut super::DeviceBuffer<f32>,
            &super::DeviceBuffer<f32>,
            &oxicuda_driver::stream::Stream,
        ) -> super::CudaResult<()> = super::copy_dtod_async;
    }

    #[test]
    fn async_pinned_htod_signature_compiles() {
        let _f: fn(
            &mut super::DeviceBuffer<f32>,
            &super::PinnedBuffer<f32>,
            &oxicuda_driver::stream::Stream,
        ) -> super::CudaResult<()> = super::copy_htod_async;
    }

    #[test]
    fn region_dtoh_signature_compiles() {
        type RegionDtohFn = fn(
            &mut super::PinnedBuffer<f32>,
            &super::DeviceBuffer<f32>,
            usize,
            usize,
            &oxicuda_driver::stream::Stream,
        ) -> super::CudaResult<()>;
        let _f: RegionDtohFn = super::copy_dtoh_region_async;
    }

    #[test]
    fn region_htod_signature_compiles() {
        type RegionHtodFn = fn(
            &mut super::DeviceBuffer<f32>,
            usize,
            &super::PinnedBuffer<f32>,
            usize,
            &oxicuda_driver::stream::Stream,
        ) -> super::CudaResult<()>;
        let _f: RegionHtodFn = super::copy_htod_region_async;
    }

    /// Regression test for F035: `copy_dtod_async` must actually enqueue a
    /// real `cuMemcpyDtoDAsync_v2` on the given stream (rather than silently
    /// falling back to a synchronous legacy-stream copy) and produce
    /// correct data once the stream is synchronised.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn copy_dtod_async_round_trips_on_device() {
        if oxicuda_driver::init().is_err() {
            eprintln!("skipping: CUDA init failed");
            return;
        }
        let Ok(dev) = oxicuda_driver::device::Device::get(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let Ok(ctx) = oxicuda_driver::context::Context::new(&dev) else {
            eprintln!("skipping: context creation failed");
            return;
        };
        let ctx = std::sync::Arc::new(ctx);
        let Ok(stream) = oxicuda_driver::stream::Stream::new(&ctx) else {
            eprintln!("skipping: stream creation failed");
            return;
        };

        let host_src: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
        let Ok(src) = super::DeviceBuffer::<f32>::from_host(&host_src) else {
            eprintln!("skipping: src alloc failed");
            return;
        };
        let Ok(mut dst) = super::DeviceBuffer::<f32>::from_host(&vec![0.0f32; 1024]) else {
            eprintln!("skipping: dst alloc failed");
            return;
        };

        match super::copy_dtod_async(&mut dst, &src, &stream) {
            Ok(()) => {}
            Err(oxicuda_driver::error::CudaError::NotSupported) => {
                eprintln!("skipping: driver lacks cuMemcpyDtoDAsync_v2");
                return;
            }
            Err(e) => panic!("copy_dtod_async failed: {e:?}"),
        }
        stream.synchronize().expect("stream sync failed");

        let mut host_out = vec![0.0f32; 1024];
        dst.copy_to_host(&mut host_out).expect("copy back failed");
        assert_eq!(host_out, host_src);
    }
}
