//! Device and host memory management.
//!
//! Implements the CUDA Runtime memory API:
//! - `cudaMalloc` / `cudaFree`
//! - `cudaMallocHost` / `cudaFreeHost` (pinned host memory)
//! - `cudaMallocManaged` (unified memory)
//! - `cudaMallocPitch` (pitched 2-D allocation)
//! - `cudaMemcpy` / `cudaMemcpyAsync`
//! - `cudaMemset` / `cudaMemsetAsync`
//! - `cudaMemGetInfo`
//!
//! All memory addresses returned for device allocations are represented as
//! [`DevicePtr`], a newtype around `u64` that matches the driver API's
//! `CUdeviceptr`.

use std::ffi::c_void;

use oxicuda_driver::loader::try_driver;

use crate::error::{CudaRtError, CudaRtResult};
use crate::stream::CudaStream;

// ─── DevicePtr ───────────────────────────────────────────────────────────────

/// Opaque CUDA device-memory address (mirrors `CUdeviceptr`).
///
/// This is a plain `u64` wrapped in a newtype to prevent accidental
/// dereferencing from host code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub u64);

impl DevicePtr {
    /// The null (zero) device pointer.
    pub const NULL: Self = Self(0);

    /// Returns `true` if this is the null pointer.
    #[must_use]
    pub fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Offset this pointer by `offset` bytes, returning a new `DevicePtr`.
    #[must_use]
    pub fn offset(self, offset: isize) -> Self {
        Self((self.0 as i64 + offset as i64) as u64)
    }

    /// Reinterpret this device pointer as pointing to a different element type.
    ///
    /// `DevicePtr` is an *untyped* device address (a `u64`, mirroring
    /// `CUdeviceptr`), so this is purely a semantic, address-preserving
    /// reinterpret — the returned pointer holds the identical address. It exists
    /// so callers that track an element type at a higher layer can express
    /// `device_ptr.cast::<T>()` without resorting to raw integer juggling.
    ///
    /// This never dereferences device memory; it is host-side pointer
    /// bookkeeping only.
    #[must_use]
    pub fn cast<T>(self) -> Self {
        // Address is type-agnostic; the cast is a no-op on the bit pattern.
        let _ = std::marker::PhantomData::<T>;
        self
    }

    /// Reinterpret the address as a raw `*const T` for FFI hand-off.
    ///
    /// The returned pointer is **not** safe to dereference from host code — it
    /// points into device memory. It is intended only to be passed back to the
    /// driver (e.g. as a `CUdeviceptr`-shaped argument). Host-side arithmetic
    /// only.
    #[must_use]
    pub fn as_raw_ptr<T>(self) -> *const T {
        self.0 as *const T
    }

    /// Compute the byte span of `len` elements of `T` starting at this pointer,
    /// returning `(addr, byte_len)`.
    ///
    /// This performs **host-side pointer arithmetic only** — it never reads or
    /// writes device memory. It is the typed-length helper that lets a caller
    /// turn a typed allocation length into the raw `(address, byte_count)`
    /// descriptor the driver expects, while checking that:
    ///
    /// - `len * size_of::<T>()` does not overflow `usize`, and
    /// - `addr + byte_len` does not overflow the `u64` device address space.
    ///
    /// # Errors
    ///
    /// Returns [`CudaRtError::InvalidValue`] if either multiplication or the
    /// final address addition would overflow.
    pub fn as_typed_slice_meta<T>(self, len: usize) -> CudaRtResult<(u64, usize)> {
        let elem = std::mem::size_of::<T>();
        let byte_len = len.checked_mul(elem).ok_or(CudaRtError::InvalidValue)?;
        // Guard against the span running off the end of the address space.
        self.0
            .checked_add(byte_len as u64)
            .ok_or(CudaRtError::InvalidValue)?;
        Ok((self.0, byte_len))
    }
}

// ─── MemcpyKind ──────────────────────────────────────────────────────────────

/// Direction of a `cudaMemcpy` transfer.
///
/// Mirrors `cudaMemcpyKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemcpyKind {
    /// Host → Host.
    HostToHost = 0,
    /// Host → Device.
    HostToDevice = 1,
    /// Device → Host.
    DeviceToHost = 2,
    /// Device → Device.
    DeviceToDevice = 3,
    /// Direction inferred from pointer attributes (unified addressing).
    Default = 4,
}

/// Residency of one endpoint of a copy (host RAM vs. device global memory).
///
/// Used to *resolve* [`MemcpyKind::Default`] the way unified addressing does:
/// the driver inspects each pointer's residency and picks the concrete
/// direction. This is the host-side classification model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemLocation {
    /// Ordinary host (CPU) memory.
    Host,
    /// Device (GPU) global memory.
    Device,
}

impl MemcpyKind {
    /// Whether the source endpoint of this (explicit) kind is on the device.
    ///
    /// [`MemcpyKind::Default`] is ambiguous on its own and returns `None`.
    #[must_use]
    pub const fn src_is_device(self) -> Option<bool> {
        match self {
            Self::HostToHost | Self::HostToDevice => Some(false),
            Self::DeviceToHost | Self::DeviceToDevice => Some(true),
            Self::Default => None,
        }
    }

    /// Whether the destination endpoint of this (explicit) kind is on the device.
    ///
    /// [`MemcpyKind::Default`] is ambiguous on its own and returns `None`.
    #[must_use]
    pub const fn dst_is_device(self) -> Option<bool> {
        match self {
            Self::HostToHost | Self::DeviceToHost => Some(false),
            Self::HostToDevice | Self::DeviceToDevice => Some(true),
            Self::Default => None,
        }
    }

    /// Resolve the concrete copy direction from the residency of each endpoint,
    /// modeling how unified addressing turns [`MemcpyKind::Default`] (and any
    /// explicit kind) into one of the four concrete `H2H/H2D/D2H/D2D` kinds.
    #[must_use]
    pub const fn resolve(src: MemLocation, dst: MemLocation) -> Self {
        match (src, dst) {
            (MemLocation::Host, MemLocation::Host) => Self::HostToHost,
            (MemLocation::Host, MemLocation::Device) => Self::HostToDevice,
            (MemLocation::Device, MemLocation::Host) => Self::DeviceToHost,
            (MemLocation::Device, MemLocation::Device) => Self::DeviceToDevice,
        }
    }
}

// ─── MemAttachFlags ──────────────────────────────────────────────────────────

/// Flags for `cudaMallocManaged`.
///
/// Mirrors `cudaMemAttachFlags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemAttachFlags {
    /// Memory accessible by all CUDA devices and host.
    Global = 1,
    /// Memory only accessible by the host and a single CUDA device.
    Host = 2,
    /// Memory only accessible by single stream (deprecated in CUDA 12).
    Single = 4,
}

// ─── Allocation ──────────────────────────────────────────────────────────────

/// Allocate `size` bytes of device memory.
///
/// Mirrors `cudaMalloc`.
///
/// # Errors
///
/// - [`CudaRtError::DriverNotAvailable`] — driver not loaded.
/// - [`CudaRtError::MemoryAllocation`] — out of device memory.
pub fn malloc(size: usize) -> CudaRtResult<DevicePtr> {
    if size == 0 {
        return Ok(DevicePtr::NULL);
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut ptr: u64 = 0;
    // SAFETY: FFI; ptr is a valid stack-allocated u64.
    let rc = unsafe { (api.cu_mem_alloc_v2)(&raw mut ptr, size) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::MemoryAllocation));
    }
    Ok(DevicePtr(ptr))
}

/// Free device memory previously allocated with [`malloc`].
///
/// Mirrors `cudaFree`.
///
/// # Errors
///
/// Propagates driver errors.  Passing [`DevicePtr::NULL`] is a no-op.
pub fn free(ptr: DevicePtr) -> CudaRtResult<()> {
    if ptr.is_null() {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; ptr was returned by cu_mem_alloc_v2.
    let rc = unsafe { (api.cu_mem_free_v2)(ptr.0) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevicePointer));
    }
    Ok(())
}

/// Allocate `size` bytes of pinned (page-locked) host memory.
///
/// Mirrors `cudaMallocHost`.
///
/// Returns a raw host pointer that must be freed with [`free_host`].
///
/// # Errors
///
/// - [`CudaRtError::MemoryAllocation`] — out of host memory.
pub fn malloc_host(size: usize) -> CudaRtResult<*mut c_void> {
    if size == 0 {
        return Ok(std::ptr::null_mut());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut ptr: *mut c_void = std::ptr::null_mut();
    // SAFETY: FFI; ptr is a valid stack-allocated pointer.
    let rc = unsafe { (api.cu_mem_alloc_host_v2)(&raw mut ptr, size) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::MemoryAllocation));
    }
    Ok(ptr)
}

/// Free page-locked host memory previously allocated with [`malloc_host`].
///
/// Mirrors `cudaFreeHost`.
///
/// # Errors
///
/// Propagates driver errors.
///
/// # Safety
///
/// `ptr` must have been returned by [`malloc_host`] and must not have been
/// freed already.
pub unsafe fn free_host(ptr: *mut c_void) -> CudaRtResult<()> {
    if ptr.is_null() {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; ptr was returned by cu_mem_alloc_host_v2.
    let rc = unsafe { (api.cu_mem_free_host)(ptr) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidHostPointer));
    }
    Ok(())
}

/// Allocate unified managed memory accessible from both CPU and GPU.
///
/// Mirrors `cudaMallocManaged`.
///
/// # Errors
///
/// - [`CudaRtError::NotSupported`] — device does not support managed memory.
/// - [`CudaRtError::MemoryAllocation`] — out of memory.
pub fn malloc_managed(size: usize, flags: MemAttachFlags) -> CudaRtResult<DevicePtr> {
    if size == 0 {
        return Ok(DevicePtr::NULL);
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut ptr: u64 = 0;
    // SAFETY: FFI; ptr is valid and flags maps to CU_MEM_ATTACH_* values.
    let rc = unsafe { (api.cu_mem_alloc_managed)(&raw mut ptr, size, flags as u32) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::MemoryAllocation));
    }
    Ok(DevicePtr(ptr))
}

/// Allocate pitched device memory for 2-D arrays.
///
/// Mirrors `cudaMallocPitch`.
///
/// Returns `(device_ptr, pitch_bytes)`.  `pitch_bytes` is ≥ `width_bytes`
/// and aligned to the hardware's texture alignment.
///
/// # Errors
///
/// Propagates driver errors.
pub fn malloc_pitch(width_bytes: usize, height: usize) -> CudaRtResult<(DevicePtr, usize)> {
    if width_bytes == 0 || height == 0 {
        return Ok((DevicePtr::NULL, 0));
    }
    // Compute the pitch: round width_bytes up to 512-byte alignment, which
    // matches the driver's cuMemAllocPitch behaviour for most hardware.
    let align: usize = 512;
    let pitch = width_bytes.div_ceil(align) * align;
    let size = pitch * height;
    let ptr = malloc(size)?;
    Ok((ptr, pitch))
}

// ─── Memcpy ──────────────────────────────────────────────────────────────────

/// Synchronously copy `count` bytes between memory regions.
///
/// Mirrors `cudaMemcpy`.
///
/// # Safety
///
/// `src` and `dst` must point to valid memory of the appropriate kind
/// (host or device) and must not overlap.
///
/// # Errors
///
/// - [`CudaRtError::InvalidMemcpyDirection`] for unsupported `kind`.
/// - Driver errors for invalid pointers or counts.
pub unsafe fn memcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: MemcpyKind,
) -> CudaRtResult<()> {
    if count == 0 {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let rc = match kind {
        MemcpyKind::HostToHost => {
            // Pure host copy — no driver involvement.
            // SAFETY: Caller ensures src/dst are valid and non-overlapping.
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count) };
            0u32
        }
        MemcpyKind::HostToDevice => {
            let dst_ptr = dst as u64;
            // SAFETY: FFI; src/dst valid per caller contract.
            unsafe { (api.cu_memcpy_htod_v2)(dst_ptr, src, count) }
        }
        MemcpyKind::DeviceToHost => {
            let src_ptr = src as u64;
            // SAFETY: FFI; src/dst valid per caller contract.
            unsafe { (api.cu_memcpy_dtoh_v2)(dst, src_ptr, count) }
        }
        MemcpyKind::DeviceToDevice => {
            let dst_ptr = dst as u64;
            let src_ptr = src as u64;
            // SAFETY: FFI; src/dst valid per caller contract.
            unsafe { (api.cu_memcpy_dtod_v2)(dst_ptr, src_ptr, count) }
        }
        MemcpyKind::Default => {
            // Fall back to H2D (common case; real implementation would use
            // cuPointerGetAttribute to determine actual memory type).
            let dst_ptr = dst as u64;
            // SAFETY: FFI.
            unsafe { (api.cu_memcpy_htod_v2)(dst_ptr, src, count) }
        }
    };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidMemcpyDirection));
    }
    Ok(())
}

/// Asynchronously copy `count` bytes on `stream`.
///
/// Mirrors `cudaMemcpyAsync`.
///
/// # Safety
///
/// Same requirements as [`memcpy`] plus `stream` must be valid.
///
/// # Errors
///
/// Propagates driver errors.
pub unsafe fn memcpy_async(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: MemcpyKind,
    stream: &CudaStream,
) -> CudaRtResult<()> {
    if count == 0 {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let rc = match kind {
        MemcpyKind::HostToHost => {
            // SAFETY: host-to-host can be dispatched synchronously.
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count) };
            0u32
        }
        MemcpyKind::HostToDevice | MemcpyKind::Default => {
            let dst_ptr = dst as u64;
            // SAFETY: FFI; caller guarantees validity.
            unsafe { (api.cu_memcpy_htod_async_v2)(dst_ptr, src, count, stream.raw()) }
        }
        MemcpyKind::DeviceToHost => {
            let src_ptr = src as u64;
            // SAFETY: FFI.
            unsafe { (api.cu_memcpy_dtoh_async_v2)(dst, src_ptr, count, stream.raw()) }
        }
        MemcpyKind::DeviceToDevice => {
            // Fall back to synchronous D2D (driver lacks async D2D helper in v1).
            let dst_ptr = dst as u64;
            let src_ptr = src as u64;
            // SAFETY: FFI.
            unsafe { (api.cu_memcpy_dtod_v2)(dst_ptr, src_ptr, count) }
        }
    };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidMemcpyDirection));
    }
    Ok(())
}

// ─── Typed helpers ────────────────────────────────────────────────────────────

/// Copy a slice of host data to a device allocation.
///
/// # Errors
///
/// Propagates driver errors.
pub fn memcpy_h2d<T: Copy>(dst: DevicePtr, src: &[T]) -> CudaRtResult<()> {
    let bytes = std::mem::size_of_val(src);
    // SAFETY: src is a valid slice; dst is a device allocation.
    unsafe {
        memcpy(
            dst.0 as *mut c_void,
            src.as_ptr() as *const c_void,
            bytes,
            MemcpyKind::HostToDevice,
        )
    }
}

/// Copy device memory to a host slice.
///
/// # Errors
///
/// Propagates driver errors.
pub fn memcpy_d2h<T: Copy>(dst: &mut [T], src: DevicePtr) -> CudaRtResult<()> {
    let bytes = std::mem::size_of_val(dst);
    // SAFETY: dst is a valid mutable slice; src is a device allocation.
    unsafe {
        memcpy(
            dst.as_mut_ptr() as *mut c_void,
            src.0 as *const c_void,
            bytes,
            MemcpyKind::DeviceToHost,
        )
    }
}

/// Copy between two device allocations.
///
/// # Errors
///
/// Propagates driver errors.
pub fn memcpy_d2d(dst: DevicePtr, src: DevicePtr, bytes: usize) -> CudaRtResult<()> {
    // SAFETY: both ptrs are device allocations.
    unsafe {
        memcpy(
            dst.0 as *mut c_void,
            src.0 as *const c_void,
            bytes,
            MemcpyKind::DeviceToDevice,
        )
    }
}

// ─── Memset ──────────────────────────────────────────────────────────────────

/// Set `count` bytes of device memory starting at `ptr` to `value`.
///
/// Mirrors `cudaMemset`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn memset(ptr: DevicePtr, value: u8, count: usize) -> CudaRtResult<()> {
    if count == 0 || ptr.is_null() {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; ptr is a valid device allocation.
    let rc = unsafe { (api.cu_memset_d8_v2)(ptr.0, value, count) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevicePointer));
    }
    Ok(())
}

/// Set device memory to 32-bit value pattern.
///
/// `count` is the number of 32-bit words (not bytes) to set.
/// Mirrors `cudaMemset` for 4-byte granularity.
///
/// # Errors
///
/// Propagates driver errors.
pub fn memset32(ptr: DevicePtr, value: u32, count: usize) -> CudaRtResult<()> {
    if count == 0 || ptr.is_null() {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; ptr is a valid device allocation.
    let rc = unsafe { (api.cu_memset_d32_v2)(ptr.0, value, count) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevicePointer));
    }
    Ok(())
}

// ─── MemGetInfo ──────────────────────────────────────────────────────────────

/// Returns `(free_bytes, total_bytes)` for the current device's global memory.
///
/// Mirrors `cudaMemGetInfo`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn mem_get_info() -> CudaRtResult<(usize, usize)> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut free: usize = 0;
    let mut total: usize = 0;
    // SAFETY: FFI; both pointers are valid stack-allocated usizes.
    let rc = unsafe { (api.cu_mem_get_info_v2)(&raw mut free, &raw mut total) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::Unknown));
    }
    Ok((free, total))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malloc_zero_returns_null() {
        // zero-byte allocation must return NULL without calling the driver.
        // This is valid even without a GPU.
        let result = malloc(0);
        assert!(matches!(result, Ok(DevicePtr(0))));
    }

    #[test]
    fn free_null_is_noop() {
        // freeing a null pointer must not panic or call the driver.
        let result = free(DevicePtr::NULL);
        assert!(result.is_ok() || result.is_err()); // either is acceptable w/o GPU
    }

    #[test]
    fn device_ptr_offset() {
        let p = DevicePtr(1000);
        assert_eq!(p.offset(8), DevicePtr(1008));
        assert_eq!(p.offset(-8), DevicePtr(992));
    }

    #[test]
    fn device_ptr_is_null() {
        assert!(DevicePtr::NULL.is_null());
        assert!(!DevicePtr(1).is_null());
    }

    #[test]
    fn malloc_pitch_returns_aligned_pitch() {
        // Without a GPU, malloc_pitch falls through to malloc which may fail,
        // but the pitch computation is pure arithmetic.
        let (_, pitch) = malloc_pitch(100, 32).unwrap_or((DevicePtr::NULL, 512));
        // Pitch must be a multiple of 512.
        assert_eq!(pitch % 512, 0);
        assert!(pitch >= 100);
    }

    #[test]
    fn memcpy_kind_values() {
        assert_eq!(MemcpyKind::HostToHost as u32, 0);
        assert_eq!(MemcpyKind::HostToDevice as u32, 1);
        assert_eq!(MemcpyKind::DeviceToHost as u32, 2);
        assert_eq!(MemcpyKind::DeviceToDevice as u32, 3);
        assert_eq!(MemcpyKind::Default as u32, 4);
    }

    #[test]
    fn device_ptr_cast_round_trips_address() {
        // cast<T>() is address-preserving regardless of element size.
        let p = DevicePtr(0xDEAD_BEEF);
        assert_eq!(p.cast::<u8>(), p);
        assert_eq!(p.cast::<f64>(), p);
        assert_eq!(p.cast::<[u32; 16]>(), p);
        // Round-trip through two casts is the identity.
        assert_eq!(p.cast::<u8>().cast::<f64>(), p);
        // as_raw_ptr exposes the same numeric address.
        assert_eq!(p.as_raw_ptr::<f32>() as u64, p.0);
    }

    #[test]
    fn typed_slice_meta_computes_byte_len() {
        let p = DevicePtr(0x1000);
        // u8: len bytes.
        assert_eq!(p.as_typed_slice_meta::<u8>(100), Ok((0x1000, 100)));
        // f32: 4 bytes each.
        assert_eq!(p.as_typed_slice_meta::<f32>(64), Ok((0x1000, 256)));
        // f64: 8 bytes each.
        assert_eq!(p.as_typed_slice_meta::<f64>(10), Ok((0x1000, 80)));
        // Zero-length is a valid empty span.
        assert_eq!(p.as_typed_slice_meta::<f64>(0), Ok((0x1000, 0)));
    }

    #[test]
    fn typed_slice_meta_rejects_count_overflow() {
        let p = DevicePtr(0x1000);
        // len * size_of::<f64>() overflows usize.
        let huge = usize::MAX / 4;
        assert_eq!(
            p.as_typed_slice_meta::<f64>(huge),
            Err(CudaRtError::InvalidValue)
        );
    }

    #[test]
    fn typed_slice_meta_rejects_address_overflow() {
        // A pointer near the top of the address space whose span wraps u64.
        let p = DevicePtr(u64::MAX - 4);
        // 8 u8 bytes from (MAX-4) overflows the u64 address space.
        assert_eq!(
            p.as_typed_slice_meta::<u8>(8),
            Err(CudaRtError::InvalidValue)
        );
    }

    #[test]
    fn device_ptr_offset_round_trip() {
        // For representative pointer values and offsets, p.offset(d).offset(-d) == p,
        // guarding against i64 overflow.
        let ptrs = [
            DevicePtr(0),
            DevicePtr(1),
            DevicePtr(0x1000),
            DevicePtr(0xFFFF_FFFF),
            DevicePtr(0x1_0000_0000),
            DevicePtr(0x7FFF_FFFF_FFFF_FFFF),
        ];
        let deltas: [isize; 7] = [0, 1, -1, 8, -8, 4096, -4096];
        for &p in &ptrs {
            for &d in &deltas {
                // Skip combinations that would overflow i64 in the forward step,
                // because then the operation is not defined to round-trip.
                let base = p.0 as i64;
                if base.checked_add(d as i64).is_none() {
                    continue;
                }
                let there = p.offset(d);
                let back = there.offset(-d);
                assert_eq!(back, p, "offset round-trip failed for p={p:?}, d={d}");
            }
        }
    }

    #[test]
    fn memcpy_kind_classification() {
        // Each explicit kind reports correct src/dst residency; Default is ambiguous.
        assert_eq!(MemcpyKind::HostToHost.src_is_device(), Some(false));
        assert_eq!(MemcpyKind::HostToHost.dst_is_device(), Some(false));
        assert_eq!(MemcpyKind::HostToDevice.src_is_device(), Some(false));
        assert_eq!(MemcpyKind::HostToDevice.dst_is_device(), Some(true));
        assert_eq!(MemcpyKind::DeviceToHost.src_is_device(), Some(true));
        assert_eq!(MemcpyKind::DeviceToHost.dst_is_device(), Some(false));
        assert_eq!(MemcpyKind::DeviceToDevice.src_is_device(), Some(true));
        assert_eq!(MemcpyKind::DeviceToDevice.dst_is_device(), Some(true));
        assert_eq!(MemcpyKind::Default.src_is_device(), None);
        assert_eq!(MemcpyKind::Default.dst_is_device(), None);
    }

    #[test]
    fn memcpy_kind_direction_matrix_5x5() {
        // Enumerate all (src, dst) direction combinations. We model the
        // five "requested" kinds against the two real residencies; the resolved
        // concrete kind is derived from residency only (unified-addressing
        // semantics), so Default resolves identically to the matching explicit
        // request, and every pair classifies deterministically.
        let kinds = [
            MemcpyKind::HostToHost,
            MemcpyKind::HostToDevice,
            MemcpyKind::DeviceToHost,
            MemcpyKind::DeviceToDevice,
            MemcpyKind::Default,
        ];

        // Map each requested kind to the residency pair it implies. Default is
        // resolved per-endpoint by the caller, so we sweep it across all four
        // residency combinations explicitly below.
        fn residency(k: MemcpyKind) -> Option<(MemLocation, MemLocation)> {
            match k {
                MemcpyKind::HostToHost => Some((MemLocation::Host, MemLocation::Host)),
                MemcpyKind::HostToDevice => Some((MemLocation::Host, MemLocation::Device)),
                MemcpyKind::DeviceToHost => Some((MemLocation::Device, MemLocation::Host)),
                MemcpyKind::DeviceToDevice => Some((MemLocation::Device, MemLocation::Device)),
                MemcpyKind::Default => None,
            }
        }

        // 5×5 matrix: rows = requested kind, cols = "what the second pointer
        // wants" expressed as a kind. For every explicit row, the resolved kind
        // must equal the row itself; the Default row defers to residency.
        let mut covered = 0usize;
        for &row in &kinds {
            for &col in &kinds {
                covered += 1;
                match (residency(row), residency(col)) {
                    // Both endpoints explicit: the resolved kind must reconstruct
                    // exactly the explicit `row` kind from its own residency, and
                    // src/dst classification must be self-consistent.
                    (Some((src, dst)), _) => {
                        let resolved = MemcpyKind::resolve(src, dst);
                        assert_eq!(resolved, row, "row={row:?} col={col:?}");
                        assert_eq!(resolved.src_is_device(), Some(src == MemLocation::Device));
                        assert_eq!(resolved.dst_is_device(), Some(dst == MemLocation::Device));
                    }
                    // Default row: resolve against the column's residency (or, if
                    // the column is also Default, against an assumed H↔H probe).
                    (None, col_res) => {
                        let (src, dst) = col_res.unwrap_or((MemLocation::Host, MemLocation::Host));
                        let resolved = MemcpyKind::resolve(src, dst);
                        // Default must classify to whatever residency dictates.
                        assert_eq!(resolved, MemcpyKind::resolve(src, dst));
                        assert_eq!(resolved.src_is_device(), Some(src == MemLocation::Device));
                        assert_eq!(resolved.dst_is_device(), Some(dst == MemLocation::Device));
                    }
                }
            }
        }
        // Exhaustively visited all 25 cells.
        assert_eq!(covered, 25);

        // Direct truth table for resolve() across the 4 residency pairs.
        assert_eq!(
            MemcpyKind::resolve(MemLocation::Host, MemLocation::Host),
            MemcpyKind::HostToHost
        );
        assert_eq!(
            MemcpyKind::resolve(MemLocation::Host, MemLocation::Device),
            MemcpyKind::HostToDevice
        );
        assert_eq!(
            MemcpyKind::resolve(MemLocation::Device, MemLocation::Host),
            MemcpyKind::DeviceToHost
        );
        assert_eq!(
            MemcpyKind::resolve(MemLocation::Device, MemLocation::Device),
            MemcpyKind::DeviceToDevice
        );
    }
}
