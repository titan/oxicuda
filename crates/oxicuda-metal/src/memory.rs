//! Metal buffer manager — allocates, copies, and frees `metal::Buffer` objects
//! using the shared-memory storage mode so the CPU can read and write them
//! directly without a staging copy.
//!
//! All buffers are tracked by opaque `u64` handles (starting at 1) that mirror
//! the CUDA device-pointer model used by the rest of OxiCUDA.

#[cfg(target_os = "macos")]
use std::sync::atomic::Ordering;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, atomic::AtomicU64},
};

use crate::{
    device::MetalDevice,
    error::{MetalError, MetalResult},
};

// ─── Internal buffer record ──────────────────────────────────────────────────

/// Provenance of a tracked Metal buffer — decides whether [`MetalMemoryManager::free`]
/// (and manager drop) is allowed to release the underlying `MTLBuffer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BufferOwnership {
    /// Allocated by this manager via [`MetalMemoryManager::alloc`]. The manager
    /// holds the sole retain and releases it on `free`/drop.
    ///
    /// Only constructed on macOS — on other platforms [`MetalMemoryManager::alloc`]
    /// returns [`MetalError::UnsupportedPlatform`] instead of producing a buffer.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    Owned,
    /// Imported from an external caller (e.g. a consumer's buffer cache). The
    /// manager holds its **own independent retain** (taken at import time) and
    /// releases only *that* retain on `free`/drop — it never deallocates the
    /// caller's buffer, which the caller keeps alive via its own retain.
    External,
}

/// Bookkeeping entry for a single tracked Metal buffer.
pub(crate) struct MetalBufferInfo {
    /// The GPU-resident buffer. For [`BufferOwnership::Owned`] entries this is a
    /// freshly allocated shared-mode buffer; for [`BufferOwnership::External`]
    /// entries it is an independent retain of a caller-provided buffer.
    #[cfg(target_os = "macos")]
    pub(crate) buffer: metal::Buffer,
    /// Byte size of the allocation (the imported logical length for external
    /// buffers, which bounds `copy_to_device` / `copy_from_device`).
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) size: u64,
    /// Whether this manager owns the allocation (and may release it) or is only
    /// borrowing an external buffer via its own retain.
    pub(crate) ownership: BufferOwnership,
}

// ─── Memory manager ──────────────────────────────────────────────────────────

/// Manages a pool of Metal buffers, returning opaque `u64` handles.
///
/// Uses `MTLResourceOptions::StorageModeShared` so the same physical pages are
/// accessible from both CPU and GPU without explicit synchronisation — the same
/// model used by Metal's unified-memory architecture on Apple Silicon.
///
/// All public methods take `&self` so the manager can be shared behind `Arc`.
pub struct MetalMemoryManager {
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    device: Arc<MetalDevice>,
    buffers: Mutex<HashMap<u64, MetalBufferInfo>>,
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    next_handle: AtomicU64,
}

impl MetalMemoryManager {
    /// Create a new memory manager backed by `device`.
    pub fn new(device: Arc<MetalDevice>) -> Self {
        Self {
            device,
            buffers: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        }
    }

    /// Lock the internal buffer map, returning a guard for buffer access.
    ///
    /// Used by the backend to bind Metal buffers to compute command encoders.
    #[cfg(target_os = "macos")]
    pub(crate) fn lock_buffers(
        &self,
    ) -> MetalResult<std::sync::MutexGuard<'_, HashMap<u64, MetalBufferInfo>>> {
        self.buffers
            .lock()
            .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))
    }

    /// Allocate `bytes` bytes of shared-mode device memory.
    ///
    /// Returns an opaque handle.  The caller must eventually call [`free`](Self::free).
    pub fn alloc(&self, bytes: usize) -> MetalResult<u64> {
        #[cfg(target_os = "macos")]
        {
            let buffer = self
                .device
                .device
                .new_buffer(bytes as u64, metal::MTLResourceOptions::StorageModeShared);
            let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
            self.buffers
                .lock()
                .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?
                .insert(
                    handle,
                    MetalBufferInfo {
                        buffer,
                        size: bytes as u64,
                        ownership: BufferOwnership::Owned,
                    },
                );
            Ok(handle)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = bytes;
            Err(MetalError::UnsupportedPlatform)
        }
    }

    /// Import an **externally owned** `metal::Buffer` and return a handle usable
    /// by every compute op (`gemm`, `copy_*`, …) without copying through the host.
    ///
    /// The manager takes its **own** retain on `buffer` (so the buffer stays
    /// alive while the handle is live) but is flagged
    /// `BufferOwnership::External`: [`free`](Self::free) and manager drop
    /// release only the manager's own retain and **never deallocate the caller's
    /// buffer**, which the caller keeps alive via its own retain (e.g. an
    /// `Arc<metal::Buffer>` in a buffer cache).
    ///
    /// `len_bytes` is the logical byte length the handle exposes; it bounds
    /// host copies via [`copy_to_device`](Self::copy_to_device) /
    /// [`copy_from_device`](Self::copy_from_device). It must not exceed the
    /// buffer's actual `length()`.
    ///
    /// # Errors
    /// * [`MetalError::UnsupportedPlatform`] on non-macOS.
    /// * [`MetalError::InvalidArgument`] if `len_bytes` exceeds the buffer's
    ///   physical length.
    #[cfg(target_os = "macos")]
    pub fn import_external(&self, buffer: &metal::Buffer, len_bytes: usize) -> MetalResult<u64> {
        let physical = buffer.length();
        if len_bytes as u64 > physical {
            return Err(MetalError::InvalidArgument(format!(
                "import len_bytes {len_bytes} exceeds buffer length {physical}"
            )));
        }
        // The host copy helpers (`copy_to_device` / `copy_from_device`)
        // dereference `buffer.contents()`, which Metal returns as NULL for
        // storage modes that are not CPU-accessible (Private / Memoryless).
        // Reject those at import time so a later host copy cannot deref null.
        let mode = buffer.storage_mode();
        if mode == metal::MTLStorageMode::Private || mode == metal::MTLStorageMode::Memoryless {
            return Err(MetalError::InvalidArgument(format!(
                "import_external requires a CPU-accessible buffer \
                 (Shared or Managed); got storage mode {mode:?}"
            )));
        }
        // Take an independent retain so the buffer survives for the handle's
        // lifetime regardless of what the caller does with its own reference.
        let retained = buffer.to_owned();
        let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.buffers
            .lock()
            .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?
            .insert(
                handle,
                MetalBufferInfo {
                    buffer: retained,
                    size: len_bytes as u64,
                    ownership: BufferOwnership::External,
                },
            );
        Ok(handle)
    }

    /// Release the buffer associated with `handle`.
    ///
    /// For an `BufferOwnership::Owned` allocation this drops the manager's sole
    /// retain (freeing the GPU memory). For an `BufferOwnership::External`
    /// import this drops only the manager's own retain — the caller's buffer is
    /// untouched (the caller holds an independent retain). Unknown handles are
    /// silently ignored (idempotent free).
    pub fn free(&self, handle: u64) -> MetalResult<()> {
        let removed = self
            .buffers
            .lock()
            .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?
            .remove(&handle);
        if let Some(info) = removed {
            // The dropped `MetalBufferInfo` releases exactly one retain — the one
            // this manager holds. The `ownership` flag distinguishes whether that
            // retain was a sole allocation (owned) or an independent import retain
            // (external); in the external case the caller's buffer survives.
            match info.ownership {
                BufferOwnership::Owned => {
                    tracing::trace!(handle, "freed owned Metal buffer");
                }
                BufferOwnership::External => {
                    tracing::trace!(
                        handle,
                        "released import retain for external Metal buffer (caller's buffer untouched)"
                    );
                }
            }
            // `info` (and its `metal::Buffer`, on macOS) drops at the end of this
            // block, releasing the manager's single retain.
        }
        Ok(())
    }

    /// Report whether `handle` refers to an imported (`BufferOwnership::External`)
    /// buffer (`Some(true)`), an owned allocation (`Some(false)`), or is unknown
    /// (`None`).
    ///
    /// Lets a consumer assert that a cache-backed buffer was registered as
    /// external (so [`free`](Self::free) will not deallocate it).
    pub fn is_external(&self, handle: u64) -> MetalResult<Option<bool>> {
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?;
        Ok(buffers
            .get(&handle)
            .map(|info| info.ownership == BufferOwnership::External))
    }

    /// Upload host bytes `src` into the device buffer identified by `handle`.
    ///
    /// Because the buffer uses shared storage, this is a direct CPU `memcpy`.
    pub fn copy_to_device(&self, handle: u64, src: &[u8]) -> MetalResult<()> {
        #[cfg(target_os = "macos")]
        {
            // Resolve the buffer and validate the size under the lock, then take
            // an independent retain so the `memcpy` runs *outside* the critical
            // section — concurrent alloc/free/copy on unrelated handles are not
            // serialised behind this transfer, and the retain guarantees the
            // buffer cannot be freed out from under the copy.
            let buffer = {
                let buffers = self
                    .buffers
                    .lock()
                    .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?;
                let info = buffers.get(&handle).ok_or_else(|| {
                    MetalError::InvalidArgument(format!("unknown handle {handle}"))
                })?;
                if src.len() as u64 > info.size {
                    return Err(MetalError::InvalidArgument(format!(
                        "copy_to_device: src length {} exceeds buffer length {}",
                        src.len(),
                        info.size
                    )));
                }
                info.buffer.to_owned()
            };
            // SAFETY: Metal Shared/Managed buffers are CPU-accessible; `contents()`
            // returns a valid `*mut c_void` for the buffer's lifetime, which the
            // retain above extends across this copy. `src.len() <= buffer length`
            // was verified, so the write stays in bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    buffer.contents() as *mut u8,
                    src.len(),
                );
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (handle, src);
            Err(MetalError::UnsupportedPlatform)
        }
    }

    /// Download device buffer `handle` into `dst`.
    ///
    /// Because the buffer uses shared storage, this is a direct CPU `memcpy`.
    pub fn copy_from_device(&self, dst: &mut [u8], handle: u64) -> MetalResult<()> {
        #[cfg(target_os = "macos")]
        {
            // Resolve + validate under the lock, retain, then copy outside the
            // critical section (see `copy_to_device` for the rationale).
            let buffer = {
                let buffers = self
                    .buffers
                    .lock()
                    .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?;
                let info = buffers.get(&handle).ok_or_else(|| {
                    MetalError::InvalidArgument(format!("unknown handle {handle}"))
                })?;
                if dst.len() as u64 > info.size {
                    return Err(MetalError::InvalidArgument(format!(
                        "copy_from_device: dst length {} exceeds buffer length {}",
                        dst.len(),
                        info.size
                    )));
                }
                info.buffer.to_owned()
            };
            // SAFETY: Metal Shared/Managed buffers are CPU-accessible; `contents()`
            // returns a valid `*const c_void` for the buffer's lifetime, extended
            // across the copy by the retain. `dst.len() <= buffer length` was
            // verified, so the read stays in bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.contents() as *const u8,
                    dst.as_mut_ptr(),
                    dst.len(),
                );
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (dst, handle);
            Err(MetalError::UnsupportedPlatform)
        }
    }

    /// Copy `len_bytes` from device buffer `src` to device buffer `dst`,
    /// **device-to-device** with no host round-trip.
    ///
    /// Both handles may be owned or imported, in any combination. For the
    /// shared-mode buffers this manager tracks, the copy is a direct
    /// CPU-visible `memcpy` between the two buffers' unified-memory pages, so it
    /// never stages through a host `Vec`. `len_bytes` is clamped to each
    /// buffer's tracked length.
    ///
    /// # Errors
    /// * [`MetalError::UnsupportedPlatform`] on non-macOS.
    /// * [`MetalError::InvalidArgument`] for an unknown `src`/`dst` handle, if
    ///   `src == dst`, or if `len_bytes` exceeds either buffer's length.
    pub fn copy_device_to_device(&self, dst: u64, src: u64, len_bytes: usize) -> MetalResult<()> {
        #[cfg(target_os = "macos")]
        {
            if src == dst {
                return Err(MetalError::InvalidArgument(
                    "copy_device_to_device requires distinct src and dst handles".into(),
                ));
            }
            // Resolve + validate under the lock, retain both buffers, then copy
            // outside the critical section (see `copy_to_device`).
            let (src_buf, dst_buf) = {
                let buffers = self
                    .buffers
                    .lock()
                    .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?;
                let src_info = buffers.get(&src).ok_or_else(|| {
                    MetalError::InvalidArgument(format!("unknown src handle {src}"))
                })?;
                let dst_info = buffers.get(&dst).ok_or_else(|| {
                    MetalError::InvalidArgument(format!("unknown dst handle {dst}"))
                })?;
                if len_bytes as u64 > src_info.size {
                    return Err(MetalError::InvalidArgument(format!(
                        "len_bytes {len_bytes} exceeds src length {}",
                        src_info.size
                    )));
                }
                if len_bytes as u64 > dst_info.size {
                    return Err(MetalError::InvalidArgument(format!(
                        "len_bytes {len_bytes} exceeds dst length {}",
                        dst_info.size
                    )));
                }
                (src_info.buffer.to_owned(), dst_info.buffer.to_owned())
            };
            let src_ptr = src_buf.contents() as *const u8;
            let dst_ptr = dst_buf.contents() as *mut u8;
            // Distinct handles can still alias the *same* physical MTLBuffer
            // (e.g. the same buffer imported twice), so `src_ptr == dst_ptr` or
            // overlapping byte ranges are possible. `copy_nonoverlapping`
            // requires disjoint regions; fall back to the overlap-safe `copy`
            // (memmove) whenever the ranges overlap.
            let src_addr = src_ptr as usize;
            let dst_addr = dst_ptr as usize;
            let overlaps = src_addr < dst_addr + len_bytes && dst_addr < src_addr + len_bytes;
            // SAFETY: both Shared/Managed buffers are CPU-accessible for their
            // full lifetime (extended across the copy by the retains above); the
            // validated `len_bytes` fits within both. `copy` is used for the
            // aliasing/overlapping case, `copy_nonoverlapping` only when the
            // regions are provably disjoint.
            unsafe {
                if overlaps {
                    std::ptr::copy(src_ptr, dst_ptr, len_bytes);
                } else {
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len_bytes);
                }
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (dst, src, len_bytes);
            Err(MetalError::UnsupportedPlatform)
        }
    }
}

impl std::fmt::Debug for MetalMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.buffers.lock().map(|b| b.len()).unwrap_or(0);
        write!(f, "MetalMemoryManager(buffers={count})")
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    fn try_get_device() -> Option<Arc<MetalDevice>> {
        MetalDevice::new().ok().map(Arc::new)
    }

    #[test]
    fn alloc_and_free_requires_device() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        let h = mm.alloc(256).expect("alloc 256 bytes");
        assert!(h > 0);
        mm.free(h).expect("free");
        // Double-free is silently ignored.
        mm.free(h).expect("double-free is a no-op");
    }

    #[test]
    fn copy_roundtrip_requires_device() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);

        let src: Vec<u8> = (0u8..64).collect();
        let h = mm.alloc(src.len()).expect("alloc");
        mm.copy_to_device(h, &src).expect("copy_to_device");

        let mut dst = vec![0u8; src.len()];
        mm.copy_from_device(&mut dst, h).expect("copy_from_device");

        assert_eq!(src, dst);
        mm.free(h).expect("free");
    }

    #[test]
    fn unknown_handle_returns_error() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        let err = mm.copy_to_device(9999, b"hello").unwrap_err();
        assert!(matches!(err, MetalError::InvalidArgument(_)));
    }

    #[test]
    fn debug_impl_smoke() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        let s = format!("{mm:?}");
        assert!(s.contains("MetalMemoryManager"));
    }
}
