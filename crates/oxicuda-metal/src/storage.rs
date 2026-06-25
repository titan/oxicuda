//! Metal resource storage-mode selection and buffer-descriptor logic.
//!
//! Metal exposes several `MTLStorageMode`s with very different performance
//! characteristics:
//!
//! * `Shared` — CPU and GPU share the same backing pages (zero-copy on Apple
//!   Silicon unified memory).
//! * `Managed` — separate CPU/GPU copies kept coherent by explicit
//!   `didModifyRange` / `synchronize` calls (Intel/discrete Macs).
//! * `Private` — GPU-only memory; fastest for GPU-resident working sets but
//!   not CPU-mappable (requires a blit to upload/download).
//! * `Memoryless` — transient tile/render-target memory with no backing store.
//!
//! Choosing the right mode depends on the GPU family (unified vs discrete) and
//! the buffer's access pattern.  This module encodes that decision plus the
//! alignment math Metal requires for buffer offsets and lengths — all pure and
//! unit-testable without a device.

use crate::device_family::MetalDeviceCapabilities;
use std::fmt;

/// Default Metal buffer alignment in bytes (256 for argument-buffer offsets and
/// the conservative `minimumBufferOffsetAlignment` on Apple Silicon).
pub const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Page size used for `Managed`/`Private` heap suballocation (16 KiB on Apple
/// Silicon).
pub const METAL_PAGE_SIZE: usize = 16 * 1024;

// ─── MetalStorageMode ──────────────────────────────────────────────────────────

/// Resource storage mode, mirroring `MTLStorageMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetalStorageMode {
    /// CPU+GPU shared pages (zero-copy, unified memory).
    Shared,
    /// Separate CPU/GPU copies with explicit coherency (discrete GPUs).
    Managed,
    /// GPU-only memory; not CPU-mappable.
    Private,
    /// Transient tile memory with no backing store.
    Memoryless,
}

impl MetalStorageMode {
    /// `true` when the CPU can directly read/write a buffer in this mode
    /// (via `contents()`), without a staging blit.
    pub fn is_cpu_accessible(self) -> bool {
        matches!(self, Self::Shared | Self::Managed)
    }

    /// `true` when an explicit `didModifyRange` / `synchronize` is required to
    /// propagate CPU writes to the GPU (and vice versa).
    pub fn needs_explicit_sync(self) -> bool {
        matches!(self, Self::Managed)
    }

    /// `true` when this mode has a host-visible backing store at all.
    pub fn has_backing_store(self) -> bool {
        !matches!(self, Self::Memoryless)
    }
}

impl fmt::Display for MetalStorageMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Shared => "Shared",
            Self::Managed => "Managed",
            Self::Private => "Private",
            Self::Memoryless => "Memoryless",
        };
        f.write_str(s)
    }
}

// ─── BufferAccess ──────────────────────────────────────────────────────────────

/// How a buffer will be accessed across CPU/GPU — drives storage-mode choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferAccess {
    /// CPU writes once, GPU reads (e.g. constants, weights uploaded at init).
    UploadThenGpuRead,
    /// GPU writes, CPU reads back (e.g. results).
    GpuWriteThenDownload,
    /// CPU and GPU both touch the buffer frequently (interleaved).
    SharedReadWrite,
    /// GPU-only intermediate; CPU never touches it.
    GpuPrivate,
    /// Transient on-chip scratch (tile memory).
    Transient,
}

// ─── MetalBufferDescriptor ─────────────────────────────────────────────────────

/// A host-side description of a Metal buffer to be allocated.
///
/// Holds the requested length, the chosen storage mode, and the (aligned)
/// allocation length.  Produced by [`StoragePlanner::plan_buffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalBufferDescriptor {
    /// Requested logical length in bytes.
    pub length: usize,
    /// Allocation length after rounding up to [`METAL_BUFFER_ALIGNMENT`].
    pub aligned_length: usize,
    /// Selected storage mode.
    pub storage_mode: MetalStorageMode,
    /// `true` when uploads/downloads require a staging blit (Private mode).
    pub needs_staging: bool,
}

// ─── StoragePlanner ────────────────────────────────────────────────────────────

/// Selects storage modes and computes buffer descriptors for a target device.
#[derive(Debug, Clone, Copy)]
pub struct StoragePlanner {
    caps: MetalDeviceCapabilities,
}

impl StoragePlanner {
    /// Create a planner for a device with the given capabilities.
    pub fn new(caps: MetalDeviceCapabilities) -> Self {
        Self { caps }
    }

    /// Choose the optimal storage mode for the given access pattern.
    ///
    /// On unified-memory (Apple Silicon) devices `Shared` is almost always
    /// best, since there is no separate GPU copy.  On discrete (`Mac2`) GPUs,
    /// CPU-visible buffers must use `Managed`, and GPU-private working sets
    /// benefit from `Private`.
    pub fn select_mode(&self, access: BufferAccess) -> MetalStorageMode {
        let unified = self.caps.unified_memory;
        match access {
            BufferAccess::Transient => MetalStorageMode::Memoryless,
            BufferAccess::GpuPrivate => MetalStorageMode::Private,
            BufferAccess::SharedReadWrite => {
                if unified {
                    MetalStorageMode::Shared
                } else {
                    MetalStorageMode::Managed
                }
            }
            BufferAccess::UploadThenGpuRead | BufferAccess::GpuWriteThenDownload => {
                if unified {
                    // Zero-copy: CPU and GPU touch the same pages.
                    MetalStorageMode::Shared
                } else {
                    // Discrete GPUs: keep a GPU-resident copy, sync explicitly.
                    MetalStorageMode::Managed
                }
            }
        }
    }

    /// Build a full buffer descriptor for the requested length and access.
    pub fn plan_buffer(&self, length: usize, access: BufferAccess) -> MetalBufferDescriptor {
        let storage_mode = self.select_mode(access);
        let aligned_length = align_up(length.max(1), METAL_BUFFER_ALIGNMENT);
        MetalBufferDescriptor {
            length,
            aligned_length,
            storage_mode,
            needs_staging: storage_mode == MetalStorageMode::Private,
        }
    }
}

// ─── Alignment helpers ─────────────────────────────────────────────────────────

/// Round `value` up to the next multiple of `alignment` (a power of two).
///
/// `alignment` must be non-zero; if it is zero this returns `value` unchanged.
pub fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return value;
    }
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment must be a power of two"
    );
    (value + alignment - 1) & !(alignment - 1)
}

/// Round `value` down to the previous multiple of `alignment` (a power of two).
pub fn align_down(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return value;
    }
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment must be a power of two"
    );
    value & !(alignment - 1)
}

/// `true` when `value` is a multiple of `alignment`.
pub fn is_aligned(value: usize, alignment: usize) -> bool {
    alignment != 0 && value % alignment == 0
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_family::MetalGpuFamily;

    fn unified_planner() -> StoragePlanner {
        StoragePlanner::new(MetalDeviceCapabilities::from_family(MetalGpuFamily::Apple8))
    }

    fn discrete_planner() -> StoragePlanner {
        StoragePlanner::new(MetalDeviceCapabilities::from_family(MetalGpuFamily::Mac2))
    }

    #[test]
    fn storage_mode_properties() {
        assert!(MetalStorageMode::Shared.is_cpu_accessible());
        assert!(MetalStorageMode::Managed.is_cpu_accessible());
        assert!(!MetalStorageMode::Private.is_cpu_accessible());
        assert!(!MetalStorageMode::Memoryless.is_cpu_accessible());

        assert!(MetalStorageMode::Managed.needs_explicit_sync());
        assert!(!MetalStorageMode::Shared.needs_explicit_sync());

        assert!(!MetalStorageMode::Memoryless.has_backing_store());
        assert!(MetalStorageMode::Private.has_backing_store());
    }

    #[test]
    fn unified_prefers_shared() {
        let p = unified_planner();
        assert_eq!(
            p.select_mode(BufferAccess::UploadThenGpuRead),
            MetalStorageMode::Shared
        );
        assert_eq!(
            p.select_mode(BufferAccess::SharedReadWrite),
            MetalStorageMode::Shared
        );
        assert_eq!(
            p.select_mode(BufferAccess::GpuWriteThenDownload),
            MetalStorageMode::Shared
        );
    }

    #[test]
    fn discrete_uses_managed_for_cpu_visible() {
        let p = discrete_planner();
        assert_eq!(
            p.select_mode(BufferAccess::UploadThenGpuRead),
            MetalStorageMode::Managed
        );
        assert_eq!(
            p.select_mode(BufferAccess::SharedReadWrite),
            MetalStorageMode::Managed
        );
    }

    #[test]
    fn private_and_transient_modes() {
        let p = unified_planner();
        assert_eq!(
            p.select_mode(BufferAccess::GpuPrivate),
            MetalStorageMode::Private
        );
        assert_eq!(
            p.select_mode(BufferAccess::Transient),
            MetalStorageMode::Memoryless
        );
    }

    #[test]
    fn plan_buffer_aligns_and_flags_staging() {
        let p = unified_planner();
        let d = p.plan_buffer(100, BufferAccess::UploadThenGpuRead);
        assert_eq!(d.length, 100);
        assert_eq!(d.aligned_length, 256); // rounded up to 256
        assert_eq!(d.storage_mode, MetalStorageMode::Shared);
        assert!(!d.needs_staging);

        let priv_d = p.plan_buffer(300, BufferAccess::GpuPrivate);
        assert_eq!(priv_d.aligned_length, 512);
        assert!(priv_d.needs_staging);
    }

    #[test]
    fn plan_buffer_zero_length_is_one_unit() {
        let p = unified_planner();
        let d = p.plan_buffer(0, BufferAccess::SharedReadWrite);
        assert_eq!(d.aligned_length, 256);
    }

    #[test]
    fn align_up_cases() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(100, 16), 112);
    }

    #[test]
    fn align_down_cases() {
        assert_eq!(align_down(257, 256), 256);
        assert_eq!(align_down(256, 256), 256);
        assert_eq!(align_down(255, 256), 0);
        assert_eq!(align_down(100, 16), 96);
    }

    #[test]
    fn is_aligned_cases() {
        assert!(is_aligned(256, 256));
        assert!(is_aligned(512, 256));
        assert!(!is_aligned(255, 256));
        assert!(!is_aligned(100, 256));
        assert!(!is_aligned(100, 0));
    }

    #[test]
    fn alignment_zero_is_identity() {
        assert_eq!(align_up(123, 0), 123);
        assert_eq!(align_down(123, 0), 123);
    }

    #[test]
    fn display_modes() {
        assert_eq!(MetalStorageMode::Shared.to_string(), "Shared");
        assert_eq!(MetalStorageMode::Private.to_string(), "Private");
        assert_eq!(MetalStorageMode::Managed.to_string(), "Managed");
        assert_eq!(MetalStorageMode::Memoryless.to_string(), "Memoryless");
    }
}
