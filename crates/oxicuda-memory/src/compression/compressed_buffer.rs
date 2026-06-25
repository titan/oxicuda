//! Hardware-accelerated lossless memory compression bookkeeping.
//!
//! On Ampere (compute capability 8.0) and newer GPUs the virtual-memory
//! management API (`cuMemCreate`) accepts a
//! `CU_MEM_ALLOCATION_COMPRESSION_TYPE_GENERIC` allocation flag that enables
//! the memory controller's lossless compression engine.  Compressible
//! allocations transparently reduce the number of bytes that travel across the
//! L2 ↔ DRAM interface, raising effective bandwidth for compressible data
//! (e.g. activation tensors with large runs of zeros).
//!
//! This module models the *host-side bookkeeping* required to drive that
//! feature: capability gating, compression-granularity alignment, an
//! effective-bandwidth model, and a [`CompressedDeviceBuffer`] descriptor that
//! tracks the logical / physical footprint of a compressed allocation.  The
//! actual `cuMemCreate` call requires a GPU and lives behind the device-gated
//! path; everything here is deterministic and unit-testable on the host.
//!
//! # Example
//!
//! ```rust
//! # use oxicuda_memory::compression::compressed_buffer::*;
//! // Ampere supports generic compression; the granularity is 2 MiB.
//! let support = CompressionSupport::for_compute_capability(8, 0);
//! assert!(support.is_supported());
//!
//! // A 5 MiB request is rounded up to a multiple of the granularity.
//! let plan = CompressionPlan::new(5 * 1024 * 1024, support)?;
//! assert_eq!(plan.physical_bytes() % support.granularity(), 0);
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use oxicuda_driver::error::{CudaError, CudaResult};

// ---------------------------------------------------------------------------
// CompressionType
// ---------------------------------------------------------------------------

/// The kind of memory compression requested for an allocation.
///
/// Mirrors the `CUmemAllocationCompType` enumeration from the CUDA virtual
/// memory management API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CompressionType {
    /// No compression.  Equivalent to `CU_MEM_ALLOCATION_COMP_NONE`.
    None = 0,
    /// Generic hardware lossless compression.  Equivalent to
    /// `CU_MEM_ALLOCATION_COMP_GENERIC`.
    Generic = 1,
}

impl CompressionType {
    /// Returns `true` if this requests an actual compression engine.
    #[inline]
    #[must_use]
    pub fn is_compressed(self) -> bool {
        matches!(self, Self::Generic)
    }
}

impl std::fmt::Display for CompressionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Generic => write!(f, "generic"),
        }
    }
}

// ---------------------------------------------------------------------------
// CompressionSupport
// ---------------------------------------------------------------------------

/// The default compression page granularity on supported hardware (2 MiB).
///
/// `cuMemCreate` allocations must be a multiple of the granularity returned by
/// `cuMemGetAllocationGranularity`.  On all currently shipping Ampere/Hopper
/// parts this is the large-page size of 2 MiB.
pub const DEFAULT_COMPRESSION_GRANULARITY: usize = 2 * 1024 * 1024;

/// Describes whether and how a given device supports generic compression.
///
/// Construct via [`CompressionSupport::for_compute_capability`] to apply the
/// capability gate (compression requires compute capability ≥ 8.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionSupport {
    /// Whether generic compression is available at all.
    supported: bool,
    /// The allocation granularity in bytes.  Always non-zero.
    granularity: usize,
}

impl CompressionSupport {
    /// Returns the compression support for a device with the given compute
    /// capability `(major, minor)`.
    ///
    /// Generic memory compression is an Ampere+ feature, so any capability
    /// with `major >= 8` is considered supported.  Older devices report
    /// unsupported with the granularity still set to the default (so that
    /// alignment math remains well-defined).
    #[must_use]
    pub fn for_compute_capability(major: u32, minor: u32) -> Self {
        let _ = minor;
        Self {
            supported: major >= 8,
            granularity: DEFAULT_COMPRESSION_GRANULARITY,
        }
    }

    /// Builds an explicit support descriptor.
    ///
    /// `granularity` is clamped to a minimum of 1 so that alignment never
    /// divides by zero.
    #[must_use]
    pub fn new(supported: bool, granularity: usize) -> Self {
        Self {
            supported,
            granularity: granularity.max(1),
        }
    }

    /// Returns `true` if generic compression is available.
    #[inline]
    #[must_use]
    pub fn is_supported(&self) -> bool {
        self.supported
    }

    /// Returns the allocation granularity in bytes.
    #[inline]
    #[must_use]
    pub fn granularity(&self) -> usize {
        self.granularity
    }
}

// ---------------------------------------------------------------------------
// CompressionPlan
// ---------------------------------------------------------------------------

/// Rounds `value` up to the next multiple of `granularity`.
///
/// `granularity` is assumed non-zero (guaranteed by [`CompressionSupport`]).
/// Returns `None` on overflow.
#[inline]
fn round_up(value: usize, granularity: usize) -> Option<usize> {
    if value == 0 {
        return Some(0);
    }
    let blocks = value.checked_add(granularity - 1)? / granularity;
    blocks.checked_mul(granularity)
}

/// A validated, granularity-aligned plan for a compressed allocation.
///
/// Records the logically requested size and the physically reserved size
/// (rounded up to the compression granularity).  The physical footprint is the
/// number of bytes that `cuMemCreate` would reserve; the *effective* footprint
/// after compression depends on the data and is modelled separately by
/// [`CompressedDeviceBuffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionPlan {
    requested_bytes: usize,
    physical_bytes: usize,
    granularity: usize,
}

impl CompressionPlan {
    /// Builds a compression plan for `requested_bytes` against `support`.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `requested_bytes` is zero or rounding
    ///   to the granularity overflows.
    /// * [`CudaError::NotSupported`] if `support` reports that the device does
    ///   not support generic compression.
    pub fn new(requested_bytes: usize, support: CompressionSupport) -> CudaResult<Self> {
        if requested_bytes == 0 {
            return Err(CudaError::InvalidValue);
        }
        if !support.is_supported() {
            return Err(CudaError::NotSupported);
        }
        let physical_bytes =
            round_up(requested_bytes, support.granularity()).ok_or(CudaError::InvalidValue)?;
        Ok(Self {
            requested_bytes,
            physical_bytes,
            granularity: support.granularity(),
        })
    }

    /// The number of bytes the caller asked for.
    #[inline]
    #[must_use]
    pub fn requested_bytes(&self) -> usize {
        self.requested_bytes
    }

    /// The number of physical bytes reserved (granularity-aligned).
    #[inline]
    #[must_use]
    pub fn physical_bytes(&self) -> usize {
        self.physical_bytes
    }

    /// The granularity this plan was aligned to.
    #[inline]
    #[must_use]
    pub fn granularity(&self) -> usize {
        self.granularity
    }

    /// The padding bytes added by granularity rounding (`physical - requested`).
    #[inline]
    #[must_use]
    pub fn padding_bytes(&self) -> usize {
        self.physical_bytes.saturating_sub(self.requested_bytes)
    }
}

// ---------------------------------------------------------------------------
// CompressedDeviceBuffer
// ---------------------------------------------------------------------------

/// Host-side descriptor for a compressed device allocation.
///
/// Tracks the logical size, the physically reserved (granularity-aligned)
/// size, and an *observed* compression ratio that the application can update
/// as it learns how compressible its data is.  The descriptor never owns a GPU
/// pointer here — that is filled in by the device-gated path — but it provides
/// the bandwidth and footprint accounting that callers use to decide whether
/// compression is worthwhile.
///
/// The compression ratio is defined as `physical / effective`: a ratio of
/// `2.0` means the data occupies half its physical footprint over the memory
/// bus, doubling effective bandwidth for that region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressedDeviceBuffer {
    plan: CompressionPlan,
    comp_type: CompressionType,
    /// Observed compression ratio (`physical / effective`).  `1.0` means data
    /// did not compress at all.  Always `>= 1.0`.
    ratio: f64,
}

impl CompressedDeviceBuffer {
    /// Creates a descriptor for a freshly planned compressed buffer.
    ///
    /// The compression ratio starts at `1.0` (no measured compression yet).
    #[must_use]
    pub fn new(plan: CompressionPlan) -> Self {
        Self {
            plan,
            comp_type: CompressionType::Generic,
            ratio: 1.0,
        }
    }

    /// Builds a descriptor directly from a request, applying capability gating
    /// and granularity alignment in one step.
    ///
    /// # Errors
    ///
    /// Forwards errors from [`CompressionPlan::new`].
    pub fn alloc(requested_bytes: usize, support: CompressionSupport) -> CudaResult<Self> {
        Ok(Self::new(CompressionPlan::new(requested_bytes, support)?))
    }

    /// Returns the underlying allocation plan.
    #[inline]
    #[must_use]
    pub fn plan(&self) -> CompressionPlan {
        self.plan
    }

    /// Returns the compression type recorded for this buffer.
    #[inline]
    #[must_use]
    pub fn compression_type(&self) -> CompressionType {
        self.comp_type
    }

    /// The logical (requested) size in bytes.
    #[inline]
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        self.plan.requested_bytes()
    }

    /// The physical (granularity-aligned) footprint in bytes.
    #[inline]
    #[must_use]
    pub fn physical_bytes(&self) -> usize {
        self.plan.physical_bytes()
    }

    /// Records an observed compression ratio (`physical / effective`).
    ///
    /// Values below `1.0` are clamped to `1.0` (data cannot expand under
    /// lossless compression as far as bus traffic is concerned).  Non-finite
    /// inputs are ignored.
    pub fn set_ratio(&mut self, ratio: f64) {
        if ratio.is_finite() {
            self.ratio = ratio.max(1.0);
        }
    }

    /// Returns the recorded compression ratio (`>= 1.0`).
    #[inline]
    #[must_use]
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// The effective number of bytes that traverse the memory bus given the
    /// recorded compression ratio.
    ///
    /// Equal to `physical_bytes / ratio`, rounded down.  With a ratio of `1.0`
    /// this equals the physical footprint.
    #[must_use]
    pub fn effective_bus_bytes(&self) -> usize {
        if self.ratio <= 1.0 {
            return self.physical_bytes();
        }
        (self.physical_bytes() as f64 / self.ratio) as usize
    }

    /// Estimates the effective bandwidth seen by a transfer of this buffer's
    /// physical footprint, given a raw `dram_gbps` DRAM bandwidth.
    ///
    /// Because compression reduces bus traffic by `ratio`, the *effective*
    /// bandwidth observed for compressible data is `dram_gbps * ratio`.
    /// Non-positive bandwidth returns `0.0`.
    #[must_use]
    pub fn effective_bandwidth_gbps(&self, dram_gbps: f64) -> f64 {
        if dram_gbps <= 0.0 {
            return 0.0;
        }
        dram_gbps * self.ratio
    }

    /// The number of physical bytes saved on the bus relative to an
    /// uncompressed transfer (`physical - effective`).
    #[inline]
    #[must_use]
    pub fn bytes_saved(&self) -> usize {
        self.physical_bytes()
            .saturating_sub(self.effective_bus_bytes())
    }
}

impl std::fmt::Display for CompressedDeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CompressedDeviceBuffer(type={}, logical={} B, physical={} B, ratio={:.2})",
            self.comp_type,
            self.logical_bytes(),
            self.physical_bytes(),
            self.ratio,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_type_is_compressed() {
        assert!(!CompressionType::None.is_compressed());
        assert!(CompressionType::Generic.is_compressed());
    }

    #[test]
    fn support_gated_on_compute_capability() {
        // Ampere (8.0), Hopper (9.0) -> supported.
        assert!(CompressionSupport::for_compute_capability(8, 0).is_supported());
        assert!(CompressionSupport::for_compute_capability(9, 0).is_supported());
        // Turing (7.5), Volta (7.0), Pascal (6.1) -> unsupported.
        assert!(!CompressionSupport::for_compute_capability(7, 5).is_supported());
        assert!(!CompressionSupport::for_compute_capability(7, 0).is_supported());
        assert!(!CompressionSupport::for_compute_capability(6, 1).is_supported());
    }

    #[test]
    fn support_granularity_default() {
        let s = CompressionSupport::for_compute_capability(8, 6);
        assert_eq!(s.granularity(), 2 * 1024 * 1024);
    }

    #[test]
    fn support_new_clamps_zero_granularity() {
        let s = CompressionSupport::new(true, 0);
        assert_eq!(s.granularity(), 1);
    }

    #[test]
    fn round_up_exact_multiple_unchanged() {
        let g = 2 * 1024 * 1024;
        assert_eq!(round_up(g, g), Some(g));
        assert_eq!(round_up(2 * g, g), Some(2 * g));
    }

    #[test]
    fn round_up_partial_block_rounds_up() {
        let g = 2 * 1024 * 1024;
        // 1 byte -> one full granularity block.
        assert_eq!(round_up(1, g), Some(g));
        // g + 1 -> two blocks.
        assert_eq!(round_up(g + 1, g), Some(2 * g));
    }

    #[test]
    fn round_up_zero_is_zero() {
        assert_eq!(round_up(0, 4096), Some(0));
    }

    #[test]
    fn round_up_overflow_returns_none() {
        assert_eq!(round_up(usize::MAX, 2 * 1024 * 1024), None);
    }

    #[test]
    fn plan_rejects_zero() {
        let s = CompressionSupport::for_compute_capability(8, 0);
        assert_eq!(CompressionPlan::new(0, s), Err(CudaError::InvalidValue));
    }

    #[test]
    fn plan_rejects_unsupported_device() {
        let s = CompressionSupport::for_compute_capability(7, 5);
        assert_eq!(CompressionPlan::new(4096, s), Err(CudaError::NotSupported));
    }

    #[test]
    fn plan_aligns_to_granularity() {
        let s = CompressionSupport::for_compute_capability(8, 0);
        let g = s.granularity();
        // 5 MiB request with 2 MiB granularity -> 6 MiB physical (3 blocks).
        let plan = CompressionPlan::new(5 * 1024 * 1024, s).expect("plan");
        assert_eq!(plan.requested_bytes(), 5 * 1024 * 1024);
        assert_eq!(plan.physical_bytes(), 3 * g);
        assert_eq!(plan.physical_bytes() % g, 0);
        // padding = 6 MiB - 5 MiB = 1 MiB.
        assert_eq!(plan.padding_bytes(), 1024 * 1024);
    }

    #[test]
    fn plan_exact_multiple_has_no_padding() {
        let s = CompressionSupport::for_compute_capability(8, 0);
        let plan = CompressionPlan::new(s.granularity(), s).expect("plan");
        assert_eq!(plan.padding_bytes(), 0);
        assert_eq!(plan.physical_bytes(), s.granularity());
    }

    #[test]
    fn buffer_alloc_defaults_to_generic_ratio_one() {
        let s = CompressionSupport::for_compute_capability(8, 0);
        let buf = CompressedDeviceBuffer::alloc(1024 * 1024, s).expect("alloc");
        assert_eq!(buf.compression_type(), CompressionType::Generic);
        assert!((buf.ratio() - 1.0).abs() < 1e-12);
        // With ratio 1.0, effective == physical and nothing is saved.
        assert_eq!(buf.effective_bus_bytes(), buf.physical_bytes());
        assert_eq!(buf.bytes_saved(), 0);
    }

    #[test]
    fn buffer_ratio_clamped_and_effective_bytes() {
        let s = CompressionSupport::for_compute_capability(8, 0);
        let mut buf = CompressedDeviceBuffer::alloc(2 * 1024 * 1024, s).expect("alloc");
        // physical is exactly 2 MiB.
        assert_eq!(buf.physical_bytes(), 2 * 1024 * 1024);
        buf.set_ratio(2.0);
        assert!((buf.ratio() - 2.0).abs() < 1e-12);
        // 2 MiB / 2.0 = 1 MiB effective on the bus.
        assert_eq!(buf.effective_bus_bytes(), 1024 * 1024);
        assert_eq!(buf.bytes_saved(), 1024 * 1024);
    }

    #[test]
    fn buffer_ratio_below_one_clamped() {
        let s = CompressionSupport::for_compute_capability(8, 0);
        let mut buf = CompressedDeviceBuffer::alloc(2 * 1024 * 1024, s).expect("alloc");
        buf.set_ratio(0.5);
        assert!((buf.ratio() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn buffer_ignores_non_finite_ratio() {
        let s = CompressionSupport::for_compute_capability(8, 0);
        let mut buf = CompressedDeviceBuffer::alloc(2 * 1024 * 1024, s).expect("alloc");
        buf.set_ratio(3.0);
        buf.set_ratio(f64::NAN);
        buf.set_ratio(f64::INFINITY);
        // Unchanged from the last finite value.
        assert!((buf.ratio() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn buffer_effective_bandwidth_scales_with_ratio() {
        let s = CompressionSupport::for_compute_capability(8, 0);
        let mut buf = CompressedDeviceBuffer::alloc(2 * 1024 * 1024, s).expect("alloc");
        buf.set_ratio(2.5);
        // 1000 GB/s DRAM with 2.5x compression -> 2500 GB/s effective.
        assert!((buf.effective_bandwidth_gbps(1000.0) - 2500.0).abs() < 1e-9);
        assert_eq!(buf.effective_bandwidth_gbps(0.0), 0.0);
    }

    #[test]
    fn buffer_display_contains_fields() {
        let s = CompressionSupport::for_compute_capability(8, 0);
        let buf = CompressedDeviceBuffer::alloc(1024 * 1024, s).expect("alloc");
        let text = format!("{buf}");
        assert!(text.contains("generic"));
        assert!(text.contains("physical="));
    }
}
