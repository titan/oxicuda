//! AMD GPU architecture (`gfx*`) capability tables.
//!
//! Provides a static, hardware-derived description of each supported AMD GPU
//! ISA target — CDNA1/2/3 (Instinct MI-series) and RDNA2/3 (Radeon RX) — so
//! that host-side planners (occupancy, launch configuration, MFMA tile
//! selection) can reason about resource limits **without** a live HIP runtime.
//!
//! All limits are taken from the AMD ISA reference manuals and the LLVM
//! AMDGPU back-end:
//!
//! - VGPR file sizes per SIMD (CDNA: 256 VGPR/lane, i.e. a 64 KiB file over 64
//!   lanes; RDNA2/3: 1024 VGPR/lane, i.e. a 128 KiB file over the 32-lane
//!   SIMD32).
//! - LDS (Local Data Share) size per compute unit (CDNA1/2/3: 64 KiB,
//!   RDNA2/3: 64 KiB).
//! - Wavefront width (CDNA: 64 lanes, RDNA: 32 lanes natively, 64 in
//!   wave64 mode).
//! - Matrix-core capability (MFMA on CDNA, WMMA on RDNA3).
//!
//! These are *modeling* tables: they do not require any AMD GPU to be present.

// ─── GfxArch ───────────────────────────────────────────────────────────────────

/// A concrete AMD GPU ISA target.
///
/// The naming follows the LLVM `--gpu-architecture=gfxNNN` convention used by
/// `hipcc` and `hiprtcCompileProgram`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GfxArch {
    /// CDNA1, MI100 (`gfx908`).
    Gfx908,
    /// CDNA1 / Vega20, MI50/MI60 (`gfx906`).
    Gfx906,
    /// CDNA2, MI210/MI250/MI250X (`gfx90a`).
    Gfx90a,
    /// CDNA3, MI300A (`gfx940`).
    Gfx940,
    /// CDNA3, MI300 variant (`gfx941`).
    Gfx941,
    /// CDNA3, MI300X (`gfx942`).
    Gfx942,
    /// RDNA2, Radeon RX 6000 series (`gfx1030`).
    Gfx1030,
    /// RDNA3, Radeon RX 7000 series (`gfx1100`).
    Gfx1100,
}

/// The micro-architecture generation a [`GfxArch`] belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchFamily {
    /// CDNA1 — first-generation Instinct compute architecture.
    Cdna1,
    /// CDNA2 — MI200 series, BF16 + FP64 matrix cores.
    Cdna2,
    /// CDNA3 — MI300 series, FP8 matrix cores, split-die XCD.
    Cdna3,
    /// RDNA2 — Radeon RX 6000, no matrix cores.
    Rdna2,
    /// RDNA3 — Radeon RX 7000, WMMA matrix cores.
    Rdna3,
}

impl GfxArch {
    /// All architectures known to this backend, in canonical order.
    pub const ALL: [GfxArch; 8] = [
        GfxArch::Gfx906,
        GfxArch::Gfx908,
        GfxArch::Gfx90a,
        GfxArch::Gfx940,
        GfxArch::Gfx941,
        GfxArch::Gfx942,
        GfxArch::Gfx1030,
        GfxArch::Gfx1100,
    ];

    /// The canonical `gfx*` identifier string passed to the compiler.
    pub fn target_id(self) -> &'static str {
        match self {
            GfxArch::Gfx906 => "gfx906",
            GfxArch::Gfx908 => "gfx908",
            GfxArch::Gfx90a => "gfx90a",
            GfxArch::Gfx940 => "gfx940",
            GfxArch::Gfx941 => "gfx941",
            GfxArch::Gfx942 => "gfx942",
            GfxArch::Gfx1030 => "gfx1030",
            GfxArch::Gfx1100 => "gfx1100",
        }
    }

    /// Parse a `gfx*` identifier (case-insensitive) back into a [`GfxArch`].
    ///
    /// Returns `None` for unknown targets.
    pub fn from_target_id(id: &str) -> Option<GfxArch> {
        let id = id.trim().to_ascii_lowercase();
        GfxArch::ALL.into_iter().find(|a| a.target_id() == id)
    }

    /// Best-effort detection from a device name reported by
    /// `hipGetDeviceProperties` (e.g. `"AMD Instinct MI250X"`).
    ///
    /// Returns `None` when the name cannot be mapped to a known architecture.
    pub fn from_device_name(name: &str) -> Option<GfxArch> {
        let n = name.to_ascii_lowercase();
        // Explicit gfx ids win.
        if let Some(a) = GfxArch::ALL.into_iter().find(|a| n.contains(a.target_id())) {
            return Some(a);
        }
        // Marketing names. Check the more-specific MI300A before MI300/MI300X,
        // since "MI300A" also contains the "mi300" substring.
        if n.contains("mi300a") {
            return Some(GfxArch::Gfx940);
        }
        if n.contains("mi300x") || n.contains("mi300") {
            return Some(GfxArch::Gfx942);
        }
        if n.contains("mi250") || n.contains("mi210") {
            return Some(GfxArch::Gfx90a);
        }
        if n.contains("mi100") {
            return Some(GfxArch::Gfx908);
        }
        if n.contains("mi50") || n.contains("mi60") || n.contains("vega 20") {
            return Some(GfxArch::Gfx906);
        }
        if n.contains("rx 7") || n.contains("navi 3") {
            return Some(GfxArch::Gfx1100);
        }
        if n.contains("rx 6") || n.contains("navi 2") {
            return Some(GfxArch::Gfx1030);
        }
        None
    }

    /// The micro-architecture family.
    pub fn family(self) -> ArchFamily {
        match self {
            GfxArch::Gfx906 | GfxArch::Gfx908 => ArchFamily::Cdna1,
            GfxArch::Gfx90a => ArchFamily::Cdna2,
            GfxArch::Gfx940 | GfxArch::Gfx941 | GfxArch::Gfx942 => ArchFamily::Cdna3,
            GfxArch::Gfx1030 => ArchFamily::Rdna2,
            GfxArch::Gfx1100 => ArchFamily::Rdna3,
        }
    }

    /// `true` for the CDNA (Instinct compute) lineage.
    pub fn is_cdna(self) -> bool {
        matches!(
            self.family(),
            ArchFamily::Cdna1 | ArchFamily::Cdna2 | ArchFamily::Cdna3
        )
    }

    /// `true` for the RDNA (Radeon graphics/compute) lineage.
    pub fn is_rdna(self) -> bool {
        matches!(self.family(), ArchFamily::Rdna2 | ArchFamily::Rdna3)
    }

    /// Native wavefront width in lanes (CDNA → 64, RDNA → 32).
    pub fn native_wavefront(self) -> u32 {
        if self.is_cdna() { 64 } else { 32 }
    }

    /// Number of 32-bit vector general-purpose registers (VGPRs) addressable
    /// per lane by a single wavefront.
    ///
    /// CDNA/GCN: a 64 KiB VGPR file shared by 64 lanes → `64*1024 / 4 / 64 =
    /// 256` VGPRs/lane.  RDNA2/3: a 128 KiB file over the 32-lane SIMD32 →
    /// `128*1024 / 4 / 32 = 1024` VGPRs/lane.
    pub fn vgprs_per_simd(self) -> u32 {
        if self.is_rdna() { 1024 } else { 256 }
    }

    /// Number of scalar general-purpose registers (SGPRs) addressable per
    /// wavefront.  Architecturally 102 usable (others reserved) on GCN/CDNA;
    /// RDNA exposes 106.
    pub fn sgprs_per_wave(self) -> u32 {
        if self.is_rdna() { 106 } else { 102 }
    }

    /// LDS (Local Data Share / `__shared__`) capacity per compute unit, in
    /// bytes.  64 KiB across CDNA1/2/3 and RDNA2/3.
    pub fn lds_bytes_per_cu(self) -> u32 {
        64 * 1024
    }

    /// Number of SIMD units per compute unit (CU).  4 on both CDNA and RDNA.
    pub fn simds_per_cu(self) -> u32 {
        4
    }

    /// VGPR allocation granularity in registers — wavefront VGPR allocations
    /// round up to a multiple of this value.  CDNA allocates in blocks of 4,
    /// RDNA in blocks of 8 (per the AMDGPU back-end).
    pub fn vgpr_alloc_granularity(self) -> u32 {
        if self.is_rdna() { 8 } else { 4 }
    }

    /// Maximum number of wavefronts that can be resident on a single SIMD.
    ///
    /// CDNA SIMDs support up to 8 concurrent (wave64) wavefronts; RDNA SIMD32
    /// supports up to 16 (wave32) wavefronts.
    pub fn max_waves_per_simd(self) -> u32 {
        if self.is_rdna() { 16 } else { 8 }
    }

    /// Maximum threads (work-items) per work-group (block).  1024 on all
    /// supported architectures.
    pub fn max_threads_per_block(self) -> u32 {
        1024
    }

    /// `true` if this architecture exposes MFMA (Matrix Fused-Multiply-Add)
    /// instructions for matrix-core acceleration (all CDNA generations).
    pub fn has_mfma(self) -> bool {
        self.is_cdna()
    }

    /// `true` if this architecture exposes WMMA (Wave Matrix Multiply-Accumulate)
    /// instructions (RDNA3 only).
    pub fn has_wmma(self) -> bool {
        matches!(self.family(), ArchFamily::Rdna3)
    }

    /// `true` if this architecture supports native FP8 (OCP E4M3 / E5M2) matrix
    /// instructions (CDNA3 only).
    pub fn has_fp8_mfma(self) -> bool {
        matches!(self.family(), ArchFamily::Cdna3)
    }

    /// `true` if this architecture supports native BF16 matrix instructions
    /// (CDNA2+ and RDNA3).
    pub fn has_bf16_mfma(self) -> bool {
        matches!(
            self.family(),
            ArchFamily::Cdna2 | ArchFamily::Cdna3 | ArchFamily::Rdna3
        )
    }

    /// `true` if this architecture supports native FP64 matrix instructions
    /// (CDNA2+ only).
    pub fn has_fp64_mfma(self) -> bool {
        matches!(self.family(), ArchFamily::Cdna2 | ArchFamily::Cdna3)
    }

    /// Number of XCD (Accelerator Complex Die) chiplets on the package.
    ///
    /// MI300X (`gfx942`) ships 8 XCDs; MI300A (`gfx940`) ships 6; everything
    /// else is monolithic (1).
    pub fn xcd_count(self) -> u32 {
        match self {
            GfxArch::Gfx942 => 8,
            GfxArch::Gfx940 | GfxArch::Gfx941 => 6,
            _ => 1,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_id_roundtrip() {
        for arch in GfxArch::ALL {
            let id = arch.target_id();
            assert_eq!(GfxArch::from_target_id(id), Some(arch));
            // Case-insensitive parse.
            assert_eq!(GfxArch::from_target_id(&id.to_uppercase()), Some(arch));
        }
        assert_eq!(GfxArch::from_target_id("gfx999"), None);
    }

    #[test]
    fn device_name_detection() {
        assert_eq!(
            GfxArch::from_device_name("AMD Instinct MI250X"),
            Some(GfxArch::Gfx90a)
        );
        assert_eq!(
            GfxArch::from_device_name("AMD Instinct MI300X"),
            Some(GfxArch::Gfx942)
        );
        assert_eq!(
            GfxArch::from_device_name("AMD Instinct MI300A"),
            Some(GfxArch::Gfx940)
        );
        assert_eq!(
            GfxArch::from_device_name("AMD Instinct MI100"),
            Some(GfxArch::Gfx908)
        );
        assert_eq!(
            GfxArch::from_device_name("AMD Radeon RX 7900 XTX"),
            Some(GfxArch::Gfx1100)
        );
        assert_eq!(
            GfxArch::from_device_name("AMD Radeon RX 6800"),
            Some(GfxArch::Gfx1030)
        );
        // Explicit gfx id embedded in name.
        assert_eq!(
            GfxArch::from_device_name("device gfx90a:sramecc+:xnack-"),
            Some(GfxArch::Gfx90a)
        );
        assert_eq!(GfxArch::from_device_name("Intel Arc A770"), None);
    }

    #[test]
    fn family_classification() {
        assert_eq!(GfxArch::Gfx908.family(), ArchFamily::Cdna1);
        assert_eq!(GfxArch::Gfx90a.family(), ArchFamily::Cdna2);
        assert_eq!(GfxArch::Gfx942.family(), ArchFamily::Cdna3);
        assert_eq!(GfxArch::Gfx1030.family(), ArchFamily::Rdna2);
        assert_eq!(GfxArch::Gfx1100.family(), ArchFamily::Rdna3);
    }

    #[test]
    fn cdna_vs_rdna() {
        assert!(GfxArch::Gfx90a.is_cdna());
        assert!(!GfxArch::Gfx90a.is_rdna());
        assert!(GfxArch::Gfx1100.is_rdna());
        assert!(!GfxArch::Gfx1100.is_cdna());
    }

    #[test]
    fn wavefront_widths() {
        assert_eq!(GfxArch::Gfx908.native_wavefront(), 64);
        assert_eq!(GfxArch::Gfx90a.native_wavefront(), 64);
        assert_eq!(GfxArch::Gfx1030.native_wavefront(), 32);
        assert_eq!(GfxArch::Gfx1100.native_wavefront(), 32);
    }

    #[test]
    fn matrix_core_capabilities() {
        // CDNA3 has everything.
        assert!(GfxArch::Gfx942.has_mfma());
        assert!(GfxArch::Gfx942.has_fp8_mfma());
        assert!(GfxArch::Gfx942.has_bf16_mfma());
        assert!(GfxArch::Gfx942.has_fp64_mfma());

        // CDNA2: BF16 + FP64 but no FP8.
        assert!(GfxArch::Gfx90a.has_mfma());
        assert!(!GfxArch::Gfx90a.has_fp8_mfma());
        assert!(GfxArch::Gfx90a.has_bf16_mfma());
        assert!(GfxArch::Gfx90a.has_fp64_mfma());

        // CDNA1: MFMA but no BF16/FP64/FP8 matrix.
        assert!(GfxArch::Gfx908.has_mfma());
        assert!(!GfxArch::Gfx908.has_bf16_mfma());
        assert!(!GfxArch::Gfx908.has_fp64_mfma());
        assert!(!GfxArch::Gfx908.has_fp8_mfma());

        // RDNA3: WMMA, no MFMA, no FP8.
        assert!(GfxArch::Gfx1100.has_wmma());
        assert!(!GfxArch::Gfx1100.has_mfma());
        assert!(GfxArch::Gfx1100.has_bf16_mfma());
        assert!(!GfxArch::Gfx1100.has_fp8_mfma());

        // RDNA2: no matrix cores at all.
        assert!(!GfxArch::Gfx1030.has_wmma());
        assert!(!GfxArch::Gfx1030.has_mfma());
    }

    #[test]
    fn resource_limits() {
        let a = GfxArch::Gfx90a;
        assert_eq!(a.vgprs_per_simd(), 256);
        assert_eq!(a.lds_bytes_per_cu(), 65536);
        assert_eq!(a.simds_per_cu(), 4);
        assert_eq!(a.max_waves_per_simd(), 8);
        assert_eq!(a.vgpr_alloc_granularity(), 4);
        assert_eq!(a.max_threads_per_block(), 1024);

        let r = GfxArch::Gfx1100;
        assert_eq!(r.max_waves_per_simd(), 16);
        assert_eq!(r.vgpr_alloc_granularity(), 8);
        assert_eq!(r.sgprs_per_wave(), 106);
        // RDNA SIMD32 exposes a 128 KiB VGPR file → 1024 VGPRs/lane, 4x CDNA.
        assert_eq!(r.vgprs_per_simd(), 1024);
    }

    #[test]
    fn xcd_chiplet_counts() {
        assert_eq!(GfxArch::Gfx942.xcd_count(), 8);
        assert_eq!(GfxArch::Gfx940.xcd_count(), 6);
        assert_eq!(GfxArch::Gfx90a.xcd_count(), 1);
        assert_eq!(GfxArch::Gfx1100.xcd_count(), 1);
    }
}
