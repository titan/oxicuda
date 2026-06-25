//! Host-side occupancy calculator for AMD GPUs.
//!
//! Computes how many wavefronts (and therefore work-groups) can be resident on
//! a single compute unit (CU) for a given kernel resource footprint — VGPRs,
//! SGPRs, LDS bytes, and work-group size — on a given [`GfxArch`].
//!
//! This mirrors the role of `hipOccupancyMaxActiveBlocksPerMultiprocessor`, but
//! is a **pure host-side model**: it derives the answer from the published
//! per-architecture resource limits in [`crate::gfx_arch`] and never touches a
//! HIP runtime, so it is fully testable on CPU-only systems.
//!
//! The limiting-resource model follows the AMD ISA reference:
//!
//! - VGPR-bound:  `waves_per_simd = vgpr_file / round_up(vgprs_used, gran)`
//! - SGPR-bound:  `waves_per_simd = sgpr_limit / sgprs_used` (modeled per-wave)
//! - LDS-bound:   `groups_per_cu  = lds_per_cu / round_up(lds_used, 256)`
//! - Wave-cap:    `waves_per_simd <= max_waves_per_simd`
//!
//! The achieved occupancy is the *minimum* across every limiter.

use crate::gfx_arch::GfxArch;

// ─── KernelResources ────────────────────────────────────────────────────────

/// The static resource footprint of a compiled kernel, as would be reported by
/// the AMDGPU assembler (`.vgpr_count`, `.sgpr_count`, `.group_segment_fixed_size`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelResources {
    /// VGPRs used per work-item (lane).
    pub vgprs: u32,
    /// SGPRs used per wavefront.
    pub sgprs: u32,
    /// Static LDS (`__shared__`) bytes used per work-group.
    pub lds_bytes: u32,
    /// Work-group (block) size in work-items.
    pub workgroup_size: u32,
}

impl KernelResources {
    /// Construct a resource record.
    pub fn new(vgprs: u32, sgprs: u32, lds_bytes: u32, workgroup_size: u32) -> Self {
        Self {
            vgprs,
            sgprs,
            lds_bytes,
            workgroup_size,
        }
    }
}

/// Which resource caps occupancy for a given kernel/architecture pairing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitingResource {
    /// Vector register file is the bottleneck.
    Vgpr,
    /// Scalar register file is the bottleneck.
    Sgpr,
    /// Local Data Share (`__shared__`) capacity is the bottleneck.
    Lds,
    /// The hardware wave-per-SIMD cap is the bottleneck.
    WaveSlots,
    /// The kernel is invalid (zero work-group size).
    Invalid,
}

// ─── Occupancy result ───────────────────────────────────────────────────────

/// The computed occupancy of a kernel on a specific architecture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Occupancy {
    /// Maximum resident wavefronts per compute unit.
    pub waves_per_cu: u32,
    /// Maximum resident work-groups (blocks) per compute unit.
    pub blocks_per_cu: u32,
    /// The resource that limited occupancy.
    pub limited_by: LimitingResource,
    /// Theoretical occupancy as a fraction of the hardware wave cap (`0.0..=1.0`).
    pub fraction: f32,
}

/// Round `value` up to the next multiple of `granule` (which must be non-zero).
fn round_up(value: u32, granule: u32) -> u32 {
    if granule == 0 {
        return value;
    }
    value.div_ceil(granule) * granule
}

// ─── OccupancyCalculator ────────────────────────────────────────────────────

/// Computes occupancy for kernels on a fixed target architecture.
#[derive(Debug, Clone, Copy)]
pub struct OccupancyCalculator {
    arch: GfxArch,
}

impl OccupancyCalculator {
    /// Create a calculator for `arch`.
    pub fn new(arch: GfxArch) -> Self {
        Self { arch }
    }

    /// The architecture this calculator targets.
    pub fn arch(&self) -> GfxArch {
        self.arch
    }

    /// Wavefronts allowed per SIMD given the VGPR footprint.
    fn waves_per_simd_vgpr(&self, vgprs: u32) -> u32 {
        let file = self.arch.vgprs_per_simd();
        let granule = self.arch.vgpr_alloc_granularity();
        let alloc = round_up(vgprs.max(1), granule);
        (file / alloc).max(1)
    }

    /// Wavefronts allowed per SIMD given the SGPR footprint.
    fn waves_per_simd_sgpr(&self, sgprs: u32) -> u32 {
        // SGPRs are allocated per-wave; the file holds `sgprs_per_wave`
        // architectural registers, but in practice the wave count is capped by
        // the hardware wave-slot limit once SGPR usage is below the budget.
        let budget = self.arch.sgprs_per_wave();
        if sgprs <= budget {
            self.arch.max_waves_per_simd()
        } else {
            1
        }
    }

    /// Work-groups allowed per CU given the LDS footprint.
    fn blocks_per_cu_lds(&self, lds_bytes: u32) -> u32 {
        if lds_bytes == 0 {
            return u32::MAX;
        }
        let per_cu = self.arch.lds_bytes_per_cu();
        // LDS is allocated in 256-byte (DWORDx64) granules on AMDGPU.
        let alloc = round_up(lds_bytes, 256);
        per_cu / alloc
    }

    /// Compute the occupancy of a kernel described by `res`.
    ///
    /// Returns the number of resident wavefronts and work-groups per CU, plus
    /// the limiting resource.
    pub fn compute(&self, res: KernelResources) -> Occupancy {
        if res.workgroup_size == 0 {
            return Occupancy {
                waves_per_cu: 0,
                blocks_per_cu: 0,
                limited_by: LimitingResource::Invalid,
                fraction: 0.0,
            };
        }

        let wave_width = self.arch.native_wavefront();
        let waves_per_block = res.workgroup_size.div_ceil(wave_width);
        let simds = self.arch.simds_per_cu();
        let wave_cap_per_cu = self.arch.max_waves_per_simd() * simds;

        // Per-SIMD wave limits from each register file.
        let vgpr_waves_simd = self.waves_per_simd_vgpr(res.vgprs);
        let sgpr_waves_simd = self.waves_per_simd_sgpr(res.sgprs);
        let slot_waves_simd = self.arch.max_waves_per_simd();

        // The most restrictive per-SIMD wave count, then scaled to the CU.
        let reg_waves_per_cu = vgpr_waves_simd.min(sgpr_waves_simd).min(slot_waves_simd) * simds;

        // LDS constrains whole work-groups at the CU level.
        let lds_blocks = self.blocks_per_cu_lds(res.lds_bytes);
        let lds_waves_per_cu = lds_blocks.saturating_mul(waves_per_block);

        // Resident waves = min over all limiters, also bounded by the hw cap.
        let mut limited_by = LimitingResource::WaveSlots;
        let mut waves = wave_cap_per_cu;

        if reg_waves_per_cu < waves {
            waves = reg_waves_per_cu;
            // Decide whether VGPR or SGPR was the binding register file.
            limited_by = if vgpr_waves_simd <= sgpr_waves_simd {
                LimitingResource::Vgpr
            } else {
                LimitingResource::Sgpr
            };
        }
        if lds_waves_per_cu < waves {
            waves = lds_waves_per_cu;
            limited_by = LimitingResource::Lds;
        }

        // Resident waves must be a whole number of work-groups.
        let blocks = waves / waves_per_block;
        let waves = blocks * waves_per_block;

        let fraction = if wave_cap_per_cu == 0 {
            0.0
        } else {
            waves as f32 / wave_cap_per_cu as f32
        };

        Occupancy {
            waves_per_cu: waves,
            blocks_per_cu: blocks,
            limited_by,
            fraction,
        }
    }

    /// Search work-group sizes in `[wave_width, max]` (stepping by `wave_width`)
    /// and return the size achieving the highest occupancy, given a per-lane
    /// VGPR / per-wave SGPR cost and a fixed per-group LDS cost.
    ///
    /// Mirrors `hipOccupancyMaxPotentialBlockSize`.  Returns `(block_size,
    /// occupancy)`.
    pub fn max_potential_block_size(
        &self,
        vgprs: u32,
        sgprs: u32,
        lds_bytes: u32,
    ) -> (u32, Occupancy) {
        let wave_width = self.arch.native_wavefront();
        let max = self.arch.max_threads_per_block();
        let mut best_size = wave_width;
        let mut best = self.compute(KernelResources::new(vgprs, sgprs, lds_bytes, wave_width));

        let mut size = wave_width;
        while size <= max {
            let occ = self.compute(KernelResources::new(vgprs, sgprs, lds_bytes, size));
            // Prefer higher resident-wave count; break ties toward larger blocks.
            if occ.waves_per_cu > best.waves_per_cu
                || (occ.waves_per_cu == best.waves_per_cu && size > best_size)
            {
                best = occ;
                best_size = size;
            }
            size += wave_width;
        }
        (best_size, best)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_workgroup_is_invalid() {
        let calc = OccupancyCalculator::new(GfxArch::Gfx90a);
        let occ = calc.compute(KernelResources::new(32, 16, 0, 0));
        assert_eq!(occ.limited_by, LimitingResource::Invalid);
        assert_eq!(occ.waves_per_cu, 0);
        assert_eq!(occ.fraction, 0.0);
    }

    #[test]
    fn light_kernel_hits_wave_cap() {
        // A 64-thread wave64 block using few registers and no LDS should reach
        // the maximum wave occupancy (8 waves/SIMD * 4 SIMD = 32 waves/CU).
        let calc = OccupancyCalculator::new(GfxArch::Gfx90a);
        let occ = calc.compute(KernelResources::new(16, 8, 0, 64));
        assert_eq!(occ.waves_per_cu, 32);
        assert_eq!(occ.blocks_per_cu, 32);
        assert_eq!(occ.limited_by, LimitingResource::WaveSlots);
        assert!((occ.fraction - 1.0).abs() < 1e-6);
    }

    #[test]
    fn high_vgpr_kernel_is_vgpr_bound() {
        // 128 VGPRs/lane on gfx90a → 256/128 = 2 waves/SIMD → 8 waves/CU.
        let calc = OccupancyCalculator::new(GfxArch::Gfx90a);
        let occ = calc.compute(KernelResources::new(128, 16, 0, 64));
        assert_eq!(occ.limited_by, LimitingResource::Vgpr);
        assert_eq!(occ.waves_per_cu, 8);
        assert!(occ.fraction < 1.0);
    }

    #[test]
    fn vgpr_allocation_rounds_up() {
        // 100 VGPRs rounds up to 100 (mult of 4) → 256/100 = 2 waves/SIMD.
        let calc = OccupancyCalculator::new(GfxArch::Gfx90a);
        let occ = calc.compute(KernelResources::new(100, 16, 0, 64));
        assert_eq!(occ.waves_per_cu, 8); // 2/SIMD * 4 SIMD
        // 130 VGPRs rounds up to 132 → 256/132 = 1 wave/SIMD → 4 waves/CU.
        let occ = calc.compute(KernelResources::new(130, 16, 0, 64));
        assert_eq!(occ.waves_per_cu, 4);
    }

    #[test]
    fn lds_bound_kernel() {
        // 32 KiB LDS per 64-thread block on gfx90a (64 KiB/CU) → 2 blocks/CU.
        let calc = OccupancyCalculator::new(GfxArch::Gfx90a);
        let occ = calc.compute(KernelResources::new(16, 8, 32 * 1024, 64));
        assert_eq!(occ.limited_by, LimitingResource::Lds);
        assert_eq!(occ.blocks_per_cu, 2);
        assert_eq!(occ.waves_per_cu, 2); // 1 wave per block * 2 blocks
    }

    #[test]
    fn lds_rounds_to_256_granule() {
        // 200-byte LDS rounds up to 256; 64 KiB / 256 = 256 blocks possible, so
        // the wave cap (not LDS) limits a 64-thread block.
        let calc = OccupancyCalculator::new(GfxArch::Gfx90a);
        let occ = calc.compute(KernelResources::new(16, 8, 200, 64));
        assert_eq!(occ.limited_by, LimitingResource::WaveSlots);
    }

    #[test]
    fn rdna_has_more_wave_slots() {
        // RDNA3: 16 waves/SIMD * 4 = 64 waves/CU, wave32 → 32-thread block = 1 wave.
        let calc = OccupancyCalculator::new(GfxArch::Gfx1100);
        let occ = calc.compute(KernelResources::new(16, 8, 0, 32));
        assert_eq!(occ.waves_per_cu, 64);
        assert_eq!(occ.blocks_per_cu, 64);
    }

    #[test]
    fn multi_wave_block_reduces_block_count() {
        // 256-thread wave64 block = 4 waves/block. Wave cap 32/CU → 8 blocks.
        let calc = OccupancyCalculator::new(GfxArch::Gfx90a);
        let occ = calc.compute(KernelResources::new(16, 8, 0, 256));
        assert_eq!(occ.waves_per_cu, 32);
        assert_eq!(occ.blocks_per_cu, 8);
    }

    #[test]
    fn max_potential_block_size_returns_full_occupancy() {
        let calc = OccupancyCalculator::new(GfxArch::Gfx90a);
        let (size, occ) = calc.max_potential_block_size(16, 8, 0);
        assert!(size >= 64);
        assert_eq!(occ.waves_per_cu, 32);
        // The picked block size must be a multiple of the wavefront width.
        assert_eq!(size % 64, 0);
    }

    #[test]
    fn max_potential_block_size_respects_vgpr_pressure() {
        let calc = OccupancyCalculator::new(GfxArch::Gfx90a);
        // Heavy VGPR use: occupancy capped regardless of block size, but the
        // search still returns a valid wavefront-aligned size.
        let (size, occ) = calc.max_potential_block_size(200, 8, 0);
        assert_eq!(size % 64, 0);
        assert!(occ.waves_per_cu <= 8);
    }
}
