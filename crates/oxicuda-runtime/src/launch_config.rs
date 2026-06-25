//! CPU-side launch-configuration validation and occupancy modelling.
//!
//! Unlike [`crate::launch`], which forwards `cudaLaunchKernel` straight to the
//! driver and therefore requires a live GPU, this module is a **pure CPU model**
//! of the CUDA Runtime's launch-configuration and occupancy logic.  It lets
//! callers validate a `<<<grid, block, shared, stream>>>` configuration against
//! a device's published limits and compute occupancy
//! (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` /
//! `cudaOccupancyMaxPotentialBlockSize`) entirely on the host — exactly the
//! arithmetic the runtime performs before it ever touches the driver.
//!
//! Everything here is deterministic and GPU-free; it is exercised by the unit
//! tests below with hand-computed expected values.
//!
//! # What this models
//!
//! - **Launch-bound validation**: grid/block dimensions against
//!   `maxGridSize` / `maxThreadsDim`, total threads-per-block against
//!   `maxThreadsPerBlock`, and dynamic shared memory against the per-block
//!   shared-memory budget (`cudaErrorInvalidConfiguration` on violation).
//! - **Occupancy calculator**: blocks-per-SM limited simultaneously by the
//!   warp/thread budget, the register-file budget, the shared-memory budget,
//!   and the hard per-SM block cap — the minimum across all four, exactly as
//!   the CUDA occupancy calculator does.
//! - **`max_potential_block_size`**: sweep block sizes (multiples of the warp
//!   size) and return the one maximising resident warps per SM.
//! - **Cooperative-launch grid sizing**: the largest grid that still co-resides
//!   so every block runs concurrently (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`
//!   × SM count).

use crate::device::CudaDeviceProp;
use crate::error::{CudaRtError, CudaRtResult};
use crate::launch::Dim3;

// ─── KernelResourceUsage ─────────────────────────────────────────────────────

/// Per-thread / per-block resource usage of a kernel.
///
/// Mirrors the subset of `cudaFuncAttributes` that drives occupancy: registers
/// per thread, statically-allocated shared memory per block, and the kernel's
/// own `maxThreadsPerBlock` launch bound (0 = unbounded / use device default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct KernelResourceUsage {
    /// 32-bit registers consumed by each thread.
    pub registers_per_thread: u32,
    /// Statically-allocated shared memory per block in bytes.
    pub static_shared_bytes: u32,
    /// `__launch_bounds__` maximum threads per block (0 = device maximum).
    pub max_threads_per_block: u32,
}

impl KernelResourceUsage {
    /// Construct a usage record.
    #[must_use]
    pub const fn new(
        registers_per_thread: u32,
        static_shared_bytes: u32,
        max_threads_per_block: u32,
    ) -> Self {
        Self {
            registers_per_thread,
            static_shared_bytes,
            max_threads_per_block,
        }
    }
}

// ─── DeviceLaunchLimits ──────────────────────────────────────────────────────

/// The device-side limits an occupancy / launch computation needs.
///
/// Can be derived from a [`CudaDeviceProp`] via [`DeviceLaunchLimits::from_prop`]
/// or synthesised for a compute capability via
/// [`DeviceLaunchLimits::for_compute_capability`] so the model runs without a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceLaunchLimits {
    /// Number of streaming multiprocessors.
    pub sm_count: u32,
    /// Warp size (threads per warp, normally 32).
    pub warp_size: u32,
    /// Maximum resident threads per SM.
    pub max_threads_per_sm: u32,
    /// Maximum resident blocks per SM.
    pub max_blocks_per_sm: u32,
    /// 32-bit registers available per SM.
    pub registers_per_sm: u32,
    /// Shared memory available per SM in bytes.
    pub shared_mem_per_sm: u32,
    /// Maximum threads in a single block.
    pub max_threads_per_block: u32,
    /// Maximum dynamic + static shared memory usable per block in bytes.
    pub shared_mem_per_block: u32,
    /// Maximum block dimensions `[x, y, z]`.
    pub max_block_dim: [u32; 3],
    /// Maximum grid dimensions `[x, y, z]`.
    pub max_grid_dim: [u32; 3],
    /// Register-file allocation granularity (registers rounded up per block).
    pub register_alloc_unit: u32,
    /// Warp allocation granularity (warps per block rounded up to this).
    pub warp_alloc_unit: u32,
    /// Shared-memory allocation granularity in bytes.
    pub shared_alloc_unit: u32,
}

impl DeviceLaunchLimits {
    /// Build limits from a [`CudaDeviceProp`] queried from a real device.
    #[must_use]
    pub fn from_prop(prop: &CudaDeviceProp) -> Self {
        Self {
            sm_count: prop.multi_processor_count,
            warp_size: prop.warp_size,
            max_threads_per_sm: prop.max_threads_per_multi_processor,
            max_blocks_per_sm: prop.max_blocks_per_multi_processor,
            registers_per_sm: prop.regs_per_multiprocessor,
            shared_mem_per_sm: prop.shared_mem_per_multiprocessor as u32,
            max_threads_per_block: prop.max_threads_per_block,
            shared_mem_per_block: prop.shared_mem_per_block as u32,
            max_block_dim: prop.max_threads_dim,
            max_grid_dim: prop.max_grid_size,
            // Granularities are not exposed by cudaDeviceProp; use the values
            // documented for sm_70+ which apply to every shipping architecture.
            register_alloc_unit: 256,
            warp_alloc_unit: 4,
            shared_alloc_unit: 256,
        }
    }

    /// Synthesise limits for a compute capability without a live GPU.
    ///
    /// Covers Turing through Blackwell; unknown capabilities fall back to
    /// Ampere GA10x (sm_86) defaults.  Register / warp / shared granularities
    /// are the architecture-published values.
    #[must_use]
    pub fn for_compute_capability(major: u32, minor: u32) -> Self {
        // (sm_count, max_threads_per_sm, max_blocks_per_sm, regs_per_sm,
        //  shared_per_sm, shared_per_block_optin)
        let (sm_count, threads_sm, blocks_sm, shared_sm, shared_block) = match (major, minor) {
            (7, 5) => (68, 1024, 16, 65536, 65536),
            (8, 0) => (108, 2048, 32, 167936, 166912),
            (8, 6) => (84, 1536, 16, 102400, 101376),
            (8, 9) => (76, 1536, 24, 101376, 101376),
            (9, 0) => (132, 2048, 32, 232448, 227328),
            (10, 0) => (132, 2048, 32, 262144, 232448),
            (12, 0) => (148, 2048, 32, 262144, 232448),
            _ => (84, 1536, 16, 102400, 101376),
        };
        Self {
            sm_count,
            warp_size: 32,
            max_threads_per_sm: threads_sm,
            max_blocks_per_sm: blocks_sm,
            registers_per_sm: 65536,
            shared_mem_per_sm: shared_sm,
            max_threads_per_block: 1024,
            shared_mem_per_block: shared_block,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [2_147_483_647, 65535, 65535],
            register_alloc_unit: 256,
            warp_alloc_unit: 4,
            shared_alloc_unit: 128,
        }
    }

    /// Maximum warps that can be resident on one SM.
    #[must_use]
    pub fn max_warps_per_sm(&self) -> u32 {
        self.max_threads_per_sm
            .checked_div(self.warp_size)
            .unwrap_or(0)
    }
}

// ─── Launch configuration ────────────────────────────────────────────────────

/// A kernel launch configuration to validate (`<<<grid, block, shared, stream>>>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaunchConfig {
    /// Grid dimensions (number of blocks per axis).
    pub grid: Dim3,
    /// Block dimensions (number of threads per axis).
    pub block: Dim3,
    /// Dynamic shared memory per block in bytes.
    pub dynamic_shared_bytes: u32,
}

impl LaunchConfig {
    /// Construct a launch configuration.
    #[must_use]
    pub const fn new(grid: Dim3, block: Dim3, dynamic_shared_bytes: u32) -> Self {
        Self {
            grid,
            block,
            dynamic_shared_bytes,
        }
    }

    /// Threads per block (`block.x * block.y * block.z`).
    #[must_use]
    pub fn threads_per_block(&self) -> u64 {
        self.block.volume()
    }

    /// Validate this configuration against device limits and (optionally) the
    /// kernel's own resource usage.
    ///
    /// This mirrors the checks `cudaLaunchKernel` performs before submitting to
    /// the driver — a zero-sized or oversized block, an oversized grid, or a
    /// shared-memory request beyond the per-block budget all yield
    /// [`CudaRtError::InvalidConfiguration`].  A register / shared request that
    /// makes even a single block unschedulable yields
    /// [`CudaRtError::LaunchOutOfResources`].
    ///
    /// # Errors
    ///
    /// - [`CudaRtError::InvalidConfiguration`] for malformed grid/block/shared.
    /// - [`CudaRtError::LaunchOutOfResources`] if no block fits on an SM.
    pub fn validate(
        &self,
        limits: &DeviceLaunchLimits,
        kernel: &KernelResourceUsage,
    ) -> CudaRtResult<()> {
        // A zero dimension means zero blocks / threads — never valid.
        if self.block.x == 0 || self.block.y == 0 || self.block.z == 0 {
            return Err(CudaRtError::InvalidConfiguration);
        }
        if self.grid.x == 0 || self.grid.y == 0 || self.grid.z == 0 {
            return Err(CudaRtError::InvalidConfiguration);
        }

        // Per-axis block / grid bounds.
        if self.block.x > limits.max_block_dim[0]
            || self.block.y > limits.max_block_dim[1]
            || self.block.z > limits.max_block_dim[2]
        {
            return Err(CudaRtError::InvalidConfiguration);
        }
        if self.grid.x > limits.max_grid_dim[0]
            || self.grid.y > limits.max_grid_dim[1]
            || self.grid.z > limits.max_grid_dim[2]
        {
            return Err(CudaRtError::InvalidConfiguration);
        }

        // Total threads per block against the device and kernel launch bound.
        let threads = self.threads_per_block();
        if threads > u64::from(limits.max_threads_per_block) {
            return Err(CudaRtError::InvalidConfiguration);
        }
        if kernel.max_threads_per_block != 0 && threads > u64::from(kernel.max_threads_per_block) {
            return Err(CudaRtError::InvalidConfiguration);
        }

        // Shared-memory budget: static + dynamic must fit per block.
        let shared_total =
            u64::from(kernel.static_shared_bytes) + u64::from(self.dynamic_shared_bytes);
        if shared_total > u64::from(limits.shared_mem_per_block) {
            return Err(CudaRtError::InvalidConfiguration);
        }

        // Register budget: the whole block's registers must fit in one SM, else
        // not even one block is schedulable.
        let regs_block = u64::from(kernel.registers_per_thread) * threads;
        if kernel.registers_per_thread != 0 && regs_block > u64::from(limits.registers_per_sm) {
            return Err(CudaRtError::LaunchOutOfResources);
        }

        // Shared memory must also leave room for at least one block on an SM.
        if shared_total > u64::from(limits.shared_mem_per_sm) {
            return Err(CudaRtError::LaunchOutOfResources);
        }

        Ok(())
    }
}

// ─── Occupancy ───────────────────────────────────────────────────────────────

/// Result of an occupancy computation for a particular block size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Occupancy {
    /// Active (co-resident) blocks per SM at this block size.
    pub active_blocks_per_sm: u32,
    /// Active warps per SM (= `active_blocks_per_sm * warps_per_block`).
    pub active_warps_per_sm: u32,
    /// The limiting resource that capped `active_blocks_per_sm`.
    pub limiter: OccupancyLimiter,
}

impl Occupancy {
    /// Occupancy as a fraction of the SM's maximum resident warps (`0.0..=1.0`).
    #[must_use]
    pub fn ratio(&self, limits: &DeviceLaunchLimits) -> f64 {
        let max_warps = limits.max_warps_per_sm();
        if max_warps == 0 {
            0.0
        } else {
            f64::from(self.active_warps_per_sm) / f64::from(max_warps)
        }
    }
}

/// Which resource limited the number of co-resident blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OccupancyLimiter {
    /// Limited by the warp / thread budget of the SM.
    Warps,
    /// Limited by the register file.
    Registers,
    /// Limited by shared memory.
    SharedMemory,
    /// Limited by the hard per-SM block cap.
    Blocks,
    /// The block cannot be scheduled at all (zero blocks per SM).
    None,
}

/// CPU model of the CUDA occupancy calculator.
///
/// Computes blocks-per-SM and the optimal block size from device limits and a
/// kernel's resource usage — the same arithmetic
/// `cudaOccupancyMaxActiveBlocksPerMultiprocessor` performs, but on the host.
#[derive(Debug, Clone, Copy)]
pub struct OccupancyCalculator {
    limits: DeviceLaunchLimits,
}

impl OccupancyCalculator {
    /// Wrap a set of device limits.
    #[must_use]
    pub fn new(limits: DeviceLaunchLimits) -> Self {
        Self { limits }
    }

    /// The device limits this calculator was built with.
    #[must_use]
    pub fn limits(&self) -> &DeviceLaunchLimits {
        &self.limits
    }

    /// Round `value` up to the next multiple of `unit` (unit ≥ 1).
    fn round_up(value: u32, unit: u32) -> u32 {
        if unit <= 1 {
            return value;
        }
        value.div_ceil(unit) * unit
    }

    /// Active blocks per SM for a given block size and dynamic shared memory.
    ///
    /// Mirrors `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.  The result is
    /// the **minimum** of the warp limit, the register limit, the shared-memory
    /// limit and the hard per-SM block cap, accounting for allocation
    /// granularity exactly as the hardware allocator does.
    #[must_use]
    pub fn active_blocks_per_sm(
        &self,
        block_size: u32,
        dynamic_shared_bytes: u32,
        kernel: &KernelResourceUsage,
    ) -> Occupancy {
        let l = &self.limits;
        if block_size == 0
            || l.warp_size == 0
            || block_size > l.max_threads_per_block
            || (kernel.max_threads_per_block != 0 && block_size > kernel.max_threads_per_block)
        {
            return Occupancy {
                active_blocks_per_sm: 0,
                active_warps_per_sm: 0,
                limiter: OccupancyLimiter::None,
            };
        }

        let warps_per_block = block_size.div_ceil(l.warp_size);

        // 1. Warp / thread limit (with warp allocation granularity).
        let warps_alloc = Self::round_up(warps_per_block, l.warp_alloc_unit).max(1);
        let warp_limit = l.max_warps_per_sm() / warps_alloc;

        // 2. Register limit. Registers are allocated per warp, rounded up to the
        //    allocation unit, then a whole-block's worth must fit in the SM file.
        let reg_limit = if kernel.registers_per_thread == 0 {
            u32::MAX
        } else {
            let regs_per_warp = Self::round_up(
                kernel.registers_per_thread * l.warp_size,
                l.register_alloc_unit,
            );
            let regs_per_block = regs_per_warp.saturating_mul(warps_alloc).max(1);
            l.registers_per_sm / regs_per_block
        };

        // 3. Shared-memory limit (static + dynamic, rounded to alloc unit).
        let shared_per_block = Self::round_up(
            kernel.static_shared_bytes + dynamic_shared_bytes,
            l.shared_alloc_unit,
        );
        let shared_limit = l
            .shared_mem_per_sm
            .checked_div(shared_per_block)
            .unwrap_or(u32::MAX);

        // 4. Hard per-SM block cap.
        let block_cap = l.max_blocks_per_sm;

        // The occupancy is the minimum of all limits.
        let mut active = warp_limit.min(reg_limit).min(shared_limit).min(block_cap);

        // Determine which resource is the binding constraint.
        let limiter = if active == 0 {
            OccupancyLimiter::None
        } else if active == warp_limit {
            OccupancyLimiter::Warps
        } else if active == reg_limit {
            OccupancyLimiter::Registers
        } else if active == shared_limit {
            OccupancyLimiter::SharedMemory
        } else {
            OccupancyLimiter::Blocks
        };

        // Guard against an unschedulable config collapsing to a non-zero value
        // through MAX sentinels — if any genuine limit is zero, occupancy is 0.
        if warp_limit == 0 || reg_limit == 0 || shared_limit == 0 || block_cap == 0 {
            active = 0;
        }

        Occupancy {
            active_blocks_per_sm: active,
            active_warps_per_sm: active * warps_per_block,
            limiter: if active == 0 {
                OccupancyLimiter::None
            } else {
                limiter
            },
        }
    }

    /// Suggest the block size that maximises resident warps per SM.
    ///
    /// Mirrors `cudaOccupancyMaxPotentialBlockSize`.  Returns
    /// `(min_grid_size, block_size)` where `min_grid_size` is the block count
    /// needed to fill every SM at that block size.  `dynamic_shared_for` lets a
    /// caller model shared memory that scales with the block size (the
    /// runtime's shared-memory callback); pass `|_| 0` for a fixed kernel.
    #[must_use]
    pub fn max_potential_block_size<F>(
        &self,
        kernel: &KernelResourceUsage,
        mut dynamic_shared_for: F,
    ) -> (u32, u32)
    where
        F: FnMut(u32) -> u32,
    {
        let l = &self.limits;
        let cap = if kernel.max_threads_per_block == 0 {
            l.max_threads_per_block
        } else {
            kernel.max_threads_per_block.min(l.max_threads_per_block)
        };

        let mut best_block = 0u32;
        let mut best_warps = 0u32;
        let mut best_blocks = 0u32;

        // Sweep block sizes in warp-size steps (the CUDA calculator's grid).
        let mut block_size = l.warp_size;
        while block_size <= cap {
            let dyn_shared = dynamic_shared_for(block_size);
            let occ = self.active_blocks_per_sm(block_size, dyn_shared, kernel);
            if occ.active_warps_per_sm > best_warps {
                best_warps = occ.active_warps_per_sm;
                best_block = block_size;
                best_blocks = occ.active_blocks_per_sm;
            }
            block_size += l.warp_size;
        }

        let min_grid = best_blocks.saturating_mul(l.sm_count);
        (min_grid, best_block)
    }

    /// Largest grid (in blocks) such that every block is simultaneously
    /// resident — the requirement for a cooperative launch.
    ///
    /// Mirrors the sizing behind `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
    /// used by `cudaLaunchCooperativeKernel`: at most
    /// `active_blocks_per_sm * sm_count` blocks may co-reside.
    #[must_use]
    pub fn max_cooperative_grid_size(
        &self,
        block_size: u32,
        dynamic_shared_bytes: u32,
        kernel: &KernelResourceUsage,
    ) -> u32 {
        let occ = self.active_blocks_per_sm(block_size, dynamic_shared_bytes, kernel);
        occ.active_blocks_per_sm
            .saturating_mul(self.limits.sm_count)
    }

    /// Validate that a cooperative-launch grid fits (every block co-resident).
    ///
    /// # Errors
    ///
    /// [`CudaRtError::CooperativeLaunchTooLarge`] if the grid exceeds the
    /// concurrent-resident capacity.
    pub fn validate_cooperative_grid(
        &self,
        grid_blocks: u32,
        block_size: u32,
        dynamic_shared_bytes: u32,
        kernel: &KernelResourceUsage,
    ) -> CudaRtResult<()> {
        let max = self.max_cooperative_grid_size(block_size, dynamic_shared_bytes, kernel);
        if grid_blocks > max {
            return Err(CudaRtError::CooperativeLaunchTooLarge);
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ampere() -> DeviceLaunchLimits {
        DeviceLaunchLimits::for_compute_capability(8, 6)
    }

    #[test]
    fn valid_config_passes() {
        let limits = ampere();
        let kernel = KernelResourceUsage::new(32, 0, 0);
        let cfg = LaunchConfig::new(Dim3::one_d(1024), Dim3::one_d(256), 0);
        assert!(cfg.validate(&limits, &kernel).is_ok());
    }

    #[test]
    fn zero_block_dim_is_invalid_config() {
        let limits = ampere();
        let kernel = KernelResourceUsage::default();
        let cfg = LaunchConfig::new(Dim3::one_d(1), Dim3::three_d(0, 1, 1), 0);
        assert_eq!(
            cfg.validate(&limits, &kernel),
            Err(CudaRtError::InvalidConfiguration)
        );
    }

    #[test]
    fn oversized_block_is_invalid_config() {
        let limits = ampere();
        let kernel = KernelResourceUsage::default();
        // 2048 threads > 1024 max per block.
        let cfg = LaunchConfig::new(Dim3::one_d(1), Dim3::one_d(2048), 0);
        assert_eq!(
            cfg.validate(&limits, &kernel),
            Err(CudaRtError::InvalidConfiguration)
        );
    }

    #[test]
    fn launch_bound_violation_is_invalid_config() {
        let limits = ampere();
        // Kernel compiled with __launch_bounds__(128).
        let kernel = KernelResourceUsage::new(0, 0, 128);
        let cfg = LaunchConfig::new(Dim3::one_d(1), Dim3::one_d(256), 0);
        assert_eq!(
            cfg.validate(&limits, &kernel),
            Err(CudaRtError::InvalidConfiguration)
        );
    }

    #[test]
    fn excess_shared_is_invalid_config() {
        let limits = ampere();
        let kernel = KernelResourceUsage::default();
        // Request far more dynamic shared than the per-block budget.
        let cfg = LaunchConfig::new(
            Dim3::one_d(1),
            Dim3::one_d(64),
            limits.shared_mem_per_block + 1,
        );
        assert_eq!(
            cfg.validate(&limits, &kernel),
            Err(CudaRtError::InvalidConfiguration)
        );
    }

    #[test]
    fn register_starvation_is_out_of_resources() {
        let limits = ampere();
        // 1024 threads * 255 regs/thread = 261_120 regs > 65_536 per SM.
        let kernel = KernelResourceUsage::new(255, 0, 0);
        let cfg = LaunchConfig::new(Dim3::one_d(1), Dim3::one_d(1024), 0);
        assert_eq!(
            cfg.validate(&limits, &kernel),
            Err(CudaRtError::LaunchOutOfResources)
        );
    }

    #[test]
    fn occupancy_warp_limited_full() {
        // 256-thread block, no register/shared pressure → warp-limited.
        // sm_86: 1536 threads/SM, warp 32 → 48 warps/SM. 256 threads = 8 warps.
        // warp_alloc_unit 4 → 8 (already aligned). 48/8 = 6 blocks. But the hard
        // block cap on sm_86 is 16, so 6 wins.
        let calc = OccupancyCalculator::new(ampere());
        let kernel = KernelResourceUsage::new(32, 0, 0);
        let occ = calc.active_blocks_per_sm(256, 0, &kernel);
        assert_eq!(occ.active_blocks_per_sm, 6);
        assert_eq!(occ.active_warps_per_sm, 48);
        // 48 of 48 warps resident → 100% occupancy.
        assert!((occ.ratio(calc.limits()) - 1.0).abs() < 1e-9);
        assert_eq!(occ.limiter, OccupancyLimiter::Warps);
    }

    #[test]
    fn occupancy_warp_granularity_caps_tiny_blocks() {
        // Tiny 32-thread blocks on sm_86: 1 warp/block, but the warp allocation
        // granularity (4) means each block reserves 4 warps' worth → 48/4 = 12
        // blocks, which beats the hard 16-block cap → warp-limited at 12.
        let calc = OccupancyCalculator::new(ampere());
        let kernel = KernelResourceUsage::new(16, 0, 0);
        let occ = calc.active_blocks_per_sm(32, 0, &kernel);
        assert_eq!(occ.active_blocks_per_sm, 12);
        assert_eq!(occ.limiter, OccupancyLimiter::Warps);
    }

    #[test]
    fn occupancy_block_cap_limited() {
        // A synthetic device with warp granularity 1 so the hard per-SM block
        // cap genuinely binds: 64 warps/SM, 1-warp blocks → warp limit 64, but
        // the device caps blocks at 8 per SM → block-cap limited at 8.
        let limits = DeviceLaunchLimits {
            sm_count: 10,
            warp_size: 32,
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 8,
            registers_per_sm: 65536,
            shared_mem_per_sm: 65536,
            max_threads_per_block: 1024,
            shared_mem_per_block: 49152,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [i32::MAX as u32, 65535, 65535],
            register_alloc_unit: 256,
            warp_alloc_unit: 1,
            shared_alloc_unit: 128,
        };
        let calc = OccupancyCalculator::new(limits);
        let kernel = KernelResourceUsage::new(16, 0, 0);
        let occ = calc.active_blocks_per_sm(32, 0, &kernel);
        assert_eq!(occ.active_blocks_per_sm, 8);
        assert_eq!(occ.limiter, OccupancyLimiter::Blocks);
    }

    #[test]
    fn occupancy_shared_limited() {
        // 49_152 bytes static shared, sm_86 has 102_400/SM → only 2 blocks fit
        // by shared memory (102_400 / 49_152 = 2), well under warp/block limits.
        let calc = OccupancyCalculator::new(ampere());
        let kernel = KernelResourceUsage::new(0, 49_152, 0);
        let occ = calc.active_blocks_per_sm(256, 0, &kernel);
        assert_eq!(occ.active_blocks_per_sm, 2);
        assert_eq!(occ.limiter, OccupancyLimiter::SharedMemory);
    }

    #[test]
    fn occupancy_register_limited() {
        // 128 regs/thread, 256-thread block.
        // warps_per_block = 8, warp_alloc 4 → 8.
        // regs/warp = round_up(128*32=4096, 256) = 4096.
        // regs/block = 4096 * 8 = 32_768. 65_536 / 32_768 = 2 blocks.
        let calc = OccupancyCalculator::new(ampere());
        let kernel = KernelResourceUsage::new(128, 0, 0);
        let occ = calc.active_blocks_per_sm(256, 0, &kernel);
        assert_eq!(occ.active_blocks_per_sm, 2);
        assert_eq!(occ.limiter, OccupancyLimiter::Registers);
    }

    #[test]
    fn occupancy_zero_for_unschedulable_block() {
        let calc = OccupancyCalculator::new(ampere());
        let kernel = KernelResourceUsage::default();
        // block size beyond device max → unschedulable.
        let occ = calc.active_blocks_per_sm(4096, 0, &kernel);
        assert_eq!(occ.active_blocks_per_sm, 0);
        assert_eq!(occ.limiter, OccupancyLimiter::None);
        assert_eq!(occ.ratio(calc.limits()), 0.0);
    }

    #[test]
    fn max_potential_block_size_prefers_full_occupancy() {
        let calc = OccupancyCalculator::new(ampere());
        let kernel = KernelResourceUsage::new(32, 0, 0);
        let (min_grid, block) = calc.max_potential_block_size(&kernel, |_| 0);
        // Any block size that achieves the 48-warp maximum is acceptable; the
        // sweep returns the first such size. 256 threads → 48 warps resident.
        let occ = calc.active_blocks_per_sm(block, 0, &kernel);
        assert_eq!(occ.active_warps_per_sm, calc.limits().max_warps_per_sm());
        assert_eq!(min_grid, occ.active_blocks_per_sm * calc.limits().sm_count);
        assert!(block > 0 && block % calc.limits().warp_size == 0);
    }

    #[test]
    fn cooperative_grid_sizing_and_validation() {
        let calc = OccupancyCalculator::new(ampere());
        let kernel = KernelResourceUsage::new(32, 0, 0);
        // 256-thread blocks → 6 blocks/SM × 84 SMs = 504 max co-resident.
        let max = calc.max_cooperative_grid_size(256, 0, &kernel);
        assert_eq!(max, 6 * 84);
        assert!(calc.validate_cooperative_grid(max, 256, 0, &kernel).is_ok());
        assert_eq!(
            calc.validate_cooperative_grid(max + 1, 256, 0, &kernel),
            Err(CudaRtError::CooperativeLaunchTooLarge)
        );
    }

    #[test]
    fn limits_from_compute_capabilities_are_distinct() {
        let hopper = DeviceLaunchLimits::for_compute_capability(9, 0);
        let turing = DeviceLaunchLimits::for_compute_capability(7, 5);
        assert_eq!(hopper.max_threads_per_sm, 2048);
        assert_eq!(turing.max_threads_per_sm, 1024);
        assert_eq!(hopper.max_warps_per_sm(), 64);
        assert_eq!(turing.max_warps_per_sm(), 32);
    }
}
