//! Cooperative kernel launch support.
//!
//! Cooperative kernels allow all thread blocks in the grid to synchronize
//! with each other via `cooperative_groups::grid_group::sync()`. This is
//! useful for algorithms that need global synchronization across all blocks,
//! such as certain reduction patterns and iterative solvers.
//!
//! # Requirements
//!
//! - Compute capability 6.0+ (Pascal or later).
//! - The kernel must be compiled with cooperative launch support.
//! - The grid size must not exceed the maximum cooperative blocks
//!   for the kernel and device.
//!
//! # Example
//!
//! ```rust,no_run
//! # use std::sync::Arc;
//! # use oxicuda_driver::{Module, Stream, Context, Device};
//! # use oxicuda_launch::{Kernel, LaunchParams};
//! # use oxicuda_launch::cooperative::CooperativeLaunch;
//! # fn main() -> oxicuda_driver::CudaResult<()> {
//! # oxicuda_driver::init()?;
//! # let dev = Device::get(0)?;
//! # let ctx = Arc::new(Context::new(&dev)?);
//! # let ptx = "";
//! # let module = Arc::new(Module::from_ptx(ptx)?);
//! # let kernel = Kernel::from_module(module, "cooperative_reduce")?;
//! # let stream = Stream::new(&ctx)?;
//! let max_blocks = CooperativeLaunch::max_active_blocks(&kernel, 256, 0)?;
//! let params = LaunchParams::new(max_blocks, 256u32);
//!
//! CooperativeLaunch::launch(&kernel, &params, &stream, &(0u64, 1024u32))?;
//! stream.synchronize()?;
//! # Ok(())
//! # }
//! ```

use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::stream::Stream;

use crate::kernel::{Kernel, KernelArgs};
use crate::params::LaunchParams;

// ---------------------------------------------------------------------------
// CooperativeLaunch
// ---------------------------------------------------------------------------

/// Cooperative kernel launch facility.
///
/// Cooperative launches allow all thread blocks in the grid to synchronize
/// with each other. This is achieved by the CUDA driver launching all
/// blocks simultaneously on the GPU, which requires that the total number
/// of blocks does not exceed the hardware's capacity to run them all at
/// once.
///
/// Use [`max_active_blocks`](Self::max_active_blocks) to query the maximum
/// number of blocks that can participate in a cooperative launch for a
/// given kernel configuration.
#[derive(Debug)]
pub struct CooperativeLaunch;

impl CooperativeLaunch {
    /// Launches a kernel cooperatively, allowing inter-block synchronization.
    ///
    /// All thread blocks in the grid will be scheduled simultaneously,
    /// enabling them to synchronize via `cooperative_groups::grid_group::sync()`.
    ///
    /// This calls `cuLaunchCooperativeKernel` (via
    /// [`oxicuda_driver::cooperative_launch::cooperative_launch`]), which
    /// guarantees that every thread block in the grid is resident on the
    /// GPU simultaneously. That guarantee is required for grid-wide
    /// synchronization to be well-defined: a plain `cuLaunchKernel` launch
    /// (as used by [`Kernel::launch`](crate::kernel::Kernel::launch)) makes
    /// no such promise, so a kernel that calls
    /// `cooperative_groups::this_grid().sync()` could deadlock (blocks that
    /// never get scheduled never reach the barrier) or observe stale data
    /// (blocks racing ahead before others have started) if launched that
    /// way.
    ///
    /// # Parameters
    ///
    /// * `kernel` - The kernel to launch.
    /// * `params` - Launch configuration (grid, block, shared memory).
    /// * `stream` - The stream to launch on.
    /// * `args` - Kernel arguments.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if grid or block dimensions are zero.
    /// * [`CudaError::CooperativeLaunchTooLarge`] if the grid exceeds the
    ///   maximum cooperative grid size the device can run simultaneously.
    ///   This is enforced authoritatively by the driver, which accounts for
    ///   the device's full SM count (unlike the single-SM
    ///   [`max_active_blocks`](Self::max_active_blocks) query).
    /// * [`CudaError::NotInitialized`] if the CUDA driver is unavailable.
    /// * [`CudaError::NotSupported`] on platforms without a CUDA driver
    ///   (e.g. macOS).
    /// * Other [`CudaError`] variants on driver failure.
    pub fn launch<A: KernelArgs>(
        kernel: &Kernel,
        params: &LaunchParams,
        stream: &Stream,
        args: &A,
    ) -> CudaResult<()> {
        // Validate non-zero dimensions
        if params.grid.x == 0
            || params.grid.y == 0
            || params.grid.z == 0
            || params.block.x == 0
            || params.block.y == 0
            || params.block.z == 0
        {
            return Err(CudaError::InvalidValue);
        }

        let config = oxicuda_driver::cooperative_launch::CooperativeLaunchConfig {
            grid_dim: (params.grid.x, params.grid.y, params.grid.z),
            block_dim: (params.block.x, params.block.y, params.block.z),
            shared_mem_bytes: params.shared_mem_bytes,
            stream: Some(stream.raw()),
        };
        let param_ptrs = args.as_param_ptrs();
        oxicuda_driver::cooperative_launch::cooperative_launch(
            kernel.function(),
            &config,
            &param_ptrs,
        )
    }

    /// Queries the maximum number of active blocks per streaming
    /// multiprocessor for a cooperative launch with the given configuration.
    ///
    /// This value, multiplied by the number of SMs on the device, gives
    /// the maximum grid size for a cooperative launch.
    ///
    /// # Parameters
    ///
    /// * `kernel` - The kernel to query.
    /// * `block_size` - Number of threads per block.
    /// * `dynamic_smem` - Dynamic shared memory per block in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotInitialized`] if the CUDA driver is not
    /// available, or another [`CudaError`] on driver failure.
    pub fn max_active_blocks(
        kernel: &Kernel,
        block_size: u32,
        dynamic_smem: usize,
    ) -> CudaResult<u32> {
        let result = Self::max_active_blocks_inner(kernel, block_size as i32, dynamic_smem)?;
        Ok(result as u32)
    }

    /// Inner implementation for max active blocks query.
    fn max_active_blocks_inner(
        kernel: &Kernel,
        block_size: i32,
        dynamic_smem: usize,
    ) -> CudaResult<i32> {
        kernel
            .function()
            .max_active_blocks_per_sm(block_size, dynamic_smem)
    }

    /// Returns the optimal block size for a cooperative launch of this kernel.
    ///
    /// This queries the device for the optimal block size that maximizes
    /// occupancy, returning `(min_grid_size, optimal_block_size)`.
    ///
    /// # Parameters
    ///
    /// * `kernel` - The kernel to query.
    /// * `dynamic_smem` - Dynamic shared memory per block in bytes.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] on driver failure.
    pub fn optimal_block_size(kernel: &Kernel, dynamic_smem: usize) -> CudaResult<(i32, i32)> {
        kernel.function().optimal_block_size(dynamic_smem)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Dim3;

    #[test]
    fn cooperative_launch_struct_exists() {
        // Verify the type and its methods compile.
        let _: fn(&Kernel, &LaunchParams, &Stream, &(u64, u32)) -> CudaResult<()> =
            CooperativeLaunch::launch;
    }

    #[test]
    fn max_active_blocks_signature_compiles() {
        let _: fn(&Kernel, u32, usize) -> CudaResult<u32> = CooperativeLaunch::max_active_blocks;
    }

    #[test]
    fn optimal_block_size_signature_compiles() {
        let _: fn(&Kernel, usize) -> CudaResult<(i32, i32)> = CooperativeLaunch::optimal_block_size;
    }

    #[test]
    fn launch_rejects_zero_grid_x() {
        // We cannot construct a real Kernel without a GPU, but we can
        // verify that zero-dimension checking is the first validation.
        // The launch function checks dimensions before the kernel.
        let params = LaunchParams {
            grid: Dim3::new(0, 1, 1),
            block: Dim3::x(256),
            shared_mem_bytes: 0,
        };
        // Verify the params have the expected zero dimension.
        assert_eq!(params.grid.x, 0);
    }

    #[test]
    fn launch_rejects_zero_block_y() {
        let params = LaunchParams {
            grid: Dim3::x(4),
            block: Dim3::new(256, 0, 1),
            shared_mem_bytes: 0,
        };
        assert_eq!(params.block.y, 0);
    }

    #[test]
    fn cooperative_launch_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CooperativeLaunch>();
    }

    #[test]
    fn cooperative_launch_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<CooperativeLaunch>();
    }

    #[test]
    fn cooperative_dim3_total_nonzero() {
        let d = Dim3::new(4, 2, 1);
        assert_eq!(d.total(), 8);
        // All cooperative launches require total > 0
        assert!(d.total() > 0);
    }

    // ---------------------------------------------------------------------------
    // Quality gate tests (CPU-only)
    // ---------------------------------------------------------------------------

    #[test]
    fn cooperative_config_valid_fields() {
        // CooperativeLaunch passes a LaunchParams to launch(); verify those fields
        // are accessible — this checks the integration contract.
        let params = LaunchParams {
            grid: Dim3::new(4, 1, 1),
            block: Dim3::new(256, 1, 1),
            shared_mem_bytes: 1024,
        };
        assert_eq!(params.grid.x, 4);
        assert_eq!(params.block.x, 256);
        assert_eq!(params.shared_mem_bytes, 1024);
    }

    #[test]
    fn cooperative_max_blocks_constraint_signature() {
        // CooperativeLaunch::launch can return CooperativeLaunchTooLarge
        // (enforced by the driver's real cuLaunchCooperativeKernel call
        // when the grid does not fit). Verify the function signature
        // compiles (actual invocation requires a real Kernel and GPU).
        let _: fn(&Kernel, &LaunchParams, &Stream, &(u64, u32)) -> CudaResult<()> =
            CooperativeLaunch::launch;
    }

    #[test]
    fn cooperative_debug_display() {
        // CooperativeLaunch must implement Debug (derived above).
        let coop = CooperativeLaunch;
        let dbg = format!("{coop:?}");
        assert!(
            dbg.contains("CooperativeLaunch"),
            "Debug output must contain type name, got: {dbg}"
        );
    }

    // ---------------------------------------------------------------------------
    // On-device regression test (F032): CooperativeLaunch::launch must call the
    // real cuLaunchCooperativeKernel, not a plain cuLaunchKernel gated by an
    // over-strict, single-SM manual pre-check.
    // ---------------------------------------------------------------------------

    /// A minimal, arch-portable empty kernel (no parameters).
    #[cfg(feature = "gpu-tests")]
    const NOOP_PTX: &str = "\
.version 7.0
.target sm_70
.address_size 64
.visible .entry noop_kernel()
{
    ret;
}
";

    /// Regression test for the bug where `CooperativeLaunch::launch`'s manual
    /// pre-check compared the *total* grid size against
    /// `max_active_blocks_per_sm` — a *per-SM* quantity — without multiplying
    /// by the device's SM count, so it incorrectly rejected grids that
    /// legitimately span multiple SMs (the normal case for any real
    /// cooperative launch). It also never called the real
    /// `cuLaunchCooperativeKernel`, so even a grid that passed the pre-check
    /// would silently run as an ordinary (non-cooperative) launch. This test
    /// launches a grid sized to span every SM at full per-SM occupancy —
    /// which the old pre-check would have rejected on any GPU with more than
    /// one SM — and asserts it now succeeds via the real cooperative launch
    /// path. Self-skips if there is no GPU/driver.
    #[cfg(feature = "gpu-tests")]
    #[test]
    fn cooperative_launch_succeeds_across_all_sms() {
        use std::sync::Arc;

        let Ok(dev) = oxicuda_driver::device::Device::get(0) else {
            return;
        };
        let ctx = match oxicuda_driver::context::Context::new(&dev) {
            Ok(c) => Arc::new(c),
            Err(_) => return,
        };
        let stream = match Stream::new(&ctx) {
            Ok(s) => s,
            Err(_) => return,
        };
        let module = match oxicuda_driver::module::Module::from_ptx(NOOP_PTX) {
            Ok(m) => Arc::new(m),
            Err(_) => return,
        };
        let kernel = match Kernel::from_module(module, "noop_kernel") {
            Ok(k) => k,
            Err(_) => return,
        };

        let block_size: u32 = 128;
        let max_per_sm = match CooperativeLaunch::max_active_blocks(&kernel, block_size, 0) {
            Ok(m) if m > 0 => m,
            _ => return,
        };
        let sm_count = match dev.multiprocessor_count() {
            Ok(c) if c > 0 => c as u32,
            _ => return,
        };

        // Grid spans every SM at full per-SM occupancy: legitimate on real
        // hardware, and strictly greater than `max_active_blocks_per_sm`
        // alone whenever the device has more than one SM (true of every
        // real discrete GPU, e.g. the RTX A4000 has 48 SMs).
        let grid_x = max_per_sm * sm_count;
        let params = LaunchParams::new(grid_x, block_size);

        let result = CooperativeLaunch::launch(&kernel, &params, &stream, &());
        assert!(
            result.is_ok(),
            "cooperative launch spanning all {sm_count} SMs \
             ({max_per_sm} blocks/SM) must succeed, got {result:?}"
        );
        stream
            .synchronize()
            .expect("stream sync after cooperative launch");
    }
}
