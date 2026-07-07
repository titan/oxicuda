//! GPU occupancy queries for performance optimisation.
//!
//! Occupancy measures how effectively GPU resources (warps, registers,
//! shared memory) are utilised. These queries help select launch
//! configurations that maximise hardware utilisation.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_driver::module::Module;
//! # fn main() -> Result<(), oxicuda_driver::error::CudaError> {
//! # let module: Module = unimplemented!();
//! let func = module.get_function("my_kernel")?;
//!
//! // Query the optimal block size for maximum occupancy.
//! let (min_grid_size, optimal_block_size) = func.optimal_block_size(0)?;
//! println!("optimal: grid >= {min_grid_size}, block = {optimal_block_size}");
//!
//! // Query active blocks per SM for a specific block size.
//! let active = func.max_active_blocks_per_sm(256, 0)?;
//! println!("active blocks per SM with 256 threads: {active}");
//! # Ok(())
//! # }
//! ```

use crate::error::CudaResult;
use crate::loader::try_driver;
use crate::module::Function;

impl Function {
    /// Returns the maximum number of active blocks per streaming
    /// multiprocessor for a given block size and dynamic shared memory.
    ///
    /// This is useful for evaluating different block sizes to find
    /// the configuration that achieves the highest occupancy.
    ///
    /// # Parameters
    ///
    /// * `block_size` — number of threads per block.
    /// * `dynamic_smem` — dynamic shared memory per block in bytes
    ///   (set to `0` if the kernel does not use dynamic shared memory).
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if the function
    /// handle is invalid or the driver call fails.
    pub fn max_active_blocks_per_sm(
        &self,
        block_size: i32,
        dynamic_smem: usize,
    ) -> CudaResult<i32> {
        let api = try_driver()?;
        let mut num_blocks: i32 = 0;
        crate::cuda_call!((api.cu_occupancy_max_active_blocks_per_multiprocessor)(
            &mut num_blocks,
            self.raw(),
            block_size,
            dynamic_smem,
        ))?;
        Ok(num_blocks)
    }

    /// Suggests an optimal launch configuration that maximises
    /// multiprocessor occupancy.
    ///
    /// Returns `(min_grid_size, optimal_block_size)` where:
    ///
    /// * `min_grid_size` — the minimum number of blocks needed to
    ///   achieve maximum occupancy across all SMs.
    /// * `optimal_block_size` — the block size (number of threads)
    ///   that achieves maximum occupancy.
    ///
    /// # Parameters
    ///
    /// * `dynamic_smem` — dynamic shared memory per block in bytes
    ///   (set to `0` if the kernel does not use dynamic shared memory).
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if the function
    /// handle is invalid or the driver call fails.
    pub fn optimal_block_size(&self, dynamic_smem: usize) -> CudaResult<(i32, i32)> {
        let api = try_driver()?;
        let mut min_grid_size: i32 = 0;
        let mut block_size: i32 = 0;
        crate::cuda_call!((api.cu_occupancy_max_potential_block_size)(
            &mut min_grid_size,
            &mut block_size,
            self.raw(),
            None, // no dynamic smem callback
            dynamic_smem,
            0, // no block size limit
        ))?;
        Ok((min_grid_size, block_size))
    }
}

#[cfg(test)]
mod tests {
    use crate::context::Context;
    use crate::device::Device;
    use crate::module::Module;

    /// A minimal, arch-portable empty kernel. `.target sm_70` JIT-recompiles
    /// forward to any newer device (Ampere/Ada/Hopper).
    const NOOP_PTX: &str = "\
.version 7.0
.target sm_70
.address_size 64
.visible .entry noop()
{
    ret;
}
";

    /// Real-hardware occupancy query: on a live device the driver must return
    /// a sensible resident-block count and an optimal block size within the
    /// architectural limits. No-op when no GPU is present.
    #[test]
    fn occupancy_query_on_real_device() {
        let Ok(dev) = Device::get(0) else {
            return;
        };
        // Keep the context alive (and current) for the occupancy queries.
        let _ctx = match Context::new(&dev) {
            Ok(c) => std::sync::Arc::new(c),
            Err(_) => return,
        };
        let module = match Module::from_ptx(NOOP_PTX) {
            Ok(m) => m,
            Err(_) => return,
        };
        let func = module.get_function("noop").expect("noop function");

        // Max resident blocks per SM for a 128-thread block must be >= 1 and
        // not exceed the hardware ceiling (no device allows > 64 blocks/SM).
        let blocks = func
            .max_active_blocks_per_sm(128, 0)
            .expect("occupancy query");
        assert!(
            (1..=64).contains(&blocks),
            "resident blocks/SM out of range: {blocks}"
        );

        // The suggested launch config must be a positive block size within the
        // 1024-thread/block limit and a positive minimum grid.
        let (min_grid, block) = func.optimal_block_size(0).expect("optimal block size");
        assert!(
            (1..=1024).contains(&block),
            "optimal block size out of range: {block}"
        );
        assert!(
            min_grid >= 1,
            "min grid size should be >= 1, got {min_grid}"
        );
    }
}
