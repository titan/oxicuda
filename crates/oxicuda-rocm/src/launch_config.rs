//! Host-side HIP kernel launch-configuration descriptors and validation.
//!
//! Models the arguments to `hipModuleLaunchKernel` / `hipLaunchKernelGGL`:
//! a 3-D grid of work-groups, a 3-D work-group (block) of work-items, a
//! dynamic LDS (shared-memory) request, and an optional stream handle.
//!
//! All validation here is **pure host-side**: it checks the configuration
//! against the [`crate::gfx_arch`] limits of a target architecture without
//! ever launching a kernel, so it is fully testable on CPU-only systems.

use crate::error::{RocmError, RocmResult};
use crate::gfx_arch::GfxArch;

// ─── Dim3 ───────────────────────────────────────────────────────────────────

/// A 3-D extent, matching HIP's `dim3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim3 {
    /// X extent.
    pub x: u32,
    /// Y extent.
    pub y: u32,
    /// Z extent.
    pub z: u32,
}

impl Dim3 {
    /// A 1-D extent `(x, 1, 1)`.
    pub fn new_1d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// A 2-D extent `(x, y, 1)`.
    pub fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// A full 3-D extent.
    pub fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total number of elements (`x * y * z`), saturating on overflow.
    pub fn volume(self) -> u64 {
        (self.x as u64)
            .saturating_mul(self.y as u64)
            .saturating_mul(self.z as u64)
    }

    /// `true` if any axis is zero (an empty launch).
    pub fn is_empty(self) -> bool {
        self.x == 0 || self.y == 0 || self.z == 0
    }
}

impl Default for Dim3 {
    fn default() -> Self {
        Self::new_1d(1)
    }
}

// ─── LaunchConfig ────────────────────────────────────────────────────────────

/// A complete HIP kernel launch configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Number of work-groups in each dimension.
    pub grid: Dim3,
    /// Number of work-items per work-group in each dimension.
    pub block: Dim3,
    /// Dynamically-allocated LDS (`extern __shared__`) bytes per work-group.
    pub dynamic_lds_bytes: u32,
    /// Stream handle the launch is enqueued on (`0` = default/null stream).
    pub stream: u64,
}

impl LaunchConfig {
    /// Construct a launch configuration with no dynamic LDS on the default
    /// stream.
    pub fn new(grid: Dim3, block: Dim3) -> Self {
        Self {
            grid,
            block,
            dynamic_lds_bytes: 0,
            stream: 0,
        }
    }

    /// Set the dynamic LDS byte budget.
    pub fn with_dynamic_lds(mut self, bytes: u32) -> Self {
        self.dynamic_lds_bytes = bytes;
        self
    }

    /// Set the stream handle.
    pub fn with_stream(mut self, stream: u64) -> Self {
        self.stream = stream;
        self
    }

    /// Compute a 1-D launch covering `n` elements with a `block_x`-wide block.
    ///
    /// Returns an error if `block_x` is zero.
    pub fn for_elements(n: usize, block_x: u32) -> RocmResult<Self> {
        if block_x == 0 {
            return Err(RocmError::InvalidArgument(
                "block_x must be non-zero".into(),
            ));
        }
        let grid_x = (n as u64).div_ceil(block_x as u64);
        let grid_x = u32::try_from(grid_x).map_err(|_| {
            RocmError::InvalidArgument(format!("grid x {grid_x} exceeds u32 range"))
        })?;
        Ok(LaunchConfig::new(
            Dim3::new_1d(grid_x.max(1)),
            Dim3::new_1d(block_x),
        ))
    }

    /// Total work-items per work-group (`block.x * block.y * block.z`).
    pub fn threads_per_block(&self) -> u64 {
        self.block.volume()
    }

    /// Total work-items across the whole grid.
    pub fn total_threads(&self) -> u64 {
        self.grid.volume().saturating_mul(self.block.volume())
    }

    /// Number of wavefronts per work-group for `arch`'s native wave width.
    pub fn waves_per_block(&self, arch: GfxArch) -> u32 {
        let tpb = self.threads_per_block().min(u32::MAX as u64) as u32;
        tpb.div_ceil(arch.native_wavefront())
    }

    /// Validate the configuration against the hardware limits of `arch`.
    ///
    /// # Errors
    ///
    /// Returns [`RocmError::InvalidArgument`] when:
    /// - any block axis is zero (degenerate work-group);
    /// - the block exceeds `max_threads_per_block` (1024);
    /// - per-block dynamic + the requested LDS exceeds the CU LDS budget;
    /// - any grid axis exceeds the HIP `INT_MAX`-derived maximum.
    pub fn validate(&self, arch: GfxArch) -> RocmResult<()> {
        if self.block.is_empty() {
            return Err(RocmError::InvalidArgument(
                "block dimensions must all be non-zero".into(),
            ));
        }
        let tpb = self.block.volume();
        if tpb > u64::from(arch.max_threads_per_block()) {
            return Err(RocmError::InvalidArgument(format!(
                "block has {tpb} threads, exceeds {} max for {}",
                arch.max_threads_per_block(),
                arch.target_id()
            )));
        }
        if self.dynamic_lds_bytes > arch.lds_bytes_per_cu() {
            return Err(RocmError::InvalidArgument(format!(
                "dynamic LDS {} B exceeds {} B per CU on {}",
                self.dynamic_lds_bytes,
                arch.lds_bytes_per_cu(),
                arch.target_id()
            )));
        }
        // HIP grid extents are bounded by INT_MAX per dimension.
        const MAX_GRID_DIM: u32 = i32::MAX as u32;
        for (axis, v) in [("x", self.grid.x), ("y", self.grid.y), ("z", self.grid.z)] {
            if v > MAX_GRID_DIM {
                return Err(RocmError::InvalidArgument(format!(
                    "grid {axis} = {v} exceeds INT_MAX limit"
                )));
            }
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim3_constructors() {
        assert_eq!(Dim3::new_1d(7), Dim3 { x: 7, y: 1, z: 1 });
        assert_eq!(Dim3::new_2d(3, 4), Dim3 { x: 3, y: 4, z: 1 });
        assert_eq!(Dim3::new_3d(2, 3, 4).volume(), 24);
        assert!(Dim3::new_3d(2, 0, 4).is_empty());
        assert_eq!(Dim3::default(), Dim3::new_1d(1));
    }

    #[test]
    fn for_elements_rounds_up_grid() {
        let cfg = LaunchConfig::for_elements(1000, 256).expect("valid");
        assert_eq!(cfg.grid.x, 4); // ceil(1000/256) = 4
        assert_eq!(cfg.block.x, 256);
        assert_eq!(cfg.total_threads(), 4 * 256);
    }

    #[test]
    fn for_elements_zero_block_errors() {
        let err = LaunchConfig::for_elements(100, 0).unwrap_err();
        assert!(matches!(err, RocmError::InvalidArgument(_)));
    }

    #[test]
    fn for_elements_zero_n_still_launches_one_block() {
        let cfg = LaunchConfig::for_elements(0, 256).expect("valid");
        assert_eq!(cfg.grid.x, 1);
    }

    #[test]
    fn builder_sets_lds_and_stream() {
        let cfg = LaunchConfig::new(Dim3::new_1d(8), Dim3::new_1d(64))
            .with_dynamic_lds(4096)
            .with_stream(42);
        assert_eq!(cfg.dynamic_lds_bytes, 4096);
        assert_eq!(cfg.stream, 42);
    }

    #[test]
    fn waves_per_block_uses_native_width() {
        // 128 threads on CDNA (wave64) = 2 waves.
        let cfg = LaunchConfig::new(Dim3::new_1d(1), Dim3::new_1d(128));
        assert_eq!(cfg.waves_per_block(GfxArch::Gfx90a), 2);
        // 128 threads on RDNA (wave32) = 4 waves.
        assert_eq!(cfg.waves_per_block(GfxArch::Gfx1100), 4);
    }

    #[test]
    fn validate_accepts_normal_config() {
        let cfg = LaunchConfig::new(Dim3::new_2d(64, 64), Dim3::new_2d(16, 16))
            .with_dynamic_lds(8 * 1024);
        assert!(cfg.validate(GfxArch::Gfx90a).is_ok());
    }

    #[test]
    fn validate_rejects_zero_block() {
        let cfg = LaunchConfig::new(Dim3::new_1d(8), Dim3::new_3d(0, 1, 1));
        assert!(cfg.validate(GfxArch::Gfx90a).is_err());
    }

    #[test]
    fn validate_rejects_oversize_block() {
        // 32*32 = 1024 is OK; 33*32 = 1056 exceeds 1024.
        let ok = LaunchConfig::new(Dim3::new_1d(1), Dim3::new_2d(32, 32));
        assert!(ok.validate(GfxArch::Gfx90a).is_ok());
        let bad = LaunchConfig::new(Dim3::new_1d(1), Dim3::new_2d(33, 32));
        assert!(bad.validate(GfxArch::Gfx90a).is_err());
    }

    #[test]
    fn validate_rejects_oversize_lds() {
        let cfg = LaunchConfig::new(Dim3::new_1d(1), Dim3::new_1d(64)).with_dynamic_lds(128 * 1024); // > 64 KiB
        let err = cfg.validate(GfxArch::Gfx90a).unwrap_err();
        assert!(matches!(err, RocmError::InvalidArgument(_)));
    }

    #[test]
    fn validate_rejects_oversize_grid() {
        let cfg = LaunchConfig::new(
            Dim3::new_1d(u32::MAX), // > INT_MAX
            Dim3::new_1d(64),
        );
        assert!(cfg.validate(GfxArch::Gfx90a).is_err());
    }

    #[test]
    fn total_threads_saturates() {
        let cfg = LaunchConfig::new(
            Dim3::new_3d(u32::MAX, u32::MAX, u32::MAX),
            Dim3::new_1d(1024),
        );
        // Must not panic; saturates to u64::MAX.
        assert_eq!(cfg.total_threads(), u64::MAX);
    }
}
