//! AMDGPU flat work-group size attribute hints.
//!
//! Emits the `__attribute__((amdgpu_flat_work_group_size(min, max)))` clause
//! that tells the AMDGPU back-end the bounds of a kernel's launch block size,
//! letting it bound register allocation for higher occupancy on divergent
//! kernels.
//!
//! This is pure host-side **codegen**: it produces and validates the attribute
//! string against the [`crate::gfx_arch`] limits without a GPU.

use crate::error::{RocmError, RocmResult};
use crate::gfx_arch::GfxArch;

/// A validated `amdgpu_flat_work_group_size(min, max)` hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlatWorkgroupHint {
    /// Minimum flat work-group size the kernel will be launched with.
    pub min: u32,
    /// Maximum flat work-group size the kernel will be launched with.
    pub max: u32,
}

impl FlatWorkgroupHint {
    /// Construct and validate a hint for `arch`.
    ///
    /// # Errors
    ///
    /// [`RocmError::InvalidArgument`] when `min == 0`, `min > max`, `max`
    /// exceeds the architecture's `max_threads_per_block`, or the bounds are
    /// not multiples of the native wavefront width.
    pub fn new(arch: GfxArch, min: u32, max: u32) -> RocmResult<Self> {
        if min == 0 {
            return Err(RocmError::InvalidArgument(
                "flat work-group minimum must be non-zero".into(),
            ));
        }
        if min > max {
            return Err(RocmError::InvalidArgument(format!(
                "flat work-group min {min} exceeds max {max}"
            )));
        }
        if max > arch.max_threads_per_block() {
            return Err(RocmError::InvalidArgument(format!(
                "flat work-group max {max} exceeds {} for {}",
                arch.max_threads_per_block(),
                arch.target_id()
            )));
        }
        let wave = arch.native_wavefront();
        if min % wave != 0 || max % wave != 0 {
            return Err(RocmError::InvalidArgument(format!(
                "flat work-group bounds must be multiples of the {wave}-lane wavefront"
            )));
        }
        Ok(Self { min, max })
    }

    /// A fixed-size hint (`min == max`).
    pub fn fixed(arch: GfxArch, size: u32) -> RocmResult<Self> {
        Self::new(arch, size, size)
    }

    /// The `__attribute__((...))` clause to splice before `__global__`.
    pub fn attribute(&self) -> String {
        format!(
            "__attribute__((amdgpu_flat_work_group_size({}, {})))",
            self.min, self.max
        )
    }

    /// Wrap a kernel signature line with the attribute, returning a complete
    /// attributed `extern "C" __global__` declaration prefix.
    pub fn decorate(&self, kernel_signature: &str) -> String {
        format!(
            "extern \"C\"\n{}\n__global__ {kernel_signature}",
            self.attribute()
        )
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_hint_emits_attribute() {
        let h = FlatWorkgroupHint::new(GfxArch::Gfx90a, 64, 256).expect("valid");
        assert_eq!(
            h.attribute(),
            "__attribute__((amdgpu_flat_work_group_size(64, 256)))"
        );
    }

    #[test]
    fn fixed_hint() {
        let h = FlatWorkgroupHint::fixed(GfxArch::Gfx90a, 128).expect("valid");
        assert_eq!(h.min, 128);
        assert_eq!(h.max, 128);
    }

    #[test]
    fn rejects_zero_min() {
        assert!(FlatWorkgroupHint::new(GfxArch::Gfx90a, 0, 256).is_err());
    }

    #[test]
    fn rejects_min_gt_max() {
        assert!(FlatWorkgroupHint::new(GfxArch::Gfx90a, 256, 64).is_err());
    }

    #[test]
    fn rejects_oversize_max() {
        assert!(FlatWorkgroupHint::new(GfxArch::Gfx90a, 64, 2048).is_err());
    }

    #[test]
    fn rejects_non_wavefront_multiple() {
        // 100 is not a multiple of 64 on CDNA.
        assert!(FlatWorkgroupHint::new(GfxArch::Gfx90a, 100, 256).is_err());
        // But 96 (3 * 32) is valid on RDNA (wave32).
        assert!(FlatWorkgroupHint::new(GfxArch::Gfx1100, 96, 256).is_ok());
        // 96 is NOT a multiple of 64 on CDNA.
        assert!(FlatWorkgroupHint::new(GfxArch::Gfx90a, 96, 256).is_err());
    }

    #[test]
    fn decorate_wraps_signature() {
        let h = FlatWorkgroupHint::fixed(GfxArch::Gfx1100, 256).expect("valid");
        let decorated = h.decorate("void my_kernel(float* p)");
        assert!(decorated.contains("extern \"C\""));
        assert!(decorated.contains("amdgpu_flat_work_group_size(256, 256)"));
        assert!(decorated.contains("__global__ void my_kernel(float* p)"));
    }
}
