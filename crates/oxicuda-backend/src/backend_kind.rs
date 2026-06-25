//! Enumeration of the compute backends OxiCUDA can dispatch to, with a
//! default preference ordering used by the selection logic.
//!
//! This is pure host-side metadata — it names the backends and ranks them;
//! it does **not** prove any of them is present. Availability is decided by
//! the [`crate::registry::BackendRegistry`], which combines this ordering
//! with runtime probes supplied by each concrete backend crate.

use std::fmt;

/// One of the concrete compute backends OxiCUDA targets.
///
/// Variants are ordered from most- to least-preferred for a typical
/// discrete-GPU workstation; see [`BackendKind::default_priority`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// NVIDIA CUDA (`oxicuda`).
    Cuda,
    /// AMD ROCm / HIP (`oxicuda-rocm`).
    Rocm,
    /// Intel oneAPI Level Zero (`oxicuda-levelzero`).
    LevelZero,
    /// Khronos Vulkan compute (`oxicuda-vulkan`).
    Vulkan,
    /// Apple Metal (`oxicuda-metal`).
    Metal,
    /// W3C WebGPU (`oxicuda-webgpu`).
    WebGpu,
    /// Pure-Rust host fallback (the CPU reference backend).
    Cpu,
}

impl BackendKind {
    /// Every backend kind, in declaration order.
    pub const ALL: [BackendKind; 7] = [
        BackendKind::Cuda,
        BackendKind::Rocm,
        BackendKind::LevelZero,
        BackendKind::Vulkan,
        BackendKind::Metal,
        BackendKind::WebGpu,
        BackendKind::Cpu,
    ];

    /// Short lowercase identifier matching the backend's `name()`
    /// (`"cuda"`, `"rocm"`, `"level_zero"`, `"vulkan"`, `"metal"`,
    /// `"webgpu"`, `"cpu"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::LevelZero => "level_zero",
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
            Self::WebGpu => "webgpu",
            Self::Cpu => "cpu",
        }
    }

    /// Default selection priority — **higher wins**.
    ///
    /// Native vendor stacks (CUDA, ROCm, Level Zero, Metal) outrank the
    /// portable APIs (Vulkan, WebGPU), which in turn outrank the CPU
    /// fallback. The CPU backend is deliberately the global minimum so it
    /// is only ever chosen when nothing else is available.
    #[must_use]
    pub const fn default_priority(self) -> u32 {
        match self {
            Self::Cuda => 100,
            Self::Rocm => 90,
            Self::Metal => 85,
            Self::LevelZero => 80,
            Self::Vulkan => 60,
            Self::WebGpu => 50,
            Self::Cpu => 1,
        }
    }

    /// `true` for every backend that targets a real GPU (i.e. everything
    /// except [`BackendKind::Cpu`]).
    #[must_use]
    pub const fn is_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Parse a backend kind from its short identifier (case-insensitive),
    /// accepting a few common aliases (`"hip"`, `"l0"`, `"wgpu"`, `"host"`).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "cuda" | "nvidia" => Some(Self::Cuda),
            "rocm" | "hip" | "amd" => Some(Self::Rocm),
            "level_zero" | "levelzero" | "l0" | "intel" => Some(Self::LevelZero),
            "vulkan" | "vk" => Some(Self::Vulkan),
            "metal" | "mtl" | "apple" => Some(Self::Metal),
            "webgpu" | "wgpu" => Some(Self::WebGpu),
            "cpu" | "host" | "reference" => Some(Self::Cpu),
            _ => None,
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn all_covers_every_variant_uniquely() {
        let set: HashSet<_> = BackendKind::ALL.iter().copied().collect();
        assert_eq!(set.len(), 7);
        // ALL must include the CPU fallback and the four native stacks.
        assert!(set.contains(&BackendKind::Cpu));
        assert!(set.contains(&BackendKind::Cuda));
        assert!(set.contains(&BackendKind::Metal));
    }

    #[test]
    fn cpu_is_global_minimum_priority() {
        let cpu = BackendKind::Cpu.default_priority();
        for k in BackendKind::ALL {
            if k != BackendKind::Cpu {
                assert!(
                    k.default_priority() > cpu,
                    "{k} must outrank the CPU fallback"
                );
            }
        }
    }

    #[test]
    fn cuda_is_global_maximum_priority() {
        let cuda = BackendKind::Cuda.default_priority();
        for k in BackendKind::ALL {
            if k != BackendKind::Cuda {
                assert!(cuda >= k.default_priority());
            }
        }
    }

    #[test]
    fn priority_ordering_is_strict_per_pair() {
        // Native stacks outrank portable APIs.
        assert!(BackendKind::Rocm.default_priority() > BackendKind::Vulkan.default_priority());
        assert!(BackendKind::Metal.default_priority() > BackendKind::WebGpu.default_priority());
        assert!(BackendKind::LevelZero.default_priority() > BackendKind::Vulkan.default_priority());
    }

    #[test]
    fn is_gpu_is_true_for_all_but_cpu() {
        for k in BackendKind::ALL {
            assert_eq!(k.is_gpu(), k != BackendKind::Cpu);
        }
    }

    #[test]
    fn name_round_trips() {
        for k in BackendKind::ALL {
            assert_eq!(BackendKind::from_name(k.as_str()), Some(k));
        }
    }

    #[test]
    fn name_aliases_and_case() {
        assert_eq!(BackendKind::from_name("HIP"), Some(BackendKind::Rocm));
        assert_eq!(BackendKind::from_name(" L0 "), Some(BackendKind::LevelZero));
        assert_eq!(BackendKind::from_name("WGPU"), Some(BackendKind::WebGpu));
        assert_eq!(BackendKind::from_name("Host"), Some(BackendKind::Cpu));
        assert_eq!(BackendKind::from_name("nvidia"), Some(BackendKind::Cuda));
        assert_eq!(BackendKind::from_name("opencl"), None);
    }

    #[test]
    fn display_matches_as_str() {
        assert_eq!(BackendKind::LevelZero.to_string(), "level_zero");
        assert_eq!(BackendKind::Cpu.to_string(), "cpu");
    }
}
