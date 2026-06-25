//! Backend capability discovery and device enumeration.
//!
//! These types let consumers query, at runtime, what a backend can do
//! (mixed-precision support, Tensor Cores, unified memory, …) and what
//! devices it exposes — without coupling to any concrete GPU API.
//!
//! Everything here is **host-side data**: a backend fills the structs in
//! from its own driver queries, and consumers (autotuners, schedulers)
//! read them to pick algorithms and tile shapes. No device execution is
//! involved, so the logic is fully testable on a machine with no GPU.

use std::fmt;

/// Per-backend capability report.
///
/// A concrete backend populates this from its driver during
/// [`ComputeBackend::capabilities`](crate::ComputeBackend::capabilities).
/// Consumers use it to decide whether a mixed-precision or Tensor-Core
/// path is available before issuing work.
///
/// The [`Default`] value is the **conservative CPU profile**: no
/// accelerated precisions, no Tensor Cores, no peer access, but generous
/// thread/shared-memory limits that any host can honour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    /// Native FP16 (half precision) compute is available.
    pub supports_fp16: bool,
    /// Native BF16 (bfloat16) compute is available.
    pub supports_bf16: bool,
    /// FP8 (E4M3/E5M2) compute is available.
    pub supports_fp8: bool,
    /// Dedicated matrix/Tensor-Core (WMMA / `mma.sync`) units are present.
    pub tensor_cores: bool,
    /// Peer-to-peer access between devices of this backend is possible.
    pub peer_access: bool,
    /// Unified / managed memory (host and device share an address space).
    pub unified_memory: bool,
    /// Thread-block clusters (Hopper `sm_90`+) are supported.
    pub cluster_launch: bool,
    /// Asynchronous global→shared copy (`cp.async`, Ampere `sm_80`+).
    pub async_copy: bool,
    /// Maximum threads per block / workgroup.
    pub max_threads_per_block: u32,
    /// Maximum shared / local memory per block in bytes.
    pub max_shared_mem_per_block: u32,
    /// Warp / wavefront / subgroup width in lanes.
    pub warp_size: u32,
}

impl Capabilities {
    /// Conservative capability profile for a pure-CPU reference backend.
    ///
    /// No accelerated precisions or matrix units, but limits large enough
    /// that the consumer never rejects a launch for being "too big".
    #[must_use]
    pub const fn cpu() -> Self {
        Self {
            supports_fp16: false,
            supports_bf16: false,
            supports_fp8: false,
            tensor_cores: false,
            peer_access: false,
            unified_memory: true, // host memory is trivially "unified"
            cluster_launch: false,
            async_copy: false,
            max_threads_per_block: 1024,
            max_shared_mem_per_block: 48 * 1024,
            warp_size: 1,
        }
    }

    /// Returns `true` if **any** reduced-precision compute path is available.
    #[must_use]
    pub const fn supports_mixed_precision(&self) -> bool {
        self.supports_fp16 || self.supports_bf16 || self.supports_fp8
    }

    /// Returns `true` if a GEMM of the given shape can plausibly use a
    /// Tensor-Core path: matrix units present and all three dimensions are
    /// multiples of the smallest standard WMMA tile (16).
    #[must_use]
    pub const fn can_use_tensor_cores(&self, m: usize, n: usize, k: usize) -> bool {
        self.tensor_cores && m % 16 == 0 && n % 16 == 0 && k % 16 == 0
    }
}

impl Default for Capabilities {
    fn default() -> Self {
        Self::cpu()
    }
}

impl fmt::Display for Capabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "fp16={} bf16={} fp8={} tensor_cores={} peer={} unified={} cluster={} async_copy={} \
             max_threads/block={} max_smem/block={}B warp={}",
            self.supports_fp16,
            self.supports_bf16,
            self.supports_fp8,
            self.tensor_cores,
            self.peer_access,
            self.unified_memory,
            self.cluster_launch,
            self.async_copy,
            self.max_threads_per_block,
            self.max_shared_mem_per_block,
            self.warp_size,
        )
    }
}

/// A GEMM tile-shape hint `(tile_m, tile_n, tile_k)` returned by
/// [`ComputeBackend::recommended_tile_for`](crate::ComputeBackend::recommended_tile_for).
///
/// Autotuners use this as a starting point before searching nearby shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileShape {
    /// Tile rows (M dimension).
    pub tile_m: usize,
    /// Tile columns (N dimension).
    pub tile_n: usize,
    /// Tile depth (K dimension / inner accumulation).
    pub tile_k: usize,
}

impl TileShape {
    /// Construct a tile shape.
    #[must_use]
    pub const fn new(tile_m: usize, tile_n: usize, tile_k: usize) -> Self {
        Self {
            tile_m,
            tile_n,
            tile_k,
        }
    }

    /// Number of output elements in one tile (`tile_m * tile_n`).
    #[must_use]
    pub const fn output_elems(&self) -> usize {
        self.tile_m * self.tile_n
    }
}

impl fmt::Display for TileShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}x{}", self.tile_m, self.tile_n, self.tile_k)
    }
}

/// Default tile heuristic shared by the trait's
/// [`recommended_tile_for`](crate::ComputeBackend::recommended_tile_for):
/// small problems get a small tile (less wasted compute on the fringe),
/// large problems get the canonical 128×128×8 tile, mid-size gets 64×64×16.
///
/// `caps` lets the heuristic snap to a Tensor-Core-friendly tile when the
/// backend reports matrix units.
#[must_use]
pub fn default_tile_for(m: usize, n: usize, k: usize, caps: &Capabilities) -> TileShape {
    let smallest = m.min(n).min(k);
    if caps.tensor_cores && m % 16 == 0 && n % 16 == 0 && k % 16 == 0 {
        // WMMA-aligned: prefer wide tiles that are multiples of 16.
        return if smallest >= 512 {
            TileShape::new(128, 128, 32)
        } else {
            TileShape::new(64, 64, 16)
        };
    }
    if smallest <= 64 {
        TileShape::new(16, 16, 16)
    } else if smallest <= 512 {
        TileShape::new(64, 64, 16)
    } else {
        TileShape::new(128, 128, 8)
    }
}

/// Memory-type classification, abstracting each backend's notion of where
/// a buffer physically lives. Used by [`DeviceInfo`] and cross-backend
/// transfer planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    /// Dedicated device (VRAM) memory, not host-addressable.
    Device,
    /// Pinned/page-locked host memory.
    HostPinned,
    /// Unified / managed memory addressable from host and device.
    Unified,
    /// Plain pageable host memory (the CPU reference backend).
    Host,
}

impl fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device => write!(f, "device"),
            Self::HostPinned => write!(f, "host-pinned"),
            Self::Unified => write!(f, "unified"),
            Self::Host => write!(f, "host"),
        }
    }
}

/// A single device exposed by a backend, in a backend-agnostic shape.
///
/// `compute_capability` is interpreted per backend: `(major, minor)` for
/// CUDA, the GCN/RDNA arch level for ROCm, the shader-model for Vulkan/
/// Metal/WebGPU, etc. A value of `(0, 0)` means "not applicable" (e.g. the
/// CPU reference device).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Backend-local device ordinal (0-based).
    pub ordinal: usize,
    /// Human-readable device name.
    pub name: String,
    /// `(major, minor)` capability/shader-model, or `(0, 0)` if N/A.
    pub compute_capability: (u32, u32),
    /// Total memory available on the device in bytes.
    pub total_memory_bytes: u64,
    /// Predominant memory kind for buffers on this device.
    pub memory_kind: MemoryKind,
    /// Per-device capability report.
    pub capabilities: Capabilities,
}

impl DeviceInfo {
    /// Build the canonical descriptor for the host CPU reference device.
    #[must_use]
    pub fn cpu_reference(total_memory_bytes: u64) -> Self {
        Self {
            ordinal: 0,
            name: "CPU (reference)".to_string(),
            compute_capability: (0, 0),
            total_memory_bytes,
            memory_kind: MemoryKind::Host,
            capabilities: Capabilities::cpu(),
        }
    }
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mem_mb = self.total_memory_bytes / (1024 * 1024);
        let (major, minor) = self.compute_capability;
        write!(
            f,
            "Device[{}] {} (cc {major}.{minor}, {mem_mb} MB, {})",
            self.ordinal, self.name, self.memory_kind,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_capabilities_are_conservative() {
        let caps = Capabilities::cpu();
        assert!(!caps.supports_fp16);
        assert!(!caps.supports_bf16);
        assert!(!caps.supports_fp8);
        assert!(!caps.tensor_cores);
        assert!(!caps.peer_access);
        assert!(caps.unified_memory);
        assert!(!caps.supports_mixed_precision());
        assert_eq!(Capabilities::default(), Capabilities::cpu());
    }

    #[test]
    fn mixed_precision_detection() {
        let mut caps = Capabilities::cpu();
        assert!(!caps.supports_mixed_precision());
        caps.supports_bf16 = true;
        assert!(caps.supports_mixed_precision());
    }

    #[test]
    fn tensor_core_eligibility_requires_alignment_and_units() {
        let mut caps = Capabilities::cpu();
        // No units: never eligible regardless of shape.
        assert!(!caps.can_use_tensor_cores(256, 256, 256));
        caps.tensor_cores = true;
        // Aligned shape: eligible.
        assert!(caps.can_use_tensor_cores(256, 256, 256));
        // Misaligned K: not eligible.
        assert!(!caps.can_use_tensor_cores(256, 256, 255));
    }

    #[test]
    fn tile_heuristic_scales_with_problem_size() {
        let caps = Capabilities::cpu();
        assert_eq!(
            default_tile_for(32, 32, 32, &caps),
            TileShape::new(16, 16, 16)
        );
        assert_eq!(
            default_tile_for(256, 256, 256, &caps),
            TileShape::new(64, 64, 16)
        );
        assert_eq!(
            default_tile_for(4096, 4096, 4096, &caps),
            TileShape::new(128, 128, 8)
        );
    }

    #[test]
    fn tile_heuristic_snaps_to_wmma_when_tensor_cores() {
        let mut caps = Capabilities::cpu();
        caps.tensor_cores = true;
        // Large aligned problem → wide WMMA tile.
        let t = default_tile_for(1024, 1024, 1024, &caps);
        assert_eq!(t, TileShape::new(128, 128, 32));
        assert_eq!(t.tile_m % 16, 0);
        assert_eq!(t.tile_n % 16, 0);
        assert_eq!(t.tile_k % 16, 0);
        // Small aligned problem → smaller aligned tile.
        assert_eq!(
            default_tile_for(64, 64, 64, &caps),
            TileShape::new(64, 64, 16)
        );
    }

    #[test]
    fn tile_output_elems() {
        assert_eq!(TileShape::new(128, 64, 8).output_elems(), 128 * 64);
    }

    #[test]
    fn memory_kind_display() {
        assert_eq!(MemoryKind::Device.to_string(), "device");
        assert_eq!(MemoryKind::HostPinned.to_string(), "host-pinned");
        assert_eq!(MemoryKind::Unified.to_string(), "unified");
        assert_eq!(MemoryKind::Host.to_string(), "host");
    }

    #[test]
    fn device_info_cpu_reference() {
        let dev = DeviceInfo::cpu_reference(8 * 1024 * 1024 * 1024);
        assert_eq!(dev.ordinal, 0);
        assert_eq!(dev.compute_capability, (0, 0));
        assert_eq!(dev.memory_kind, MemoryKind::Host);
        assert_eq!(dev.capabilities, Capabilities::cpu());
        assert!(dev.to_string().contains("CPU (reference)"));
        assert!(dev.to_string().contains("8192 MB"));
    }

    #[test]
    fn capabilities_display_round_trips_fields() {
        let s = Capabilities::cpu().to_string();
        assert!(s.contains("unified=true"));
        assert!(s.contains("tensor_cores=false"));
        assert!(s.contains("warp=1"));
    }
}
