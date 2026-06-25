//! OxiCUDA ROCm backend — GPU compute via AMD HIP runtime.
//!
//! # Platform Support
//!
//! | Platform | Status |
//! |----------|--------|
//! | Linux (AMD GPU) | Full support via libamdhip64.so |
//! | Windows | Not supported (`UnsupportedPlatform`) |
//! | macOS | Not supported (`UnsupportedPlatform`) |

pub mod backend;
pub mod device;
pub mod error;
pub mod flat_workgroup;
pub mod gfx_arch;
pub mod hip_graph;
pub mod hip_kernels;
pub mod hip_kernels_advanced;
pub mod hipblas;
pub mod hipblaslt;
pub mod hiprtc;
pub mod launch_config;
pub mod mem_pool;
pub mod memory;
pub mod mfma;
pub mod multi_device;
pub mod occupancy;
pub mod peer;
pub mod stream;

pub use backend::RocmBackend;
pub use error::{RocmError, RocmResult};
pub use flat_workgroup::FlatWorkgroupHint;
pub use gfx_arch::{ArchFamily, GfxArch};
pub use hip_graph::{ExecutableGraph, HipGraph, NodeKind};
pub use hipblaslt::{Epilogue, HipBlasLt, MatmulDesc, MatrixLayout};
pub use launch_config::{Dim3, LaunchConfig};
pub use mem_pool::{MemPoolStats, MemoryPool};
pub use mfma::{MatrixCoreOp, MatrixDtype};
pub use occupancy::{KernelResources, Occupancy, OccupancyCalculator};
pub use peer::{LinkKind, PeerTopology};
pub use stream::{MemcpyKind, StreamCommand, StreamPlan};
