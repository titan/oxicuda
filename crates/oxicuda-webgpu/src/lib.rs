//! OxiCUDA WebGPU backend — cross-platform GPU compute via `wgpu` and WGSL.
//!
//! Provides a [`WebGpuBackend`] that implements the [`oxicuda_backend::ComputeBackend`] trait
//! from `oxicuda-backend`, using `wgpu` to target Vulkan, Metal, Direct3D 12,
//! and the browser WebGPU API from a single Rust crate.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use oxicuda_webgpu::WebGpuBackend;
//! use oxicuda_backend::ComputeBackend;
//!
//! let mut backend = WebGpuBackend::new();
//! backend.init().expect("WebGPU init failed");
//!
//! // Allocate 1 KiB on the GPU.
//! let ptr = backend.alloc(1024).expect("alloc failed");
//! backend.free(ptr).expect("free failed");
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────┐
//! │          WebGpuBackend                   │
//! │  (implements ComputeBackend)             │
//! └──────────────┬───────────────────────────┘
//!                │
//!       ┌────────▼────────┐
//!       │  WebGpuDevice   │  ← wgpu Instance + Adapter + Device + Queue
//!       └────────┬────────┘
//!                │
//!    ┌───────────▼────────────┐
//!    │  WebGpuMemoryManager   │  ← buffer pool (u64 handle → wgpu::Buffer)
//!    └────────────────────────┘
//! ```
//!
//! # WGSL Shader Generation
//!
//! The [`shader`] module provides helpers that produce WGSL source strings for
//! common kernels (GEMM, element-wise ops, reductions).  The [`shader_ext`]
//! module adds transpose, row-wise softmax, Blelloch prefix scan, layer
//! normalisation, warp-style subgroup reductions, and emulated-f64 arithmetic,
//! while the [`fft`] module provides a radix-2 Cooley-Tukey FFT plan
//! ([`WgslFftPlan`]) plus its butterfly shaders.
//!
//! # Dispatch Planning
//!
//! The [`planner`] module resolves workgroup sizes and dispatch grids from
//! adapter [`Limits`] (auto-tuning, 2-D dispatch folding, 256-byte texture row
//! alignment) — pure host-side arithmetic requiring no GPU.

pub mod backend;
pub mod device;
pub mod error;
pub mod fft;
pub mod memory;
pub mod planner;
pub mod shader;
pub mod shader_ext;

// WASM target support — compiled on wasm32 or when the `wasm` feature is
// enabled (so that native tests can exercise the module).
#[cfg(any(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm;

pub use backend::WebGpuBackend;
pub use error::{WebGpuError, WebGpuResult};
pub use fft::{FftDirection, WgslFftPlan};
pub use planner::{DispatchGrid, Limits, Workgroup1D, Workgroup2D};
pub use shader_ext::ScanKind;
