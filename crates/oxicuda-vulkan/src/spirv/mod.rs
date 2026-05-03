//! SPIR-V compute shader generators for the Vulkan backend.
//!
//! This module provides:
//! - A lightweight [`SpvModule`] builder for emitting valid SPIR-V binaries.
//! - Generator functions for **unary**, **binary**, **reduce**, and **GEMM**
//!   compute shaders used by the backend dispatch methods.
//! - The original [`trivial_compute_shader`] placeholder used for validation.
//!
//! All generated shaders operate on `f32` buffers via SSBO (`StorageBuffer`)
//! bindings at descriptor set 0.  Parameters (element count, matrix dims, etc.)
//! are passed through an additional SSBO.
//!
//! The generated SPIR-V targets version 1.3 (supported by Vulkan 1.1+).

mod attention;
mod batched_gemm;
mod binary;
mod builder;
mod consts;
mod conv2d;
mod gemm;
mod preamble;
mod reduce;
mod subgroup;
mod trivial;
mod unary;

#[cfg(test)]
mod tests;

// ─── Public API ──────────────────────────────────────────────

pub use attention::attention_spirv;
pub use batched_gemm::batched_gemm_compute_shader;
pub use binary::binary_compute_shader;
pub use builder::SpvModule;
pub use consts::{SPIRV_GENERATOR, SPIRV_MAGIC, SPIRV_VERSION_1_2, SPIRV_VERSION_1_3};
pub use conv2d::conv2d_spirv;
pub use gemm::gemm_compute_shader;
pub use reduce::reduce_compute_shader;
pub use subgroup::{reduction_subgroup_spirv, scan_subgroup_spirv};
pub use trivial::{trivial_compute_shader, trivial_compute_shader_bytes};
pub use unary::unary_compute_shader;
