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

mod atomic_float;
mod attention;
mod batched_gemm;
mod binary;
mod builder;
mod consts;
mod conv2d;
mod cooperative_matrix;
mod gemm;
mod performance_query;
mod preamble;
mod reduce;
mod subgroup;
mod subgroup_size_control;
mod trivial;
mod unary;
mod vulkan_memory_model;

#[cfg(test)]
mod tests;

// ─── Public API ──────────────────────────────────────────────

pub use atomic_float::{AtomicFloatOp, atomic_float_reduce_spirv};
pub use attention::attention_spirv;
pub use batched_gemm::batched_gemm_compute_shader;
pub use binary::binary_compute_shader;
pub use builder::SpvModule;
pub use consts::{
    SPIRV_GENERATOR, SPIRV_MAGIC, SPIRV_VERSION_1_2, SPIRV_VERSION_1_3, SPIRV_VERSION_1_4,
    SPIRV_VERSION_1_5, SPIRV_VERSION_1_6,
};
pub use conv2d::conv2d_spirv;
pub use cooperative_matrix::{CoopMatTile, CoopMatType, cooperative_matrix_gemm_spirv};
pub use gemm::gemm_compute_shader;
pub use performance_query::{
    CounterDesc, CounterResult, CounterScope, CounterStorage, CounterUnit, PerformanceQueryPool,
};
pub use reduce::reduce_compute_shader;
pub use subgroup::{reduction_subgroup_spirv, scan_subgroup_spirv};
pub use subgroup_size_control::{
    SubgroupSizeChoice, SubgroupSizeController, SubgroupVendor, subgroup_size_aware_reduce_spirv,
};
pub use trivial::{trivial_compute_shader, trivial_compute_shader_bytes};
pub use unary::unary_compute_shader;
pub use vulkan_memory_model::{MemScope, VulkanMemModel, vulkan_memory_model_copy_spirv};
