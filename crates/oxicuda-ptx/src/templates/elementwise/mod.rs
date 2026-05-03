//! Elementwise GPU operation templates.
//!
//! This module generates complete PTX kernels for unary and binary elementwise
//! operations over device arrays. It supports basic arithmetic (`add`, `sub`, `mul`, `div`),
//! activation functions (`ReLU`, GELU, sigmoid, `SiLU`, tanh), unary math (neg, abs,
//! sqrt, rsqrt, exp, log), and fused operations (fused-add-relu, fused-scale-add).
//!
//! Each template produces a kernel that:
//! 1. Computes a global thread index
//! 2. Performs a bounds check against the array length
//! 3. Loads input element(s)
//! 4. Applies the operation
//! 5. Stores the result
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::elementwise::{ElementwiseTemplate, ElementwiseOp};
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = ElementwiseTemplate::new(
//!     ElementwiseOp::Add,
//!     PtxType::F32,
//!     SmVersion::Sm80,
//! );
//! let ptx = template.generate().expect("PTX generation failed");
//! assert!(ptx.contains("add.f32"));
//! ```
//!
//! 🤖 Refactored with [SplitRS](https://github.com/cool-japan/splitrs)

mod elementwisetemplate_impl;
mod elementwisetemplate_impl_1;
mod elementwisetemplate_type;
mod functions;
mod types;

pub use elementwisetemplate_type::ElementwiseTemplate;
pub use types::ElementwiseOp;
