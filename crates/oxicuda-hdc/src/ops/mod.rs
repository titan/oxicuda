//! HDC operations: binding, bundling, and permutation.

pub mod binding;
pub mod bundling;
pub mod permutation;
pub mod tensor_product;

pub use tensor_product::{tensor_product_bind, tensor_product_unbind, tpr_bundle, tpr_encode};
