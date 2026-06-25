//! HDC operations: binding, bundling, and permutation.

pub mod binding;
pub mod bundling;
pub mod mbat;
pub mod permutation;
pub mod permutation_bind;
pub mod tensor_product;
pub mod thinning;

pub use mbat::{
    is_perfect_square, mbat_bind, mbat_unbind_transpose, random_orthogonal_matrix, vtb_bind,
    vtb_unbind,
};
pub use permutation_bind::{PermutationRole, bind_superpose_i32, recover_i32};
pub use tensor_product::{tensor_product_bind, tensor_product_unbind, tpr_bundle, tpr_encode};
pub use thinning::{and_binary, context_dependent_thinning, density, or_binary};
