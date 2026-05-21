//! Generic tensor contraction.

pub mod einsum;
pub mod network_simplify;
pub mod path;
pub mod path_optimal;

pub use einsum::{LabelledTensor, einsum_binary};
pub use network_simplify::{
    NetworkTensor, SimplifyStats, TensorNetwork, absorb_leaves, contract_network, fold_scalars,
    fuse_parallel_bonds, gauge_fix_bonds, remove_traces, simplify_chains, simplify_network,
};
pub use path::{ContractionPath, greedy_path};
pub use path_optimal::{
    ContractionPathConfig, DpEntry, OptimalPath, TensorSpec, build_index_dims, compare_with_greedy,
    contraction_flops, contraction_result_indices, greedy_flops, optimal_contraction_path,
};
