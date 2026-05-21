//! Tensor-network diagnostics: bond dimensions, entanglement entropy, Schmidt
//! spectrum, fidelity, Loschmidt echo, and dynamic structure factor.

pub mod loschmidt;
pub mod metrics;

pub use loschmidt::{
    LoschmidtConfig, LoschmidtResult, ReturnProbResult, StructureFactorConfig,
    StructureFactorResult, SzOperator, loschmidt_echo, mpo_expectation_value, mps_inner_product,
    operator_matrix, return_probability, static_structure_factor,
};
pub use metrics::{
    bond_dimension, entanglement_entropy, fidelity, max_bond_dimension, mps_overlap,
    schmidt_spectrum,
};
