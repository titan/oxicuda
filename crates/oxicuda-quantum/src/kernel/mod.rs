//! Quantum kernels for quantum machine learning.
//!
//! * [`quantum_kernel`] — the global overlap (fidelity) kernel
//!   `k(x,y)=|⟨ψ(x)|ψ(y)⟩|²`.
//! * [`projected`] — projected quantum kernels (PQK) over single-qubit reduced
//!   observables, avoiding the exponential concentration of the fidelity kernel.
//! * [`trainable`] — trainable quantum embedding kernels optimized by
//!   kernel-target alignment.

pub mod projected;
pub mod quantum_kernel;
pub mod trainable;

pub use projected::{
    PqkEmbedding, ProjectedKernelConfig, local_pauli_features, projected_kernel,
    projected_kernel_matrix,
};
pub use trainable::{TrainableKernel, TrainableKernelConfig};
