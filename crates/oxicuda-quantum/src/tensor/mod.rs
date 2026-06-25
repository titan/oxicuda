//! Tensor-network contraction back-end.
//!
//! See [`contraction`] for the [`Tensor`] type with named legs, the
//! [`TensorNetwork`] greedy contractor, and the [`amplitude`] evaluator that
//! computes `⟨bitstring| U |0…0⟩` for shallow wide circuits without forming the
//! full `2^n` state vector.

pub mod contraction;

pub use contraction::{Tensor, TensorNetwork, TnGate, amplitude};
