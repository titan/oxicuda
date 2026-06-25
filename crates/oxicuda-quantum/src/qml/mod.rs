//! Quantum machine-learning models.
//!
//! * [`qcnn`] — translation-invariant Quantum Convolutional Neural Network
//!   classifier (Cong-Choi-Lukin 2019).
//! * [`qgan`] — quantum generative model / qGAN-style distribution loader trained
//!   on the maximum mean discrepancy (Zoufal 2019, Liu-Wang Born machine 2018).

pub mod qcnn;
pub mod qgan;

pub use qcnn::{CONV_PARAMS, POOL_PARAMS, Qcnn};
pub use qgan::QuantumGenerator;
