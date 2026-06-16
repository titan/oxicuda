//! Quantum Fourier Transform (QFT) and Quantum Phase Estimation (QPE).
pub mod qft;
pub mod qpe;
pub use qft::{qft_inplace, qft_inverse_inplace};
pub use qpe::{PhaseEstimationResult, phase_estimation};
