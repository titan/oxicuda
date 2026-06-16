//! Compressed-sensing measurement (sensing) matrices and RIP estimator.

pub mod bernoulli_matrix;
pub mod coded_diffraction;
pub mod gaussian_matrix;
pub mod partial_fourier;
pub mod rip_estimator;

pub use bernoulli_matrix::bernoulli_matrix;
pub use coded_diffraction::{CodedDiffraction, MaskKind, WirtingerConfig, phase_aligned_error};
pub use gaussian_matrix::gaussian_matrix;
pub use partial_fourier::partial_fourier;
pub use rip_estimator::rip_estimator;
