//! Compressed-sensing measurement (sensing) matrices and RIP estimator.

pub mod bernoulli_matrix;
pub mod gaussian_matrix;
pub mod partial_fourier;
pub mod rip_estimator;

pub use bernoulli_matrix::bernoulli_matrix;
pub use gaussian_matrix::gaussian_matrix;
pub use partial_fourier::partial_fourier;
pub use rip_estimator::rip_estimator;
