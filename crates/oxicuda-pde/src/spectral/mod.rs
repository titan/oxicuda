//! Spectral methods: Chebyshev collocation and FFT-based pseudo-spectral.

pub mod chebyshev;
pub mod fft_spectral;

pub use chebyshev::{cheb_diff_matrix, cheb_nodes, solve_poisson_chebyshev};
pub use fft_spectral::{periodic_diff2, periodic_poisson_solve};
