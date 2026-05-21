//! Spectral methods: Chebyshev collocation and FFT-based pseudo-spectral.

pub mod chebyshev;
pub mod fft_spectral;
pub mod fourier_2d;

pub use chebyshev::{cheb_diff_matrix, cheb_nodes, solve_poisson_chebyshev};
pub use fft_spectral::{periodic_diff2, periodic_poisson_solve};
pub use fourier_2d::{Fourier2dConfig, solve_poisson_2d_fft};
