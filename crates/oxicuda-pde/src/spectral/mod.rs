//! Spectral methods: Chebyshev collocation, FFT-based pseudo-spectral, and the
//! Gauss–Lobatto–Legendre nodal spectral-element method.

pub mod chebyshev;
pub mod chebyshev_2d;
pub mod fft_spectral;
pub mod fourier_2d;
pub mod fourier_3d;
pub mod spectral_element;

pub use chebyshev::{cheb_diff_matrix, cheb_nodes, solve_poisson_chebyshev};
pub use chebyshev_2d::{Rectangle, cheb_nodes_mapped, chebyshev_2d_grid, chebyshev_2d_poisson};
pub use fft_spectral::{periodic_diff2, periodic_poisson_solve};
pub use fourier_2d::{Fourier2dConfig, solve_poisson_2d_fft};
pub use fourier_3d::{Fourier3dConfig, neg_laplacian_3d_spectral, solve_poisson_3d_fft};
pub use spectral_element::{GllBasis, SpectralElementMesh1d, gll_nodes, gll_weights};
