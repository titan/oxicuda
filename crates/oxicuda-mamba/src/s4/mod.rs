//! S4 (Structured State Space Sequence) model implementation.
//!
//! The S4 model is a sequence-to-sequence layer built on the structured
//! State Space Model (SSM) framework.  Its key innovations are:
//!
//! 1. **HiPPO-LegS initialization** — The continuous-time A matrix is
//!    initialized via the Legendre polynomial projection theory, providing
//!    theoretically grounded memory of the input history.
//!
//! 2. **DPLR parameterization** — The A matrix is expressed as a Diagonal
//!    Plus Low Rank (DPLR) update, enabling efficient computation of the
//!    SSM convolution kernel via the Cauchy kernel identity.
//!
//! 3. **Convolutional mode** — During training (fixed `Δ`), the SSM is
//!    equivalent to a (non-circular) convolution with a structured kernel.
//!    This makes the forward pass `O(L log L)` with FFT (here implemented
//!    as `O(L² )` for correctness / portability).
//!
//! ## Modules
//!
//! - [`hippo`] — HiPPO-LegS A/B matrices and NPLR decomposition.
//! - [`dplr`]  — Diagonal Plus Low Rank parameterization and SSM kernel.
//! - [`s4_layer`] — Full S4 sequence layer (multi-channel, optional bidirectional).
//! - [`s4_fft`] — FFT-based `O(L log L)` long convolution (radix-2 Cooley-Tukey).

pub mod dplr;
pub mod hippo;
pub mod s4_fft;
pub mod s4_layer;

pub use s4_fft::{fft, fft_conv1d, s4_fft_conv};
