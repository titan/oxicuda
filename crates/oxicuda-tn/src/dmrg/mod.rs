//! Density Matrix Renormalisation Group.
//!
//! Five flavours are provided:
//! * [`dmrg_two_site`] — classic two-site DMRG with SVD truncation.
//! * [`single_site_dmrg`] — single-site DMRG with subspace-expansion noise
//!   (Hubig et al. 2015), cheaper per sweep but requires explicit noise to
//!   avoid local minima.
//! * [`block_lanczos`] — block-Lanczos eigensolver (Golub-Underwood 1977 /
//!   Wu-Simon 2000) for simultaneously finding the `n_target` lowest
//!   eigenpairs; essential when the ground state is degenerate.
//! * [`idmrg()`] — Infinite DMRG (McCulloch 2008) for translationally-invariant
//!   1D ground states, growing the system one unit cell at a time.
//! * [`finite_t`] — Finite-temperature DMRG via purification (Feiguin & White
//!   2005, White 2009): imaginary-time TEBD in a doubled Hilbert space.

pub mod dmrg;
pub mod finite_t;
pub mod idmrg;
pub mod lanczos;
pub mod lanczos_block;
pub mod single_site;
pub mod two_site_excited;

pub use dmrg::{DmrgConfig, DmrgResult, dmrg_two_site};
pub use finite_t::{
    FiniteTConfig, FiniteTResult, finite_t_expectation, finite_t_run, heisenberg_gate_doubled,
    purification_init, trotter_sweep_doubled,
};
pub use idmrg::{
    IDmrgConfig, IDmrgResult, build_heisenberg_mpo_unit_cell, build_zero_mpo_unit_cell, idmrg,
};
pub use lanczos::{LanczosResult, lanczos_smallest};
pub use lanczos_block::{BlockLanczosConfig, BlockLanczosResult, block_lanczos};
pub use single_site::{SingleSiteDmrgConfig, SingleSiteDmrgResult, single_site_dmrg};
pub use two_site_excited::{
    ExcitedDmrgConfig, ExcitedDmrgResult, mps_inner_product, two_site_excited_dmrg,
};
