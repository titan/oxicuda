//! Quantum linear-systems algorithms.
//!
//! * [`hhl`] — the Harrow–Hassidim–Lloyd algorithm for `A x = b` with Hermitian
//!   `A` (Harrow, Hassidim, Lloyd 2009).

pub mod hhl;

pub use hhl::{HermitianMatrix, HhlConfig, HhlResult, hhl_solve};
