//! Function approximation methods.
//!
//! Currently provides Padé rational approximation ([`pade`]) built from a Taylor
//! series, complementing the polynomial and spline approximants in [`crate::interp`].

pub mod pade;

pub use pade::PadeApprox;
