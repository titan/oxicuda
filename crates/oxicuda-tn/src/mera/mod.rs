//! Multi-scale Entanglement Renormalisation Ansatz (MERA).
//!
//! Binary-MERA layer primitives: disentangler + isometry, and the ascending /
//! descending super-operators that implement one entanglement-renormalisation
//! RG step on a 1D lattice.

pub mod mera;

pub use mera::MeraLayer;
