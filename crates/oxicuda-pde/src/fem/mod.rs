//! Finite-element method (P1 linear triangles).

pub mod dirichlet_apply;
pub mod mass_stiffness;
pub mod p1_triangle;

pub use dirichlet_apply::apply_dirichlet_csr;
pub use mass_stiffness::{FemAssembly, assemble_mass_stiffness};
pub use p1_triangle::{p1_local_mass, p1_local_stiffness};
