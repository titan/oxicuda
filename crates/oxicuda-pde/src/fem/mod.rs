//! Finite-element method (P1 linear and P2 quadratic triangles).

pub mod dirichlet_apply;
pub mod mass_stiffness;
pub mod p1_tet;
pub mod p1_triangle;
pub mod p2_triangle;
pub mod quadrilateral;

pub use dirichlet_apply::apply_dirichlet_csr;
pub use mass_stiffness::{FemAssembly, assemble_mass_stiffness};
pub use p1_tet::{
    P1_TET_N_DOFS, p1_tet_local_load, p1_tet_local_mass, p1_tet_local_stiffness,
    p1_tet_matrix_is_symmetric, p1_tet_shape_grad, tet_volume,
};
pub use p1_triangle::{p1_local_mass, p1_local_stiffness};
pub use p2_triangle::{
    P2_N_DOFS, gauss7, p2_jacobian, p2_local_load, p2_local_mass, p2_local_stiffness,
    p2_node_coords, p2_partition_of_unity, p2_shape_fn, p2_shape_grad, p2_stiffness_is_symmetric,
};
pub use quadrilateral::{
    Q1_N_DOFS, gauss2x2, q1_jacobian, q1_local_load, q1_local_mass, q1_local_stiffness,
    q1_matrix_is_symmetric, q1_partition_of_unity, q1_shape_fn, q1_shape_grad,
};
