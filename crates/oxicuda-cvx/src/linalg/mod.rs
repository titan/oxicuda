//! Private linear-algebra helpers.

pub mod cg;
pub mod cholesky;
pub mod matvec;
pub mod qr;
pub mod solve;

pub use cg::cg_solve;
pub use cholesky::{cholesky_factor, cholesky_solve};
pub use matvec::{add_scaled, axpy, dot, gemv, mat_t_mat, mat_t_vec, mat_vec, norm2, scale};
pub use qr::{householder_qr, qr_solve};
pub use solve::{lu_decompose, lu_solve, solve_dense, solve_lower, solve_upper};
