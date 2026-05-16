//! Iterative and direct linear solvers for sparse systems.

pub mod cg;
pub mod ilu0;
pub mod jacobi;
pub mod pcg;
pub mod sparse;
pub mod ssor;

pub use cg::cg_solve;
pub use ilu0::{ilu0_factor, ilu0_solve};
pub use jacobi::jacobi_solve;
pub use pcg::{pcg_ilu0, pcg_jacobi, pcg_ssor};
pub use sparse::SparseCsr;
pub use ssor::ssor_apply;
