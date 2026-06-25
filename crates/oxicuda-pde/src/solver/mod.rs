//! Iterative and direct linear solvers for sparse systems.

pub mod bicgstab;
pub mod cg;
pub mod eigensolver;
pub mod gmres;
pub mod ilu0;
pub mod jacobi;
pub mod multigrid_pcg;
pub mod pcg;
pub mod preconditioner_amg;
pub mod saddle_point;
pub mod sparse;
pub mod ssor;

pub use bicgstab::{BicgstabConfig, BicgstabResult, bicgstab};
pub use cg::cg_solve;
pub use eigensolver::{EigenPair, LanczosConfig, LanczosResult, Which, lanczos, lanczos_csr};
pub use gmres::{GmresConfig, GmresResult, gmres};
pub use ilu0::{ilu0_factor, ilu0_solve};
pub use jacobi::jacobi_solve;
pub use multigrid_pcg::{
    GeometricMgPreconditioner, MgPcgConfig, MgPcgResult, mg_pcg, poisson_1d_interior_csr,
};
pub use pcg::{pcg_ilu0, pcg_jacobi, pcg_ssor};
pub use preconditioner_amg::{AmgPcgConfig, AmgPcgResult, AmgPreconditioner, amg_pcg};
pub use saddle_point::{MinresConfig, SaddleResult, UzawaConfig, minres, uzawa};
pub use sparse::SparseCsr;
pub use ssor::ssor_apply;
