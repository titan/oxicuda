//! Geometric multigrid V-cycle (1D and 2D).

pub mod restrict_prolong;
pub mod smoother;
pub mod vcycle;

pub use restrict_prolong::{prolong_1d, prolong_2d, restrict_1d, restrict_2d};
pub use smoother::weighted_jacobi_smooth;
pub use vcycle::v_cycle_1d;
