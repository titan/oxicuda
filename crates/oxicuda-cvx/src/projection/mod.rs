//! Projections onto common convex sets.

pub mod box_proj;
pub mod dykstra_pocs;
pub mod halfspace;
pub mod l1_ball;
pub mod l2_ball;
pub mod psd_cone;
pub mod simplex;
pub mod soc_cone;

pub use box_proj::project_box;
pub use dykstra_pocs::{DykstraResult, dykstra_pocs};
pub use halfspace::project_halfspace;
pub use l1_ball::project_l1_ball;
pub use l2_ball::project_l2_ball;
pub use psd_cone::project_psd_cone;
pub use simplex::project_simplex;
pub use soc_cone::project_soc;
