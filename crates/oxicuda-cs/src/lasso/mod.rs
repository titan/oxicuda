//! LASSO and its variants: coordinate descent, LARS, FISTA, group/fused/elastic-net, SLOPE, Dantzig.

pub mod coord_descent;
pub mod dantzig;
pub mod elastic_net;
pub mod fista_lasso;
pub mod fused_lasso;
pub mod group_lasso;
pub mod lars;
pub mod slope;

pub use coord_descent::{coord_descent_lasso, lasso_path};
pub use dantzig::{DantzigConfig, dantzig_selector as dantzig_selector_lasso};
pub use elastic_net::elastic_net;
pub use fista_lasso::fista_lasso;
pub use fused_lasso::fused_lasso;
pub use group_lasso::group_lasso;
pub use lars::{LarsPath, lars};
pub use slope::{Slope, SlopeConfig, sorted_l1_prox};
