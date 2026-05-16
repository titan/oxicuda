//! LASSO and its variants: coordinate descent, LARS, FISTA, group/fused/elastic-net.

pub mod coord_descent;
pub mod elastic_net;
pub mod fista_lasso;
pub mod fused_lasso;
pub mod group_lasso;
pub mod lars;

pub use coord_descent::{coord_descent_lasso, lasso_path};
pub use elastic_net::elastic_net;
pub use fista_lasso::fista_lasso;
pub use fused_lasso::fused_lasso;
pub use group_lasso::group_lasso;
pub use lars::{LarsPath, lars};
